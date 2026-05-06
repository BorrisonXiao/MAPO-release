import json
import os
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

from swift.rewards.orm import ORM, orms

_ANSWER_PATTERN = re.compile(r'<answer>(.*?)</answer>', re.DOTALL | re.IGNORECASE)
_FORMAT_PATTERN = re.compile(
    r'^\s*<reasoning>.*?</reasoning>\s*<answer>.*?</answer>\s*$',
    re.DOTALL | re.IGNORECASE,
)
_THINK_BLOCK_PATTERN = re.compile(r'<think>.*?</think>', re.DOTALL | re.IGNORECASE)
_LABELLED_ANSWER_PATTERN = re.compile(r'^(?:final\s+answer|answer)\s*[:\-]\s*(.+)$', re.IGNORECASE)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {'1', 'true', 't', 'yes', 'y', 'on'}


def _normalize_whitespace(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def _clean_answer_candidate(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    compact = _normalize_whitespace(text)
    if not compact:
        return None
    compact = compact.strip('`*_').strip()
    if len(compact) >= 2 and compact[0] in {'"', "'"} and compact[-1] == compact[0]:
        compact = compact[1:-1].strip()
    compact = compact.rstrip('.,;:!?').strip()
    return compact or None


def _non_empty_lines(text: str) -> List[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _extract_tagged_answer(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    match = _ANSWER_PATTERN.search(text)
    if not match:
        return None
    return _clean_answer_candidate(match.group(1))


def _extract_line_answer(line: str) -> Optional[str]:
    if not isinstance(line, str):
        return None
    candidate = line.strip()
    if not candidate:
        return None
    labelled_match = _LABELLED_ANSWER_PATTERN.match(candidate)
    if labelled_match:
        candidate = labelled_match.group(1).strip()
    return _clean_answer_candidate(candidate)


def _is_plausible_final_answer(candidate: str) -> bool:
    if not isinstance(candidate, str):
        return False
    if '<' in candidate or '>' in candidate:
        return False
    if len(candidate) > 180:
        return False
    token_count = len(candidate.split())
    return 0 < token_count <= 24


def _extract_final_answer_from_tail(tail_text: str) -> Optional[str]:
    if not isinstance(tail_text, str):
        return None

    tagged = _extract_tagged_answer(tail_text)
    if tagged is not None:
        return tagged

    lines = _non_empty_lines(tail_text)
    if not lines:
        return None

    candidate = _extract_line_answer(lines[-1])
    if candidate is None or not _is_plausible_final_answer(candidate):
        return None
    return candidate


def _extract_post_think_answer(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    think_matches = list(_THINK_BLOCK_PATTERN.finditer(text))
    if not think_matches:
        return None
    tail = text[think_matches[-1].end():]
    return _extract_final_answer_from_tail(tail)


def _extract_plain_answer(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    lines = _non_empty_lines(text)
    if not lines:
        return None

    last_line = lines[-1]
    labelled_match = _LABELLED_ANSWER_PATTERN.match(last_line)
    if labelled_match:
        labelled_answer = _clean_answer_candidate(labelled_match.group(1))
        if labelled_answer is not None and _is_plausible_final_answer(labelled_answer):
            return labelled_answer

    if len(lines) == 1:
        single_line_answer = _extract_line_answer(last_line)
        if single_line_answer is not None and _is_plausible_final_answer(single_line_answer):
            return single_line_answer
    return None


def _extract_answer_with_source(
    text: str,
) -> Tuple[Optional[str], Optional[str]]:
    if not isinstance(text, str):
        return None, None
    extractors = (
        ('answer_tag', _extract_tagged_answer),
        ('post_think_tail', _extract_post_think_answer),
        ('plain_text', _extract_plain_answer),
    )
    for source, extractor in extractors:
        answer = extractor(text)
        if answer is not None:
            return answer, source
    return None, None


def _extract_answer(text: str) -> Optional[str]:
    answer, _ = _extract_answer_with_source(text)
    return answer


def _normalize_for_match(text: Optional[str]) -> Optional[str]:
    if not isinstance(text, str):
        return None
    candidate = _clean_answer_candidate(text)
    if candidate is None:
        return None
    candidate = re.sub(r'^(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)\s*', '', candidate, flags=re.IGNORECASE)
    candidate = _clean_answer_candidate(candidate)
    if candidate is None:
        return None
    return candidate.casefold()


def _matches_reasoning_answer_format(text: str) -> bool:
    return isinstance(text, str) and _FORMAT_PATTERN.match(text) is not None


def _matches_think_answer_format(text: str) -> bool:
    return _extract_post_think_answer(text) is not None


def _matches_supported_format(text: str) -> bool:
    return _matches_reasoning_answer_format(text) or _matches_think_answer_format(text)


# ── Counting-aware MSE reward utilities ──────────────────────────────────────

_COUNTING_KW_PATTERN = re.compile(
    r'how\s+many|how\s+much|how\s+often|how\s+frequent'
    r'|(?:what|what\'s)\s+(?:is\s+)?(?:the\s+)?(?:total\s+)?(?:number|count|amount)\s+of'
    r'|count\s+(?:the|how)',
    re.IGNORECASE,
)

# Splits on MCQ choice markers like "A. ", "A) ", "(A) ", "A: ", "A- "
_MCQ_CHOICE_SPLIT = re.compile(r'(?:^|\s)\(?([A-Da-d])\)?\s*[.):\-]\s*')

_WORD_NUMBERS: Dict[str, float] = {
    'zero': 0, 'no': 0, 'none': 0, 'null': 0,
    'one': 1, 'once': 1, 'single': 1,
    'two': 2, 'twice': 2, 'pair': 2,
    'three': 3, 'thrice': 3,
    'four': 4, 'five': 5, 'six': 6, 'seven': 7,
    'eight': 8, 'nine': 9, 'ten': 10,
    'eleven': 11, 'twelve': 12, 'thirteen': 13,
    'fourteen': 14, 'fifteen': 15, 'sixteen': 16,
    'seventeen': 17, 'eighteen': 18, 'nineteen': 19,
    'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50,
    'hundred': 100, 'thousand': 1000,
}


def _try_parse_number(text: str) -> Optional[float]:
    """Best-effort extraction of a numeric value from a choice or answer string.

    Tries, in order: direct float parse, first numeric token in the string
    (handles "3 speakers", "~5"), and English word-to-number lookup.
    """
    if not isinstance(text, str):
        return None
    text = text.strip()
    if not text:
        return None
    # 1) Direct parse
    try:
        return float(text)
    except ValueError:
        pass
    # 2) First numeric token (e.g. "3 speakers" → 3)
    m = re.search(r'(\d+(?:\.\d+)?)', text)
    if m:
        return float(m.group(1))
    # 3) Word numbers
    for token in text.lower().split():
        token = token.strip('.,;:!?()')
        if token in _WORD_NUMBERS:
            return _WORD_NUMBERS[token]
    return None


def _extract_mcq_choices(prompt_text: str) -> Dict[str, str]:
    """Extract ``{letter: choice_text}`` from a prompt string.

    Handles common MCQ formats: ``A. val``, ``A) val``, ``(A) val``.
    """
    choices: Dict[str, str] = {}
    if not isinstance(prompt_text, str):
        return choices
    parts = _MCQ_CHOICE_SPLIT.split(prompt_text)
    # split → [preamble, letter1, text1, letter2, text2, ...]
    i = 1
    while i + 1 < len(parts):
        letter = parts[i].upper()
        raw_value = parts[i + 1].strip()
        # Take first line; choices are normally single-line.
        raw_value = raw_value.split('\n')[0].strip().rstrip('.,;:!?')
        if raw_value:
            choices[letter] = raw_value
        i += 2
    return choices


def _is_counting_question(prompt_text: str) -> bool:
    """Heuristic: does this prompt ask a counting / quantity question?"""
    return isinstance(prompt_text, str) and bool(_COUNTING_KW_PATTERN.search(prompt_text))


def _compute_mse_reward(
    pred_raw: Optional[str],
    truth_raw: Optional[str],
    choices: Dict[str, str],
) -> Optional[float]:
    """Compute ``1 - |pred - truth| / range`` from MCQ numeric choices.

    *pred_raw* / *truth_raw* can be a choice letter ("B") or a raw
    number string ("3").  Returns ``None`` if the computation cannot be
    performed (non-numeric choices, missing mapping, etc.), in which
    case the caller should fall back to binary accuracy.
    """
    if pred_raw is None or truth_raw is None:
        return None

    # Build letter → number map
    numeric_map: Dict[str, float] = {}
    for letter, text in choices.items():
        val = _try_parse_number(text)
        if val is not None:
            numeric_map[letter] = val

    if len(numeric_map) < 2:
        return None

    vals = list(numeric_map.values())
    val_range = max(vals) - min(vals)
    if val_range <= 0:
        return None

    # Resolve truth value: letter mapping first, then raw parse
    truth_key = truth_raw.strip().upper()
    if truth_key in numeric_map:
        truth_val = numeric_map[truth_key]
    else:
        truth_val_parsed = _try_parse_number(truth_raw)
        if truth_val_parsed is None:
            return None
        truth_val = truth_val_parsed

    # Resolve pred value
    pred_key = pred_raw.strip().upper()
    if pred_key in numeric_map:
        pred_val = numeric_map[pred_key]
    else:
        pred_val_parsed = _try_parse_number(pred_raw)
        if pred_val_parsed is None:
            return None
        pred_val = pred_val_parsed

    return max(0.0, 1.0 - abs(pred_val - truth_val) / val_range)


def _preview(text: Any, max_chars: int = 160) -> Optional[str]:
    if not isinstance(text, str):
        return None
    compact = re.sub(r'\s+', ' ', text).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars - 3] + '...'


def _batched_get(values: Any, idx: int) -> Any:
    if isinstance(values, (list, tuple)) and idx < len(values):
        return values[idx]
    return None


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _format_trace_entry_text(entry: Dict[str, Any]) -> str:
    long_text_keys = ['prompt_full', 'solution_full', 'completion_full']
    lines: List[str] = []
    lines.append('=' * 80)
    ordered_keys = [
        'reward_name',
        'call_index',
        'idx_in_batch',
        'global_step',
        'epoch',
        'rank',
        'pid',
        'request_id',
        'gen_step',
        'reward',
        'solution_answer',
        'rollout_answer',
        'rollout_answer_source',
        'match_mode',
        'has_answer_tag',
        'starts_with_think_tag',
        'starts_with_reasoning_tag',
        'matches_reasoning_answer_format',
        'matches_think_answer_format',
        'prompt_preview',
        'completion_preview',
    ]
    for key in ordered_keys:
        if key in entry and entry[key] is not None:
            lines.append(f'{key}: {entry[key]}')

    for key, value in entry.items():
        if key in ordered_keys or key in long_text_keys or value is None:
            continue
        lines.append(f'{key}: {value}')

    for key in long_text_keys:
        if key in entry and entry[key] is not None:
            lines.append(f'--- {key} ---')
            lines.append(str(entry[key]))

    lines.append('')
    return '\n'.join(lines)


class _RewardTraceMixin:

    def __init__(self,
                 output_dir: Optional[str] = None,
                 save: Optional[str] = None,
                 verbose: bool = False,
                 reward_trace_to_file: bool = True,
                 reward_trace_dirname: str = 'rewards',
                 reward_trace_preview_chars: int = 180):
        super().__init__()
        self.output_dir = output_dir or save or os.getenv('MAPO_REWARD_OUTPUT_DIR')
        self.verbose = _env_flag('MAPO_REWARD_VERBOSE', verbose) or _env_flag('GRPO_REWARD_VERBOSE', False)
        self.reward_trace_to_file = _env_flag('MAPO_REWARD_LOG_TO_FILE', reward_trace_to_file)
        self.reward_trace_dirname = reward_trace_dirname
        self.reward_trace_preview_chars = reward_trace_preview_chars
        self.rank = str(os.getenv('RANK') or os.getenv('LOCAL_RANK') or '0')
        self.pid = os.getpid()
        self._trace_lock = threading.Lock()
        self._trace_call_index = 0
        self._trace_path_cache: Dict[str, str] = {}
        self._trace_error_reported = False

    def _trace_enabled(self) -> bool:
        return self.verbose or (self.reward_trace_to_file and bool(self.output_dir))

    def _trace_path(self, reward_name: str) -> Optional[str]:
        if not (self.reward_trace_to_file and self.output_dir):
            return None
        cached = self._trace_path_cache.get(reward_name)
        if cached:
            return cached
        safe_name = re.sub(r'[^0-9A-Za-z_.-]+', '_', reward_name)
        trace_dir = os.path.join(self.output_dir, self.reward_trace_dirname)
        os.makedirs(trace_dir, exist_ok=True)
        path = os.path.join(trace_dir, f'{safe_name}.rank{self.rank}.txt')
        self._trace_path_cache[reward_name] = path
        return path

    def _log_reward_rows(self, reward_name: str, row_payloads: List[Dict[str, Any]], **kwargs) -> None:
        if not self._trace_enabled():
            return

        trainer_state = kwargs.get('trainer_state')
        global_step = getattr(trainer_state, 'global_step', None) if trainer_state is not None else None
        epoch = getattr(trainer_state, 'epoch', None) if trainer_state is not None else None
        request_ids = kwargs.get('request_id')
        gen_steps = kwargs.get('gen_step')
        prompts = kwargs.get('prompt')

        with self._trace_lock:
            call_index = self._trace_call_index
            self._trace_call_index += 1

        entries = []
        for idx, payload in enumerate(row_payloads):
            entry: Dict[str, Any] = {
                'reward_name': reward_name,
                'call_index': call_index,
                'idx_in_batch': idx,
                'global_step': global_step,
                'epoch': epoch,
                'rank': self.rank,
                'pid': self.pid,
                'request_id': _json_safe(_batched_get(request_ids, idx)),
                'gen_step': _json_safe(_batched_get(gen_steps, idx)),
                'prompt_preview': _preview(_batched_get(prompts, idx), self.reward_trace_preview_chars),
                'prompt_full': _batched_get(prompts, idx),
            }
            entry.update({k: _json_safe(v) for k, v in payload.items()})
            entries.append(entry)

        if self.verbose and self.rank == '0':
            for entry in entries:
                reward_val = entry.get('reward')
                solution_answer = entry.get('solution_answer')
                rollout_answer = entry.get('rollout_answer')
                print(
                    f"[{reward_name}] step={entry.get('global_step')} gen_step={entry.get('gen_step')} "
                    f"idx={entry.get('idx_in_batch')} reward={reward_val} "
                    f"solution={solution_answer!r} rollout={rollout_answer!r}",
                    flush=True,
                )

        try:
            trace_path = self._trace_path(reward_name)
        except Exception as exc:  # noqa: BLE001
            if self.verbose and self.rank == '0' and not self._trace_error_reported:
                print(f"[{reward_name}] failed to prepare reward trace path: {exc}", flush=True)
            self._trace_error_reported = True
            return
        if not trace_path:
            return

        try:
            with open(trace_path, 'a', encoding='utf-8') as f:
                for entry in entries:
                    f.write(_format_trace_entry_text(entry))
                f.flush()
        except Exception as exc:  # noqa: BLE001
            # Never break reward computation because of debug logging.
            if self.verbose and self.rank == '0' and not self._trace_error_reported:
                print(f"[{reward_name}] failed to write reward trace: {exc}", flush=True)
            self._trace_error_reported = True


class MCQAAccuracy(_RewardTraceMixin, ORM):
    def __call__(self, completions, solution, **kwargs) -> List[float]:
        rewards = []
        trace_rows: List[Dict[str, Any]] = []
        for content, sol in zip(completions, solution):
            truth_raw = _extract_answer(sol)
            pred_raw, pred_source = _extract_answer_with_source(content)
            if truth_raw is None:
                reward = 0.0
                trace_rows.append({
                    'reward': reward,
                    'solution_answer': None,
                    'rollout_answer': pred_raw,
                    'rollout_answer_source': pred_source,
                    'match_mode': 'missing_solution_answer_tag',
                    'completion_preview': _preview(content, self.reward_trace_preview_chars),
                    'completion_full': content,
                    'solution_full': sol,
                })
                rewards.append(reward)
                continue

            truth_norm = _normalize_for_match(truth_raw)
            pred_norm = _normalize_for_match(pred_raw)
            match_mode = 'no_match'
            reward = 0.0

            if pred_norm is not None and truth_norm is not None:
                if pred_norm == truth_norm:
                    reward = 1.0
                    match_mode = 'normalized_exact'
                elif pred_norm.startswith(truth_norm):
                    reward = 1.0
                    match_mode = 'normalized_prefix'
                elif len(truth_norm) == 1 and pred_norm.startswith(truth_norm):
                    reward = 1.0
                    match_mode = 'single_choice_prefix'
                else:
                    reward = 0.0
                    match_mode = 'extracted_answer_mismatch'
            else:
                content_text = content if isinstance(content, str) else ''
                content_upper = content_text.upper()
                truth_upper = truth_raw.upper()
                matches = re.findall(r'([A-D])\.', content_text)
                if matches and matches[-1].upper() == truth_upper:
                    reward = 1.0
                    match_mode = 'choice_token_last'
                elif matches and matches[0].upper() == truth_upper:
                    reward = 1.0
                    match_mode = 'choice_token_first'
                elif truth_upper in content_upper:
                    if (
                        content_text.strip().upper().startswith(truth_upper)
                        or f"**{truth_upper}" in content_upper
                        or f"{truth_upper}." in content_upper
                    ):
                        reward = 1.0
                        match_mode = 'raw_text_heuristic'
                    else:
                        reward = 0.0
                        match_mode = 'raw_text_partial_only'
                else:
                    reward = 0.0
                    match_mode = 'no_answer_tag_or_heuristic_match'

            trace_rows.append({
                'reward': reward,
                'solution_answer': truth_raw,
                'rollout_answer': pred_raw,
                'rollout_answer_source': pred_source,
                'match_mode': match_mode,
                'completion_preview': _preview(content, self.reward_trace_preview_chars),
                'completion_full': content,
                'solution_full': sol,
            })
            rewards.append(reward)

        self._log_reward_rows(self.__class__.__name__, trace_rows, **kwargs)
        return rewards


class ExternalFormat(_RewardTraceMixin, ORM):
    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a supported output format."""
        rewards = []
        trace_rows: List[Dict[str, Any]] = []
        solutions = kwargs.get('solution')
        for idx, content in enumerate(completions):
            rollout_answer, rollout_answer_source = _extract_answer_with_source(content)
            has_reasoning_answer_format = _matches_reasoning_answer_format(content)
            has_think_answer_format = _matches_think_answer_format(content)
            reward = 1.0 if _matches_supported_format(content) else 0.0
            content_text = content if isinstance(content, str) else ''
            rewards.append(reward)
            trace_rows.append({
                'reward': reward,
                'solution_answer': _extract_answer(_batched_get(solutions, idx)),
                'rollout_answer': rollout_answer,
                'rollout_answer_source': rollout_answer_source,
                'has_answer_tag': _extract_tagged_answer(content_text) is not None,
                'starts_with_think_tag': content_text.lstrip().lower().startswith('<think>'),
                'starts_with_reasoning_tag': content_text.lstrip().lower().startswith('<reasoning>'),
                'matches_reasoning_answer_format': has_reasoning_answer_format,
                'matches_think_answer_format': has_think_answer_format,
                'completion_preview': _preview(content, self.reward_trace_preview_chars),
                'completion_full': content,
                'solution_full': _batched_get(solutions, idx),
            })

        self._log_reward_rows(self.__class__.__name__, trace_rows, **kwargs)
        return rewards


class _VLLMChecker:
    """Lightweight synchronous client for a vLLM OpenAI-compatible server
    used to evaluate consistency between reasoning and answer."""

    def __init__(self,
                 model_name: Optional[str] = None,
                 base_url: Optional[str] = None,
                 timeout: float = 600.0,
                 max_retries: int = 3):
        import requests as _requests  # noqa: F811
        self._requests = _requests

        # Resolve base_url: explicit arg > env var > default using ROLLOUT_NODE_IP
        if base_url is None:
            base_url = os.getenv('CHECKER_BASE_URL')
        if base_url is None:
            rollout_ip = os.getenv('ROLLOUT_NODE', '127.0.0.1')
            checker_port = os.getenv('CHECKER_PORT', '9000')
            base_url = f'http://{rollout_ip}:{checker_port}/v1'
        print(f'Using checker base_url: {base_url}')

        if model_name is None:
            model_name = os.getenv('CHECKER_MODEL_NAME', 'Qwen/Qwen3-30B-A3B-Instruct-2507')

        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries

    def _chat(self, prompt: str, max_tokens: int = 128) -> str:
        import time
        payload = {
            'model': self.model_name,
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': max_tokens,
            'temperature': 0.0,
        }
        url = f'{self.base_url}/chat/completions'

        for attempt in range(self.max_retries):
            try:
                resp = self._requests.post(url, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                text = resp.json()['choices'][0]['message']['content'].strip()
                return text.strip().split('\n')[-1]
            except self._requests.Timeout:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
            except self._requests.RequestException as exc:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    print(f'[_VLLMChecker] request failed after {self.max_retries} retries: {exc}', flush=True)
            except Exception as exc:
                print(f'[_VLLMChecker] unexpected error: {exc}', flush=True)
                break
        return ''

    def check_consistency(self, think_text: str, answer_text: str) -> bool:
        prompt = (
            'You are a reasoning consistency evaluator.\n'
            'Given a model\'s thinking process and its final answer, '
            'your task is to evaluate it against two specific failure modes.\n\n'
            f'Thinking Process:\n{think_text}\n\n'
            f'Final Answer:\n{answer_text}\n\n'
            'Respond "NO" ONLY if at least one of the following is true:\n'
            '1. The conclusion reached in the thinking process does not agree with or contradicts the final answer.\n'
            '2. The thinking process is visibly incomplete or cut off prematurely.\n\n'
            'If neither of these failure modes is present, respond "YES".\n'
            'Only output YES or NO.'
        )
        resp = self._chat(prompt, max_tokens=16).upper()
        return resp.startswith('Y')


class ConsistencyReward(_RewardTraceMixin, ORM):
    """Reward = 1 iff the answer is correct AND the reasoning is consistent with
    the answer (as judged by an external text-only LLM checker)."""

    def __init__(self,
                 output_dir: Optional[str] = None,
                 save: Optional[str] = None,
                 verbose: bool = False,
                 reward_trace_to_file: bool = True,
                 reward_trace_dirname: str = 'rewards',
                 reward_trace_preview_chars: int = 180,
                 model_name: Optional[str] = None,
                 base_url: Optional[str] = None,
                 conditional_on_correct: bool = True,
                 **kwargs):
        super().__init__(
            output_dir=output_dir,
            save=save,
            verbose=verbose,
            reward_trace_to_file=reward_trace_to_file,
            reward_trace_dirname=reward_trace_dirname,
            reward_trace_preview_chars=reward_trace_preview_chars,
            **kwargs,
        )
        self._checker = _VLLMChecker(model_name=model_name, base_url=base_url)
        self.conditional_on_correct = _env_flag(
            'MAPO_CONSISTENCY_CONDITIONAL_ON_CORRECT',
            conditional_on_correct,
        )

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        rewards: List[float] = []
        trace_rows: List[Dict[str, Any]] = []

        for idx, (content, sol) in enumerate(zip(completions, solution)):
            truth_raw = _extract_answer(sol)
            pred_raw, pred_source = _extract_answer_with_source(content)

            # Determine correctness
            truth_norm = _normalize_for_match(truth_raw)
            pred_norm = _normalize_for_match(pred_raw)
            is_correct = (
                truth_norm is not None
                and pred_norm is not None
                and (pred_norm == truth_norm or pred_norm.startswith(truth_norm))
            )

            # Extract reasoning block
            think_text: Optional[str] = None
            for pattern_name, pattern in [
                ('reasoning', re.compile(r'<reasoning>(.*?)</reasoning>', re.DOTALL | re.IGNORECASE)),
                ('think', re.compile(r'<think>(.*?)</think>', re.DOTALL | re.IGNORECASE)),
            ]:
                m = pattern.search(content if isinstance(content, str) else '')
                if m:
                    think_text = m.group(1).strip()
                    break

            # Compute consistency
            consistent = False
            skipped = False
            if think_text and pred_raw:
                if self.conditional_on_correct and not is_correct:
                    # Skip the (expensive) checker call for wrong answers
                    skipped = True
                else:
                    consistent = self._checker.check_consistency(think_text, pred_raw)

            reward = 1.0 if consistent else 0.0
            rewards.append(reward)

            trace_rows.append({
                'reward': reward,
                'solution_answer': truth_raw,
                'rollout_answer': pred_raw,
                'rollout_answer_source': pred_source,
                'is_correct': is_correct,
                'has_think': think_text is not None,
                'consistent': consistent,
                'skipped_checker': skipped,
                'completion_preview': _preview(content, self.reward_trace_preview_chars),
                'completion_full': content,
                'solution_full': sol,
            })

        self._log_reward_rows(self.__class__.__name__, trace_rows, **kwargs)
        return rewards


class CountingMSEAccuracy(_RewardTraceMixin, ORM):
    """Drop-in replacement for MCQAAccuracy that gives **partial credit** when
    the MCQ choices are numeric, via range-normalised MSE.

    For **every** question whose choices can be parsed as numbers, the reward
    is::

        1 - |pred - truth| / (max(choice_values) - min(choice_values))

    clamped to [0, 1].  No keyword heuristic is needed: if the choices are
    non-numeric, :func:`_compute_mse_reward` returns ``None`` and we fall
    back to the exact-match binary logic from :class:`MCQAAccuracy`.

    Register as ``external_counting_mse_accuracy`` (or overwrite
    ``external_mcqa_accuracy`` in *start_train.sh* to swap it in).
    """

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        rewards: List[float] = []
        trace_rows: List[Dict[str, Any]] = []
        prompts = kwargs.get('prompt')

        for idx, (content, sol) in enumerate(zip(completions, solution)):
            prompt_text = _batched_get(prompts, idx) or ''

            truth_raw = _extract_answer(sol)
            pred_raw, pred_source = _extract_answer_with_source(content)

            is_counting = _is_counting_question(prompt_text)  # diagnostic only
            mse_reward: Optional[float] = None
            mse_choices: Dict[str, str] = {}
            reward_mode = 'binary'

            # ── Attempt MSE reward for any question with numeric choices ─
            if truth_raw is not None and pred_raw is not None:
                mse_choices = _extract_mcq_choices(prompt_text)
                mse_reward = _compute_mse_reward(pred_raw, truth_raw, mse_choices)
                if mse_reward is not None:
                    reward_mode = 'counting_mse'

            # ── MSE path ────────────────────────────────────────────
            if reward_mode == 'counting_mse':
                reward = mse_reward  # type: ignore[assignment]
                match_mode = 'counting_mse'
            # ── Binary fallback (same logic as MCQAAccuracy) ────────
            else:
                if truth_raw is None:
                    reward = 0.0
                    match_mode = 'missing_solution_answer_tag'
                    trace_rows.append({
                        'reward': reward,
                        'solution_answer': None,
                        'rollout_answer': pred_raw,
                        'rollout_answer_source': pred_source,
                        'match_mode': match_mode,
                        'reward_mode': reward_mode,
                        'is_counting_q': is_counting,
                        'completion_preview': _preview(content, self.reward_trace_preview_chars),
                        'completion_full': content,
                        'solution_full': sol,
                    })
                    rewards.append(reward)
                    continue

                truth_norm = _normalize_for_match(truth_raw)
                pred_norm = _normalize_for_match(pred_raw)
                match_mode = 'no_match'
                reward = 0.0

                if pred_norm is not None and truth_norm is not None:
                    if pred_norm == truth_norm:
                        reward = 1.0
                        match_mode = 'normalized_exact'
                    elif pred_norm.startswith(truth_norm):
                        reward = 1.0
                        match_mode = 'normalized_prefix'
                    elif len(truth_norm) == 1 and pred_norm.startswith(truth_norm):
                        reward = 1.0
                        match_mode = 'single_choice_prefix'
                    else:
                        reward = 0.0
                        match_mode = 'extracted_answer_mismatch'
                else:
                    content_text = content if isinstance(content, str) else ''
                    content_upper = content_text.upper()
                    truth_upper = truth_raw.upper()
                    matches = re.findall(r'([A-D])\.', content_text)
                    if matches and matches[-1].upper() == truth_upper:
                        reward = 1.0
                        match_mode = 'choice_token_last'
                    elif matches and matches[0].upper() == truth_upper:
                        reward = 1.0
                        match_mode = 'choice_token_first'
                    elif truth_upper in content_upper:
                        if (
                            content_text.strip().upper().startswith(truth_upper)
                            or f"**{truth_upper}" in content_upper
                            or f"{truth_upper}." in content_upper
                        ):
                            reward = 1.0
                            match_mode = 'raw_text_heuristic'
                        else:
                            reward = 0.0
                            match_mode = 'raw_text_partial_only'
                    else:
                        reward = 0.0
                        match_mode = 'no_answer_tag_or_heuristic_match'

            # ── Logging ─────────────────────────────────────────────
            trace_entry: Dict[str, Any] = {
                'reward': reward,
                'solution_answer': truth_raw,
                'rollout_answer': pred_raw,
                'rollout_answer_source': pred_source,
                'match_mode': match_mode,
                'reward_mode': reward_mode,
                'is_counting_q': is_counting,
                'completion_preview': _preview(content, self.reward_trace_preview_chars),
                'completion_full': content,
                'solution_full': sol,
            }
            if reward_mode == 'counting_mse':
                # Extra diagnostics for MSE path
                truth_key = truth_raw.strip().upper() if truth_raw else ''
                pred_key = pred_raw.strip().upper() if pred_raw else ''
                trace_entry['mse_choices'] = json.dumps(mse_choices)
                trace_entry['mse_truth_key'] = truth_key
                trace_entry['mse_pred_key'] = pred_key
                truth_val = _try_parse_number(mse_choices.get(truth_key, truth_raw or ''))
                pred_val = _try_parse_number(mse_choices.get(pred_key, pred_raw or ''))
                trace_entry['mse_truth_val'] = truth_val
                trace_entry['mse_pred_val'] = pred_val
            trace_rows.append(trace_entry)
            rewards.append(reward)

        self._log_reward_rows(self.__class__.__name__, trace_rows, **kwargs)
        return rewards


orms['external_mcqa_accuracy'] = MCQAAccuracy
orms['external_format'] = ExternalFormat
orms['external_consistency'] = ConsistencyReward
orms['external_counting_mse_accuracy'] = CountingMSEAccuracy
