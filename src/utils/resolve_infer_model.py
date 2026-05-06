#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import shlex
import shutil
import sys
from typing import Tuple


def _is_adapter_checkpoint(path: str) -> bool:
    if not path or not os.path.isdir(path):
        return False
    has_adapter = (
        os.path.isfile(os.path.join(path, "adapter_config.json"))
        or os.path.isfile(os.path.join(path, "default", "adapter_config.json"))
        or os.path.isdir(os.path.join(path, "reft"))
    )
    has_full_model_config = os.path.isfile(os.path.join(path, "config.json"))
    return has_adapter and not has_full_model_config


def _read_base_model_from_args(adapter_path: str) -> str:
    args_path = os.path.join(adapter_path, "args.json")
    if not os.path.isfile(args_path):
        return ""
    try:
        with open(args_path, "r", encoding="utf-8") as f:
            args_data = json.load(f)
    except Exception:
        return ""
    model = args_data.get("model")
    return model if isinstance(model, str) else ""


def _bool_arg(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "y", "on"}


def _maybe_patch_adapter_config(adapter_path: str, cache_root: str) -> Tuple[str, bool]:
    config_path = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.isfile(config_path):
        return adapter_path, False

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)
    except Exception:
        return adapter_path, False

    target_modules = config_data.get("target_modules")
    if not isinstance(target_modules, str):
        return adapter_path, False

    # Avoid matching container modules like thinker.visual.merger_list.0.
    replacement_pairs = (
        (r"thinker.visual.merger.*\.(2|0)", r"thinker.visual.merger\.(2|0)"),
        (r"thinker.visual.merger.*\.(0|2)", r"thinker.visual.merger\.(0|2)"),
    )
    patched_target_modules = target_modules
    for old_pattern, new_pattern in replacement_pairs:
        patched_target_modules = patched_target_modules.replace(old_pattern, new_pattern)
    if patched_target_modules == target_modules:
        return adapter_path, False

    config_data["target_modules"] = patched_target_modules
    os.makedirs(cache_root, exist_ok=True)
    digest = hashlib.sha1(
        f"{adapter_path}\n{patched_target_modules}".encode("utf-8")
    ).hexdigest()[:12]
    adapter_basename = os.path.basename(os.path.normpath(adapter_path)) or "adapter"
    patched_dir = os.path.join(cache_root, f"{adapter_basename}-cfgpatch-{digest}")

    if not os.path.isdir(patched_dir):
        staging_dir = f"{patched_dir}.tmp.{os.getpid()}"
        if os.path.exists(staging_dir):
            shutil.rmtree(staging_dir)
        os.makedirs(staging_dir, exist_ok=True)
        for entry in os.listdir(adapter_path):
            src = os.path.join(adapter_path, entry)
            dst = os.path.join(staging_dir, entry)
            if entry == "adapter_config.json":
                continue
            src_abs = os.path.abspath(src)
            os.symlink(src_abs, dst, target_is_directory=os.path.isdir(src_abs))
        with open(os.path.join(staging_dir, "adapter_config.json"), "w", encoding="utf-8") as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        try:
            os.replace(staging_dir, patched_dir)
        except OSError:
            if os.path.isdir(patched_dir):
                shutil.rmtree(staging_dir, ignore_errors=True)
            else:
                raise

    return patched_dir, True


def _is_parent_writable(path: str) -> bool:
    probe = os.path.abspath(path)
    while probe and not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    if not os.path.exists(probe):
        return False
    return os.access(probe, os.W_OK | os.X_OK)


def _default_adapter_cache_root(model_path: str, adapter_path: str) -> str:
    candidates = []
    if adapter_path:
        candidates.append(os.path.join(os.path.dirname(os.path.abspath(adapter_path)), ".adapter-cache"))
    if model_path and os.path.isdir(model_path):
        candidates.append(os.path.join(os.path.dirname(os.path.abspath(model_path)), ".adapter-cache"))
    mapo_root = os.environ.get("MAPO_ROOT")
    if mapo_root:
        candidates.append(os.path.join(os.path.abspath(mapo_root), "exp", ".adapter-cache"))
    candidates.append(os.path.join(os.path.expanduser("~"), ".cache", "mapo-adapter-cache"))
    candidates.append("/tmp/mapo-adapter-cache")

    for candidate in candidates:
        if _is_parent_writable(candidate):
            return candidate
    return candidates[-1]


def resolve_paths(
    model: str,
    base_model: str,
    adapter: str,
    prepare_adapter_config: bool,
    adapter_cache_root: str,
) -> Tuple[str, str, bool, bool]:
    detected_adapter = False
    patched_adapter_config = False
    resolved_model = model or ""
    resolved_adapter = adapter or ""

    if not resolved_adapter and _is_adapter_checkpoint(resolved_model):
        resolved_adapter = resolved_model
        detected_adapter = True

    if resolved_adapter:
        if not os.path.isdir(resolved_adapter):
            raise ValueError(f"adapter checkpoint not found: {resolved_adapter}")
        inferred_base = _read_base_model_from_args(resolved_adapter)
        if base_model:
            resolved_model = base_model
        elif inferred_base:
            resolved_model = inferred_base
        if not resolved_model:
            raise ValueError(
                "unable to resolve base model for adapter checkpoint. "
                "Please pass --base-model-name-or-path."
            )
    elif base_model:
        resolved_model = base_model

    if not resolved_model:
        raise ValueError("model_name_or_path is empty after resolution.")

    if resolved_adapter and prepare_adapter_config:
        if not adapter_cache_root:
            adapter_cache_root = _default_adapter_cache_root(resolved_model, resolved_adapter)
        try:
            resolved_adapter, patched_adapter_config = _maybe_patch_adapter_config(
                resolved_adapter, adapter_cache_root
            )
        except OSError as e:
            fallback_root = _default_adapter_cache_root("", "")
            if adapter_cache_root != fallback_root:
                print(
                    f"resolve_infer_model.py: adapter cache root `{adapter_cache_root}` failed ({e}); "
                    f"falling back to `{fallback_root}`.",
                    file=sys.stderr,
                )
                resolved_adapter, patched_adapter_config = _maybe_patch_adapter_config(
                    resolved_adapter, fallback_root
                )
            else:
                raise

    return resolved_model, resolved_adapter, detected_adapter, patched_adapter_config


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resolve swift infer model/adapters from model_name_or_path."
    )
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--base-model-name-or-path", default="")
    parser.add_argument("--adapter-name-or-path", default="")
    parser.add_argument("--prepare-adapter-config", default="true")
    parser.add_argument("--adapter-cache-root", default="")
    args = parser.parse_args()

    try:
        model, adapter, detected_adapter, patched_adapter_config = resolve_paths(
            model=args.model_name_or_path,
            base_model=args.base_model_name_or_path,
            adapter=args.adapter_name_or_path,
            prepare_adapter_config=_bool_arg(args.prepare_adapter_config),
            adapter_cache_root=args.adapter_cache_root,
        )
    except ValueError as e:
        print(f"resolve_infer_model.py: {e}", file=sys.stderr)
        return 2

    print(f"RESOLVED_MODEL_NAME_OR_PATH={shlex.quote(model)}")
    print(f"RESOLVED_ADAPTER_NAME_OR_PATH={shlex.quote(adapter)}")
    print(f"DETECTED_ADAPTER_ONLY_MODEL={'1' if detected_adapter else '0'}")
    print(f"PATCHED_ADAPTER_CONFIG={'1' if patched_adapter_config else '0'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
