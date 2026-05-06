# Third-Party Notices

`libs/ms-swift` is vendored from ModelScope ms-swift and is distributed under the Apache License 2.0. The vendored copy includes local MAPO changes so the public repository does not need a git submodule.

Megatron-LM is not vendored. The training launcher uses `MEGATRON_LM_PATH` when set, or clones NVIDIA Megatron-LM `core_r0.15.0` into `libs/Megatron-LM` on first use.
