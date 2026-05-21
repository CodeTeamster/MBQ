# Repository Guidelines

## Project Structure & Module Organization
- Core code lives in `qmllm/`, organized by responsibility: `methods/` (quantization algorithms such as MBQ/AWQ/SmoothQuant/RTN), `models/` (model adapters), `quantization/` (low-level quant ops), `calibration/`, `datasets/`, and `utils/`.
- Entry points are `main_quant.py` (quantization search) and `main.py` (evaluation/inference with quantized settings).
- Ready-to-run YAML configs are under `configs/<model>/{MBQ_search,Eval}/`.
- Third-party dependencies are vendored in `3rdparty/LLaVA-NeXT` and `3rdparty/lmms-eval`; treat them as external unless a change explicitly requires patching.
- Static assets and figures are in `figures/`.

## Build, Test, and Development Commands
- Create env: `conda create -n qmllm python=3.10 && conda activate qmllm`
- Install package: `pip install -r requirements.txt && pip install -e .`
- Install bundled third-party repos:
  - `cd 3rdparty/LLaVA-NeXT && pip install -e .`
  - `cd 3rdparty/lmms-eval && pip install -e .`
- Run quantization search (config-driven):
  - `python3 -W ignore main_quant.py --config configs/internvl2/MBQ_search/8b_weight_only.yaml`
- Run evaluation (config-driven):
  - `python3 -W ignore main.py --config configs/internvl2/Eval/eval.yaml`

## Coding Style & Naming Conventions
- Follow existing Python style in `qmllm`: 4-space indentation, `snake_case` for functions/variables, `PascalCase` for classes, lowercase module filenames.
- Keep new config filenames descriptive and aligned with current patterns (example: `7b_weight_only.yaml`).
- Prefer small, targeted changes; avoid broad refactors in the same PR.

## Testing Guidelines
- There is no first-party unit test suite at repo root today.
- Validate changes with task-focused smoke runs:
  - quantization path via `main_quant.py` with a minimal sample count,
  - evaluation path via `main.py` and an existing `configs/*/Eval/eval.yaml`.
- When modifying model/method code, include exact command(s) used and key output metrics/log paths in the PR.

## Commit & Pull Request Guidelines
- Recent history uses concise, bracketed prefixes (for example: `[update] ...`, `[debug] ...`, `[minor] ...`, `[init] ...`). Keep this convention.
- Commit format: `[type] short imperative summary`.
- PRs should include: purpose, scope, reproducible commands, changed config/model targets, and any result deltas. Link related issues when applicable.
