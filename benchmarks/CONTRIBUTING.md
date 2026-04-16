# Contributing Benchmark Results

We welcome results from other Apple Silicon hardware! Here's how to submit yours.

## How to Run

1. Clone the repo and set up Bonsai ([see README](README.md#setup-bonsai))
2. Run the full suite: `./autorun.sh`
3. Results will be saved to `results/<your-hardware>/` (auto-detected)

## How to Submit

1. Fork this repo
2. Run `./autorun.sh` (runs all supported models)
3. Commit your results directory
4. Open a PR with:
   - Your hardware (chip, RAM, macOS version)
   - Any anomalies or notes
   - The summary table from the autorun output

## What We're Looking For

- M1, M2, M3, M4 (any variant: base, Pro, Max, Ultra)
- Different RAM configurations
- macOS version differences
- Results from additional models (Llama, Phi, Gemma, etc.)

## Adding New Models

Edit the `MODEL_REPO` dict in `autorun.sh` and `MODELS` dict in `compare_results.py` / `generate_report.py` to add new model entries. Then run and submit.

## Guidelines

- Don't modify benchmark test files unless fixing a genuine bug
- Include the full `NOTES.md` from your run
- One PR per hardware configuration
- Use `./autorun.sh` to ensure consistent methodology
