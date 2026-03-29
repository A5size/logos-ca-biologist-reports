# Simulation Code

This directory contains the simulation engine used to generate the biological report data.

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="your-api-key-here"
```

## Files

| File | Description |
|------|-------------|
| `logos_ca.py` | LOGOS-CA simulation engine (adapted for structured biological reports) |
| `biologist_reports.py` | Experiment runner with grid initialization and configuration |

## Usage

Edit the configuration in `biologist_reports.py` (grid size, model, steps, etc.), then run:

```bash
python biologist_reports.py
```

Results are saved to the configured output JSON file (default: `biologist_reports_result.json`). The simulation supports resume — if interrupted, run the same command again to continue from the last completed step.

## Converting Results for the Viewer

After a simulation completes, use `convert_data.py` (in the repository root) to convert the output for the web viewer:

```bash
pip install Pillow

python ../convert_data.py \
  --id 04-03 \
  --desc "10x10 grid, GPT-5-mini, 50 steps." \
  biologist_reports_result.json \
  --outdir ..
```

This generates `exp-04-03-data.json` and updates `experiments.json` in the repository root.

## Configuration

Key parameters in `biologist_reports.py`:

| Parameter | Description |
|-----------|-------------|
| `grid_size_x`, `grid_size_y` | Grid dimensions |
| `max_steps` | Number of simulation steps |
| `model_name` | OpenAI model (e.g. `gpt-5-mini`) |
| `temperature` | LLM sampling temperature |
| `max_workers` | Parallel API calls |
| `output_json` | Output file path |

## Cost Warning

Running simulations incurs OpenAI API charges. A 10×10 grid for 50 steps makes 5,000 API calls. Start with small grids and few steps to estimate costs.

## References

This code is adapted from the [LOGOS-CA](https://github.com/A5size/LOGOS-CA) framework by K. Utimula.
