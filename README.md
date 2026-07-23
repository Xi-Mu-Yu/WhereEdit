
## Overview

WhereEdit is a one-step image editing framework built on SD-Turbo. It combines:

- **AutoMask**: attention-guided automatic edit-region localization from prompt token differences
- **ACT** (Amplified Conditional Transport): stronger target-oriented semantic transport for localized editing

Core modules:

| Module | File | Role |
|--------|------|------|
| WhereEdit Pipeline | `pipeline_whereedit.py` | End-to-end one-step editing |
| AutoMask | `auto_mask.py` | Cross-attention edit mask generation |
| ACT | `act.py` | Amplified conditional transport field |

## 1. Environment

- Python 3.12
- PyTorch 2.5.0
- SD-Turbo weights: https://huggingface.co/stabilityai/sd-turbo
- Model root should contain: `unet/`, `scheduler/`, `text_encoder/`, `tokenizer/`, `vae/`

## 2. Install Dependencies

```bash
pip install -r requirement.txt
```

## 3. Run the Web Demo

```bash
python app.py --model-root ./sd-turbo --server-port 7860
```

- Left panel: upload the original image, set source prompt, target prompt, and tuning parameters.
- Right panel: view the edited output image.
- Bottom section: click built-in examples to auto-fill inputs.


## 4. Run PIE Benchmark

Export edited images:

```bash
python run_pie_bench.py --model-root ./sd-turbo --pie-root ./sd-turbo/pie_bench --overwrite
```

Run official PIE metrics and summarize results:

```bash
python eval/evaluate.py
python eval/summarize_metrics.py
```

See [eval/README.md](eval/README.md) for details.

`--pie-root` should contain:

1. `annotation_images/` — original PIE-Bench images
2. `mapping_file.json` — prompts, instructions, and masks

For PIE-Bench data preparation, see https://github.com/cure-lab/PnPInversion




export CUDA_VISIBLE_DEVICES=9

conda activate WhereEdit





python run_pie_bench.py --model-root ./sd-turbo --pie-root ./sd-turbo/pie_bench --overwrite


python eval/evaluate.py


python eval/summarize_metrics.py

