# WhereEdit Evaluation

`eval/` contains a self-contained copy of the official PIE-Bench evaluation scripts.  
Run all commands from the **repository root**.

## 1. Export PIE-Bench Results

```bash
python run_pie_bench.py --model-root ./sd-turbo --pie-root ./sd-turbo/pie_bench --overwrite
```

Outputs should appear at:

`sd-turbo/pie_bench/output/WhereEdit/annotation_images/`

## 2. Run PIE Metrics

Full 8-metric evaluation (same as the paper table):

```bash
python eval/evaluate.py
```

Results are written to `eval/evaluation_result.csv`.

Override edited-image folder if needed:

```bash
python eval/evaluate.py --tgt-image-folder ./sd-turbo/pie_bench/output/WhereEdit/annotation_images
```

## 3. Summarize Metrics

```bash
python eval/summarize_metrics.py
```

Reads `eval/evaluation_result.csv` and writes `eval/metrics_summary.csv`.

## Files

| File | Description |
|------|-------------|
| `evaluate.py` | Per-sample PIE metrics (from PnPInversion) |
| `matrics_calculator.py` | PSNR / LPIPS / CLIP / structure distance calculators |
| `summarize_metrics.py` | Mean summary over the result CSV |

## Dependencies

Metric evaluation requires: `torch`, `torchmetrics`, `torchvision`, `pandas`, `Pillow`, `numpy`.

`structure_distance` additionally downloads DINO via `torch.hub` on first run.
