# PlanSight

Tile-and-stitch region-of-interest detection for high-resolution architectural drawings.

A single site-plan sheet at 600 dpi exceeds the input budget of any detector or VLM in one pass. PlanSight processes a page as overlapping tiles, detects candidate symbols per tile, and reconciles the results in page space with non-maximum suppression across the tile seams — turning a dense sheet into a short list of grounded regions instead of reading every pixel.

**Live demo:** https://plansight-henna.vercel.app

## Two parts

**1 · Live pipeline (`index.html`)**
A client-side, interactive visualization of the full pipeline — tile → detect → flag seam duplicates → cross-seam NMS → rank. Adjustable tile size, overlap, and IoU threshold, with live precision/recall against ground truth on a procedurally generated sheet, plus upload for arbitrary drawings. The per-tile detector here is a classical region proposer (luminance threshold → morphological dilation → connected components → geometric filtering), implemented behind a fixed interface a learned model can replace.

**2 · Fine-tuned detector (`notebooks/finetune_symbols.ipynb`)**
The trained-model counterpart. Generates a synthetic architectural-symbol dataset, splits it **by document** (not by image — holding out by tile leaks an architect's style into validation), fine-tunes YOLOv8, reports mAP per class, then runs the same tile-and-stitch inference with seam NMS on a held-out sheet and catalogs the failure modes. Runs end to end on a free Colab T4.

## The three challenges it targets

- **ROI detection** — rank regions by density so downstream inference only reads what carries an answer.
- **Tile-based inference + seam NMS** — split, detect per tile, remap to page coordinates, suppress overlap-zone duplicates.
- **Symbol recognition** — the pluggable detector slot; the notebook fills it with a classifier that names each symbol.

## Run locally

```bash
# static demo — no build step
npx serve .        # or open index.html directly

# detector — open the notebook in Colab, set GPU, run all
notebooks/finetune_symbols.ipynb
```
