# Maros--Meszaros QP collector

This directory contains a collector for FA_CP summary rows. It does not include
benchmark inputs or completed solver outputs.

Run the solver with the batch script under [`QP/FA_CP`](../../QP/FA_CP), then
normalize the resulting summary:

```bash
python experiments/qp_maros/collect_qp_maros.py \
  --summary /path/to/summary.csv \
  --output-dir /path/to/qp_aggregate
```

The collector validates the input rows and writes normalized rows, aggregate
statistics, and a local audit file to the selected output directory.
