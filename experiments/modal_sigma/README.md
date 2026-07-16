# Controlled modal-penalty experiments

The scripts in this directory implement the controlled one-step and
restarted-epoch penalty experiments described in the paper. They generate
their synthetic QPs internally and write all outputs to a user-selected local
directory.

Install the Python dependencies:

```bash
python -m pip install -r experiments/modal_sigma/requirements.txt
```

Run the one-step experiment:

```bash
python experiments/modal_sigma/run_modal_sigma_controlled.py \
  --output-dir /path/to/modal_sigma_controlled
```

Run the restarted-epoch experiment:

```bash
python experiments/modal_sigma/run_modal_sigma_epoch.py \
  --output-dir /path/to/modal_sigma_epoch
```

The output directories contain generated configurations, per-instance rows,
aggregates, and figures. These files are local run products and are not tracked
by this repository.
