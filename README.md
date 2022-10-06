# End-to-End Infant Engagement Analysis

Code for the article "End-to-End Learning and Analysis of Infant Engagement During Guided Play: Prediction and Explainability" (to appear in ICMI'22).

## Reproducing the Work

### Installing

This project uses [Poetry](https://python-poetry.org/) for version control. Follow the official installation instructions, and then run this command from the repository's top-level directory:

```
poetry install
```

*__Note:__ Currently, [PyTorch doesn't play nice with Poetry](https://github.com/python-poetry/poetry/issues/6409). The `pyproject.toml` file contains a workaround that only works for 64-bit Linux using Python3.8. If you use a different system, fix the URL according to your needs.*

### Processing Data

*__Note:__ For data protection reasons, we cannot freely share the original videos, or the associated annotation files.*

Assuming you have access to the source material, it should be stored in the following folder structure:

```
data
└── raw
    ├── engagement
    │   ├── <session>_<annotator>_<date>.eaf
    │   └── ...
    └── video
        ├── <session>.mp4
        └── ...

```

To train, you need to run the following scripts in the correct order. *Always run them from the top-level directory*.

* To generate the CSV files needed for training, run two scripts in order:
    1. `scripts/data_processing/process_elan.py` pre-processes the EAF files in `data/raw/engagement/` into a CSV file `data/processed/engagement/original_annotation_spans.csv`, and validates that the data is correct and complete.
    2. `scripts/data_processing/processed_elan_to_annotation_format.py` processes `data/processed/engagement/original_annotation_spans.csv` into a training-ready format, saved as `data/processed/engagement/filled_annotation_spans.csv`.
* To prepare the videos for training, run `scripts/data_processing/process_videos.py`. It reads MP4 files in `data/raw/video/`, and outputs downsampled copies to `data/processed/video/`.
* Once the CSV annotations and MP4 videos have been generated, run `scripts/data_processing/stratify_samples.py` in order stratify all samples into evenly-distributed folds. This will generate `data/processed/engagement/stratified_annotation_spans.csv` as a binarized, fold-separated annotation file; and `data/processed/fold_statistics.json` as a summary of empirical probabilities, pixel means and deviations, and which sessions belong to each fold.
* To pre-bake samples for training the classification head, run `scripts/data_processing/bake_samples.py`. It will save tensor packs to `data/processed/baked_samples/`, separated by fold and type of data augmentation (`train` or `test`). The training fold (last fold) only gets `test` augmentation.

### Training Networks

There are three training scripts: two for hyper-parameter searches (classifier head training vs. whole-network fine-tuning), and one for the final training. Each is contained in a folder in `scripts/training/`. You can run them locally in a CUDA-accelerated machine with enough RAM by calling the respective `local.sh` script. It is highly recommended that you run the fine-tuning search in a GPU cluster; `jobfile.sh` handles this for SLURM systems.

All the scripts are designed to create timestamped folders under `artifacts/<training type>/`, containing all relevant artifacts (pickled models, run statistics, logs...)

* To perform a hyper-parameter search for the *classifier head* training, run `scripts/training/head_search/local.sh`. Artifacts are saved in `artifacts/head_search/`.
* To perform a hyper-parameter search for the *fine-tuning* step, run `scripts/training/finetune_search/local.sh`. Artifacts are saved in `artifacts/finetune_search/`.
* To perform the final training, run `scripts/training/final/local.sh`. Artifacts are saved in `artifacts/final/`.

### Using the Distributed Code

The distributed code assumes you have access to a GPU cluster that uses SLURM for job management and Singularity for containerization. To run `training/finetune_search/jobfile.sh` or `training/finetune_search/distributed.sh`, you need to create a directory named `img/` at the top level of this repository, and create an image inside it. The image should have the dependencies listed in `pyproject.toml` / `poetry.lock`.

### Human Attention Annotations

* To choose snippets to paint, run `scripts/human_attention/choose_snippets.py`. This procedure uses rejection sampling to ensure good properties on the selected samples.
* To paint your own annotations on the chosen snippets, run the notebook `scripts/human_attention/annotator.ipynb`.
