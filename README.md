# Shelf Object Detection

Shelf object detection project using the SKU-110K dataset. Currently, the
implemented script downloads and extracts the dataset; training and prediction
scripts are still empty local placeholders and are excluded from this repository.

## Download the dataset

Use Python 3 and run from the project root:

```sh
python src/data_download.py
```

The script uses only the Python standard library. It saves the archive to
`data/SKU110K_fixed.tar.gz` and extracts the dataset to `data/SKU-110K/`.
Paths are resolved relative to the script, so the project can be cloned into
any folder without editing the code.

The dataset, model artifacts, and generated output are ignored by Git.

## Implementing the placeholders

When a local placeholder is implemented, remove its corresponding entry from
`.gitignore` before adding it to Git.
