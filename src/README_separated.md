# smDeepFLUOR (separated code)

This folder is a **refactor** of `smDeepFLUOR.ipynb` into reusable python modules under `src/`.

## Folder structure

- `src/io_tiff.py` : TIFF loading -> standardize to (T,Y,X)
- `src/cropping.py` : particle-tracker CSV parsing, 7x7 cropping, TIFF->NPZ conversion
- `src/split_dataset.py` : train/test split (80/20) into `training/` and `test/`
- `src/data_loading.py` : NPZ loading + sliding-window extraction
- `src/standardize.py` : per-sample z-score + train/test split + channel expand
- `src/model_def.py` : model architecture + compile
- `src/train.py` : training wrapper + callbacks + best-model saving
- `src/visualization.py` : training curve plots
- `src/validate.py` : folder-level validation (majority vote over windows)

## Notebook

- `smDeepFLUOR_separate_code.ipynb` runs the same pipeline but imports from `src/`.

