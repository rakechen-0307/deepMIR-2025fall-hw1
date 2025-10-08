# DeepMIR 2025 HW1: Singer Classification
R14942086 陳柏睿

## Prepare Environments
Using Python 3.10
### Full Installation (For Running All Data Preprocessing, Training & Inference)
- Install `ffmpeg` 

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

- Install pip dependencies

```bash
pip install -r requirements.txt
```

### Minimal Installation (For Running Inference Only)
- Install pip dependencies
```bash
pip install -r requirements-inference.txt
```

## Execute Scripts
### Download Data
Run the following command:
```bash
bash scripts/preprocess/download_data.sh
```

### Download Model Checkpoint
Run the following command:
```bash
bash scripts/preprocess/download_ckpt.sh
```
The ML model checkpoint is in folder `exp/ml` while the DL model checkpoint is in folder `exp/dl`

### Vocals Separation
For executing source separation, run the following command:
```bash
python -m src.preprocess.vocals_separation --input_dir ${folder of data before processed} --output_dir ${folder of data after processed}
```
or simply run:
```bash
bash scripts/preprocess/vocals_separation.sh
```
if the data is downloaded by `bash scripts/preprocess/download_data.sh`

### Inference
- For ML-based model, run the following command:
```bash
python -m src.ml.inference --vocals_test_dir ${vocals stem of test files} --inst_test_dir ${instrumental stem of test files} --exp_dir ${model checkpoint directory} --jobs ${# of parallel jobs} --split_audio --use_boosting
```
or simply run:
```bash
bash scripts/ml/inference.sh
```
if the data is downloaded by `bash scripts/preprocess/download_data.sh` and preprocessed by `bash scripts/preprocess/vocals_separation.sh`.
The output results can be found in `exp_dir`

- For DL-based model, run the following command:
```bash
python -m src.dl.inference --test_dir ${(vocals stem of) test files} --exp_dir ${model checkpoint directory} --split_audio
```
or simply run:
```bash
bash scripts/dl/inference.sh
```
if the data is downloaded by `bash scripts/preprocess/download_data.sh` and preprocessed by `bash scripts/preprocess/vocals_separation.sh`.
The output results can be found in `exp_dir`

### Training
- For ML-based model, run the following command:
```bash
python -m src.ml.train --vocals_dir ${all vocals stem files(including train.json & val.json)} --inst_dir ${all instrumental stem files} --output_dir ${output directory} --jobs ${# of parallel jobs} --split_audio --num_augments ${# of augmented samples} --use_boosting
```
or simply run:
```bash
bash scripts/ml/train.sh
```
if the data is downloaded by `bash scripts/preprocess/download_data.sh` and preprocessed by `bash scripts/preprocess/vocals_separation.sh`.
The trained model can be found in `output_dir`

- For DL-based model, run the following command:
```bash
python -m src.dl.train --data_dir ${all (vocals stem of) data files(including train.json & val.json)} --output_dir ${output directory} --split_audio
```
or simply run:
```bash
bash scripts/dl/train.sh
```
if the data is downloaded by `bash scripts/preprocess/download_data.sh` and preprocessed by `bash scripts/preprocess/vocals_separation.sh`.
The trained model can be found in `output_dir`