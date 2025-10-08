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
bash scripts/ml/inference.sh
```
The output results can be found in the folder `exp/ml`

- For DL-based model, run the following command:
```bash
bash scripts/dl/inference.sh
```
The output results can be found in the folder `exp/dl`
### Training
- For ML-based model, run the following command:
```bash
bash scripts/ml/train.sh
```
The trained model can be found in the folder `exp/ml`
- For DL-based model, run the following command:
```bash
bash scripts/dl/train.sh
```
The trained model can be found in the folder `exp/dl`