import os
import json
import shlex
import joblib
import argparse
import subprocess
from tqdm import tqdm
from omegaconf import OmegaConf
from contextlib import contextmanager

@contextmanager
def tqdm_joblib(tqdm_object):
    """
    Context manager to patch joblib to report into tqdm progress bar given as argument
    """
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_cb = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_cb
        tqdm_object.close()

def process_file(input_path, output_dir, output_name, bitrate='32k', sr=16000):
    """
    Process a single audio file to separate vocals using demucs.
    """
    subprocess.run(
        shlex.split(
            f"demucs --two-stems vocals {input_path} -o {output_dir}"
        ),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True
    )

    src_wav = os.path.join(output_dir, "htdemucs", os.path.basename(input_path).replace(".mp3", ""), "vocals.wav")
    dst_mp3 = os.path.join(output_dir, output_name)

    subprocess.run(
        shlex.split(f"ffmpeg -y -i {src_wav} -ar {sr} -b:a {bitrate} -loglevel quiet {dst_mp3}"),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True
    )
    return output_name

def main():
    # Load configurations
    parser = argparse.ArgumentParser(description="Separate vocals from songs using demucs.")
    parser.add_argument("--config", required=True, type=str, help="config file path")
    args = parser.parse_args()
    configs = OmegaConf.load(args.config)

    input_dir = configs.dir.original
    train_names_json = os.path.join(input_dir, "train.json")
    val_names_json = os.path.join(input_dir, "val.json")
    test_dir = os.path.join(input_dir, "test")
    output_dir = configs.dir.processed
    output_trainval_dir = os.path.join(output_dir, "trainval")
    output_test_dir = os.path.join(output_dir, "test")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_trainval_dir, exist_ok=True)
    os.makedirs(output_test_dir, exist_ok=True)

    with open(train_names_json, "r") as f:
        train_names = json.load(f)
    with open(val_names_json, "r") as f:
        val_names = json.load(f)

    # Process training data
    with tqdm_joblib(tqdm(desc="Processing training files", total=len(train_names), ncols=80)):
        output_train_names = joblib.Parallel(n_jobs=configs.joblib.jobs, verbose=0)(
            joblib.delayed(process_file)(
                os.path.join(input_dir, name),
                output_trainval_dir,
                "_".join(name.split("/")[2:])
            )
            for name in train_names
        )

    with open(os.path.join(output_dir, "train.json"), "w") as f:
        json.dump([os.path.join("trainval", name) for name in output_train_names], f, indent=4)

    # Process validation data
    with tqdm_joblib(tqdm(desc="Processing validation files", total=len(val_names), ncols=80)):
        output_val_names = joblib.Parallel(n_jobs=configs.joblib.jobs, verbose=0)(
            joblib.delayed(process_file)(
                os.path.join(input_dir, name),
                output_trainval_dir,
                "_".join(name.split("/")[2:])
            )
            for name in val_names
        )

    with open(os.path.join(output_dir, "val.json"), "w") as f:
        json.dump([os.path.join("trainval", name) for name in output_val_names], f, indent=4)

    subprocess.run(shlex.split(f"rm -r {os.path.join(output_trainval_dir, 'htdemucs')}"))  # Clean up intermediate files

    # Processing test data
    test_names = [f for f in os.listdir(test_dir) if f.endswith('.mp3')]
    with tqdm_joblib(tqdm(desc="Processing test files", total=len(test_names), ncols=80)):
        _ = joblib.Parallel(n_jobs=configs.joblib.jobs, verbose=0)(
            joblib.delayed(process_file)(
                os.path.join(test_dir, name),
                output_test_dir,
                name
            )
            for name in test_names
        )
    
    subprocess.run(shlex.split(f"rm -r {os.path.join(output_test_dir, 'htdemucs')}"))  # Clean up intermediate files

if __name__ == "__main__":
    main()