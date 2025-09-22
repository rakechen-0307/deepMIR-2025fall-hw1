import json
import shlex
import joblib
import pathlib
import argparse
import subprocess
from tqdm import tqdm

from ..commons import tqdm_joblib

def parse_args():
    parser = argparse.ArgumentParser(description="Separate vocals from songs using demucs.")
    parser.add_argument("--input_dir", required=True, type=str, help="input audio file path")
    parser.add_argument("--output_dir", required=True, type=str, help="output directory for separated vocals")
    parser.add_argument("--jobs", default=1, type=int, help="number of parallel jobs")
    return parser.parse_args()

def process_file(input_path, type, output_dir, output_name, bitrate='32k', sr=16000):
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

    src_vocals = output_dir / "htdemucs" / str(input_path.name).replace(".mp3", "") / "vocals.wav"
    dst_vocals = output_dir / "vocals" / type / output_name
    dst_vocals.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        shlex.split(f"ffmpeg -y -i {src_vocals} -ar {sr} -b:a {bitrate} -loglevel quiet {dst_vocals}"),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True
    )

    src_inst = output_dir / "htdemucs" / str(input_path.name).replace(".mp3", "") / "no_vocals.wav"
    dst_inst = output_dir / "inst" / type / output_name
    dst_inst.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        shlex.split(f"ffmpeg -y -i {src_inst} -ar {sr} -b:a {bitrate} -loglevel quiet {dst_inst}"),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True
    )

    return output_name

def main():
    args = parse_args()
    input_dir = pathlib.Path(args.input_dir)
    input_test_dir = input_dir / "test"
    train_names_json = input_dir / "train.json"
    val_names_json = input_dir / "val.json"
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(train_names_json, "r") as f:
        train_names = json.load(f)
    with open(val_names_json, "r") as f:
        val_names = json.load(f)

    # Process training data
    if args.jobs == 1:
        output_train_names = []
        for name in tqdm(train_names, desc="Processing training files", ncols=80):
            output_name = process_file(
                input_path=input_dir / name,
                type="train_val",
                output_dir=output_dir,
                output_name='/'.join(name.split('/')[2:])
            )
            output_train_names.append(output_name)
    else:
        with tqdm_joblib(tqdm(desc="Processing training files", total=len(train_names), ncols=80)):
            output_train_names = joblib.Parallel(n_jobs=args.jobs, verbose=0)(
                joblib.delayed(process_file)(
                    input_path=input_dir / name,
                    type="train_val",
                    output_dir=output_dir,
                    output_name='/'.join(name.split('/')[2:])
                )
                for name in train_names
            )

    with open(output_dir / "vocals" / "train.json", "w") as f:
        json.dump([f"./train_val/{name}" for name in output_train_names], f, indent=4)

    # Process validation data
    if args.jobs == 1:
        output_val_names = []
        for name in tqdm(val_names, desc="Processing validation files", ncols=80):
            output_name = process_file(
                input_path=input_dir / name,
                type="train_val",
                output_dir=output_dir,
                output_name='/'.join(name.split('/')[2:])
            )
            output_val_names.append(output_name)
    else:
        with tqdm_joblib(tqdm(desc="Processing validation files", total=len(val_names), ncols=80)):
            output_val_names = joblib.Parallel(n_jobs=args.jobs, verbose=0)(
                joblib.delayed(process_file)(
                    input_path=input_dir / name,
                    type="train_val",
                    output_dir=output_dir,
                    output_name='/'.join(name.split('/')[2:])
                )
                for name in val_names
            )

    with open(output_dir / "vocals" / "val.json", "w") as f:
        json.dump([f"./train_val/{name}" for name in output_val_names], f, indent=4)

    # Processing test data
    test_names = sorted([f.name for f in input_test_dir.iterdir() if str(f).endswith('.mp3')])
    if args.jobs == 1:
        for name in tqdm(test_names, desc="Processing test files", ncols=80):
            process_file(
                input_path=input_test_dir / name,
                type="test",
                output_dir=output_dir,
                output_name=name
            )
    else:
        with tqdm_joblib(tqdm(desc="Processing test files", total=len(test_names), ncols=80)):
            _ = joblib.Parallel(n_jobs=args.jobs, verbose=0)(
                joblib.delayed(process_file)(
                    input_path=input_test_dir / name,
                    type="test",
                    output_dir=output_dir,
                    output_name=name
                )
                for name in test_names
            )

    subprocess.run(shlex.split(f"rm -r {output_dir / 'htdemucs'}"))  # Clean up intermediate files

if __name__ == "__main__":
    main()