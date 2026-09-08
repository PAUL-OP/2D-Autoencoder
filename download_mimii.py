"""
Download + unzip one MIMII dataset split from Zenodo.

Run this on your own machine (not inside a network-restricted sandbox —
zenodo.org needs to be reachable). It just automates:
    1. downloading "<snr>_dB_<machine>.zip" from Zenodo record 3384388
    2. unzipping it into --out_dir

Usage:
    python download_mimii.py --machine fan --snr 6 --out_dir ./mimii_data
    python mimii_pipeline.py --data_root ./mimii_data --machine fan
"""
import argparse
import os
import zipfile
import urllib.request

# Files hosted on the MIMII Zenodo record: https://zenodo.org/record/3384388
BASE_URL = "https://zenodo.org/record/3384388/files"


def download_file(url, dest_path):
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, dest_path)
    print(f"Saved to {dest_path}")


def main():
    parser = argparse.ArgumentParser(description="Download and unzip a MIMII dataset split")
    parser.add_argument("--machine", required=True, choices=["fan", "pump", "slider", "valve"])
    parser.add_argument("--snr", required=True, choices=["-6", "0", "6"], help="Signal-to-noise ratio split")
    parser.add_argument("--out_dir", default="./mimii_data")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    zip_name = f"{args.snr}_dB_{args.machine}.zip"
    url = f"{BASE_URL}/{zip_name}"
    zip_path = os.path.join(args.out_dir, zip_name)

    if not os.path.exists(zip_path):
        download_file(url, zip_path)
    else:
        print(f"{zip_path} already exists, skipping download.")

    print(f"Unzipping {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(args.out_dir)

    print(
        f"\nDone. Your data should now be under:\n"
        f"  {args.out_dir}/{args.snr}_dB_{args.machine}/{args.machine}/id_XX/{{normal,abnormal}}/*.wav\n\n"
        f"Point mimii_pipeline.py at the parent of the '{args.machine}' folder, e.g.:\n"
        f"  python mimii_pipeline.py --data_root {args.out_dir}/{args.snr}_dB_{args.machine} --machine {args.machine}"
    )


if __name__ == "__main__":
    main()
