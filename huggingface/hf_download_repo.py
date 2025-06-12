#!/usr/bin/env python3  
"""  
Recursively download files from a Hugging Face Hub repository.  
  
Examples  
========  
1) Entire repo  
   python hf_download.py yili18/Hamster_dev  
  
2) Specific branch  
   python hf_download.py yili18/Hamster_dev -r dev  
  
3) Only one folder  
   python hf_download.py yili18/Hamster_dev \  
       -s "VILA1.5-13b-robopoint_1432k+rlbench_all_tasks_256_1000_eps_sketch_v5_alpha+droid_train99_sketch_v5_alpha_fix+bridge_data_v2_train90_10k_sketch_v5_alpha-e1-LR1e-5" \  
       -o ./hf_files  
"""  
import argparse  
from pathlib import Path  
from huggingface_hub import snapshot_download  
  
  
def parse_args() -> argparse.Namespace:  
    p = argparse.ArgumentParser(  
        description="Recursively download all (or part of) a Hugging Face repo."  
    )  
    p.add_argument("repo_id", help="Repo name, e.g. 'yili18/Hamster_dev'")  
    p.add_argument(  
        "-r", "--branch", default="main",  
        help="Branch / tag / commit SHA to download (default: main)"  
    )  
    p.add_argument(  
        "-s", "--subdir", default="",  
        help="Path inside the repo to download (default: whole repo)"  
    )  
    p.add_argument(  
        "-o", "--local-dir", default=".",  
        help="Destination directory on disk (default: current directory)"  
    )  
    p.add_argument(  
        "--no-symlinks", action="store_true",  
        help="Copy files instead of symlinking to HF cache."  
    )  
    return p.parse_args()  
  
  
def main() -> None:  
    args = parse_args()  
  
    # Build allow_patterns only when the user asked for a sub-directory  
    allow_patterns = None  
    if args.subdir:  
        # include the folder itself (if it holds a file) and everything below it  
        allow_patterns = [f"{args.subdir}/*", args.subdir]  
  
    dst: Path = Path(args.local_dir).expanduser().resolve()  
    dst.mkdir(parents=True, exist_ok=True)  
  
    print(  
        f"Downloading from repo: {args.repo_id}\n"  
        f"revision          : {args.branch}\n"  
        f"sub-directory     : {args.subdir or '(whole repo)'}\n"  
        f"destination       : {dst}\n"  
    )  
  
    snapshot_download(  
        repo_id=args.repo_id,  
        revision=args.branch,  
        local_dir=str(dst),  
        local_dir_use_symlinks=not args.no_symlinks,  
        allow_patterns=allow_patterns,  
    )  
  
    print("\n✅ Done.")  
  
  
if __name__ == "__main__":  
    main()  