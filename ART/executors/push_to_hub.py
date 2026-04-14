import os
import argparse
from huggingface_hub import HfApi, create_repo, upload_folder
from transformers import AutoConfig, AutoModelForSequenceClassification


def build_argparser():
    p = argparse.ArgumentParser(description="Push a saved HF model directory to Hugging Face Hub")
    p.add_argument("--model_dir", type=str, required=True, help="Directory produced by save_pretrained")
    p.add_argument("--repo_id", type=str, required=True, help="org/name or username/name on the Hub")
    p.add_argument("--private", action="store_true", default=False)
    p.add_argument("--token", type=str, default=None, help="Hugging Face token; if not provided, will use env HF_TOKEN or stored login")
    return p


def main():
    args = build_argparser().parse_args()

    # Sanity check the folder can be loaded
    _ = AutoConfig.from_pretrained(args.model_dir)
    _ = AutoModelForSequenceClassification.from_pretrained(args.model_dir)

    token = args.token or os.environ.get("HF_TOKEN")

    api = HfApi()
    create_repo(repo_id=args.repo_id, private=args.private, exist_ok=True, token=token)

    print(f"Uploading folder {args.model_dir} to {args.repo_id} ...")
    upload_folder(
        repo_id=args.repo_id,
        folder_path=args.model_dir,
        path_in_repo=".",
        token=token,
    )
    print("Done.")


if __name__ == "__main__":
    main()
