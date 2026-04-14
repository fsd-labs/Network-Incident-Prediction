import os
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config

ENDPOINT=""
ACCESS_KEY = ''
SECRET_KEY = ''
BUCKET=""
PREFIX_BASE = ""

local_savedir = ""

MONTH_RANGE = range(11, 13)
DAY_RANGE = range(1, 32)

def main():
    s3 = boto3.client(
        "s3",
        endpoint_url=ENDPOINT,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        config=Config(signature_version="s3v4"),
        verify=True,
    )

    def list_files(bucket, prefix):
        paginator = s3.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix)

        all_files = []
        for page in page_iterator:
            for obj in page.get('Contents', []):
                if obj['Key'].endswith(".parquet"):
                    all_files.append(obj['Key'])
        return all_files

    try:
        files = []
        for month in MONTH_RANGE:
            if month == 1:
                DAY_RANGE = range(15, 32)
            elif month == 2:
                DAY_RANGE = range(1, 11)
            for day in DAY_RANGE:
                prefix = f"{PREFIX_BASE}/month={month:02d}/day={day:02d}/"
                print(f"Checking: {prefix}")
                
                day_files = list_files(BUCKET, prefix)
                print(f"  → found {len(day_files)} files")
                if day_files:
                    print("    sample:", day_files[:3])

                files.extend(day_files)

        if not files:
            print(f"No files found for any day in month={month:02d}-day={day:02d}/year=2025.")
            return

        print(f"\nTotal files to download: {len(files)}\n")

        for key in files:
            local_path = os.path.join(local_savedir, key)

            if os.path.exists(local_path):
                print(f"Skip (exists): {key}")
                continue

            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            try:
                s3.download_file(BUCKET, key, local_path)
                print(f"Downloaded {key}")
            except ClientError as e:
                print(f"Failed to download {key}: {e}")

    except ClientError as e:
        print(f"S3 error: {e}")


if __name__ == "__main__":
    main()
