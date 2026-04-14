import os
import boto3
from botocore.exceptions import ClientError

ENDPOINT=""
ACCESS_KEY = ''
SECRET_KEY = ''
BUCKET=""
PREFIX_BASE = ""

local_savedir = ""

def main():
    """Test connection to s3 data, list all files"""
    s3 = boto3.client(
        "s3",
        endpoint_url=ENDPOINT,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        config=boto3.session.Config(signature_version="s3v4"),
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
        prefix = f"{PREFIX_BASE}/month=01/day=15/"
        print(f"Checking: {prefix}")
        
        all_files = list_files(BUCKET, prefix)
        print(f"  → found {len(all_files)} files")
        if all_files:
            print("    sample:", all_files[:3])
        else:
            print(f"No files found for prefix: {prefix}")
    except Exception as e:
        print(f"Error listing files: {e}")

if __name__ == "__main__":
    main()
