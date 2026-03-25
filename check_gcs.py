from src.services.gcp_storage_service import GCPStorageService
from src.config.settings import IMAGE_BUCKET_NAME
import os
import pandas as pd

def check():
    service = GCPStorageService()
    print(f"🔍 Checking bucket: {IMAGE_BUCKET_NAME}")
    
    blobs = service.list_blobs(IMAGE_BUCKET_NAME, prefix='magic-eye/')
    print(f"Total blobs: {len(blobs)}")
    
    has_meta = any(b.name == 'magic-eye/metadata.csv' for b in blobs)
    print(f"Has metadata.csv: {has_meta}")
    
    if has_meta:
        local_path = "./temp_metadata_remote.csv"
        if service.download_file(IMAGE_BUCKET_NAME, 'magic-eye/metadata.csv', local_path):
            df = pd.read_csv(local_path)
            print("\nRemote Metadata Split Distribution:")
            print(df['split'].value_counts())
            os.remove(local_path)
        else:
            print("❌ Failed to download metadata.csv")
    
    # Check a few random folders
    folders = set()
    for b in blobs:
        parts = b.name.split('/')
        if len(parts) > 2:
            folders.add(parts[1])
    
    print(f"\nFolders found in GCS: {sorted(list(folders))}")

if __name__ == "__main__":
    check()
