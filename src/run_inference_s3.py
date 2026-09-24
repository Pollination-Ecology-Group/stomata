#!/usr/bin/env python3
import os
import io
import time
import pandas as pd
import concurrent.futures
from PIL import Image

import boto3
from botocore.client import Config
from ultralytics import YOLO

# Import our processing function
from extract_traits import process_image

S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "https://s3.cl4.du.cesnet.cz")
S3_BUCKET = os.environ.get("S3_BUCKET", "bucket")
TILE_PREFIX = os.environ.get("TILE_PREFIX", "Ostatni/Stomata/extracted/")
MODEL_PATH = os.environ.get("MODEL_PATH", "/tmp/best.pt")
S3_MODEL_KEY = os.environ.get("S3_MODEL_KEY")
FETCH_WORKERS = int(os.environ.get("FETCH_WORKERS", "16"))
OUTPUT_CSV = os.environ.get("OUTPUT_CSV", "/tmp/stomata_traits.csv")
S3_OUTPUT_KEY = os.environ.get("S3_OUTPUT_KEY", "Ostatni/Stomata/stomata_traits.csv")

def get_s3_client():
    return boto3.client('s3',
                        endpoint_url=S3_ENDPOINT,
                        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
                        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
                        config=Config(signature_version='s3v4'))

def download_image(s3_client, key):
    try:
        obj = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
        image_data = obj['Body'].read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        return key, image
    except Exception as e:
        print(f"Error downloading {key}: {e}")
        return key, None

def main():
    s3 = get_s3_client()
    
    if S3_MODEL_KEY and not os.path.exists(MODEL_PATH):
        print(f"Downloading model {S3_MODEL_KEY} to {MODEL_PATH}...")
        s3.download_file(S3_BUCKET, S3_MODEL_KEY, MODEL_PATH)
        print("Model downloaded.")
        
    print(f"Loading YOLO model from {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    print(f"Listing images in {S3_BUCKET}/{TILE_PREFIX}")
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=S3_BUCKET, Prefix=TILE_PREFIX)
    
    image_keys = []
    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                if key.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_keys.append(key)
                    
    print(f"Found {len(image_keys)} images.")
    
    all_results = []
    
    start_time = time.time()
    
    # Process images using thread pool for downloading
    with concurrent.futures.ThreadPoolExecutor(max_workers=FETCH_WORKERS) as executor:
        # We submit in batches to avoid high memory usage if there are millions of images
        batch_size = 100
        for i in range(0, len(image_keys), batch_size):
            batch_keys = image_keys[i:i+batch_size]
            
            # Download images concurrently
            futures = [executor.submit(download_image, s3, key) for key in batch_keys]
            
            for future in concurrent.futures.as_completed(futures):
                key, img = future.result()
                if img is not None:
                    # Run inference (YOLO handles PIL images directly)
                    # We pass the basename as image_name, or the full key if preferred
                    image_name = os.path.basename(key)
                    res = process_image(model, img, image_name)
                    # Add the full S3 key to the results as well
                    for r in res:
                        r['s3_key'] = key
                    all_results.extend(res)
            
            if i % 1000 == 0 and i > 0:
                print(f"Processed {i}/{len(image_keys)} images... Time elapsed: {time.time() - start_time:.2f}s")
                # Save partial results locally
                pd.DataFrame(all_results).to_csv(OUTPUT_CSV, index=False)
                
    # Final save
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"Saved {len(all_results)} stomata measurements to {OUTPUT_CSV}")
        
        # Upload back to S3
        print(f"Uploading results to s3://{S3_BUCKET}/{S3_OUTPUT_KEY}")
        s3.upload_file(OUTPUT_CSV, S3_BUCKET, S3_OUTPUT_KEY)
        print("Upload complete!")
    else:
        print("No stomata detected in any image.")

if __name__ == "__main__":
    main()
