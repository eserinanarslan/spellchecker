"""
Pull the training file de_corrections.txt and product index file
from the SEBE bucket and pushes it to the bucket in the DSC project.
"""

import sys
import os
sys.path.append(os.path.join('..', 'notebooks'))
import config
import utils

# Gets the date of the product index from the SEBE bucket
## and creates a log in the version file.
# prod_index_date = utils.blob_metadata(
#     config.SEBE_PROD_INDEX_BUCKET,
#     f'{sys.argv[1]}/{config.PROD_INDEX_FILE}')

prod_index_date = sys.argv[1]
# dir in gcp bucket
prefix = os.path.join(config.RAW_DATA_DIR, prod_index_date)
# dir in local storage
data_dir = os.path.join(config.ARTIFACTS_DIR, prefix)
os.makedirs(data_dir, exist_ok=True)

# Download the training file de_corrections.txt from the sebe bucket
print("1. Downloading the training file corrections.txt")
utils.download_from_bucket(
    bucket_name=config.SEBE_TRAINING_SOURCE_BUCKET,
    prefix=f'corrections/{config.LANG}_corrections.txt',
    dest_folder=data_dir, project=config.SEBE_GCP_PROJECT)
# Download the product index file from the sebe bucket
print("2. Downloading the product index file")
utils.download_from_bucket(
    bucket_name=config.SEBE_PROD_INDEX_BUCKET,
    prefix=sys.argv[1],
    dest_folder=data_dir,
    project=config.SEBE_GCP_PROJECT)

print(f"Uploading files to gs://{config.DSC_BUCKET}/{config.RAW_DATA_DIR}")
utils.upload_to_bucket(project=config.DSC_GCP_PROJECT,
    bucket_name=config.DSC_BUCKET, folder_name=data_dir,
    prefix=prefix)

