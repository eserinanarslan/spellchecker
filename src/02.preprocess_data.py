"""
Refinement of the product index words from SEBE uisng doc_count
* Input - terms_map.json from bucket search-ml-spellchecker in de-ecom-1505-conradsearch
* Output - limited_terms_map_v2_dup.json pushed to bucket data_spellchecker in ecom-ai-poc
"""

import json
import time
import sys
import os
sys.path.append(os.path.join('..', 'notebooks'))

import utils
from data_utils import clean_dictionary
import config


def truncate_product_index(trigger_date, prod_index_ver, max_count):
    """
    Download the product index file from SEBE and truncates it to a doc_count
    Args: 
        trigger_date(str): Date when product index was create at SEBE side
        max_count (int): Count below which terms will not be considered in the 
                         spellchecker dictionary
    """
    prefix = os.path.join(
        config.RAW_DATA_DIR,
        trigger_date,
        prod_index_ver,
        config.PROD_INDEX_FILE)
    utils.download_from_bucket(
        project=config.DSC_GCP_PROJECT,
        bucket_name=config.DSC_BUCKET,
        prefix=prefix,
        dest_folder=os.path.join(config.ARTIFACTS_DIR))
    with open(os.path.join(config.ARTIFACTS_DIR, prefix), 'r') as file_:
        product_index = json.load(file_)

    limited_dict = clean_dictionary(product_index, max_count=max_count)
    return limited_dict


def main():
    timestamp = int(time.time())
    username = sys.argv[1]
    prod_index_ver = sys.argv[2]
    # Uploading the output of the pipeline step to the storage bucket
    # trigger_date = utils.blob_metadata(config.SEBE_PROD_INDEX_BUCKET,
    #                                   f'{prod_index_ver}/{config.PROD_INDEX_FILE}')
    truncated_dict = truncate_product_index(
        prod_index_ver, prod_index_ver, max_count=config.MAX_COUNT)
    
    # Serialize the truncated dictionary in pickle format
    prefix = os.path.join(username, config.PROD_INDEX_DIR, str(timestamp))
    pre_process_dir = os.path.join(config.ARTIFACTS_DIR,
        config.PROD_INDEX_DIR, str(timestamp))
    os.makedirs(pre_process_dir, exist_ok=True)
    with open(os.path.join(
        pre_process_dir, config.PROCESSED_PROD_INDEX_FILE), 'w') as fp:
        json.dump(truncated_dict, fp, indent=4)

    product_index_metadata_path = os.path.join(
        pre_process_dir, "product_index_metadata.json")
    with open(product_index_metadata_path, 'w') as fp:
        json.dump({"generation_date": timestamp,
            "dictionary_len": len(truncated_dict),
            "data_date": prod_index_ver,
            "max_count": config.MAX_COUNT
        }, fp)
    utils.upload_to_bucket(
        project=config.DSC_GCP_PROJECT,
        bucket_name=config.DSC_BUCKET, folder_name=pre_process_dir,
        prefix=prefix)

    print(f"Data successfully pre-processed and uploaded to gs://{config.DSC_BUCKET}/{prefix}")
    logs_dir = os.path.join(config.ARTIFACTS_DIR, 'logs')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir, exist_ok=True)
    with open(
        os.path.join(logs_dir, f'preprocessing_{os.getpid()}.txt'), 'w') as file_:
        file_.write(f'timestamp|{timestamp}')

if __name__=='__main__':
    print("===== Preprocessing data ...")
    main()