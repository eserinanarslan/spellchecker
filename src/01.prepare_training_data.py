"""
Training data extraction and preparation
Formatting of raw dataset to a structured format used by the model.
###
* Input file - de_corrections.txt from the bucket search-external-files-mr 
in de-ecom-1505-conradsearch
* Output file - training_words_3to25_de_to_en_dup.txt pushed to  bucket data_spellchecker 
in ecom-ai-poc
"""
import pickle
import os
import json
import sys
sys.path.append(os.path.join('..', 'notebooks'))
from pytextdist.edit_distance import levenshtein_distance
import time

import utils
from data_utils import generate_samples
import config


def generate_training_data(trigger_date, max_batch_size=1000):
    """
    Downloads the training data from SEBE bucket and creates training tuples
    Args:
        trigger_date (str) - Date when the training data was made
        available in SEBE bucket
    Returns
        similar_words, non_similar_words (tuples)
    """
    print("Generating training Data ...")
    file_prefix = os.path.join(
        config.RAW_DATA_DIR,
        trigger_date,
        'corrections',
        f'{config.LANG}_corrections.txt')
    print(f"Downloading training data from {config.DSC_BUCKET}")
    dest_folder = config.ARTIFACTS_DIR
    utils.download_from_bucket(
        project=config.DSC_GCP_PROJECT,
        bucket_name=config.DSC_BUCKET,
        prefix=file_prefix,
        dest_folder=dest_folder
    )
    print(f"Data successfully downloaded to {dest_folder}/{file_prefix}")
    with open(os.path.join(dest_folder, file_prefix)) as file_:
        content = [line.strip() for line in file_.readlines()]

    similar_words, non_similar_words = generate_samples(content, max_batch_size=max_batch_size)
    return similar_words, non_similar_words


def resolve_common_sample(similar_words, non_similar_words):
    """
    Remove common samples from existing in both similar and non_similar tuples
    Args:
        similar_words, non_similar_words (tuples)
    Returns
        training_dict (dict): Dictionary format of tuples as keys and label as values.
    """
    
    training_dict = {}
    common_samples = list(set(similar_words).intersection(set(non_similar_words)))

    for samples in common_samples:
        if levenshtein_distance(samples[0], samples[1]) <= 3:
            non_similar_words.pop(samples)
        else:
            similar_words.pop(samples)

    nb_similar = len(similar_words)
    nb_non_similar = len(non_similar_words)
    training_dict.update(similar_words)
    training_dict.update(non_similar_words)
    assert abs(nb_similar - nb_non_similar) < 100
    assert len(training_dict) == len(similar_words) + len(non_similar_words)
    return training_dict


def main():
    username = sys.argv[1]
    products_index_date = sys.argv[2]
    # trigger_date = utils.blob_metadata(
    #     config.SEBE_PROD_INDEX_BUCKET,
    #     f'{sys.argv[2]}/{config.PROD_INDEX_FILE}')
    timestamp = int(time.time())
    # 1. generate the training data: similar nd non similar words datasets
    similar_words, non_similar_words = generate_training_data(trigger_date=products_index_date)
    training_data_metadata = {
        'products_index_date': products_index_date,
        'allowed_edit_distance': config.EDIT_DIST,
        'max_word_len': config.MAX_WORD_LEN
    }

    # 2. Remove common samples from similar and non simalr words datasets
    # Reason for same samples appearing in similar and non_similar sets:
    #
    # * data point - 1
    # blindnietenzange, blindneutzange, blinnietzange, blintnietzange, blindndietzange,
    # bliendnietzange, bildnietzange, blindnizange, blinfnietzange => blindnietzange
    #
    # * data point - 2
    # blindnitenzange, bliendnietenzange => blindnietenzange

    # * (blindnietenzange, blindnitenzange) will form the similar samples from data point - 1
    # * (blindnietenzange, blindnitenzange) will form the non similar samples from data point - 1 and
    # data point - 2 using the words on the right side of =>.
    # Solution:
    # * If the elements of the tuple in the common samples have edit distance less than 3 then pop
    # out that element from non_similar_set else pop out that element from similar_set
    training_samples = resolve_common_sample(similar_words, non_similar_words)
    training_data_metadata['training_samples'] = len(training_samples)

    # 3. Serialize the training dictionary
    prefix = os.path.join(username, config.TRAINING_DATA_DIR, str(timestamp))
    data_dir = os.path.join(config.ARTIFACTS_DIR,
        config.TRAINING_DATA_DIR, str(timestamp))
    os.makedirs(data_dir, exist_ok=True)

    training_data_metadata['samples_generation_date'] = timestamp
    training_data_string = json.dumps(training_data_metadata)
    training_data_metadata_path = open(
        os.path.join(data_dir, "training_data_metadata.json"), "w")
    training_data_metadata_path.write(training_data_string)
    training_data_metadata_path.close()
    print("Serializing training samples")
    training_file_path = os.path.join(
        data_dir, config.TRAINING_FILE)
    with open(training_file_path, "wb") as file_:
        pickle.dump(training_samples, file_)

    # 4. Uploading the data to the cloud storage bucket
    utils.upload_to_bucket(project=config.DSC_GCP_PROJECT,
        bucket_name=config.DSC_BUCKET,
        folder_name=data_dir, prefix=prefix)

    print(f"Data preparation done and files uploaded\
 to gs://{config.DSC_BUCKET}/{prefix}")
    logs_dir = os.path.join(config.ARTIFACTS_DIR, 'logs')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir, exist_ok=True)
    with open(
        os.path.join(logs_dir,
            f'data_preparation_{os.getpid()}.txt'), 'w') as file_:
        file_.write(f'timestamp|{timestamp}')


if __name__=='__main__':
    print("===== Preparing training data ...")
    main()