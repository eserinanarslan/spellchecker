from importlib.resources import path
import os
from itertools import islice
import json
from posixpath import relpath
import sys
sys.path.append(os.path.join('..', 'src'))
import string
import time
import sys
import requests


from google.cloud import storage

def blob_metadata(bucket_name, blob_name):
    """
    Get the date of the file last updated from a bucket
    Args:
        bucket_name(str)
        blob_name(str)
    Returns:
        Date of the file updated
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.get_blob(blob_name)
    return str(blob.updated.date())
    
def chunks(lst, nb):
    """
    Chuncks generator from an input list
    Args: 
        lst (list)
        nb (int): number of sample in each chunck
    """
    for i in range(0, len(lst), nb):
        yield lst[i:i + nb]


def load_from_json(filepath: str):
    """
    Load json data from file
    Params:
        filepath: file path
    Return:
        dictionary: Json data
    """
    with open(filepath, 'r') as myfile:
        data = myfile.read()
        data_dict = json.loads(data)
    return data_dict

def write_to_json(filepath: str, data_dict: dict, mode="w"):
    """
    Writes data to json file
    Params:
        filepath: file path
        data_dict: data to be written
    Return:
        .json file for the dictionary
    """
    with open(filepath, mode) as file_json:  
        json.dump(data_dict, file_json)


def upload_to_bucket(project: str, bucket_name: str, prefix: str, folder_name: str):
    """Upload files to GCP bucket while keeping the original directories
        structure
    """
    print(f"Uploading files from {folder_name} to gs://{bucket_name}/{prefix}")
    storage_client = storage.Client(project=project)
    bucket = storage_client.get_bucket(bucket_name)
    success = False
    retries = 1
    max_retries = 2
    while not success:
        try:
            if retries == max_retries:
                break
            for path_, _, files in os.walk(folder_name):
                for file_name in files:
                    relpath_ = path_.replace(folder_name, '').lstrip('/')
                    blob_path = os.path.join(prefix, relpath_, file_name)
                    print(f'rel {prefix} {relpath_} {file_name}')
                    print(f"blob from {os.path.join(path_, file_name)} to {blob_path}")
                    blob = bucket.blob(blob_path)
                    blob.upload_from_filename(os.path.join(path_, file_name), timeout=300)
                    success = True
        except requests.exceptions.ConnectionError as exp:
            wait = 10 # waiting for 10 seconds before retrying
            print(f"Timeout while downloading file! Waiting {wait} secs and re-trying...")
            sys.stdout.flush()
            time.sleep(wait)
            retries += 1

    print(f"Files successfully uploaded to gs://{bucket_name}/{prefix}")


def download_from_bucket(project: str, bucket_name: str, prefix: str, dest_folder: str):
    """
    Downloading files from gs://<bucket_name>/<prefix>
    """
    print(f"Downloading files from gs://{bucket_name}/{prefix} to {dest_folder}")
    storage_client = storage.Client(project=project)
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    try:  
        for blob in blobs:
            if blob.name.endswith("/"):
                print(f"skipping {blob.name}")
                continue
            file_split = blob.name.split("/")
            directory = "/".join(file_split[0:-1])
            os.makedirs(os.path.join(dest_folder, directory), exist_ok=True)
            blob.download_to_filename(
                os.path.join(dest_folder, blob.name))
    except Exception as exp:
        print(
            f"An exception occured while downloading file from \
gs://{bucket_name}/{prefix}", repr(exp))
        raise exp

def take(nb_els, iterable):
    """Return first n items of the iterable as a list"
    params:
        nb_els(int): number of elements to extract
        iterable: iterbale object
    return:
        list of n iterable values
    """
    return list(islice(iterable, nb_els))
