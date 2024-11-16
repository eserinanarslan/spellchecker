import urllib3
import json
import utils
import os
import sys
sys.path.append(os.path.join('..','src'))

from bs4 import BeautifulSoup

import config
import suggest_pipeline


def pass_version(user_name, train_date, arc_version):
    """
    Function to download a specific model based on the provided version details
    Args:
        user_name (str): Name of the user who trained the model
        train_date (str): Unix timestamp in str type of the trained model
        arc_version (str): Architecture version of the trained model
    """
    global spelling_correction
    model_path_prefix = os.path.join(
        user_name, config.MODELS_FOLDER, arc_version, train_date)
    dest_path = os.path.join(config.ARTIFACTS_DIR, model_path_prefix)
    if not os.path.exists(dest_path):
        utils.download_from_bucket(project=config.DSC_GCP_PROJECT,
            bucket_name=config.DSC_BUCKET,
            prefix=model_path_prefix,
            dest_folder=config.ARTIFACTS_DIR)
    # TODO extract model chars from the file model_metadata.json instead of using config.MODEL_CHARS
    spelling_correction = suggest_pipeline.Suggester(dest_path, \
                          model_chars=config.MODEL_CHARS)

def ml_spellchecker(input_row):
    """
    Correct users query
    """
    corrected_list = list()
    for tuple_element in input_row:
        test_dict = dict()
        if tuple_element[1] != 0:
            corrected_word = spelling_correction.correct_suggestion(tuple_element[1])[0][0]
            test_dict['actual'] = tuple_element[0]
            test_dict['noisy'] = tuple_element[1]
            test_dict['edit_dist'] = tuple_element[2]
            test_dict['ml_predicted'] = corrected_word
            corrected_list.append(test_dict)
    return corrected_list


def be_spellchecker(input_row, api_endpoint): 
    corrected_list = list()
    for tuple_element in input_row:
        test_dict = dict()
        if tuple_element[1] != 0:
        
            data = {
                "query": tuple_element[1]
            }
            req_headers = {
                'Content-Type': 'application/json',
            }

            http = urllib3.PoolManager()
            encoded_data = json.dumps(data).encode('utf-8')
            r = http.request('POST', api_endpoint,
                        headers=req_headers,
                        body=encoded_data)

            resp_body = r.data.decode('utf-8')
            resp_dict = json.loads(r.data.decode('utf-8'))

            test_dict['actual'] = tuple_element[0]
            test_dict['noisy'] = tuple_element[1]
            test_dict['edit_dist'] = tuple_element[2]
            if resp_dict.get('meta'):
                if resp_dict['meta']['correction'] is None:
                    test_dict['be_predicted'] = 'None'
                else:
                    soup = BeautifulSoup(resp_dict['meta']['correction']['corrected'],
                                        features="html.parser")
                    test_dict['be_predicted'] = soup.em.string
            else:
                print("Exception:", resp_dict)
            
            corrected_list.append(test_dict)
    return corrected_list


def create_label(row):
    """
    Check whether actual word is equal to the predicted word from ML and BE spellcheckers
    Params:
        tuple: Tuple of actual, first_model_result, second_model_result
            e.g. (1, 1, 1)
    """
    if row[0] == row[1] and row[0] == row[2]:
        return 1, 1
    elif row[0] != row[1] and row[0] != row[2]:
        return 0, 0
    elif row[0] != row[1] and row[0] == row[2]:
        return 0, 1
    elif row[0] == row[1] and row[0] != row[2]:
        return 1, 0