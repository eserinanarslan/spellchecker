import pathlib
import os
import string

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent

RAW_DATA_DIR = 'raw_data'
LANG = 'de'
SEBE_GCP_PROJECT = 'de-ecom-1505-conradsearch'
# de.corrections.txt is stored
SEBE_TRAINING_SOURCE_BUCKET = 'search-external-files-mr'
# Product index is stored
SEBE_PROD_INDEX_BUCKET = 'search-ml-spellchecker'
PROD_INDEX_FILE = 'terms_map_v2.json'

#----------------------------------------------------------------#
DEV_ENV = os.getenv('PIPELINE_DEV_ENV', 'dev')
_ENV_CONFIGS = { # dev_environment -> (gcp_project, artifacts_bucket)
    'dev': ('datascience-dev-319609', 'dsc-ml-dev-spellchecker'),
    'staging': ('datascience-staging-319609', 'dsc-ml-staging-spellchecker'),
    'prod': ('datascience-prod-319609', 'dsc-ml-prod-spellchecker'),
}
DSC_GCP_PROJECT, DSC_BUCKET = _ENV_CONFIGS[DEV_ENV]
#----------------------------------------------------------------#
EDIT_DIST = 3
# The maximum length of a word
MODEL_CHARS = [*list(string.ascii_lowercase)]
MAX_WORD_LEN = len(MODEL_CHARS)

ARTIFACTS_DIR = os.path.join(os.path.expanduser("~"), "conrad_ml_spellchecker")
TRAINING_DATA_DIR = 'training_data'
TRAINING_FILE = f'edist_{EDIT_DIST}_maxlen_{MAX_WORD_LEN}_tuples.txt'
#-----------------------------------------------------------------#
PROD_INDEX_DIR =  'preprocessed_product_index'
MAX_COUNT=10 # maximum number of times a word appears in the products index
PROCESSED_PROD_INDEX_FILE = f'max_count_{MAX_COUNT}_product_index.json'
#-----------------------------------------------------------------#
MODELS_FOLDER = "models"
EMB_DIM = 100 # embedding vector size
PATIENCE = 1
VAL_SPLIT = 0.05
BATCH_SIZE = 64
#------------------------------------------------------------------#
VECTORS_FILE = "train.bin"
VECTORIZED_DICTIONARY_FOLDER = 'vectorized_dictionary'
#------------------------------------------------------------------#