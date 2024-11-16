import os
import logging
import pickle
import time
import json
import sys
sys.path.append(os.path.join('..', 'notebooks'))

import numpy as np
from sklearn.utils import shuffle
import click

import char2vec_model
import config
import utils


_logger = logging.getLogger(__name__)


def load_data(file_name, nb_samples=None):
    """
    Load the training data for the spell checker and shuffles its order for good training weights optimization.
    Args:
        file_name (str): path for the training file.
        
    Returns:
        X_train (tuple): tuples of word pairs.
        y_train (int): correctness label of word pairs .
    """
    print(f"Loading Data from {file_name}")
    with open(file_name, "rb") as file_:
        tr_data = pickle.load(file_)
    X_train = list(tr_data.keys())
    y_train = list(tr_data.values())
    X_train, y_train = shuffle(X_train, y_train)
    _logger.info("Raw data for training loaded successfully")
    if nb_samples:
        return X_train[:nb_samples], y_train[:nb_samples]
    else:
        return X_train, y_train


def train_model(embedding_dim, X_train, y_train, model_chars,
                max_word_len, max_epochs, patience, validation_split,
                batch_size, model_path):
    """
    Initialize the NN architecture,compile and fit to the training data, eventually save the trained model.
    Args:
        embedding_dim (int): output dimension of model
        X_train (tuple): tuple of word pairs
        y_train (int): label for correctness
        model_chars (list): list of characters for the model (a to z)
        max_word_len(int): parameter 'maxlen' for keras pad_sequences transform.
        model_path (str): Target model path
    Returns:
        Fitted model
    """
    char_to_ix = {ch: i for i, ch in enumerate(model_chars)}
    c2v_model = char2vec_model.Chars2Vec(embedding_dim, char_to_ix, max_word_len)

    targets = np.array(y_train)
    c2v_model.fit(X_train, targets, max_epochs, patience, validation_split,
        batch_size, model_path)
    return c2v_model


@click.command()
@click.option('--n_samples', '-n', type=int, 
    help='Number of training samples',
    default=None)
@click.option('--username', '-u', type=str,
    help='user name. Default: local system username')
@click.option('--data_date', '-d',  type=str, help='Training data date')
@click.option('--arc_version', '-v', type=str,
    help='Short description for Model architecture or version number')
@click.option('--n_epochs', '-e', type=int,
    help='Number of epochs')
def main(n_samples, username, data_date, arc_version, n_epochs):
    # Training_data params
    print(f"=== Training the model for {n_samples} samples and {n_epochs} epochs")
    training_data_prefix = os.path.join(
        username, config.TRAINING_DATA_DIR, data_date)
    data_path = os.path.join(config.ARTIFACTS_DIR)

    # Model training
    print("Downloading and loading the training data ...")
    utils.download_from_bucket(
        bucket_name=config.DSC_BUCKET,
        prefix=training_data_prefix, dest_folder=data_path,
        project=config.DSC_GCP_PROJECT)
    
    data_filename = os.path.join(
        data_path, training_data_prefix, config.TRAINING_FILE)
    X_train, y_train = load_data(
        file_name=data_filename, nb_samples=n_samples)
    nb_training_samples = len(X_train)
    print(f"Nb training samples: {nb_training_samples}")
    _logger.info("Dataset loaded successfully")
    
    timestamp = int(time.time())
    prefix = os.path.join(
        username, config.MODELS_FOLDER,
        arc_version, str(timestamp))
    model_path = os.path.join(
        config.ARTIFACTS_DIR,
        config.MODELS_FOLDER,
        arc_version, str(timestamp))
    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)

    print(f"Training the model then saving it to {model_path} ...")
    train_model(
        config.EMB_DIM, X_train, y_train,
        config.MODEL_CHARS,
        config.MAX_WORD_LEN,
        n_epochs, config.PATIENCE,
        config.VAL_SPLIT, config.BATCH_SIZE,
        model_path
    )

    model_details = {
        'train_data_date': data_date,
        'training_date': timestamp,
        'n_data_samples': n_samples,
        'nb_training_samples': nb_training_samples,
        'data_file': config.TRAINING_FILE,
        'emd_dim': config.EMB_DIM,
        'n_epochs': n_epochs,
        'model_chars': config.MODEL_CHARS,
        'validation_split': config.VAL_SPLIT,
        'early_stopping_patience': config.PATIENCE,
        'batch_size': config.BATCH_SIZE,
        'max_word_len': config.MAX_WORD_LEN,
        'language': config.LANG,
        'nb_train_samples': len(X_train)
    }
    metadata_filepath = os.path.join(model_path, "model_metadata.json")
    with open(metadata_filepath, 'w') as outfile:
        json.dump(model_details, outfile, indent=4)

    utils.upload_to_bucket(project=config.DSC_GCP_PROJECT,
        bucket_name=config.DSC_BUCKET,
        prefix=prefix,
        folder_name=model_path)

    print(f"Model successfully trained and uploaded to gs://{config.DSC_BUCKET}/{prefix}")
    logs_dir = os.path.join(config.ARTIFACTS_DIR, 'logs')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir, exist_ok=True)
    with open(
        os.path.join(logs_dir,
            f'training_{os.getpid()}.txt'), 'w') as file_:
        file_.write(f'timestamp|{timestamp}')


if __name__=='__main__':
    main()
