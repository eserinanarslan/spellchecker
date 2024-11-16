import os
import logging
import config
import argparse
import sys
sys.path.append(os.path.join('..', 'notebooks'))

import gensim
import numpy as np
from numpy import float32

from vectorizer import Vectorizer
import utils

_logger = logging.getLogger(__name__)

def export_word2vec(fname, vocab, vectors, binary=True, total_vec=None):
    """
    Convert the dictionary {word:embedding} into
        word2vec format consumable by the gensim library.
    Args:
        fname (str): file location to save the word2vec C format file
        vocab (list): words from the clean dictionary from SEBE
        vectors (numpy.ndarray): 200 dimension vectors obtained from the trained model
    """
    if not (vocab or vectors):
        raise RuntimeError(
            f"No input was provided to export word2vec: {__name__}")
    total_vec = total_vec or len(vocab)
    vector_size = vectors.shape[1]
    assert (len(vocab), vector_size) == vectors.shape
    with gensim.utils.open(fname, 'wb') as fout:
        fout.write(gensim.utils.to_utf8("%s %s\n" % (total_vec, vector_size)))
        for word, row in vocab.items():
            fout.write(gensim.utils.to_utf8("%s %s\n" % (word, ' '.join(repr(val) for val in row))))


def export_binary_word2vec(fname, vocab, vectors, total_vec=None):
    """
    Convert the dictionary {word:embedding}
        into word2vec bin format consumable by the gensim library.
    Args:
        fname (str): file location to save the word2vec C format file
        vocab (list): words from the clean dictionary from SEBE
        vectors (numpy.ndarray): 200 dimension vectors obtained from the trained model
    """
    if not (vocab or vectors):
        raise RuntimeError(
            f"No input was provided to export word2vec: {__name__}")
    total_vec = total_vec or len(vocab)
    vector_size = vectors.shape[1]
    assert (len(vocab), vector_size) == vectors.shape
    with gensim.utils.open(fname, 'wb') as fout:
        fout.write(gensim.utils.to_utf8("%s %s\n" % (total_vec, vector_size)))
        for word, row in vocab.items():
            row = row.astype(float32)
            fout.write(gensim.utils.to_utf8(word) + b" " + row.tostring())


def save_w2v_model(model_path: str, data_prefix: str, emb_dim: int, bin_file_path: str):
    """
    Orchestrates the functions for creating the binary word2vec
        C format file from the dictionary of word:embedding.
    """
    data_path = os.path.join(config.ARTIFACTS_DIR, data_prefix)
    utils.download_from_bucket(
        project=config.DSC_GCP_PROJECT,
        bucket_name=config.DSC_BUCKET,
        prefix=data_prefix,
        dest_folder=os.path.join(config.ARTIFACTS_DIR))
        
    models_folder = os.path.join(config.ARTIFACTS_DIR, model_path)
    utils.download_from_bucket(
        project=config.DSC_GCP_PROJECT,
        bucket_name=config.DSC_BUCKET,
        prefix=model_path,
        dest_folder=os.path.join(config.ARTIFACTS_DIR))

    #  Load the clean dictionary provided by the search backend after removing the zero hit terms and 
    # different inflections of a single word.
    data_dict = utils.load_from_json(data_path)
    words = list(data_dict.keys())

    vectorizer = Vectorizer(model_path=models_folder, model_chars=config.MODEL_CHARS)

    keyed_vecs = gensim.models.keyedvectors.Word2VecKeyedVectors(emb_dim)

    embs = vectorizer.vectorize_words(words)
    word_embs = dict(zip(words, embs))
    keyed_vecs.vocab = word_embs
    keyed_vecs.vectors = np.array(embs)

    # Serialize the word2vec vocab and vectors in binary format
    os.makedirs(os.path.dirname(bin_file_path), exist_ok=True)
    export_binary_word2vec(
        fname=bin_file_path,
        total_vec=len(word_embs),
        vocab=keyed_vecs.vocab,
        vectors=keyed_vecs.vectors)
    _logger.info("w2v conversion successful")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Vectorizing Details')

    parser.add_argument('--prod_index_date', dest='prod_index_date', type=str, 
                        help='Date of product index')

    parser.add_argument('--username', dest='username', type=str, 
                        help='User who trained')

    parser.add_argument('--arc_version', dest='arc_version', type=str, 
                        help='Model architecture')

    parser.add_argument('--train_timestamp', dest='train_timestamp', type=str, 
                        help='Model training date')

    args = parser.parse_args()
    print("===== Vectorizing the dictionary ...")
    
    model_path = os.path.join(args.username, config.MODELS_FOLDER,
        args.arc_version, args.train_timestamp)
    bin_folder_path = os.path.join(config.ARTIFACTS_DIR, model_path, config.VECTORIZED_DICTIONARY_FOLDER)
    data_prefix = os.path.join(args.username, config.PROD_INDEX_DIR, args.prod_index_date,
        config.PROCESSED_PROD_INDEX_FILE)
    
    save_w2v_model(model_path=model_path, data_prefix=data_prefix, emb_dim=config.EMB_DIM,
        bin_file_path=os.path.join(bin_folder_path, config.VECTORS_FILE))
    
    gcp_path = os.path.join(model_path, config.VECTORIZED_DICTIONARY_FOLDER)
    local_path = os.path.join(config.ARTIFACTS_DIR, gcp_path)
    utils.upload_to_bucket(project=config.DSC_GCP_PROJECT,
        bucket_name=config.DSC_BUCKET,
        prefix=gcp_path,
        folder_name=local_path)
        