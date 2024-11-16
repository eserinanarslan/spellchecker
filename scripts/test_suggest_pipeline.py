# /usr/bin/env python
"""
Script for testing the misspellings corrections pipeline
"""
import os
import json
import sys
sys.path.append(os.path.join('..', 'src'))
import suggest_pipeline
import config
import utils


if __name__=='__main__':
    username = sys.argv[1]
    model_train_date = sys.argv[2]
    arc_version = sys.argv[3]
    input_words = sys.argv[4]

    model_path_prefix = os.path.join(
        username, config.MODELS_FOLDER, arc_version, model_train_date)
    dest_path = os.path.join(config.ARTIFACTS_DIR, model_path_prefix)
    if not os.path.exists(dest_path):
        utils.download_from_bucket(project=config.DSC_GCP_PROJECT,
            bucket_name=config.DSC_BUCKET,
            prefix=model_path_prefix,
            dest_folder=config.ARTIFACTS_DIR)
    # TODO extract model chars from the file model_metadata.json instead of using config.MODEL_CHARS
    spellchecker = suggest_pipeline.Suggester(dest_path, model_chars=config.MODEL_CHARS)
    words_list = input_words.strip().split()
    res = {}
    for word in words_list:
        res[word] = spellchecker.correct_suggestion(word, topn=5)
    print(json.dumps(res, indent=4))
