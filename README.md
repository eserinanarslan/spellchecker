# ML Spellchecker
The objective of the project is to develop a machine learning based spellchecker module that would produce corrected suggestions for a misspelled token on our webshop SEARCH bar.
## Owner Team
Data Science Team
GCP project = 'https://confluence.onconrad.com/display/DataScience/04.+Spell+Checker'

# Table of Contents
1. [Description](#desc)
2. [CBR and project dependencies](#cbr)
3. [Full Documentation](#doc)
4. [Dataset(s)](#dataset)
5. [Project Overview](#prj)
	- [Architecture](#architecture)
	- [Building blocks explanation](#blocks)
	- [Project folders Structure](#structure)
6. [Model and performance](#model)
7. [Setting up the environment](#env)
8. [Running the scripts](#scripts)
9. [Testing the whole pipeline](#testing)

<a name="desc"></a>
# Description
The objective of this project is to alleviate the inconsistencies present in the current plugin (in the form of Jazzy) by designing a Machine Learning based Spellchecker. The model will be trained to learn the spelling of a term from a subset of induced errors in the same term. By adopting this approach we will be able to diminish the zero hits recorded on our conrad webshop majorly due to the misspelled words.

<a name="cbr"></a>
# CBR and project dependencies
Development of a ML-based Spell Checker - https://jira.onconrad.com/browse/CBR-6216
SEBE Epic - https://jira.onconrad.com/browse/SEBE-14779


<a name="doc"></a>
# Full Documentation
Confluence documentation - https://confluence.onconrad.com/display/DataScience/04.+Spell+Checker

<a name="dataset"></a>
# Dataset(s)
The source of the dataset was found from the [confluence page](https://confluence.onconrad.com/display/SearchContext/Spell+Checker) where already a vocabulary for different languages has been published under the directory gs://search-external-files/corrections/. This directory contains a regularly updated de_corrections.txt which has a mapping between the misspelled word and its correction. For the initial start we will restrict the misspelled words that are less than or equal to two edits away from the correct word. Later we can extend the dataset to include misspelled words that are upto 3 edits away.

<a name="proj"></a>
# Project overview
Overall steps of the used methodology to train the ML model.

	1. Formatting of raw dataset (de_corrections.txt) to structured dataset (gs://data_spellchecker/training_words_3to25_de_to_en.txt)
	2. Model training
	3. SEBE product index extraction and refinement (gs://search-ml-spellchecker/terms_map.json)
	4. Vectorization to generate the embedding dictionary file 
	5. Evaluation using 3500 words from the tracking data for a period of 10 days.

<a name="architecture"></a>
## Architecture
[Project structure - Click Here](https://drive.google.com/file/d/1m4ynytqXMDrE5FGqwnPHZ3jaYAi4ZT07/view?usp=sharing)

<a name="blocks"></a>
## Building blocks explanation
 - Offline Mode: The Dataset generation, Formatting, Model training, Dictionary vectorization will be performed locally (cloud) and the resultant outcome i.e trained model and vector file will be uploaded to the GCP bucket.
 - Online Mode: The byproducts of the above operation will be loaded. Any misspelled word tracked online will be provide to the system and the closest possible correction will be returned back by performing a cosine similarity check against the vectorized file.
<a name="structure"></a>
## Project folders Structure
```
├── notebooks                               <- Jupyter Notebooks
│   ├── training_dataset_preparation.ipynb  <- Formatting of raw dataset to structured one acceptable by the model.
│   ├── training_dictionary_cleaning.ipynb  <- Refinement of the product index words from SEBE
│   └── utils.py                            <- All utility functions for dataset cleaning.
│
├── src                             <- Python scripts.
│   ├── 01.train_model.py           <- Dataset loading and model training with save.
│   ├── 02.vectorize_dictionary.py  <- Utilizing the trained model to generate the vectorized file 
│   ├── 03.evaluate_model.py        <- Downloads the tracking data for evaluation and produces a folder that shows metrics and graphs.
│   ├── char2vec_model.py           <- Model architecture and other function for fit and model 
│   ├── config.py              <- Configuration constants used accross all the scripts.
│   ├── logging_config.py           <- Logging of successfull passing of different modules.
│   ├── model_retriever.py          <- Downloading and uploading of models to the GCP bucket.for the conrad vocab provided by SEBE. 
│   ├── suggest_pipeline.py         <- Suggesting nearest neighbors for the misspelled word.
│   └── vectorize.py                <- Vectorize the input word used for dict vectorization and suggestion.
```
<a name="env"></a>
# Setting up the environment
To run the project notebooks you need to setup the required packages within a virtual environment as follows.
- Create a conda virtual environment from yaml file `conda env create --file environment.yml`
- You can verify that the new environment was correctly installed using `conda env list`
- Activate the new environment `conda activate spellchecker`
- Run `$CONDA_PREFIX/bin/jupyter lab`

To update existing environment using the environment yaml file, run the following command

```bash
conda env update --name spellchecker --file environment.yml --prune
```

<a name="scripts"></a>
# Running the scipts
## Spelling correction
```python
import suggest_pipeline
spellchecker = suggest_pipeline.Suggest(user_name, architect_version, model_train_date)
spellchecker.correct_suggestion(word_string)
```
For example
```python
import suggest_pipeline
spellchecker = suggest_pipeline.Suggest('shrikanth', 'version_bilstm_100d', '2022-03-08-11h-01m')
spellchecker.correct_suggestion('kipschalter')
```

## Model training 

```bash
cd src ; python 01.train_model.py \
	--data_date <data_generation_date> \
	--username  <your-username> \
	--arc_version <model_version_description> \
	--n_epochs <number_of_epochs> \
	--n_samples <n_samples>
```
## Dictionary vectorization

```bash
cd src ; python 02.vectorize_dictionary.py \
	--prod_index_date <prodindex_generation_timestamp> \
	--username <your-username> \
	--arc_version <model_version_description> \
	--train_date <timestamp>
```

## Model evaluation

```bash
cd src; python 03.evaluate_model.py
```

<a name="scripts"></a>
# Testing the whole pipeline

For running all the steps from training the model to performing a correction of a set of misspeled words, you can run the script `test_whole_pipeline.sh` located under the folder scripts.
```bash
Usage: ./test_whole_pipeline.sh <username> <model_version_desc> <n_samples> <n_epochs>
```
Example execution command
```bash
./test_whole_pipeline.sh $USER test_model 100 1
```

If all the steps get successfully executed, you will get the following message.
```html
The whole pipeline was successfully executed
```
If you would like to test the whole pipeline for different training parameters, you should set the needed parameters in the `config.py` file.
Here are the pre-defined model parameters values set in the config file
```python
EMB_DIM = 100
PATIENCE = 1
VAL_SPLIT = 0.05
BATCH_SIZE = 64
MAX_WORD_LEN = 26
```


