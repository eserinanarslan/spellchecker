from numpy.random import randint as random_randint
import matplotlib.pyplot as plt
import suggest_pipeline
import seaborn as sns
import config
import random
import json
import os

from google.cloud import bigquery

from utils import download_from_bucketset


client = bigquery.Client()


def description_truncate(product_desc, search_keyword):

    """
    Function for picking a valid search query based on its presence in the product title clicked after the search
    Args:
        product_desc (str): Product description of the click registered after search from the tracking data.
        search_keyword (str): Search keyword from the tracking data.
    Returns:
        Matched search keyword in the product description
    """
    clean_desc = ' '.join([token.lower() for token in product_desc.split() if token.isalpha()])
    match_keyword = [search_keyword for _token in clean_desc.split() if _token == search_keyword.lower()]
    if match_keyword:
        return search_keyword.lower()
        
        
def read_test_dataset():
    """
    Function to extract the tracking data using BigQuery, returns a dataframe with product title and search keyword.
    """
    sql = """
    SELECT DISTINCT 
    productTable.productDescGer AS prod_desc_der,
    productTable.productDescEng AS prod_desc_eng,
    c1.value AS search_keyword2,
    FROM `valiant-metric-166707.36328697.ga_sessions_*` a,
    UNNEST(a.hits) b
    LEFT JOIN UNNEST(b.customDimensions) c1 ON (c1.index = 26)  #SearchKeyword
    LEFT JOIN UNNEST(b.customDimensions) c2 ON (c2.index = 5)   #Page Type
    LEFT JOIN UNNEST(b.customDimensions) c3 ON (c3.index = 91)  #FilterName
    LEFT JOIN UNNEST(b.product) f
    LEFT JOIN `conrad-cbdp-prod-core.de_conrad_dwh1000_dwh_DimProduct.DimProduct` productTable
    ON (CAST(productTable.productNo AS STRING) = CAST(f.productsku AS STRING))

    WHERE isImpression IS NULL
    AND c1.value != ''                                          #Remove searches without searchterm
    AND c2.value = 'Search Results'                             #Remove categorypage, brandpage etc.
    AND f.productSku IS NOT NULL
    AND c3.value IS NULL                                        #Remove searches where a filter is applied
    AND productTable.productDescGer IS NOT NULL
    AND productTable.productDescEng IS NOT NULL
    AND _TABLE_SUFFIX BETWEEN 
    FORMAT_DATE("%Y%m%d", DATE_SUB(CURRENT_DATE(), INTERVAL 1 MONTH)) AND
    FORMAT_DATE("%Y%m%d", DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY))
      
    """

    test_data = client.query(sql).to_dataframe()
    test_data = test_data.sample(frac=1).reset_index(drop=True)
    filtered_test_data = test_data[test_data['search_keyword2'].apply(lambda x: len(x.split())==1)]
    filtered_test_data = filtered_test_data[filtered_test_data.search_keyword2.str.isalpha()]
    filtered_test_data = filtered_test_data.reset_index(drop=True)
    filtered_test_data['Match Found'] = filtered_test_data[['prod_desc_der','search_keyword2']].apply(lambda x: description_truncate(*x), axis=1)
    filtered_test_data = filtered_test_data.dropna()
    return filtered_test_data
    
    
def add_noise_to_string(a_string):
    """
    Add some artificial spelling mistakes to the string
    Args: 
        a_string (str): Search keyword from the tracking data that has to be corrupted with spelling errors.
    Returns:
        result_string (str): Corrupted search keyword with one of the possible spelling error.
        choice (str): Type of spelling error.
    """
    
    CHARS = list("abcdefghijklmnopqrstuvwxyzüöä")
    choices = ['transpose', 'insert', 'delete', 'replace']
    random_choice = random.choice(choices)
    
    if random_choice == 'replace':
        # Replace a character with a random character
        choice = 'replace'
        random_char_position = random_randint(len(a_string))
        result_string = a_string[:random_char_position] + random.choice(CHARS[:-1]) + a_string[random_char_position + 1:]
        
    elif random_choice == 'delete':
        # Delete a character
        choice = 'delete'
        random_char_position = random_randint(len(a_string))
        result_string = a_string[:random_char_position] + a_string[random_char_position + 1:]
        
    elif random_choice == 'insert':
        # Add a random character
        choice = 'insert'
        random_char_position = random_randint(len(a_string))
        result_string = a_string[:random_char_position] + random.choice(CHARS[:-1]) + a_string[random_char_position:]
        
    elif random_choice == 'transpose':
        # Transpose 2 characters
        choice = 'transpose'
        random_char_position = random_randint(len(a_string) - 1)
        result_string = (a_string[:random_char_position] + a_string[random_char_position + 1] + a_string[random_char_position] +
                        a_string[random_char_position + 2:])
    
    return result_string, choice


def remove_umlaut(string):
    """
    Removes umlauts from strings and replaces them with the letter+e convention
    Args: 
    string: string to remove umlauts from
    Returns: un-umlauted string
    """
    u = 'ü'.encode()
    U = 'Ü'.encode()
    a = 'ä'.encode()
    A = 'Ä'.encode()
    o = 'ö'.encode()
    O = 'Ö'.encode()
    ss = 'ß'.encode()

    string = string.encode()
    string = string.replace(u, b'ue')
    string = string.replace(U, b'Ue')
    string = string.replace(a, b'ae')
    string = string.replace(A, b'Ae')
    string = string.replace(o, b'oe')
    string = string.replace(O, b'Oe')
    string = string.replace(ss, b'ss')

    string = string.decode('utf-8')
    return string
        
def refine_induce_errors(filtered_test_data):
    """
    Function to truncate the search keyowrds based on their presence in the product index file from SEBE and error induce.
    Args: 
        filtered_test_data (DataFrame): Preprocessed dataframe.
    Returns:
        DataFrame: Refined dataset for prediction test.
    """
    if os.path.isdir(config.DATASET_DIR) == False:
        download_from_bucketset()
    with open(config.CLEAN_DICTIONARY, 'r') as myfile:
        data=myfile.read()
        words = json.loads(data)
    
    filtered_test_data = filtered_test_data[filtered_test_data['search_keyword2'].isin(words)]
    filtered_test_data = filtered_test_data.drop_duplicates(subset=['search_keyword2'], keep = 'first')
    filtered_test_data = filtered_test_data[filtered_test_data['search_keyword2'].str.len() > 4]
    filtered_test_data = filtered_test_data.reset_index(drop=True)
    filtered_test_data['search_keyword2'] = filtered_test_data['search_keyword2'].apply(lambda item: remove_umlaut(item))
    filtered_test_data['corrupted_input'], filtered_test_data['corruption_typ'] = zip(*filtered_test_data['search_keyword2'].apply(lambda x: add_noise_to_string(x)))
    return filtered_test_data


def spellcheck_suggest(search_word):
    """
    Function to correct the artificially induced error in the search term
    Args:
        search_word (str): Artificially corrupted search term.
    Returns:
        suggested_words (list): List of corrected suggestions from the spellchecker module
    """
    suggested_words = suggest_pipeline.correct_suggestion(search_word, use_annoy_indexer = False)
    suggested_words = [preds[0] for preds in suggested_words]
    return suggested_words


def generate_prediction(filtered_test_data):
    """
    Function for generating corrected spelling predictions.
    Args:
        filtered_test_data (DataFrame): Containing columns of true label (correct spelling), corrupted input for prediction.
    Returns:
        DataFrame with an additional column containing the predictions from the spellchecker.
    """
    filtered_test_data['predictions'] = filtered_test_data['corrupted_input'].apply(lambda x: spellcheck_suggest(x))
    return filtered_test_data


def plot_graph(df_series, x_axis_label, y_axis_label, title, file):
    sns.set(font_scale=1.2)
    df_series.plot(kind='bar', figsize=(6, 6), rot=0)
    plt.xlabel("{}".format(x_axis_label), labelpad=14)
    plt.ylabel("{}".format(y_axis_label), labelpad=14)
    plt.title("{}".format(title), y=1.02)
    
    # TODO versioning the evaluation results and pushing them to the project bucket
    path = os.path.join(config.PACKAGE_ROOT.parent, "evaluation_results")
    os.makedirs(path, exist_ok=True)
    plt.savefig(os.path.join(path, file))
    
    
def generate_report(filtered_test_data):
    """
    Function that created a file to view the log containing the metrics for evaluation of the model performance.
    Args: 
        filtered_test_data (DataFrame): Table with all the fields for evaluation
    Returns:
        Output file of performance metrics and csv file for futher analysis if necessary.
    """
    for idx, row in filtered_test_data.iterrows():
        filtered_test_data.loc[idx,'Num of suggestions'] = int(len(row['predictions']))
        if row['search_keyword2'] in row['predictions']:
            filtered_test_data.loc[idx,'Precision'] = 1
            if row['predictions'].index(row['search_keyword2'])==0:
                filtered_test_data.loc[idx,'First_word'] = 1
            else:
                filtered_test_data.loc[idx,'First_word'] = 0
        else:
            filtered_test_data.loc[idx,'Precision'], filtered_test_data.loc[idx,'Recall'] = 0, 0
            filtered_test_data.loc[idx,'First_word'] = 0
    
    plot_graph(filtered_test_data.Precision.value_counts(), x_axis_label = 'Precision Value', 
               y_axis_label = 'Count of Samples', title = 'Count of Precision value', file='precision.png')
    plot_graph(filtered_test_data.First_word.value_counts(), x_axis_label = 'First Word', 
               y_axis_label = 'Count of Samples', title = 'Count of First Word', file='firstword.png')
    
    # TODO versioning the evaluation results and pushing them to the project bucket
    path = os.path.join(config.PACKAGE_ROOT.parent, "evaluation_results")
    with open(os.path.join(path, 'metrics.txt'), 'w') as f:
        f.write('Size of the testing dataset = {}\n'.format(len(filtered_test_data)))
        f.write('Ratio of the dataset where the expected correction was found in top-5 prediction = {}\n'.
                format(len(filtered_test_data[filtered_test_data['Precision'] > 0.0])/len(filtered_test_data)))
        f.write('Ratio of the dataset where the expected correction was found in First position = {}\n'.
                format(len(filtered_test_data[filtered_test_data['First_word'] == 1.0])/len(filtered_test_data)))           
    
    

if __name__ == "__main__":
    filtered_test_data = read_test_dataset()
    print('----------------------Dataset read successfully-------------------------')
    error_test_data = refine_induce_errors(filtered_test_data)
    print('------------------Dataset refined and errors induced successfully-----------------')
    predicted_test_data = generate_prediction(error_test_data)
    print('-----------------------Predictions generated successfully------------------------')
    generate_report(predicted_test_data)
    
    