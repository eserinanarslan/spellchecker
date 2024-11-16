import random
import operator
from itertools import combinations
from pytextdist.edit_distance import levenshtein_distance
import os
import sys
sys.path.append(os.path.join('..', 'src'))
import config
from utils import chunks
import str_utils


def clean_dictionary(obj, max_count):
    """
    Filter out the terms from product index that has count less than n
    Args:
        obj (json): Input file provided by SEBE
    """
    limit_parsed_dict = dict((k, v) for k, v in obj.items() if v >= max_count)
    return limit_parsed_dict


def get_search_queries_at(client, day, nb_samples=500):
    """
    Extract search tracking data for a particual day
    Params:
        client: BigQuery client
        day(string): date of the tracking data in string format
        nb_sample(int): Number of samples to extract.
    Returns:
        DataFrame: query result
    """
    sql = f"""
        SELECT DISTINCT 
        productTable.productDescGer AS prod_desc_ger,
        productTable.productDescEng AS prod_desc_eng,
        c1.value AS search_keyword,
        FROM `valiant-metric-166707.36328697.ga_sessions_*` a, # tracking data table
        UNNEST(a.hits) b
        LEFT JOIN UNNEST(b.customDimensions) c1 ON (c1.index = 26)  # SearchKeyword
        LEFT JOIN UNNEST(b.customDimensions) c2 ON (c2.index = 5)   # Page Type
        LEFT JOIN UNNEST(b.customDimensions) c3 ON (c3.index = 91)  # FilterName
        LEFT JOIN UNNEST(b.product) f
        LEFT JOIN `conrad-cbdp-prod-core.de_conrad_dwh1000_dwh_DimProduct.DimProduct` productTable
        ON (CAST(productTable.productNo AS STRING) = CAST(f.productsku AS STRING))

        WHERE isImpression IS NULL
        AND c1.value != ''                                          # Remove searches without searchterm
        AND c2.value = 'Search Results'                             # Remove categorypage, brandpage etc.
        AND f.productSku IS NOT NULL
        AND c3.value IS NULL                                        # Remove searches where a filter is applied
        AND productTable.productDescGer IS NOT NULL
        AND productTable.productDescEng IS NOT NULL
        AND _TABLE_SUFFIX BETWEEN '{day}' AND '{day}' # taking from GA session for 1 day
        """

    actual_data = client.query(sql).to_dataframe()
    actual_data = actual_data.sample(n=nb_samples)
    actual_data = actual_data.reset_index(drop=True)
    return actual_data


def qualify_search_keyword(search_query, inv_index_keys):
    """
        Generating synthetic noise to the correct search query
        This decides how much error should be there in the string
        based on the length of each token
        Logic: Less mistakes will occur in shorter strings
    Returns:
        list
    """
    list_data = list()
    token_query = search_query.strip().split()
    for token in token_query:
        token_len = len(token)
        noisy_token = 0
        ed_choice = 0
        label = 0
        if token in inv_index_keys and token_len >= 5:
            if token_len >= 5 and token_len <= 7:
                ed_choice = 1
            elif token_len > 7 and token_len <= 10:
                ed_choice = 2
            else:
                ed_choice = 3
            label = 1
            noisy_token = str_utils.add_noise_to_string(token, ed_choice)
        list_data.append((token, noisy_token, ed_choice, label))
        
    check_idx = list(set([lis[3] for lis in list_data]))
    if len(check_idx) == 1 and check_idx[0] == 0:
        # the rows with None value will be later dropped from the dataframe
        return 'None'
    else:
        return list_data


def filter_by_edistance(data_line: str, 
                        ed_distance: int=config.EDIT_DIST,
                        max_word_len: int=config.MAX_WORD_LEN):
    """
    Filter out the data lines from the de_corrections.txt to contain only 
    samples that are 3 edit distances away from eachother
    Args:
        data_line (str): raw data single line from the de_corrections.txt
    """
    QandA_pair=dict()
    questions, answers=data_line.split(" => ")
    len_answers = len(answers)
    replacement_dict = {'Ã¼': 'ue', 'Ã¤': 'ae', 'Ã¶': 'oe', 'ÃŸ': 'ss'}

    if len_answers >= ed_distance and len_answers <= max_word_len:
        question_elements = questions.split(', ')
        answers = str_utils.replace_chars(answers, replacement_dict)
        for q_elements in question_elements:
            q_elements = str_utils.replace_chars(q_elements, replacement_dict)
            edit_distance = levenshtein_distance(q_elements,answers)
            if edit_distance <= ed_distance:
                QandA_pair.setdefault(answers, []).append(q_elements)
    return QandA_pair

# TODO add ed_distance and max_word_len as params to the function
# and remove the default params values used in filter_by_edistance
def generate_samples(content, max_batch_size: int=1000):
    """
    Generate the similar and dissimilar samples and append the labels to the same
    params:
        content (str): String in the following format mispelling1, mispellingn => correction
        e.g. 'aafi, anafri, nafi, anaffi, anfafi => anafi'
    return:
        tuple(dict): DIctionaries of similar and non smilar words
    """
    qa_pairs = list(filter(None.__ne__,
        list(map(filter_by_edistance, content))))
    qa_pairs = {k: v for d in qa_pairs for k, v in d.items()}
    train_data, train_label = [],[]
    for q, a in qa_pairs.items():
        for _, a_item in enumerate(a):
            train_item = (q, a_item)
            train_data.append(train_item)
            train_label.append(0)
    dict_simi_samples = dict(zip(train_data, train_label))
    
    seen_combo = set()
    selected_combo = list()
    root_form = list(qa_pairs.keys())
    for batch in chunks(root_form, max_batch_size):
        combinations_object = combinations(batch, 2)
        combinations_list = list(combinations_object)
        selected_combo.extend(combinations_list)
    random.shuffle(selected_combo)
    selected_combo = random.sample(selected_combo, len(dict_simi_samples))
    labels_selected_combo = [1]*len(selected_combo)
    dict_nonsimi_samples = dict(zip(selected_combo, labels_selected_combo))
    return dict_simi_samples, dict_nonsimi_samples