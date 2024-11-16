import os
import sys
sys.path.append(os.path.join('..', 'notebooks'))

import textdistance
from gensim.similarities.index import AnnoyIndexer
from gensim.models.keyedvectors import Word2VecKeyedVectors

from vectorizer import Vectorizer
import config
import str_utils


class Suggester:
    def __init__(self, model_path: str, model_chars: list, index_num_trees=500):
        self.vectorizer = Vectorizer(model_path, model_chars=model_chars)
        self.vectors_file_path = os.path.join(model_path,
            config.VECTORIZED_DICTIONARY_FOLDER,
            config.VECTORS_FILE
        )
        self.load_keyed_vecs = Word2VecKeyedVectors.load_word2vec_format(
            self.vectors_file_path,
            binary=True,
            unicode_errors='ignore')
        self.wv = self.load_keyed_vecs.wv
        self.annoy_index = AnnoyIndexer(self.wv, index_num_trees)


    def eliminate_incorrect_suggestions(self, input_word, suggested_tuple):
        """
        Eliminates suggestions for unpredictable words eg: nudelmaschine.
        See https://jira.onconrad.com/browse/DSC-401
        Args:
            input_word (str): Misspelled input word
            suggested_tuple (tuple): Returned top-1 suggestion by the spellchecker model
        Returns:
            tuple: Final selected tuple post processing checks.
        """
        suggested_correct_word = suggested_tuple[0]
        #confidence_correct_word = suggested_tuple[1]
        edit_distance = textdistance.levenshtein.distance(input_word, suggested_correct_word)
        if edit_distance <= 2 and len(input_word) == len(suggested_correct_word):
            ecd_dist=list()
            for _, (char1, char2) in enumerate(zip(input_word, suggested_correct_word)):
                if char1 != char2:
                    ecd_dist.append(str_utils.euclidean_distance(char1, char2))
            if any(dist >= 2 for dist in ecd_dist):
                return tuple()
            else:
                suggested_tuple
        if edit_distance >= 3:
        # Do not lower the edit_distance condition above as it will affect the result greatly.
            return tuple()
        else:
            return suggested_tuple


    def correct_suggestion(self, word, topn=5):
        """
        Load the vectorized bin file for comparison against the misspelled word
        and extract the N nearest neighbors using the cosine distance.
        Args:
            word(str): input misspelled word.
            topn(int): N nearest neigbors to extract
        Returns:
            suggestions(list of tuple): top 5 nearest neighbors for the misspelled word (corrections)
        """
        vec = self.vectorizer.vectorize_words([word])
        return self.wv.most_similar(vec, topn=topn, indexer=self.annoy_index)
