import numpy as np
from tensorflow import keras


class Vectorizer:
    def __init__(self, model_path: str, model_chars: list):
        print(f"Loading the model files from {model_path}")
        try:
            self.model = keras.models.load_model(model_path)
        except Exception as exp:
            print("An exception occured while loading the model:", repr(exp))

        self.char_to_ix = {character: idx for idx, character in\
                          enumerate(model_chars)}
        self.vocab_size = len(self.char_to_ix)
        self.max_len_padseq = self.vocab_size


    def vectorize_words(self, words: list):
        """
        This function vectorizes the clean dictionary into 200 dimension vector using the trained model.
        Args:
            words (list): list of words to be vectorized.
        Returns:
            word_vectors (numpy.ndarray): word embeddings.
        """
        list_of_embeddings = []
        for current_word in words:
            current_word = current_word.lower()
            if not isinstance(current_word, str):
                raise TypeError("word must be a string")

            current_embedding = []

            for index, wrd in enumerate(current_word):
                if wrd in self.char_to_ix:
                    word_encode = np.zeros(self.vocab_size)
                    word_encode[self.char_to_ix[wrd]] = 1
                    current_embedding.append(word_encode)

                else:
                    current_embedding.append(np.zeros(self.vocab_size))

            list_of_embeddings.append(np.array(current_embedding))

        embeddings_pad_seq = keras.preprocessing.sequence.pad_sequences(
            list_of_embeddings, maxlen=self.max_len_padseq)
        new_words_vectors = self.model.predict([embeddings_pad_seq])
        return np.array(new_words_vectors)
