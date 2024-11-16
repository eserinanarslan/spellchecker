import os
import json
import sys
sys.path.append(os.path.join('..', 'notebooks'))

import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import \
    Input, Bidirectional, Subtract, Dense, Dot, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau


class Chars2Vec:

    def __init__(self, emb_dim: int, char_to_ix: dict, max_word_len: int):
        """
        Args:
            emb_dim : integer
                Number of dimension output for the given input string.
            char_to_ix : dict
                Mapping between the alphabets used to and index number.
            max_word_len : integer
                Maximum length of allowed string for padding and truncation.
        """
        
        self.char_to_ix = char_to_ix
        self.max_word_len = max_word_len
        self.ix_to_char = {char_to_ix[ch]: ch for ch in char_to_ix}
        self.vocab_size = len(self.char_to_ix)
        self.dim = emb_dim
        self.cache = {}

        lstm_input = Input(shape=(None, self.vocab_size))
        print(f"Chars2Vec INPUT TYPE ..... {type(lstm_input)}" )
        x = Bidirectional(LSTM(emb_dim, return_sequences=True))(lstm_input)
        x = Bidirectional(LSTM(emb_dim))(x)

        self.embedding_model = keras.models.Model(inputs=[lstm_input], outputs=x)

        model_input_1 = Input(shape=(None, self.vocab_size))
        model_input_2 = Input(shape=(None, self.vocab_size))

        embedding_1 = self.embedding_model(model_input_1)
        embedding_2 = self.embedding_model(model_input_2)
        x1 = Subtract()([embedding_1, embedding_2])
        x1 = Dot(1)([x1, x1])
        model_output = Dense(1, activation='sigmoid')(x1)

        self.model = Model(inputs=[model_input_1, model_input_2], outputs=model_output)
        
        #self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        sgd = SGD(lr=0.1, decay=1e-5, momentum=0.9, nesterov=True)
        self.model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
              

    def fit(self, word_pairs: list, targets: int, max_epochs: int, patience: int,
            validation_split: float, batch_size: int, model_path: str) -> None:
        """
        Model fitting
        Args:
            word_pairs (list of tuple): tuple of correct and incorrect pair of words
            targets (int): label of correctness 1 for incorrect and 0 for correct
            max_epochs (int): training epoch parameter for keras model
            patience (int): early stopping parameter during training for keras model
            validation_split (float): train and validation ratio for each epoch
            batch_size (int): number of samples per step 
        """
        x_1, x_2 = [], []

        for pair_words in word_pairs:
            emb_list_1 = []
            emb_list_2 = []

            first_word = pair_words[0].lower()
            second_word = pair_words[1].lower()

            for index in range(len(first_word)):
                if first_word[index] in self.char_to_ix:
                    x = np.zeros(self.vocab_size)
                    x[self.char_to_ix[first_word[index]]] = 1
                    emb_list_1.append(x)
                else:
                    emb_list_1.append(np.zeros(self.vocab_size))
            x_1.append(np.array(emb_list_1))

            for index in range(len(second_word)):
                if second_word[index] in self.char_to_ix:
                    x = np.zeros(self.vocab_size)
                    x[self.char_to_ix[second_word[index]]] = 1
                    emb_list_2.append(x)
                else:
                    emb_list_2.append(np.zeros(self.vocab_size))
            x_2.append(np.array(emb_list_2))

        x_1_pad_seq = keras.preprocessing.sequence.pad_sequences(x_1, maxlen=self.max_word_len)
        x_2_pad_seq = keras.preprocessing.sequence.pad_sequences(x_2, maxlen=self.max_word_len)
        print(x_1_pad_seq.shape, x_2_pad_seq.shape)
        
        call_reduce = ReduceLROnPlateau(monitor='val_accuracy', factor=0.75, patience=1, verbose=1,
                                mode='auto', min_delta=0.05, cooldown=0, min_lr=0)
        EarlyStopping_call = EarlyStopping(monitor='val_accuracy', patience=patience, mode='auto')
        callbacks_list = [call_reduce, EarlyStopping_call]
        
        self.model.fit([x_1_pad_seq, x_2_pad_seq], targets,
                       batch_size=batch_size, epochs=max_epochs,
                       validation_split=validation_split,
                       verbose=1,
                       callbacks=callbacks_list)

        self.embedding_model.save(model_path) 

             