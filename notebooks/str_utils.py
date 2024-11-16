import sys
import random
from collections import defaultdict
from itertools import product
import math

import numpy as np
from numpy.random import randint as random_randint


# TODO add geman keyboard keys as well and use that for errors generation
QWERTY_KEYBOARD_CARTESIAN = {'q': {'x':0, 'y':0}, 'w': {'x':1, 'y':0}, 'e': {'x':2, 'y':0}, 'r': {'x':3, 'y':0}, 
    't': {'x':4, 'y':0}, 'y': {'x':5, 'y':0}, 'u': {'x':6, 'y':0}, 'i': {'x':7, 'y':0}, 
    'o': {'x':8, 'y':0}, 'p': {'x':9, 'y':0}, 'a': {'x':0, 'y':1}, 'z': {'x':0, 'y':2},
    's': {'x':1, 'y':1}, 'x': {'x':1, 'y':2}, 'd': {'x':2, 'y':1}, 'c': {'x':2, 'y':2}, 
    'f': {'x':3, 'y':1}, 'b': {'x':4, 'y':2}, 'm': {'x':5, 'y':2}, 'j': {'x':6, 'y':1}, 
    'g': {'x':4, 'y':1}, 'h': {'x':5, 'y':1}, 'j': {'x':6, 'y':1}, 'k': {'x':7, 'y':1}, 
    'l': {'x':8, 'y':1}, 'v': {'x':3, 'y':2}, 'n': {'x':5, 'y':2}, 'ü': {'x':10, 'y':0}, 
    'ö': {'x':9, 'y':1}, 'ä': {'x':10, 'y':1}}
CHARS = list(QWERTY_KEYBOARD_CARTESIAN.keys())


def euclidean_distance(key1, key2):
    """
    Euclidean distance between two keyboard keys
    params:
        key1: keyboard key1
        key2: keyboard key2
    """
    x_val = (QWERTY_KEYBOARD_CARTESIAN[key1]['x'] - QWERTY_KEYBOARD_CARTESIAN[key2]['x']) ** 2
    y_val = (QWERTY_KEYBOARD_CARTESIAN[key1]['y'] - QWERTY_KEYBOARD_CARTESIAN[key2]['y']) ** 2
    return math.sqrt(x_val + y_val)


def generate_keys_distance_dict(keys):
    """
    Generate keyboard keys distance dictionary
    for each keyboard key
    Params:
        keys(list): Keyboard keys
    """
    target_dict = defaultdict(dict)
    for i, j in product(keys, keys):
        e_distance = euclidean_distance(i, j)
        if e_distance <= 2:
            target_dict[i][j] = e_distance
    return target_dict

# dictionary of proximate keys distances
#             e.g. {'q': {'q': 0.0,
#               'w': 1.0,
#               'e': 2.0,
#               'a': 1.0,
#               'z': 2.0,
#               's': 1.4142135623730951},...}
KEYS_DISTANCES = generate_keys_distance_dict(QWERTY_KEYBOARD_CARTESIAN)


def replace_at_random_pos(a_string):
    """
    replace a random character in a string
    params:
        a_string(string): a string
    """
    chars_string = list(a_string)
    rand_idx = random.randint(0, len(chars_string)-1)
    # replace one of the near keys to the selected character
    proximity_char = random.choice(
        list(KEYS_DISTANCES[chars_string[rand_idx]].keys())
    )
    chars_string[rand_idx] = proximity_char
    return  "".join(chars_string)


def delete_from_str(a_string):
    """
    Detelet a character from a string at a random position
    """
    random_char_position = random_randint(len(a_string))
    return  a_string[:random_char_position] + a_string[random_char_position + 1:] 


def insert_in_str(a_string):
    """
    Insert a char in a stringat a random position
    """
    random_char_position = random_randint(len(a_string))
    return a_string[:random_char_position] + \
        random.choice(CHARS[:-1]) + a_string[random_char_position:]

def transpose_str(a_string):
    """
    Transpose 2 characters in a string
    """
    random_char_position = random_randint(len(a_string) - 1)
    return (a_string[:random_char_position] + \
            a_string[random_char_position + 1] + \
            a_string[random_char_position] + \
            a_string[random_char_position + 2:])


def add_noise_to_string(a_string, edit_distance):
    """
    Add some artificial spelling mistakes to the string
    This decides what error should be on the string depending on the how
    Args: 
        a_string (str): Search keyword from the tracking data that has to be corrupted with spelling errors.
    Returns:
        result_string (str): Corrupted search keyword with one of the possible spelling error.
        choice (str): Type of spelling error.
    """
    edit_distance_choices = {
        1: ['insert', 'delete', 'replace'],
        2: ['transpose', 'insert', 'delete', 'replace'],
        3: ['transpose', 'insert', 'delete', 'replace']
    }
    choices = edit_distance_choices[edit_distance]
    random_choice_list = list(np.random.choice(choices, size=edit_distance))
    if 'transpose' in random_choice_list:
        random_choice_list = ['transpose']
    
    operation_function = {
        'replace': 'replace_at_random_pos',
        'delete': 'delete_from_str',
        'insert': 'insert_in_str',
        'transpose': 'transpose_str'
    }
    for random_choice in random_choice_list:
        func = getattr(sys.modules[__name__],
                       operation_function.get(random_choice, ''))
        if func:
            a_string = func(a_string)
    return a_string

# TODO implement a unit test for the function
# TODO refactor the function and do strings nomalization
# This function cannot be refactored to use any predefined python library - Because these symbols are specific to the file and 
# correspond to the german umlauts. This has been confirmed with SEBE and therefore have to use hard coded replace function.
def replace_chars(main_string: str, repl_dict: dict):
    """
    Replace a list of characters by the corresponding mappings
    provided in the input dictionary
    Args:
        main_string (str): input string to replace the characters in
        repl_dict (dict): list of characters and their respective replacement
    """
    for char, repl_str in repl_dict.items():
        main_string = main_string.replace(char, repl_str)
    return main_string

def has_digits(inputString):
    """
    Checks if the values of the input dict al are numbers 

    Args:
        inputString (list): text

    Returns:
        boolean : checks weather any char is digit or not
    """
    return any(char.isdigit() for char in inputString)