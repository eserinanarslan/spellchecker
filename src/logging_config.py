import logging
import sys

FORMATTER = logging.Formatter(
"%(asctime)s — %(name)s — %(levelname)s —" "%(funcName)s:%(lineno)d — %(message)s"
)

def get_console_handler():
    """
    Defining the log file, log format and the level of logging to be done.
    """
    #console_handler = logging.StreamHandler(sys.stdout)
    console_handler = logging.FileHandler('ml_spell_cheker.log')
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(FORMATTER)
    return console_handler

