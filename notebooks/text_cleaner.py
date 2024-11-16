import re
from unidecode import unidecode
from html.parser import HTMLParser


class HtmlUtil(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.fed = []

    def handle_data(self, d):
        self.fed.append(d)

    def get_data(self,  delimiter):
        return delimiter.join(self.fed).strip()

    def error(self, message):
        pass

    def clean(self, input, delimiter):
        self.__init__()
        self.feed(input)
        return self.get_data(delimiter).strip()


class NumberUtil:
    def __init__(self):
        pass

    @staticmethod
    def is_number(_input):
        if str.isnumeric(_input):
            return True
        try:
            float(_input)
            return True
        except ValueError:
            return False


class TextCleaner:
    def __init__(self):
        self.complete_regex = re.compile('[^a-zA-Z0-9äüöß,\"\.]')
        self.alphanumeric_regex = re.compile('[^a-zA-Z0-9äüöß]')
        self.__setup_transliteration()

        self.htmlUtil = HtmlUtil()
        self.numberUtil = NumberUtil()
        self.MIN_ALPHA_PROPORTION = 75  # this means we allow 25% for numerical values.
        self.MAX_NUM_OF_DIGITS = 7
        self.empty = ""

    def __setup_transliteration(self):
        self.umlaute = dict()
        self.umlaute["ö"] = "oe"
        self.umlaute["ü"] = "ue"
        self.umlaute["ä"] = "ae"

    def clean(self, input):
        cleaned = input.lower()
        cleaned = self.htmlUtil.clean(cleaned, '')

        if cleaned.endswith(".html") is False and \
           cleaned.endswith("cat") is False and \
           self.__is_digit(cleaned) is False and \
           self.__is_too_many_digits(cleaned) is False and \
           "promo" not in cleaned:

            if self.__contain_umlaute(cleaned):
                cleaned = self.transliterate(cleaned)

            cleaned = unidecode(cleaned)  # remove escaped characters: \xaud \xa0
            cleaned = self.complete_regex.sub(' ', cleaned).strip()
            cleaned = self.__handle_dot(cleaned)
            cleaned = self.__handle_comma(cleaned)
            cleaned = self.__handle_double_quote(cleaned)
            return cleaned.strip()
        return self.empty

    def basic_clean(self, input):
        cleaned = input.lower()
        cleaned = self.htmlUtil.clean(cleaned, '')

        if cleaned.endswith(".html") is False and \
            cleaned.endswith("cat") is False and \
            self.__is_too_many_digits(cleaned) is False and \
            "promo" not in cleaned:

            cleaned = self.__handle_dot(cleaned)
            cleaned = self.__handle_comma(cleaned)
            cleaned = self.__handle_double_quote(cleaned)
            return cleaned.strip()
        return self.empty

    # it is used for concept extraction
    def clean_all_but_not_alpha(self, input):
        cleaned = self.clean(input)

        if len(cleaned) == 0:
            return self.empty

        cleaned = self.alphanumeric_regex.sub(' ', cleaned).strip()
        tokens = cleaned.split()
        words = list()

        for token in tokens:
            if token.isdigit() is False and \
                    len(token) > 1 and \
                    self.__has_numbers(token) is False:
                words.append(token)

        return ' '.join(words).strip()

    def __is_digit(self, input):
        cleaned = self.alphanumeric_regex.sub('', input).strip()

        if cleaned.isdigit():
            return True
        return False

    def __is_too_many_digits(self, input):
        num_of_alpha = 0
        num_of_digits = 0

        for c in input:
            if c.isalpha():
                num_of_alpha += 1
            elif c.isdigit():
                num_of_digits += 1

        # ensure safe-division
        denominator = num_of_alpha + num_of_digits

        if denominator == 0:
            return True

        alpha_proportion = num_of_alpha / denominator * 100

        if alpha_proportion < self.MIN_ALPHA_PROPORTION and num_of_digits > self.MAX_NUM_OF_DIGITS:
            return True

        return False

    def __handle_dot(self, input):
        tokens = input.split()
        cleaned = list()

        for token in tokens:
            if token.find(".") >= 0:
                if self.numberUtil.is_number(token):
                    cleaned.append(token)
                else:
                    cleaned.append(token.replace(".", " ").strip())
            else:
                cleaned.append(token)

        return " ".join(cleaned)

    def __handle_comma(self, input):
        chars = list(input)
        cleaned = list()
        idx = 0
        last = len(input) - 1

        for c in chars:
            if chars[idx] != ',':
                cleaned.append(c)
            else:
                if idx > 0 and idx < last:
                    prev_char = chars[idx-1]
                    next_char = chars[idx+1]

                    if prev_char.isdigit() and next_char.isdigit():
                        cleaned.append(c)
                    else:
                        cleaned.append(' ')
            idx += 1

        tokens = ''.join(cleaned).split()
        return ' '.join(tokens)

    def __handle_double_quote(self, input):
        chars = list(input)
        cleaned = list()
        idx = 0

        for c in chars:
            if chars[idx] != '"':
                cleaned.append(c)
            elif idx > 0:
                prev_char = chars[idx-1]

                if prev_char.isdigit() is True:
                    cleaned.append(c)

            idx += 1

        tokens = ''.join(cleaned).split()
        return ' '.join(tokens)

    def __contain_umlaute(self, input):
        for char in self.umlaute:
            if char in input:
                return True

        return False

    def transliterate(self, input):
        for key in self.umlaute:
            input = input.replace(key, self.umlaute.get(key))

        return input

    def __has_numbers(self, input):
        return any(char.isdigit() for char in input)
