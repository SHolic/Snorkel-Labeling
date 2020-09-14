from abc import abstractmethod
import os

WORK_PATH = os.getcwd()

__depends__ = {
    "district_path": f"{WORK_PATH}/xner/sources/district_type2.txt",
}


class Feature:
    def __init__(self):
        pass

    @abstractmethod
    def create(self, text, mode="char"):
        pass


class BasicFeature(Feature):
    @staticmethod
    def __create_mode_char(text):
        def get_num_type(c):
            chn_num = {'一', '二', '三', '四', '五', '六', '七', '八', '九', '十'}
            chn_traditional_num = {'甲', '乙', '丙', '丁', '戊', '己', '庚', '辛', '壬', '葵'}
            if c.isdigit():
                return 'digit'
            if c in chn_num:
                return 'chn_digit'
            if c in chn_traditional_num:
                return 'traditional_digit'
            return False

        text_length = len(text)
        ret = []
        for i, char in enumerate(text):
            ret.append({
                'bias': 1.0,
                'char.lower()': char.lower(),
                'char.isdigit()': get_num_type(char),
                'char.isupper()': char.isupper(),
                'BOS': True if i == 0 else False,
                'EOS': True if i == text_length - 1 else False,
                '-1:char.lower()': text[i - 1].lower() if i > 0 else False,
                '-2:char.lower()': text[i - 2].lower() if i > 1 else False,
                '+1:char.lower()': text[i + 1].lower() if i < text_length - 1 else False,
                '+2:char.lower()': text[i + 2].lower() if i < text_length - 2 else False
            })
        return ret

    @staticmethod
    def __create_mode_word(text):
        ret = []
        text_length = len(text)
        for i, word in enumerate(text):
            ret.append({
                'bias': 1.0,
                'word.lower()': word.lower(),
                'word.isdigit()': word.isdigit(),
                'word.isupper()': word.isupper(),
                'word.len()': len(word),
                'word.suffix()': word[-1],
                'BOS': True if i == 0 else False,
                'EOS': True if i == text_length - 1 else False,
                '-1:word.lower()': text[i - 1].lower() if i > 0 else False,
                '-1:word.len()': len(text[i - 1]) if i > 0 else False,
                '-1:word.suffix()': text[i - 1][-1] if i > 0 else False,
                '-2:word.lower()': text[i - 2].lower() if i > 1 else False,
                '-2: word.len()': len(text[i - 2]) if i > 1 else False,
                '-2: word.suffix()': text[i - 2][-1] if i > 1 else False,
                '+1:char.lower()': text[i + 1].lower() if i < text_length - 1 else False,
                '+1:char.len()': len(text[i + 1]) if i < text_length - 1 else False,
                '+1:char.suffix()': text[i + 1][-1] if i < text_length - 1 else False,
                '+2:char.lower()': text[i + 2].lower() if i < text_length - 2 else False,
                '+2:char.len()': text[i + 2].lower() if i < text_length - 2 else False,
                '+2:char.suffix()': text[i + 2].lower() if i < text_length - 2 else False
            })

        return ret

    def create(self, text, mode="char"):
        if mode == "char":
            return self.__create_mode_char(text)
        if mode == "word":
            return self.__create_mode_word(text)
        return None


class NgramFeature(Feature):
    @staticmethod
    def __create_mode_char(text):
        text_length = len(text)
        ret = []
        for i, char in enumerate(text):
            ret.append({
                '-1:bi-grams': text[i - 1:i + 1] if i > 0 else False,
                '-2:tri-grams': text[i - 2:i + 1] if i > 1 else False,
                '+1:bi-grams': text[i:i + 2] if i < text_length - 1 else False,
                '+2:tri-grams': text[i:i + 3] if i < text_length - 2 else False,
            })
        return ret

    @staticmethod
    def __create_mode_word(text):
        text_length = len(text)
        ret = []
        for i, word in enumerate(text):
            ret.append({
                '-1:bi-grams': text[i - 1:i + 1] if i > 0 else False,
                '-2:tri-grams': text[i - 2:i + 1] if i > 1 else False,
                '+1:bi-grams': text[i:i + 2] if i < text_length - 1 else False,
                '+2:tri-grams': text[i:i + 3] if i < text_length - 2 else False,
            })
        return ret

    def create(self, text, mode="char"):
        if mode == "char":
            return self.__create_mode_char(text)
        if mode == "word":
            return self.__create_mode_word(text)
        return None


class DistrictFeature(Feature):
    def __init__(self, district_path=__depends__["district_path"]):
        super(DistrictFeature, self).__init__()
        self.district = dict()
        with open(district_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.strip().split("\t")
                self.district[line[0]] = line[1]

    @staticmethod
    def __detect_char_in_district(sent, i, district, district_max_length, district_type="A1"):
        code, level = False, 0
        if i < 0 or i > len(sent) - 1:
            return False, False
        sent_length = len(sent)

        for interval in range(max(district_max_length, sent_length), 0, -1):
            find = False
            start = 0 if i - interval + 1 < 0 else i - interval + 1
            end = start + interval - 1
            while end < sent_length and start <= i:
                word = "".join(sent[start: end + 1])

                if word in district.keys() and district_type in district[word]:
                    level = 1
                    code = word
                    find = True
                    break
                start, end = start + 1, end + 1

            if find:
                break
        return code, level

    def __create_mode_char(self, text):
        ret = []
        cache = dict()
        for i, char in enumerate(text):
            a1, _ = self.__detect_char_in_district(text, i, self.district, 15, district_type="A1")
            a2, _ = self.__detect_char_in_district(text, i, self.district, 15, district_type="A2")
            a3, _ = self.__detect_char_in_district(text, i, self.district, 15, district_type="A3")
            a4, _ = self.__detect_char_in_district(text, i, self.district, 15, district_type="A4")
            cache[i] = {"A1": a1, "A2": a2, "A3": a3, "A4": a4}
        for i, char in enumerate(text):
            ret.append({
                'char.is_A1': 1 if cache[i]["A1"] else 0,
                'char.is_A2': 1 if cache[i]["A2"] else 0,
                'char.is_A3': 1 if cache[i]["A3"] else 0,
                'char.is_A4': 1 if cache[i]["A4"] else 0,
                'char.before_is_same_A1':
                    0 if i == 0 or not cache[i]["A1"]
                    else (1 if cache[i]["A1"] == cache[i - 1]["A1"] else 0),
                'char.before_is_same_A2':
                    0 if i == 0 or not cache[i]["A2"]
                    else (1 if cache[i]["A2"] == cache[i - 1]["A2"] else 0),
                'char.before_is_same_A3':
                    0 if i == 0 or not cache[i]["A3"]
                    else (1 if cache[i]["A3"] == cache[i - 1]["A3"] else 0),
                'char.before_is_same_A4':
                    0 if i == 0 or not cache[i]["A4"]
                    else (1 if cache[i]["A4"] == cache[i - 1]["A4"] else 0),
                'char.after_is_same_A1':
                    0 if i == len(text) - 1 or not cache[i]["A1"]
                    else (1 if cache[i]["A1"] == cache[i + 1]["A1"] else 0),
                'char.after_is_same_A2':
                    0 if i == len(text) - 1 or not cache[i]["A2"]
                    else (1 if cache[i]["A2"] == cache[i + 1]["A2"] else 0),
                'char.after_is_same_A3':
                    False if i == len(text) - 1 or not cache[i]["A3"]
                    else (1 if cache[i]["A3"] == cache[i + 1]["A3"] else 0),
                'char.after_is_same_A4':
                    False if i == len(text) - 1 or not cache[i]["A4"]
                    else (1 if cache[i]["A4"] == cache[i + 1]["A4"] else 0),
            })
        return ret

    def __create_mode_word(self, text):
        ret = []
        text_length = len(text)
        for i, word in enumerate(text):
            ret.append({
                'word.isDistrict()': self.district[word] if word in self.district.keys() else False,
                '-1: word.isDistrict()': self.district["".join(text[i - 1:i + 1])]
                if i > 0 and "".join(text[i - 1:i + 1]) in self.district.keys()
                else False,
                '-2: word.isDistrict()': self.district["".join(text[i - 2:i + 1])]
                if i > 1 and "".join(text[i - 2:i + 1]) in self.district.keys()
                else False,
                '+1: word.isDistrict()': self.district["".join(text[i:i + 2])]
                if i < text_length - 1 and "".join(text[i:i + 2]) in self.district.keys()
                else False,
                '+2: word.isDistrict()': self.district["".join(text[i:i + 2])]
                if i < text_length - 2 and "".join(text[i:i + 2]) in self.district.keys()
                else False,
            })
        return ret

    def create(self, text: str, mode="char"):
        if mode == "char":
            return self.__create_mode_char(text)
        if mode == "word":
            return self.__create_mode_word(text)
        return None
