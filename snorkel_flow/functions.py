from . import SETTINGS, Polarity
from snorkel_flow.utils import get_replacements
from snorkel.preprocess import preprocessor
from snorkel.labeling import labeling_function
from snorkel.augmentation import transformation_function
from snorkel.slicing import slicing_function
import fast_text
from xner.models import crf
import numpy as np
import re
import os

# from snorkel.labeling import LabelingFunction
# from snorkel.slicing import SlicingFunction
# from snorkel.preprocess.core import Preprocessor

WORK_PATH = os.getcwd()


class Preprocessors:
    @staticmethod
    @preprocessor(memoize=True)
    def cls_text(x):
        cls_ret = fast_text.predict(test_data=x.text,
                                    model_path=f"{WORK_PATH}/snorkel_flow/sources/fasttext_name_model.bin")
        x.cls = cls_ret
        return x

    @staticmethod
    @preprocessor(memoize=True)
    def ner_text(x):
        ner_ret = crf.predict(test_data=x.text,
                              model_path=f"{WORK_PATH}/snorkel_flow/sources/comp_char_crf_bmeso_model.pkl",
                              return_type="dict")
        x.ner = ner_ret[0]
        return x


class LabelingFunction:
    @staticmethod
    @labeling_function()
    def lf_ind_keyword(x):
        """Many individuals includes '个体'."""
        return Polarity.INDIVIDUAL.value if "个体" in x['text'] else Polarity.ABSTAIN.value

    @staticmethod
    @labeling_function()
    def lf_short(x):
        """Individuals are often short, such as '张三'."""
        return Polarity.INDIVIDUAL.value if len(x.text) < 5 else Polarity.ABSTAIN.value

    @staticmethod
    @labeling_function(resources=dict(re_cmp=SETTINGS['re_cmp']))
    def lf_cmp_re(x, re_cmp):
        """Many companies includes '公司',etc."""
        return Polarity.COMPANY.value if (
                    (re.search(re_cmp, x.text)) and ('个体' not in x.text)) else Polarity.ABSTAIN.value

    @staticmethod
    @labeling_function(resources=dict(keywords_indus=SETTINGS['keywords_indus']))
    def lf_industry_keyword(x, keywords_indus):
        """Many companies includes industry words."""
        flag = False
        for indus in keywords_indus:
            if indus in x.text:
                flag = True
                break
        if flag and ('个体' not in x.text):
            return Polarity.COMPANY.value
        return Polarity.ABSTAIN.value

    @staticmethod
    @labeling_function(resources=dict(re_surname=SETTINGS['re_surname']))
    def lf_surname_re(x, re_surname):
        """Individuals usually startswith family name."""
        return Polarity.INDIVIDUAL.value if re.search(re_surname, x['text']) else Polarity.ABSTAIN.value

    @staticmethod
    @labeling_function(pre=[Preprocessors.cls_text])
    def industry_cls(x):
        return Polarity.COMPANY.value if ((len(x.text) < 4) and ('__label__1' not in x.cls)) else Polarity.ABSTAIN.value


replacements = get_replacements(["ADDR", "BUSINESS", "O"])


class TransformationFunction:
    @staticmethod
    @transformation_function(pre=[Preprocessors.ner_text])
    def change_addr(x):
        addr = x.ner.get("ADDR", "").split(',')
        if addr:
            name_to_replace = np.random.choice(addr)
            replacement_name = np.random.choice(replacements["ADDR"])
            x.text = x.text.replace(name_to_replace, replacement_name)
            return x

    @staticmethod
    @transformation_function(pre=[Preprocessors.ner_text])
    def change_business(x):
        business = x.ner.get("BUSINESS", "").split(',')
        if business:
            name_to_replace = np.random.choice(business)
            replacement_name = np.random.choice(replacements["BUSINESS"])
            x.text = x.text.replace(name_to_replace, replacement_name)
            return x

    @staticmethod
    @transformation_function(pre=[Preprocessors.ner_text])
    def change_o(x):
        o = x.ner.get("O", "").split(',')
        if o:
            name_to_replace = np.random.choice(o)
            replacement_name = np.random.choice(replacements["O"])
            x.text = x.text.replace(name_to_replace, replacement_name)
            return x

    @staticmethod
    @transformation_function(pre=[Preprocessors.ner_text])
    def randomly_delete(x):
        label = np.random.choice(["ADDR", "BUSINESS", "O"])
        token = x.ner.get(label, "").split(',')
        if token:
            name_to_delete = np.random.choice(token)
            if name_to_delete == x.text:
                return x
            x.text = x.text.replace(name_to_delete, "")
            return x

    @staticmethod
    @transformation_function(pre=[Preprocessors.ner_text])
    def randomly_add(x):
        keyword = x.ner.get("KEYWORDS", "").replace(",", "")
        label_candidate = [l for l in ["ADDR", "BUSINESS", "O"] if not x.ner.get(l)]
        if keyword and label_candidate:
            label = np.random.choice(label_candidate)
            name_to_add = np.random.choice(replacements[label])
            if label == "ADDR":
                x.text = name_to_add + keyword + x.ner.get("BUSINESS", "").replace(',', '') + \
                         x.ner.get("O", "").replace(',', '')
                return x

            if label == "BUSINESS":
                x.text = x.ner.get("ADDR", "").replace(',', '') + keyword + name_to_add + \
                         x.ner.get("O", "").replace(',', '')
                return x

            if label == "O":
                x.text = x.ner.get("ADDR", "").replace(',', '') + keyword + \
                         x.ner.get("BUSINESS", "").replace(',', '') + name_to_add
                return x


class SlicingFunction:
    @staticmethod
    @slicing_function()
    def short_comment(x):
        """Many individuals includes '个体'."""
        return len(x.text) < 5

    @staticmethod
    @slicing_function()
    def ind_keyword(x):
        """Many individuals includes '个体'."""
        return bool("个体" in x['text'])

    @staticmethod
    @slicing_function(resources=dict(re_cmp=SETTINGS['re_cmp']))
    def cmp_re(x, re_cmp):
        """Many companies includes '公司',etc."""
        return bool(re.search(re_cmp, x.text))

    @staticmethod
    @slicing_function(resources=dict(keywords_indus=SETTINGS['keywords_indus']))
    def industry_keyword(x, keywords_indus):
        """Many companies includes industry words."""
        for indus in keywords_indus:
            if indus in x.text:
                return True
        return False

# def keyword_lookup(x, keywords, label):
#     if any(word in x.text for word in keywords):
#         return label
#     return Polarity.ABSTAIN.value
#
#
# def make_keyword_lf(keywords, label):
#     return LabelingFunction(
#         name=f"lf_keyword_cmp",
#         f=keyword_lookup,
#         resources=dict(keywords=keywords, label=label),
#     )
#
#
# lf_keyword_cmp = make_keyword_lf(keywords=SETTINGS['keywords_cmp'], label=Polarity.COMPANY.value)


# # Keyword-based SFs
# def keyword_lookup(x, keywords):
#     return any(word in x.text for word in keywords)
#
#
# def make_keyword_sf(keywords):
#     return SlicingFunction(
#         name=f"sf_keyword_cmp",
#         f=keyword_lookup,
#         resources=dict(keywords=keywords),
#     )
#
#
# sf_keyword_cmp = make_keyword_sf(keywords=SETTINGS['keywords_cmp'])


# class NlpPreprocessor(Preprocessor):
#     def __init__(
#             self,
#             text_field,
#             doc_field,
#             pre=None,
#             memoize=False,
#             crf_model_path=f"{WORK_PATH}/sources/comp_char_crf_bmeso_model.pkl",
#             cls_model_path=f"{WORK_PATH}/sources/fasttext_name_model.bin"
#     ):
#         name = type(self).__name__
#         super().__init__(
#             name,
#             field_names=dict(text=text_field),
#             mapped_field_names=dict(doc=doc_field),
#             pre=pre,
#             memoize=memoize
#         )
#         self.crf_model_path = crf_model_path
#         self.cls_model_path = cls_model_path
#         self._nlp = dict(
#             ner=self._ner.predict(text), cls=self._cls.predict(text)
#         )
#
#     def run(self, text):
#         return dict(doc=dict(
#             ner=crf.predict(test_data=text, model_path=self.crf_model_path, return_type="dict")[0],
#             cls=fast_text.predict(test_data=text, model_path=self.cls_model_path)
#         ))
#
#
# nlp = NlpPreprocessor(text_field="text", doc_field="doc", memoize=True)
