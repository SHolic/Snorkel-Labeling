import re
import jieba
import os

from ... import SETTINGS

WORK_PATH = os.getcwd()
__depends__ = {
    "district_path": f"{WORK_PATH}/xner/sources/district_type.txt"
}

with open(__depends__["district_path"], "r", encoding="utf-8") as f:
    for line in f.readlines():
        jieba.add_word(line.strip().split("\t")[0])


def clean(text):
    return re.sub("[^\u4e00-\u9fa5\d\w\-·（）()]|_", "", text)


def _trans2word(text_list):
    return [jieba.lcut(text) for text in text_list]


def load_test_data(path, mode="char"):
    with open(path, "r", encoding="utf-8") as f:
        test_data = [clean(line.strip()) for line in f.readlines()]
    if mode == "char":
        return list(test_data)
    if mode == "word":
        return _trans2word(list(test_data))
    return None


def load_train_data(path,
                    mode="char",
                    district_path=__depends__["district_path"]):
    if mode == "word":
        with open(district_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                jieba.add_word(line.strip().split("\t")[0])

    sentences, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        _used_label = set(SETTINGS["labels"])
        for i, line in enumerate(f.readlines()):
            token = line.strip().split("\t")
            if len(token) % 2 != 0 or \
               len([label.upper() for i, label in enumerate(token) if i % 2 == 1 and label.upper() not in _used_label]) > 0:
                print(i+1, token, "train data has error!")
            addr = None
            sentence = []
            label = []
            for j, v in enumerate(token):
                if j % 2 == 0:
                    addr = v if mode == "char" else jieba.lcut(v)
                    sentence += addr
                    continue
                if v.upper() == "O":
                    label += ["O"] * len(addr)
                else:
                    if len(addr) == 1:
                        label += ["S-" + v.upper()]
                    else:
                        if SETTINGS["label_type"] == "bmeso":
                            label += ["B-" + v.upper()] + ["M-" + v.upper()] * (len(addr) - 2) + ["E-" + v.upper()]
                        elif SETTINGS["label_type"] == "biso":
                            label += ["B-" + v.upper()] + ["I-" + v.upper()] * (len(addr) - 1)
                        else:
                            raise ValueError(f"\"LABEL_TYPE\" should be \"bmeso\" or \"biso\", but currently is \"{SETTINGS['label_type']}\"")
            if mode == "char":
                sentence = "".join(sentence)
            sentences.append(sentence)
            labels.append(label)

            if len(sentence) != len(label):
                print(i+1, token, "train data has error!")
    return sentences, labels
