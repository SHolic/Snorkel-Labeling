import re
import os

from ... import SETTINGS

WORK_PATH = os.getcwd()


def clean(text):
    return re.sub("[^\u4e00-\u9fa5\d\w\-·（）()]", "", text)


def load_test_data(path):
    with open(path, "r") as f:
        return [[char for char in clean(line.strip())] for line in f.readlines()]


def load_train_data(path):
    sentences, labels = [], []
    _used_label = set(SETTINGS["labels"])
    with open(path, "r") as f:
        for i, line in enumerate(f.readlines()):
            token = line.strip().split("\t")
            if len(token) % 2 != 0 or \
                    len([label.upper() for i, label in enumerate(token) if
                         i % 2 == 1 and label.upper() not in _used_label]) > 0:
                print(i + 1, token, "train data has error!")
            addr = None
            sentence = ""
            label = []
            for j, v in enumerate(token):
                if j % 2 == 0:
                    addr = v
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
                            raise ValueError(
                                f"\"LABEL_TYPE\" should be \"bmeso\" or \"biso\", but currently is \"{SETTINGS['label_type']}\"")
            sentences.append([i for i in sentence])
            labels.append(label)
            if len(sentence) != len(label):
                print(i+1, token, "train data has error!")
    return sentences[:20], labels[:20]
