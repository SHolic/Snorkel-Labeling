import re
import jieba
import json
import os

WORK_PATH = os.getcwd()
__depends__ = {
    "label2index": f"{WORK_PATH}/fast_text/sources/label2index.json",
    "stop_words": f"{WORK_PATH}/fast_text/sources/stopwords-zh.txt"
}


def clean(text):
    return re.sub("[^\u4e00-\u9fa5\d\w\-·]", "", text)


def tokenize(sent):
    # with open(stop_words_path, "r", encoding="utf-8") as sw:
    #     stop_words_ = sw.readlines()
    # stop_words = [w.replace("\n", "") for w in stop_words_]
    return ' '.join([w for w in jieba.lcut(clean(sent.strip()))])


def load_test_data(path):
    sentences = []
    with open(path, "r", encoding="utf-8") as f:
        for sentence in f.readlines():
            sentences.append(tokenize(sentence))
    return sentences


def load_train_data(path, label2index_path=__depends__["label2index"]):
    with open(label2index_path, "r", encoding="utf-8") as li:
        label_to_index = json.load(li)

    sentences, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            sentence, label = line.split("\t")[0], line.split("\t")[1]
            sentences.append(tokenize(sentence))
            labels.append(label_to_index[label.replace("\n", "")])
    return sentences, labels


def _merge_label(sentences, labels):
    ret = []
    for sentence, label in zip(sentences, labels):
        ret.append("__label__" + str(label) + " " + sentence + "\n")
    return ret


def _index2label(l2i_path):
    with open(l2i_path, "r", encoding="utf-8") as li:
        label_to_index = json.load(li)
    index_to_label = {}
    for key, label in label_to_index.items():
        index_to_label["__label__" + str(label)] = key
    return index_to_label


def _transform(
        labels,
        label2index_path=__depends__["label2index"]
):
    index_to_label = _index2label(label2index_path)
    trans_labels = []
    for label in labels:
        trans_labels.append(index_to_label[label])
    return trans_labels
