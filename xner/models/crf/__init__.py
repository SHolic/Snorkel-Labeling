from .features import Feature, BasicFeature, NgramFeature, DistrictFeature
from .model import CRF
from .utils import load_train_data, load_test_data, _trans2word, clean
from .. import _merge_label, calc_acc_with_merged_label
from ... import SETTINGS

import time
import multiprocessing as mp
from sklearn_crfsuite import metrics
import random
import pandas as pd

ftr_name_dict = {ftr.__name__: ftr for ftr in Feature.__subclasses__()}
used_instances = [ftr_name_dict[ftr]() for ftr in SETTINGS["crf_features"]]


def _create_feature(text, mode):
    ret = []
    domain_features = []
    for ins in used_instances:
        domain_features.append(ins.create(text, mode))
    for row in zip(*[ftr for ftr in domain_features if ftr]):
        for d in row[1:]:
            row[0].update(d)
        ret.append(row[0])
    return ret


# function for using multiprocess
def __create_char_feature(text):
    return _create_feature(text, "char")


# function for using multiprocess
def __create_word_feature(text):
    return _create_feature(text, "word")


def train(train_data_path, model_path=None, model_params=None, mode="char", train_test_split_rate=0.8):
    sentences, y = load_train_data(train_data_path, mode=mode)
    # 特征生成如果很多，则用并行加快速度
    start_time = time.time()
    if len(sentences) > 1000:
        if mp.cpu_count() > 4:
            used_kernel_num = 4
        elif mp.cpu_count() > 2:
            used_kernel_num = 2
        else:
            used_kernel_num = 1
        with mp.Pool(used_kernel_num) as p:
            if mode == "char":
                x = p.map(__create_char_feature, sentences)
            else:
                x = p.map(__create_word_feature, sentences)
    else:
        x = [_create_feature(sent, mode) for sent in sentences]
    print("Create Feature Time: ", round(time.time() - start_time, 3), "s")

    # 分测试集和训练集，因为要计算合并的label的acc，因此暂不使用sklearn的函数
    random.seed(123)
    random.shuffle(x)
    random.seed(123)
    random.shuffle(y)
    random.seed(123)
    random.shuffle(sentences)

    x_train = x[:int(len(x) * train_test_split_rate)]
    x_test = x[int(len(x) * train_test_split_rate):]
    y_train = y[:int(len(y) * train_test_split_rate)]
    y_test = y[int(len(y) * train_test_split_rate):]
    sentences_train = sentences[:int(len(sentences) * 0.8)]
    sentences_test = sentences[int(len(sentences) * 0.8):]

    crf = CRF(params=model_params).train(x_train, y_train)

    # predict
    if mode == "word":
        # 需要重新切词计算特征
        new_sentence_train = _trans2word(["".join(sent) for sent in sentences_train])
        new_sentence_test = _trans2word(["".join(sent) for sent in sentences_test])
        x_train = [_create_feature(sent, mode) for sent in new_sentence_train]
        x_test = [_create_feature(sent, mode) for sent in new_sentence_test]
    pred_train = crf.predict(x_train)
    pred_test = crf.predict(x_test)

    # 把标签merge起来
    if mode == "word":
        train_pred_data = _merge_label(new_sentence_train, pred_train)
    else:
        train_pred_data = _merge_label(sentences_train, pred_train)
    train_true_data = _merge_label(sentences_train, y_train)

    train_acc = calc_acc_with_merged_label(zip(train_true_data[0], train_true_data[1]),
                                           zip(train_pred_data[0], train_pred_data[1]))

    if mode == "word":
        test_pred_data = _merge_label(new_sentence_test, pred_test)
    else:
        test_pred_data = _merge_label(sentences_test, pred_test)
    test_true_data = _merge_label(sentences_test, y_test)

    test_acc = calc_acc_with_merged_label(zip(test_true_data[0], test_true_data[1]),
                                          zip(test_pred_data[0], test_pred_data[1]))

    print("\nMerged label ACC:")
    acc = pd.DataFrame([train_acc, test_acc], index=["train", "test"])
    print(acc)

    labels = [i for i in crf.model.classes_ if i != "O"]

    if mode == "char":
        # "word"级别，切出来的词与标注词不一致，因此无法使用 metrics.flat_classification_report
        print("\nTest label report:")
        print(metrics.flat_classification_report(y_test, pred_test, labels=labels))

    if model_path:
        crf.save(model_path)


def predict(test_data_path=None, test_data=None, model_path=None, return_type=None, mode="char"):
    sentences = None
    if test_data:
        sentences = [test_data] if isinstance(test_data, str) else test_data
        sentences = [clean(sent) for sent in sentences]
        if mode == "word":
            sentences = _trans2word(sentences)
    if test_data_path:
        sentences = load_test_data(test_data_path, mode)

    # 特征生成如果很多，则用并行加快速度
    if len(sentences) > 1000:
        if mp.cpu_count() > 4:
            used_kernel_num = 4
        elif mp.cpu_count() > 2:
            used_kernel_num = 2
        else:
            used_kernel_num = 1
        with mp.Pool(used_kernel_num) as p:
            if mode == "char":
                x = p.map(__create_char_feature, sentences)
            else:
                x = p.map(__create_word_feature, sentences)
    else:
        x = [_create_feature(sent, mode) for sent in sentences]

    crf = CRF.load(model_path)
    pred = crf.predict(x)

    if not return_type:
        return [sentences, pred]
    if return_type == "merge":
        return _merge_label(sentences, pred)
    if return_type == "dict":
        ret_dict = []
        ret_sent, ret_label = _merge_label(sentences, pred)
        for _sent, _label in zip(ret_sent, ret_label):
            sent_dict = dict()
            for _s, _l in zip(_sent, _label):
                if _l not in sent_dict.keys():
                    sent_dict[_l] = _s
                else:
                    sent_dict[_l] += "," + _s
            ret_dict.append(sent_dict)
        return ret_dict
