from .model import BilstmCrf
from .utils import load_train_data, load_test_data, clean
from .. import _merge_label, calc_acc_with_merged_label

import random
import pandas as pd


def train(train_data_path, model_path=None, model_params=None, train_test_split_rate=0.9):
    x, y = load_train_data(train_data_path)

    random.seed(123)
    random.shuffle(x)
    random.seed(123)
    random.shuffle(y)

    x_train = x[:int(len(x) * train_test_split_rate)]
    x_test = x[int(len(x) * train_test_split_rate):]
    y_train = y[:int(len(y) * train_test_split_rate)]
    y_test = y[int(len(y) * train_test_split_rate):]

    bcrf = BilstmCrf(**model_params) if model_params else BilstmCrf()
    bcrf.train(x_train, y_train)

    pred_train = bcrf.predict(x_train)
    pred_test = bcrf.predict(x_test)

    # 把标签merge起来
    train_pred_data = _merge_label(["".join(x) for x in x_train], pred_train)
    train_true_data = _merge_label(["".join(x) for x in x_train], y_train)

    train_acc = calc_acc_with_merged_label(zip(train_true_data[0], train_true_data[1]),
                                           zip(train_pred_data[0], train_pred_data[1]))

    test_pred_data = _merge_label(["".join(x) for x in x_test], pred_test)
    test_true_data = _merge_label(["".join(x) for x in x_test], y_test)

    test_acc = calc_acc_with_merged_label(zip(test_true_data[0], test_true_data[1]),
                                          zip(test_pred_data[0], test_pred_data[1]))

    print("\nMerged label ACC:")
    acc = pd.DataFrame([train_acc, test_acc], index=["train", "test"])
    print(acc)

    if model_path:
        bcrf.save(model_path)


def predict(test_data_path=None, test_data=None, model_path=None, return_type=None):
    sentences = None
    if test_data:
        sentences = [test_data] if isinstance(test_data, str) else test_data
        sentences = [clean(sent) for sent in sentences]
        sentences = [[char for char in sent] for sent in sentences]

    if test_data_path:
        sentences = load_test_data(test_data_path)

    bcrf = BilstmCrf.load(model_path)
    pred = bcrf.predict(sentences)

    if not return_type:
        return pred
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
    return None
