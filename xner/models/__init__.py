from abc import abstractmethod, ABCMeta
import warnings
warnings.filterwarnings("ignore")

from .. import SETTINGS


class Model(metaclass=ABCMeta):
    @abstractmethod
    def train(self, x, y):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @staticmethod
    @abstractmethod
    def load(path):
        pass


def _merge_label(sentences, labels):
    # 确认标签类型
    if SETTINGS["label_type"] == "bmeso":
        func = _merge_label_sentence_bmeso
    elif SETTINGS["label_type"] == "biso":
        func = _merge_label_sentence_bio
    else:
        raise ValueError(
            f"\"LABEL_TYPE\" should be \"bmeso\" or \"biso\", but currently is \"{SETTINGS['label_type']}\"")

    sent_ret, label_ret = [], []
    for sentence, label in zip(sentences, labels):
        s, l = func(sentence, label)
        sent_ret.append(s)
        label_ret.append(l)
    return sent_ret, label_ret


def _merge_label_sentence_bmeso(sentence, label):
    length = len(label)
    ret_sentence, ret_label = [], []
    curr_index = 0

    while curr_index < length:
        curr_label = label[curr_index].split("-")[-1]
        curr_state = label[curr_index].split("-")[0]
        if curr_state == "S":
            ret_sentence.append(sentence[curr_index])
            ret_label.append(curr_label)
            curr_index += 1
            continue

        if curr_label == "O":
            curr_merge_label = curr_label
            curr_merge_sentence = ""
            while curr_label == "O":
                curr_merge_sentence += sentence[curr_index]
                curr_index += 1
                if curr_index >= length:
                    break
                curr_label = label[curr_index].split("-")[-1]
            ret_sentence.append(curr_merge_sentence)
            ret_label.append(curr_merge_label)
            continue

        curr_merge_label = curr_label
        curr_merge_sentence = ""
        while curr_merge_label == curr_label:
            curr_state = label[curr_index].split("-")[0]
            curr_merge_sentence += sentence[curr_index]
            if curr_state == "E":
                ret_sentence.append(curr_merge_sentence)
                ret_label.append(curr_merge_label)
                curr_merge_sentence = ""
            curr_index += 1
            if curr_index >= length:
                break
            curr_label = label[curr_index].split("-")[-1]
        if curr_merge_sentence != "":
            ret_sentence.append(curr_merge_sentence)
            ret_label.append(curr_merge_label)

    return ret_sentence, ret_label


def _merge_label_sentence_bio(sentence, label):
    length = len(label)
    ret_sentence, ret_label = [], []
    curr_index = 0

    while curr_index < length:
        curr_label = label[curr_index].split("-")[-1]
        curr_state = label[curr_index].split("-")[0]
        if curr_state == "S":
            ret_sentence.append(sentence[curr_index])
            ret_label.append(curr_label)
            curr_index += 1
            continue

        if curr_label == "O":
            curr_merge_label = curr_label
            curr_merge_sentence = ""
            while curr_label == "O":
                curr_merge_sentence += sentence[curr_index]
                curr_index += 1
                if curr_index >= length:
                    break
                curr_label = label[curr_index].split("-")[-1]
            ret_sentence.append(curr_merge_sentence)
            ret_label.append(curr_merge_label)
            continue

        curr_merge_label = curr_label
        curr_merge_sentence = ""
        while curr_merge_label == curr_label:
            curr_state = label[curr_index].split("-")[0]
            if curr_state == "B" and curr_merge_sentence != "":
                ret_sentence.append(curr_merge_sentence)
                ret_label.append(curr_merge_label)
                curr_merge_sentence = ""
            curr_merge_sentence += sentence[curr_index]
            curr_index += 1

            if curr_index >= length:
                break
            curr_label = label[curr_index].split("-")[-1]
        if curr_merge_sentence != "":
            ret_sentence.append(curr_merge_sentence)
            ret_label.append(curr_merge_label)

    return ret_sentence, ret_label


# for calc_acc_with_merged_label
def _to_dict(data):
    ret = dict()
    for k, v in zip(data[1], data[0]):
        if k not in ret.keys():
            ret[k] = []
        ret[k].append(v)
    return ret


def calc_acc_with_merged_label(true_data, pred_data):
    eval_labels = [l for l in SETTINGS["labels"] if l != "O"]
    result_dict = {
        label: {"correct": 0, "total": 0}
        for label in eval_labels
    }

    def update_result_dict(key, correct, total):
        result_dict[key]["correct"] += correct
        result_dict[key]["total"] += total

    for pred, true in zip(pred_data, true_data):
        pred_dict = _to_dict(pred)
        true_dict = _to_dict(true)
        for k, v in true_dict.items():
            if k not in eval_labels:
                continue
            if k not in pred_dict.keys():
                update_result_dict(k, 0, len(v))
            else:
                intersect = []
                v_length = len(v)
                for pred_v in pred_dict[k]:
                    if pred_v in v:
                        v.remove(pred_v)
                        intersect.append(pred_v)
                update_result_dict(k, len(intersect), v_length)

    total_, correct_ = 0, 0
    ret = dict()
    for label in eval_labels:
        total_ += result_dict[label]["total"]
        correct_ += result_dict[label]["correct"]
        if result_dict[label]["total"] != 0:
            ret.update({label: round(result_dict[label]["correct"] / result_dict[label]["total"], 3)})
    ret.update({"Total": round(correct_ / total_, 3)})
    return ret
