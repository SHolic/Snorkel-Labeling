import fasttext
from sklearn.model_selection import train_test_split
from imblearn import metrics
import time
import re
import os
import multiprocessing

WORK_PATH = os.getcwd()
CPUs = multiprocessing.cpu_count()


class FT:
    def __init__(self, train_test_split_rate=0.2, params=None):
        self.train_test_split_rate = train_test_split_rate
        self.params = {
            "lr": 1,  # learning rate [0.1]
            "dim": 100,  # size of word vectors [100]
            "ws": 5,  # size of the context window [5]
            "epoch": 25,  # number of epochs [5]
            "minCount": 1,  # minimal number of word occurences [1]
            "minCountLabel": 0,  # minimal number of label occurences [1]
            "minn": 0,  # min length of char ngram [0]
            "maxn": 0,  # max length of char ngram [0]
            "neg": 5,  # number of negatives sampled [5]
            "wordNgrams": 4,  # max length of word ngram [1]
            "loss": 'softmax',  # loss function {ns, hs, softmax, ova} [softmax]
            "bucket": 2000000,  # number of buckets [2000000]
            "thread": CPUs,  # number of threads [number of cpus]
            "lrUpdateRate": 100,  # change the rate of updates for the learning rate [100]
            "t": 0.0001,  # sampling threshold [0.0001]
            "label": '__label__',  # label prefix ['__label__']
            "verbose": 2,  # verbose [2]
            "pretrainedVectors": ''  # pretrained word vectors (.vec file) for supervised learning []
        }
        if params:
            self.params.update(params)
        self.model = None

    @staticmethod
    def _print_results(N, p, r):
        print("N\t" + str(N))
        print("P@{}\t{:.3f}".format(1, p))
        print("R@{}\t{:.3f}".format(1, r))

    def train(self, train_data, qtz, auto):
        y = []
        x = []
        for line in train_data:
            y.append(line.split(" ")[0])
            each_text = ' '.join(line.split(" ")[1:])
            each_text = re.sub('\n', '', each_text)
            x.append(each_text)

        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            test_size=self.train_test_split_rate,
                                                            random_state=0)

        train_interm_path = f"{WORK_PATH}/interm_data/train_interm.txt"
        test_interm_path = f"{WORK_PATH}/interm_data/test_interm.txt"

        train_ret = []
        for x_, y_ in zip(x_train, y_train):
            train_ret.append(y_ + " " + x_ + "\n")

        test_ret = []
        for x_, y_ in zip(x_test, y_test):
            test_ret.append(y_ + " " + x_ + "\n")

        with open(train_interm_path, "w", encoding="utf-8") as tr:
            tr.writelines(train_ret)

        with open(test_interm_path, "w", encoding="utf-8") as te:
            te.writelines(test_ret)

        if not auto:
            start_time = time.time()
            self.model = fasttext.train_supervised(input=train_interm_path, **self.params)
            print("Train Time: ", round(time.time() - start_time, 3), " s")
        else:
            start_time = time.time()
            self.model = fasttext.train_supervised(input=train_interm_path,
                                                    thread=CPUs,
                                                    verbose=2,
                                                    autotuneValidationFile=test_interm_path)
            print("Train Time: ", round(time.time() - start_time, 3), " s")

        if qtz:
            start_time = time.time()
            self.model.quantize(train_interm_path, thread=CPUs, verbose=2, retrain=True)
            print("Retrain Time: ", round(time.time() - start_time, 3), " s")

        y_train_pred = [e[0] for e in self.model.predict(x_train)[0]]
        print("train acc:")
        self._print_results(*self.model.test(train_interm_path))
        print("train label report:")
        print(metrics.classification_report_imbalanced(y_train, y_train_pred))

        y_test_pred = [e[0] for e in self.model.predict(x_test)[0]]
        print("test acc:")
        self._print_results(*self.model.test(test_interm_path))
        print("test label report:")
        print(metrics.classification_report_imbalanced(y_test, y_test_pred, labels=self.model.labels))

        return self

    def predict(self, x):
        return self.model.predict(x)

    def save(self, path):
        self.model.save_model(path)

    @staticmethod
    def load(path):
        return fasttext.load_model(path)
