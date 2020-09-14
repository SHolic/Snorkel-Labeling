import sklearn_crfsuite
import joblib
import time

from .. import Model


class CRF(Model):
    def __init__(self, params=None):
        self.params = {
            "algorithm": "ap",
            "all_possible_states": True,
            "all_possible_transitions": False,
            "max_iterations": 100,
        }
        if params:
            self.params.update(params)
        self.model = None

    def train(self, x, y):
        start_time = time.time()
        self.model = sklearn_crfsuite.CRF(**self.params)
        self.model.fit(x, y)
        print("Train Time: ", round(time.time()-start_time, 3), "s")

        return self

    def predict(self, x):
        return self.model.predict(x)

    def save(self, path):
        joblib.dump({
            "model": self.model,
            "params": self.params,
        }, path)

    @staticmethod
    def load(path):
        save_data = joblib.load(path)
        crf = CRF(params=save_data["params"])
        crf.model = save_data["model"]
        return crf
