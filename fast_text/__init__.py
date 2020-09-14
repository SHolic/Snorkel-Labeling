from .model import FT
from .utils import load_train_data, load_test_data, tokenize, _transform, _merge_label


def _get_feature(train_data_path, model_path):
    sentences, y = load_train_data(train_data_path)
    ft = FT.load(model_path)
    return [ft.get_sentence_vector(sent).tolist() for sent in sentences]


def train(train_data_path, model_path=None, model_params=None, quantize=False, autotune=False):
    sentences, y = load_train_data(train_data_path)
    x = _merge_label(sentences, y)
    ft = FT(params=model_params).train(x, qtz=quantize, auto=autotune)
    if model_path:
        ft.save(model_path)


def predict(test_data_path=None, test_data=None, model_path=None, return_type=None):
    token = []
    if test_data:
        sentences = [test_data] if isinstance(test_data, str) else test_data
        for sent in sentences:
            token.append(tokenize(sent))

    if test_data_path:
        token = load_test_data(test_data_path)

    ft = FT.load(model_path)

    pred = [e[0] for e in ft.predict(token)[0]]

    if not return_type:
        return pred
    if return_type == "transformed":
        return _transform(pred)
