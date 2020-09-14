import pandas as pd
import random
from xner.models import crf
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from snorkel.classification.data import DictDataset, DictDataLoader
import os

WORK_PATH = os.getcwd()


def get_replacements(
        token_list,
        ner_data_path=f"{WORK_PATH}/snorkel_flow/sources/sampled_company.txt",
        ner_model_path=f"{WORK_PATH}/snorkel_flow/sources/comp_char_crf_bmeso_model.pkl"
):
    ner = crf.predict(
        test_data_path=ner_data_path,
        model_path=ner_model_path,
        return_type="dict",
        mode="char"
    )
    replacements = dict()
    for tokens in ner:
        for token_name in token_list:
            if (token_name in tokens.keys()) and (token_name in replacements.keys()):
                replacements[token_name] += set(tokens.get(token_name).split(','))
            if (token_name in tokens.keys()) and (token_name not in replacements.keys()):
                replacements[token_name] = []
    return replacements


def random_pick(some_list, probabilities):
    global item
    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    for item, item_probability in zip(some_list, probabilities):
        cumulative_probability += item_probability
        if x < cumulative_probability:
            break
    return item


def add_ind(text):
    pick = random_pick([1, 0], [0.6, 0.4])
    if pick:
        front = random_pick([1, 0], [0.7, 0.3])
        if front:
            return "个体" + text
        return text + "个体"
    return text


def load_ind_dataset():
    company1 = pd.read_table(f"{WORK_PATH}/snorkel_flow/sources/Company-Names-Corpus（480W）.txt", skiprows=3,
                             header=None) \
        .sample(1100000, random_state=123)
    company2 = pd.read_table(f"{WORK_PATH}/snorkel_flow/sources/Company-Shorter-Form（28W）.txt", skiprows=3, header=None) \
        .sample(100000, random_state=123)
    company3 = pd.read_table(f"{WORK_PATH}/snorkel_flow/sources/Organization-Names-Corpus（110W）.txt", skiprows=3,
                             header=None) \
        .sample(300000, random_state=123)
    company = pd.concat([company1, company2, company3])
    company['label'] = 0

    name = pd.read_table(f"{WORK_PATH}/snorkel_flow/sources/Chinese_Names_Corpus（120W）.txt", skiprows=3, header=None)
    name['label'] = 1
    name[0] = name[0].apply(add_ind)

    df = pd.concat([company, name]).sample(frac=1, random_state=123).reset_index(drop=True)
    df.columns = ['text', 'label']
    X_train, X_test, y_train, y_test = \
        train_test_split(df[["text"]], df[["label"]], test_size=0.2, random_state=0)

    df_train = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
    df_test = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)

    return df_train, df_test


def save_sampled_company(n):
    sampled_company1 = pd.read_table(f"{WORK_PATH}/snorkel_flow/sources/Company-Names-Corpus（480W）.txt", skiprows=3,
                                     header=None) \
        .sample(n // 2, random_state=123)
    sampled_company2 = pd.read_table(f"{WORK_PATH}/snorkel_flow/sources/Organization-Names-Corpus（110W）.txt",
                                     skiprows=3, header=None) \
        .sample(n // 2, random_state=123)
    sampled_company = pd.concat([sampled_company1, sampled_company2], axis=0)
    sampled_company.to_csv(f"{WORK_PATH}/snorkel_flow/sources/sampled_company.txt", index=False)


def create_dict_dataloader(X, Y, split, **kwargs):
    """Create a DictDataLoader for bag-of-words features."""
    ds = DictDataset.from_tensors(torch.FloatTensor(X), torch.LongTensor(Y), split)
    return DictDataLoader(ds, **kwargs)


def get_pytorch_mlp(hidden_dim, num_layers):
    layers = []
    for _ in range(num_layers):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
    return nn.Sequential(*layers)