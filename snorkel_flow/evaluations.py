import numpy as np
import jieba
import os
from fast_text.model import FT
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from snorkel.analysis import Scorer
from snorkel.augmentation import RandomPolicy, MeanFieldPolicy, PandasTFApplier
from snorkel.classification import Trainer
from snorkel.labeling import LFAnalysis, PandasLFApplier, filter_unlabeled_dataframe
from snorkel.labeling.model import MajorityLabelVoter, LabelModel
from snorkel.slicing import SliceAwareClassifier, PandasSFApplier
from snorkel.utils import probs_to_preds, preds_to_probs

from .functions import LabelingFunction, TransformationFunction, SlicingFunction
from .utils import get_pytorch_mlp, create_dict_dataloader
from . import Polarity

WORK_PATH = os.getcwd()


def labeling_evaluation(df_train, df_test, label_model):
    lfs = [
        LabelingFunction.lf_ind_keyword,
        LabelingFunction.lf_short,
        LabelingFunction.lf_cmp_re,
        LabelingFunction.lf_industry_keyword,
        LabelingFunction.lf_surname_re,
        LabelingFunction.industry_cls
    ]

    applier = PandasLFApplier(lfs=lfs)
    L_train = applier.apply(df=df_train)
    L_test = applier.apply(df=df_test)
    Y_test = df_test.label.values
    analysis = LFAnalysis(L=L_train, lfs=lfs).lf_summary()

    if label_model == "majority":
        majority_model = MajorityLabelVoter()
        preds_train = majority_model.predict(L=L_train)
        majority_acc = majority_model.score(L=L_test, Y=Y_test, tie_break_policy="random")["accuracy"]
        print(f"{'Majority Vote Accuracy:':<25} {majority_acc * 100:.1f}%")

        df_train_filtered, preds_train_filtered = filter_unlabeled_dataframe(
            X=df_train, y=preds_train, L=L_train
        )
        return df_train_filtered, preds_train_filtered, analysis

    if label_model == "weighted":
        label_model = LabelModel(cardinality=len([c for c in dir(Polarity) if not c.startswith("__")]), verbose=True)
        label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=123)
        probs_train = label_model.predict_proba(L_train)
        label_model_acc = label_model.score(L=L_test, Y=Y_test, tie_break_policy="random")["accuracy"]
        print(f"{'Label Model Accuracy:':<25} {label_model_acc * 100:.1f}%")

        df_train_filtered, probs_train_filtered = filter_unlabeled_dataframe(
            X=df_train, y=probs_train, L=L_train
        )
        preds_train_filtered = probs_to_preds(probs_train_filtered)
        return df_train_filtered, probs_train_filtered, preds_train_filtered, analysis


def augmentation_evaluation(df_train, df_test, policy, p=None):
    tfs = [
        TransformationFunction.change_addr,
        TransformationFunction.change_business,
        TransformationFunction.change_o,
        TransformationFunction.randomly_delete,
        TransformationFunction.randomly_add
    ]

    if policy == "random":
        random_policy = RandomPolicy(
            len(tfs), sequence_length=2, n_per_original=2, keep_original=True
        )
        tf_applier = PandasTFApplier(tfs, random_policy)
        df_train_augmented = tf_applier.apply(df_train)
        Y_train_augmented = df_train_augmented["label"].values
        print(f"Original training set size: {len(df_train)}")
        print(f"Augmented training set size: {len(df_train_augmented)}")
        return df_train_augmented, Y_train_augmented

    if policy == "mean":
        if p is None:
            p = [0.1, 0.1, 0.1, 0.35, 0.35]
        mean_field_policy = MeanFieldPolicy(
            len(tfs),
            sequence_length=2,  # how many TFs to apply uniformly at random per data point
            n_per_original=2,  # how many augmented data points to generate per original data point
            keep_original=True,
            p=p,  # specify a sampling distribution for the TFs
        )
        tf_applier = PandasTFApplier(tfs, mean_field_policy)
        df_train_augmented = tf_applier.apply(df_train)
        Y_train_augmented = df_train_augmented["label"].values
        print(f"Original training set size: {len(df_train)}")
        print(f"Augmented training set size: {len(df_train_augmented)}")
        return df_train_augmented, Y_train_augmented


def slicing_evaluation(df_train, df_test, train_model=None):
    if train_model is None:
        train_model = "mlp"

    sfs = [
        SlicingFunction.short_comment,
        SlicingFunction.ind_keyword,
        SlicingFunction.cmp_re,
        SlicingFunction.industry_keyword
    ]

    slice_names = [sf.name for sf in sfs]
    scorer = Scorer(metrics=["f1"])

    ft = FT.load(f"{WORK_PATH}/snorkel_flow/sources/fasttext_name_model.bin")

    def get_ftr(text):
        return ft.get_sentence_vector(' '.join([w for w in jieba.lcut(text.strip())]))

    X_train = np.array(list(df_train.text.apply(get_ftr).values))
    X_test = np.array(list(df_test.text.apply(get_ftr).values))
    Y_train = df_train.label.values
    Y_test = df_test.label.values

    if train_model == "lr":
        sklearn_model = LogisticRegression(C=0.001, solver="liblinear")
        sklearn_model.fit(X=X_train, y=Y_train)
        preds_test = sklearn_model.predict(X_test)
        probs_test = preds_to_probs(preds_test, len([c for c in dir(Polarity) if not c.startswith("__")]))
        print(f"Test set F1: {100 * f1_score(Y_test, preds_test):.1f}%")
        applier = PandasSFApplier(sfs)
        S_test = applier.apply(df_test)
        analysis = scorer.score_slices(S=S_test, golds=Y_test, preds=preds_test, probs=probs_test, as_dataframe=True)
        return analysis

    if train_model == "mlp":
        # Define model architecture
        bow_dim = X_train.shape[1]
        hidden_dim = bow_dim
        mlp = get_pytorch_mlp(hidden_dim=hidden_dim, num_layers=2)

        # Initialize slice model
        slice_model = SliceAwareClassifier(
            base_architecture=mlp,
            head_dim=hidden_dim,
            slice_names=slice_names,
            scorer=scorer,
        )

        # generate the remaining S matrices with the new set of slicing functions
        applier = PandasSFApplier(sfs)
        S_train = applier.apply(df_train)
        S_test = applier.apply(df_test)

        # add slice labels to an existing dataloader
        BATCH_SIZE = 64

        train_dl = create_dict_dataloader(X_train, Y_train, "train")
        train_dl_slice = slice_model.make_slice_dataloader(
            train_dl.dataset, S_train, shuffle=True, batch_size=BATCH_SIZE
        )
        test_dl = create_dict_dataloader(X_test, Y_test, "train")
        test_dl_slice = slice_model.make_slice_dataloader(
            test_dl.dataset, S_test, shuffle=False, batch_size=BATCH_SIZE
        )

        #  fit our classifier with the training set dataloader
        trainer = Trainer(n_epochs=2, lr=1e-4, progress_bar=True)
        trainer.fit(slice_model, [train_dl_slice])

        analysis = slice_model.score_slices([test_dl_slice], as_dataframe=True)
        return analysis
