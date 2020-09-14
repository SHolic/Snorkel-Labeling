from snorkel_flow.utils import load_ind_dataset
from snorkel_flow.evaluations import labeling_evaluation, augmentation_evaluation, slicing_evaluation


df_train, df_test = load_ind_dataset()


# labeling
# A simple baseline: take the majority vote on a per-data point basis
df_train_filtered, preds_train_filtered, analysis = \
    labeling_evaluation(df_train[:200], df_test[:40], label_model="majority")
print(analysis)
# learn weights for the labeling functions
df_train_filtered, probs_train_filtered, preds_train_filtered, analysis = \
    labeling_evaluation(df_train[:200], df_test[:40], label_model="weighted")
print(analysis)


# augmentation
# apply uniformly at random per data point
df_train_augmented, Y_train_augmented = augmentation_evaluation(df_train[:200], df_test[:40], policy="random")
# specify a sampling distribution for the TFs
df_train_augmented, Y_train_augmented = augmentation_evaluation(df_train[:200], df_test[:40], policy="mean")


# slicing
# a simple classifier: LogisticRegression
analysis = slicing_evaluation(df_train[:10000], df_test[:2000], train_model="lr")
print(analysis)
# combine many slice-specific representations with an attention mechanism: SliceAwareClassifier(multi-task learning)
analysis = slicing_evaluation(df_train[:10000], df_test[:2000], train_model="mlp")
print(analysis)