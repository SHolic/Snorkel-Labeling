# Snorkel-Labeling

**本项目用于业务经验标注、数据增强、切片模型场景。snorkel还支持众包项目，多任务模型场景，本项目不包括这两个功能。
本项目场景：**

- 个体户分类

**当前可供选择的策略：**
- 业务经验标注
    - MajorityLabelVoter
    - LabelModel
- 数据增强
    - RandomPolicy
    - MeanFieldPolicy
- 切片模型
    - LogisticRegression
    - SliceAwareClassifier

# 1. 使用方法

下载网盘数据，各个模块`fast_text`，`snorkel_flow`，`fast_text`都有对应的`sources`文件

## 1.1. 项目配置
 1. **把`snorkel_flow`文件夹移动到工作目录下**
 2. `import` 这个模块
 3. 由于在nlp场景中需要用到文本分类和实体识别，这里同样把`fast_text`和`xner`模块加入进来。

- `fast_text`项目：https://git.creditx.com/shenghl/fast-text-classifier
- `xner`项目：https://git.creditx.com/baojs/NLP-NER

## 1.2. 参数配置
模块`snorkel_flow` 的基本参数目前有两个:
1. `re` : 标注需要用到的正则表达式
2. `keywords` : 标注需要用到的关键词

```python
import snorkel_flow

# 默认是个体户分类，在做公司名称标注时，则需要增加相应的正则表达式
snorkel_flow.set_option("re_cmp", '公司|集团|有限|办事处|合作社')

# 增加相应的关键词
snorkel_flow.set_option("keywords_indus", ['三方', '中介', '中心', ...])
```

也可以在`snorkel_flow/__init__.py`里直接修改
```python
# snorkel_flow/__init__.py
SETTINGS = {
    "re_cmp": '公司|集团|有限|办事处|合作社',
    "keywords_indus": ['三方', '中介', '中心', ...],
    "re_surname": '王|李|张|刘|陈|杨|黄|...'
}
```

`Polarity`用来定义

## 1.3. 输入数据格式
输入格式为`pd.DataFrame`，在nlp场景中需要两个columns: text和label。如果是其他场景需在`functions.py`中改写三类function.
在`snorkel_flow.utils`中定义了`load_ind_dataset`函数，可以直接导出`df_train`和`df_test`
```python
from snorkel_flow.utils import load_ind_dataset
df_train, df_test = load_ind_dataset()
df_train
```
```text
	text	label
0	个体崔晓冰	1
1	个体玉影	1
2	花猪	0
3	个体吴和俊	1
4	南浔古镇旅游发展公司	0
...	...	...
2131003	个体吴幼章	1
2131004	胡仁伟	1
2131005	烟台长城冶金设备有限公司	0
2131006	高连波个体	1
2131007	胡保明	1
```

## 1.4. 三种功能apply并评估
`snorkel_flow`的各种功能都提供`evaluation`接口。

### 1.4.1. 标注功能
可以根据业务先验知识对输入数据进行弱监督标注。
我们需要将LF中的标签转换为每个数据点的单个噪声感知概率（或置信度加权）标签。 
一个简单的基准是在每个数据点上进行多数表决。我们可以使用MajorityLabelVoter基线模型进行测试。
`MajorityLabelVoter`只需将`label_model`参数设为"majority".
```python
from snorkel_flow.evaluations import labeling_evaluation
# A simple baseline: take the majority vote on a per-data point basis
df_train_filtered, preds_train_filtered, analysis = \
df_train_filtered = labeling_evaluation(df_train[:200], df_test[:40], label_model="majority")
print(analysis)
```
```text
Majority Vote Accuracy:   90.0%

                     j Polarity  Coverage  Overlaps  Conflicts
lf_ind_keyword       0      [1]     0.240     0.240      0.000
lf_short             1      [1]     0.305     0.305      0.000
lf_cmp_re            2      [0]     0.415     0.415      0.415
lf_industry_keyword  3      [0]     0.440     0.440      0.440
lf_surname_re        4      [1]     1.000     0.940      0.445
industry_cls         5       []     0.000     0.000      0.000
```
但是，从上一节中我们的LF的摘要统计中可以看出，它们具有不同的属性，因此不应一视同仁。
除了具有不同的准确性和覆盖范围外，LF可能会相关，从而导致某些信号在基于多数投票的模型中被过度代表。
为了适当地解决这些问题，我们将改用更复杂的Snorkel LabelModel来组合LF的输出。
`LabelModel`只需将`label_model`参数设为"weighted".
```python
from snorkel_flow.evaluations import labeling_evaluation
# learn weights for the labeling functions
df_train_filtered, probs_train_filtered, preds_train_filtered, analysis = \
labeling_evaluation(df_train[:200], df_test[:40], label_model="weighted")
print(analysis)
```
```text
Label Model Accuracy:     92.5%

                     j Polarity  Coverage  Overlaps  Conflicts
lf_ind_keyword       0      [1]     0.240     0.240      0.000
lf_short             1      [1]     0.305     0.305      0.000
lf_cmp_re            2      [0]     0.415     0.415      0.415
lf_industry_keyword  3      [0]     0.440     0.440      0.440
lf_surname_re        4      [1]     1.000     0.940      0.445
industry_cls         5       []     0.000     0.000      0.000
```

### 1.4.2. 数据增强

309/5000
我们需要定义一个策略，以确定将哪些TF序列应用于每个数据点。
我们将从RandomPolicy开始，该策略对`sequence_length = 2`TF进行采样，以在每个数据点随机均匀地应用。
`n_per_original`参数确定每个原始数据点要生成多少个增强数据点。
这些参数为默认参数，可以在`functions.py`中修改。
`RandomPolicy`只需将`policy`参数设为"random".
```python
from snorkel_flow.evaluations import augmentation_evaluation
# apply uniformly at random per data point
df_train_augmented, Y_train_augmented = \
augmentation_evaluation(df_train[:200], df_test[:40], policy="random")
```
```text
Original training set size: 200
Augmented training set size: 590
```
在某些情况下，我们可以做得比均匀分布更好。
我们可能具有领域知识，即某些TF应该比其他TF更频繁地应用，或者已经训练了自动数据扩充模型，该模型学习了TF的采样分布。
Snorkel通过MeanFieldPolicy支持此用例，它允许您指定TF的采样分布。
默认比例为`[0.1, 0.1, 0.1, 0.35, 0.35]`，可以直接通过`p`参数更改。
`MeanFieldPolicy`只需将`policy`参数设为"mean".
```python
from snorkel_flow.evaluations import augmentation_evaluation
# specify a sampling distribution for the TFs
df_train_augmented, Y_train_augmented = \
augmentation_evaluation(df_train[:200], df_test[:40], p=[0.1, 0.1, 0.1, 0.35, 0.35], policy="mean")
```
```text
Original training set size: 200
Augmented training set size: 580
```

### 1.4.3. 切片模型
提供一个为每个切片建模的baseline方法，使用`sklearn`中的逻辑回归。
`LogisticRegression`只需将`train_model`参数设为"lr".
```python
from snorkel_flow.evaluations import slicing_evaluation
# a simple classifier: LogisticRegression
analysis = slicing_evaluation(df_train[:10000], df_test[:2000], train_model="lr")
print(analysis)
```
```text
                        f1
overall           0.848485
short_comment     0.899899
ind_keyword       0.881818
cmp_re            0.000000
industry_keyword  0.000000
```
需要一种称为“基于切片的学习”的建模方法，该方法通过向使用的任何模型添加额外的特定于切片的表示能力来提高性能。
直观地讲，我们希望建模以学习更适合处理此切片中数据点的表示形式。 
在我们`SliceAwareClassifier`方法中，我们以多任务学习的方式将每个切片建模为单独的“专家任务”。
`SliceAwareClassifier`只需将`train_model`参数设为"mlp".
```python
from snorkel_flow.evaluations import slicing_evaluation
# combine many slice-specific representations 
# with an attention mechanism: SliceAwareClassifier(multi-task learning)
analysis = slicing_evaluation(df_train[:10000], df_test[:2000], train_model="mlp")
print(analysis)
```
```text
                              label         dataset  split metric     score
0                              task  SnorkelDataset  train     f1  0.847826
1     task_slice:short_comment_pred  SnorkelDataset  train     f1  0.903357
2       task_slice:ind_keyword_pred  SnorkelDataset  train     f1  0.881818
3            task_slice:cmp_re_pred  SnorkelDataset  train     f1  0.000000
4  task_slice:industry_keyword_pred  SnorkelDataset  train     f1  0.000000
5              task_slice:base_pred  SnorkelDataset  train     f1  0.847826
```

# 2. 三种功能function编写
三种功能的编写可以在`snorkel_flow.functions`模块中编写，每个功能用一个class封装。
有一个class比较特殊，我们可以自定义需要用到的预训练模型（文本分类、实体识别）。

## 2.1. 预训练模型
实体识别主要预测地址、核心词、经营范围、其他。用来指导数据增强。
注意载入的文本分类模块的模型的分类任务要与最终任务不同且相似，在这里最终任务是个体户分类，而载入的分类模型任务是行业分类，其中
行业分类可以指导个体户分类。
```python
from snorkel.preprocess import preprocessor
import os

WORK_PATH = os.getcwd()

class Preprocessors:
    @staticmethod
    @preprocessor(memoize=True)
    def cls_text(x):
        cls_ret = fast_text.predict(test_data=x.text,
                                    model_path=f"{WORK_PATH}/snorkel_flow/sources/fasttext_name_model.bin")
        x.cls = cls_ret
        return x

    @staticmethod
    @preprocessor(memoize=True)
    def ner_text(x):
        ner_ret = crf.predict(test_data=x.text,
                              model_path=f"{WORK_PATH}/snorkel_flow/sources/comp_char_crf_bmeso_model.pkl",
                              return_type="dict")
        x.ner = ner_ret[0]
        return x
```

## 2.2. 标注功能
通过文本的长短，是否包含关键字，是否符合某种正则pattern，以及不同任务的预训练模型来确定标签。
标签可以有冲突。
```python
from . import SETTINGS, Polarity
from snorkel.labeling import labeling_function
import re

class LabelingFunction:
    @staticmethod
    @labeling_function()
    def lf_ind_keyword(x):
        """Many individuals includes '个体'."""
        return Polarity.INDIVIDUAL.value if "个体" in x['text'] else Polarity.ABSTAIN.value

    @staticmethod
    @labeling_function()
    def lf_short(x):
        """Individuals are often short, such as '张三'."""
        return Polarity.INDIVIDUAL.value if len(x.text) < 5 else Polarity.ABSTAIN.value

    @staticmethod
    @labeling_function(resources=dict(re_cmp=SETTINGS['re_cmp']))
    def lf_cmp_re(x, re_cmp):
        """Many companies includes '公司',etc."""
        return Polarity.COMPANY.value if (
                    (re.search(re_cmp, x.text)) and ('个体' not in x.text)) else Polarity.ABSTAIN.value

    @staticmethod
    @labeling_function(pre=[Preprocessors.cls_text])
    def industry_cls(x):
        return Polarity.COMPANY.value if ((len(x.text) < 4) and ('__label__1' not in x.cls)) else Polarity.ABSTAIN.value
```

## 2.3. 数据增强
通过实体识别模型增加、删除或改变实体来实现数据增强功能
```python
from snorkel.augmentation import transformation_function
from snorkel_flow.utils import get_replacements

replacements = get_replacements(["ADDR", "BUSINESS", "O"])

class TransformationFunction:
    @staticmethod
    @transformation_function(pre=[Preprocessors.ner_text])
    def change_addr(x):
        addr = x.ner.get("ADDR", "").split(',')
        if addr:
            name_to_replace = np.random.choice(addr)
            replacement_name = np.random.choice(replacements["ADDR"])
            x.text = x.text.replace(name_to_replace, replacement_name)
            return x

    @staticmethod
    @transformation_function(pre=[Preprocessors.ner_text])
    def randomly_delete(x):
        label = np.random.choice(["ADDR", "BUSINESS", "O"])
        token = x.ner.get(label, "").split(',')
        if token:
            name_to_delete = np.random.choice(token)
            if name_to_delete == x.text:
                return x
            x.text = x.text.replace(name_to_delete, "")
            return x

    @staticmethod
    @transformation_function(pre=[Preprocessors.ner_text])
    def randomly_add(x):
        keyword = x.ner.get("KEYWORDS", "").replace(",", "")
        label_candidate = [l for l in ["ADDR", "BUSINESS", "O"] if not x.ner.get(l)]
        if keyword and label_candidate:
            label = np.random.choice(label_candidate)
            name_to_add = np.random.choice(replacements[label])
            if label == "ADDR":
                x.text = name_to_add + keyword + x.ner.get("BUSINESS", "").replace(',', '') + \
                         x.ner.get("O", "").replace(',', '')
                return x

            if label == "BUSINESS":
                x.text = x.ner.get("ADDR", "").replace(',', '') + keyword + name_to_add + \
                         x.ner.get("O", "").replace(',', '')
                return x

            if label == "O":
                x.text = x.ner.get("ADDR", "").replace(',', '') + keyword + \
                         x.ner.get("BUSINESS", "").replace(',', '') + name_to_add
                return x
```

## 2.4. 切片模型
切片模型编写与标注模型类似，通过文本长度、关键词、正则表达式、预训练模型来筛选我们关注的数据集的子集的预测效果。
```python
from snorkel.slicing import slicing_function

class SlicingFunction:
    @staticmethod
    @slicing_function()
    def short_comment(x):
        """Many individuals includes '个体'."""
        return len(x.text) < 5

    @staticmethod
    @slicing_function()
    def ind_keyword(x):
        """Many individuals includes '个体'."""
        return bool("个体" in x['text'])

    @staticmethod
    @slicing_function(resources=dict(re_cmp=SETTINGS['re_cmp']))
    def cmp_re(x, re_cmp):
        """Many companies includes '公司',etc."""
        return bool(re.search(re_cmp, x.text))

    @staticmethod
    @slicing_function(resources=dict(keywords_indus=SETTINGS['keywords_indus']))
    def industry_keyword(x, keywords_indus):
        """Many companies includes industry words."""
        for indus in keywords_indus:
            if indus in x.text:
                return True
        return False
```
