# CCLUE

古文自然语言理解测评基准。包括代表性任务对应的数据集、基准（预训练）模型、评测代码。

## 排行榜

| Model                   | Score   | S&P     | NER     | CLS     | SENT    | RETR    |
|-------------------------|---------|---------|---------|---------|---------|---------|
| chinese-roberta-wwm-ext |  70.88  |  71.92  |  80.77  |  81.91  |  58.65  |  61.15  |
| guwenbert-base-fs       |  77.21  |  **80.96**  |  88.66  |  **84.99**  |  59.75  |  71.69  |
| guwenbert-base          |  **77.55**  |  80.25  |  **90.24**  |  84.56  |  **60.40**  |  **72.28** |

## Baselines

### Chinese-BERT-wwm

基于全词遮罩（Whole Word Masking）技术的中文预训练模型

https://github.com/ymcui/Chinese-BERT-wwm

### guwenbert

基于古文语料和继续训练技术的预训练语言模型

https://github.com/Ethan-yt/guwenbert

### guwenbert-fs

从头训练的古文预训练语言模型

https://1drv.ms/u/s!AuBc6K5UDq9Um1AgkbHqwCVCnB7O?e=XbGssd

## Tasks


### 断句与标点 S&P

| Model                   | SEG   | PUNC  | QUOTE | AVG   |
|-------------------------|-------|-------|-------|-------|
| chinese-roberta-wwm-ext | 85.24 | 71.10 | 59.43 | 71.92 |
| guwenbert-base-fs       | 93.01 | 80.81 | 69.06 | **80.96** |
| guwenbert-base          | 92.62 | 80.01 | 68.12 | 80.25 |

#### 断句 SEG

```shell
sh run_punctuation.sh seg [Model Name]
```

| Model                   | Precision | Recall | F1     |
|-------------------------|-----------|--------|--------|
| chinese-roberta-wwm-ext | 86.25     | 84.26  | 85.24  |
| guwenbert-base-fs       | 92.54     | 93.49  | 93.01  |
| guwenbert-base          | 92.61     | 92.64  | 92.62  |

#### 标点 PUNC

```shell
sh run_punctuation.sh punc [Model Name]
```
    
| Model                   | Precision | Recall | F1     |
|-------------------------|-----------|--------|--------|
| chinese-roberta-wwm-ext | 72.37     | 69.88  | 71.10  |
| guwenbert-base-fs       | 80.06     | 81.57  | 80.81  |
| guwenbert-base          | 79.83     | 80.19  | 80.01  |

#### 引号标注 QUOTE

```shell
sh run_punctuation.sh quote [Model Name]
```
| Model                   | Precision | Recall | F1    |
|-------------------------|-----------|--------|-------|
| chinese-roberta-wwm-ext | 59.75     | 59.12  | 59.43 |
| guwenbert-base-fs       | 63.56     | 75.61  | 69.06 |
| guwenbert-base          | 63.79     | 73.08  | 68.12 |

### 命名实体识别 NER

```shell
sh run_ner.sh [Model Name] <crf>
```

| Model                       | F1    |
|-----------------------------|-------|
| chinese-roberta-wwm-ext     | 77.79 |
| chinese-roberta-wwm-ext-crf | 80.77 |
| guwenbert-base-fs           | 87.78 |
| guwenbert-base-fs-crf       | 88.66 |
| guwenbert-base              | 88.02 |
| guwenbert-base-crf          | 90.24 |

### 古文分类 CLS

```shell
sh run_classification.sh [Model Name]
```

| Model                   | Acc     |
|-------------------------|---------|
| chinese-roberta-wwm-ext |  81.91  |
| guwenbert-base-fs       |  84.99  |
| guwenbert-base          |  84.56  |

### 古诗情感分类 SENT

```shell
sh run_fspc.sh [Model Name]
sh run_fspc_poem.sh [Model Name]
```

| 模型                      | 诗句     | 诗文     | 平均     |
|-------------------------|--------|--------|--------|
| chinese-roberta-wwm-ext | 59.90  | 57.40  | 58.65  |
| guwenbert-base-fs       | 61.70  | 57.80  | 59.75  |
| guwenbert-base          | 61.80  | 59.00  | 60.40  |

### 文白检索 RETR

```shell
sh run_retrieval.sh
```

#### 实验结果

| 模型                      | 文言-白话 | 白话-文言 | 平均     |
|-------------------------|-------|-------|--------|
| chinese-roberta-wwm-ext | 57.92 | 64.37 | 61.15  |
| guwenbert-base-fs       | 78.11 | 65.27 | 71.69  |
| guwenbert-base          | 80.39 | 64.16 | 72.28  |

