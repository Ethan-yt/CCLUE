<p align="center">
    <br>
    <img src="./assets/cclue.png" width="500"/>
    <br>
</p>
<p align="center">
<a href="https://github.com/ethan-yt/cclue/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues/ethan-yt/cclue"></a>
<a href="https://github.com/ethan-yt/cclue/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/ethan-yt/cclue"></a>
<a href="https://github.com/Ethan-yt/cclue/blob/main/LICENSE"><img alt="GitHub license" src="https://img.shields.io/github/license/ethan-yt/cclue"></a>
</p>

CCLUE是一个古文自然语言理解的测评基准，包括代表性任务对应的数据集、基准模型、评测代码，研究人员能够通过几行代码快速测评各种预训练语言模型。

## 新闻

2021/06/22 官方网站上线：https://cclue.top

## 任务介绍

该基准包含以下几个任务和数据集：

| 任务名    | 缩写   | 训练集    | 开发集   | 测试集   | 任务类型 | 指标    |
|--------|------|--------|-------|-------|------|-------|
| 断句和标点  | S&P   | 26935  | 4075  | 3992  | 序列标注 | F1 |
| 命名实体识别 | NER  | 2566   | 281   | 327   | 序列标注 | F1 |
| 古文分类   | CLS  | 160000 | 20000 | 20000 | 文本分类 | Acc   |
| 古诗情感分类 | SENT | 16000  | 2000  | 2000  | 文本分类 | Acc   |
| 文白检索   | RETR | -      | -     | 10000 | 文本检索 | Acc   |

## 快速测评

快速测评基于您提交的模型，使用默认的超参数设置微调，得到最终成绩。使用这种方式不需要下载代码本地测评。

1. 将你的模型上传至Huggingface。[查看文档](https://huggingface.co/transformers/model_sharing.html)
1. 申请测评。[链接](https://github.com/Ethan-yt/CCLUE/issues/new?assignees=Ethan-yt&labels=&template=quick_test.md&title=%5B快速测评%5D)
1. 测评结果将会在3个工作日内回复

## 本地测评方法

1. 下载数据集和测评代码: `git clone https://github.com/Ethan-yt/CCLUE.git`
1. 安装所需依赖(Python 3): `pip install -r requirements.txt`
1. 准备评测模型。支持[Hugging Face Models](https://huggingface.co/models)中列出的模型和本地相同格式的模型
1. 评测所有任务
1. 收集测评结果。测评结果位于`outputs`文件夹。模型的最终得分为几个任务的平均值。

评测代码如下：
```bash
# 断句和标点任务
sh run_punctuation.sh seg [Model Name]
sh run_punctuation.sh punc [Model Name]
sh run_punctuation.sh quote [Model Name]

# 命名实体识别任务
sh run_ner.sh [Model Name] crf

# 古文分类任务
sh run_classification.sh [Model Name]

# 古诗情感分类任务
sh run_fspc.sh [Model Name]
sh run_fspc_poem.sh [Model Name]

# 文白检索任务
sh run_retrieval.sh [Model Name]
```

## 提交结果

您可以将测评结果提交至CCLUE排行榜。所有结果必须可以复现，经过认证后可以登陆CCLUE排行榜。[提交链接](https://github.com/Ethan-yt/CCLUE/issues/new?assignees=Ethan-yt&labels=&template=approve.md&title=%5B申请认证%5D)

提交时请注明以下信息：

- 提交单位（个人 / 团队名称）
- 模型名称
- 项目/论文地址（能够描述模型的结构，超参数设置，训练过程，创新点等）
- 模型权重链接（上传至Hugging Face，或国内外其他网盘）
- 评测结果（在outputs文件夹内，包括每个任务的超参数设置，随机数种子等以便复现，不要包括模型权重）
