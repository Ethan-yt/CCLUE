# coding=utf-8
# Copyright 2020 HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
import csv
import os

import datasets

logger = datasets.logging.get_logger(__name__)

_URL = ""
_TRAINING_FILE = "train.tsv"
_DEV_FILE = "dev.tsv"
_TEST_FILE = "test.tsv"


class TextClassificationConfig(datasets.BuilderConfig):

    def __init__(self, **kwargs):
        super(TextClassificationConfig, self).__init__(**kwargs)


class TextClassification(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        TextClassificationConfig(name="punctuation", version=datasets.Version("1.0.0"),
                                 description="TextClassification dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(
                        names=['易藏', '医藏', '艺藏', '史藏', '佛藏', '集藏', '诗藏', '子藏', '儒藏', '道藏']
                    )
                }
            ),
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        urls_to_download = {
            "train": f"{_URL}{_TRAINING_FILE}",
            "dev": f"{_URL}{_DEV_FILE}",
            "test": f"{_URL}{_TEST_FILE}",
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as csv_file:
            csv_reader = csv.reader(
                csv_file, delimiter="\t",
            )
            next(csv_reader)
            for id_, row in enumerate(csv_reader):
                label, text = row
                yield id_, {"text": text, "label": label}
