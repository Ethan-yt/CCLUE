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
import os

import datasets

logger = datasets.logging.get_logger(__name__)

_URL = ""
_TRAINING_FILE = "train.txt"
_DEV_FILE = "dev.txt"
_TEST_FILE = "test.txt"


class PunctuationConfig(datasets.BuilderConfig):

    def __init__(self, **kwargs):
        super(PunctuationConfig, self).__init__(**kwargs)


class Punctuation(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        PunctuationConfig(name="punctuation", version=datasets.Version("1.0.0"), description="Punctuation dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "seg_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                'O',
                                'B',
                            ]
                        )
                    ),
                    "punc_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                "O",
                                'B-,',
                                'B-.',
                                'B-?',
                                'B-!',
                                'B-\\',
                                'B-:',
                                'B-;',
                            ]
                        )
                    ),
                    "quote_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                'O',
                                'B',
                                'I',
                            ]
                        )
                    ),
                    "book_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                'O',
                                'B',
                                'I',
                            ]
                        )
                    ),
                    "has_quote": datasets.Value("bool"),
                    "has_book": datasets.Value("bool"),
                }
            ),
            supervised_keys=None,
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
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            tokens = []
            seg_tags = []
            punc_tags = []
            quote_tags = []
            book_tags = []
            has_quote = False
            has_book = False
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if line.startswith("-DOCSTART-"):
                        has_quote = '-QUOTE-' in line
                        has_book = '-BOOK-' in line
                    if tokens:
                        yield guid, {
                            "id": str(guid),
                            "tokens": tokens,
                            "seg_tags": seg_tags,
                            "punc_tags": punc_tags,
                            "quote_tags": quote_tags,
                            "book_tags": book_tags,
                            "has_quote": has_quote,
                            "has_book": has_book,
                        }
                        guid += 1
                        tokens = []
                        seg_tags = []
                        punc_tags = []
                        quote_tags = []
                        book_tags = []
                else:
                    # conll2003 tokens are space separated
                    splits = line.split(" ")
                    tokens.append(splits[0])
                    seg_tags.append(splits[1])
                    punc_tags.append(splits[2])
                    quote_tags.append(splits[3])
                    book_tags.append(splits[4])
            # last example
            yield guid, {
                "id": str(guid),
                "tokens": tokens,
                "seg_tags": seg_tags,
                "punc_tags": punc_tags,
                "quote_tags": quote_tags,
                "book_tags": book_tags,
                "has_quote": has_quote,
                "has_book": has_book,
            }
