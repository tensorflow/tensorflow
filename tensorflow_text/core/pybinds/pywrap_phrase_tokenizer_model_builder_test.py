# coding=utf-8
# Copyright 2025 TF.Text Authors.
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

r"""Tests for pywrap_phrase_tokenizer_model_builder.cc."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import test_util
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow_text.core.pybinds import pywrap_phrase_tokenizer_model_builder

EXPECTED_MODEL_BUFFER_PATH = "tensorflow_text/python/ops/test_data/phrase_tokenizer_model_test.fb"


class PywrapPhraseBuilderTest(test_util.TensorFlowTestCase):

  def test_build(self):
    vocab = [
        "<UNK>",
        "I",
        "heard",
        "the",
        "news",
        "today",
        "have",
        "heard news today",
        "the news today",
    ]
    unk_token = "<UNK>"
    expected_model_buffer = gfile.GFile(EXPECTED_MODEL_BUFFER_PATH, "rb").read()
    self.assertEqual(
        pywrap_phrase_tokenizer_model_builder.build_phrase_model(
            vocab, unk_token, True, 0, False), expected_model_buffer)


if __name__ == "__main__":
  test.main()
