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

r"""Tests for pywrap_fast_bert_normalizer_model_builder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

from tensorflow.python.framework import test_util
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow_text.core.pybinds import pywrap_fast_bert_normalizer_model_builder

EXPECTED_MODEL_BUFFER_PATH = "tensorflow_text/python/ops/test_data/fast_bert_normalizer_model.fb"
EXPECTED_MODEL_LOWER_CASE_NFD_STRIP_ACCENTS_BUFFER_PATH = "tensorflow_text/python/ops/test_data/fast_bert_normalizer_model_lower_case_nfd_strip_accents.fb"

flags.DEFINE_boolean(
    "update_goldens", False, "Update the golden files after an ICU update."
)


class PywrapCodepointWiseTextNormalizerModelBuilderTest(
    test_util.TensorFlowTestCase):

  def test_build(self):
    result = pywrap_fast_bert_normalizer_model_builder.build_fast_bert_normalizer_model(
        False
    )
    if flags.FLAGS.update_goldens:
      gfile.GFile(EXPECTED_MODEL_BUFFER_PATH, "wb").write(result)
    else:
      expected_model_buffer = gfile.GFile(
          EXPECTED_MODEL_BUFFER_PATH, "rb"
      ).read()
      self.assertEqual(result, expected_model_buffer)

  def test_build_lower_case_nfd_strip_accents(self):
    result = pywrap_fast_bert_normalizer_model_builder.build_fast_bert_normalizer_model(
        True
    )
    if flags.FLAGS.update_goldens:
      gfile.GFile(
          EXPECTED_MODEL_LOWER_CASE_NFD_STRIP_ACCENTS_BUFFER_PATH, "wb"
      ).write(result)
    else:
      expected_model_buffer = gfile.GFile(
          EXPECTED_MODEL_LOWER_CASE_NFD_STRIP_ACCENTS_BUFFER_PATH, "rb"
      ).read()
      self.assertEqual(result, expected_model_buffer)


if __name__ == "__main__":
  test.main()
