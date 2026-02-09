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

r"""Tests for pywrap_whitespace_tokenizer_config_builder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow_text.core.pybinds import pywrap_whitespace_tokenizer_config_builder as pywrap_builder


class PywrapFastWordpieceBuilderTest(test_util.TensorFlowTestCase):

  # This is not supposed to be an exhaustive test. That is done with the
  # builder test. We just want to sanity check a couple values to show we have
  # received something.
  def test_build(self):
    # check non-empty
    config = pywrap_builder.build_whitespace_tokenizer_config()
    self.assertNotEmpty(config)
    # check space character is whitespace
    character = ord(' ')
    bits = config[character >> 3]
    mask = 1 << (character & 0x7)
    self.assertGreater(bits & mask, 0)
    # check letter is not whitespace
    character = ord('a')
    bits = config[character >> 3]
    mask = 1 << (character & 0x7)
    self.assertEqual(bits & mask, 0)


if __name__ == '__main__':
  test.main()
