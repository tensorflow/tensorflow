# Lint as: python2, python3
# # Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Tests for tensorflow.lite.python.optimize.format_converter."""

# These 3 lines below are not necessary in a Python 3-only module
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.lite.python.optimize import sparsifier
from tensorflow.python.framework import test_util
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import test


class SparsifierTest(test_util.TensorFlowTestCase):

  def test_simple(self):
    model_path = resource_loader.get_path_to_datafile(
        '../../testdata/multi_add.bin')
    dense_model = open(model_path, 'rb').read()
    converter = sparsifier.Sparsifier(dense_model)

    sparse_model = converter.sparsify()
    self.assertIsNotNone(sparse_model)


if __name__ == '__main__':
  test.main()
