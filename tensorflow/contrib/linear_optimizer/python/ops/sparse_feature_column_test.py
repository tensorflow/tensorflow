# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for sparse_feature_column.py (deprecated).

This module and all its submodules are deprecated. To UPDATE or USE linear
optimizers, please check its latest version in core:
tensorflow_estimator/python/estimator/canned/linear_optimizer/.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.linear_optimizer.python.ops.sparse_feature_column import SparseFeatureColumn
from tensorflow.python.framework import ops
from tensorflow.python.framework.test_util import TensorFlowTestCase
from tensorflow.python.platform import googletest


class SparseFeatureColumnTest(TensorFlowTestCase):
  """Tests for SparseFeatureColumn.
  """

  def testBasic(self):
    expected_example_indices = [1, 1, 1, 2]
    expected_feature_indices = [0, 1, 2, 0]
    sfc = SparseFeatureColumn(expected_example_indices,
                              expected_feature_indices, None)
    self.assertTrue(isinstance(sfc.example_indices, ops.Tensor))
    self.assertTrue(isinstance(sfc.feature_indices, ops.Tensor))
    self.assertEqual(sfc.feature_values, None)
    with self.cached_session():
      self.assertAllEqual(expected_example_indices, sfc.example_indices.eval())
      self.assertAllEqual(expected_feature_indices, sfc.feature_indices.eval())
    expected_feature_values = [1.0, 2.0, 3.0, 4.0]
    sfc = SparseFeatureColumn([1, 1, 1, 2], [0, 1, 2, 0],
                              expected_feature_values)
    with self.cached_session():
      self.assertAllEqual(expected_feature_values, sfc.feature_values.eval())


if __name__ == '__main__':
  googletest.main()
