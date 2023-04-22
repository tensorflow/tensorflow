# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for `tf.data.Dataset.__len__()`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.platform import test


class LenTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(test_base.eager_only_combinations())
  def testKnown(self):
    num_elements = 10
    ds = dataset_ops.Dataset.range(num_elements)
    self.assertLen(ds, 10)

  @combinations.generate(test_base.eager_only_combinations())
  def testInfinite(self):
    num_elements = 10
    ds = dataset_ops.Dataset.range(num_elements).repeat()
    with self.assertRaisesRegex(TypeError, 'infinite'):
      len(ds)

  @combinations.generate(test_base.eager_only_combinations())
  def testUnknown(self):
    num_elements = 10
    ds = dataset_ops.Dataset.range(num_elements).filter(lambda x: True)
    with self.assertRaisesRegex(TypeError, 'unknown'):
      len(ds)

  @combinations.generate(test_base.graph_only_combinations())
  def testGraphMode(self):
    num_elements = 10
    ds = dataset_ops.Dataset.range(num_elements)
    with self.assertRaisesRegex(TypeError, 'not supported while tracing'):
      len(ds)


if __name__ == '__main__':
  test.main()
