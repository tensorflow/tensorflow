# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for the private `_ModelDataset` transformation."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.data.experimental.ops import optimization
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import errors
from tensorflow.python.platform import test


# TODO(b/117581999): Add eager coverage for the following tests.
class ModelDatasetTest(test_base.DatasetTestBase, parameterized.TestCase):

  def testAutotuneOption(self):
    dataset = dataset_ops.Dataset.from_tensors(0)
    dataset = dataset.map(lambda x: x).apply(
        optimization.assert_next(["Model"]))
    options = dataset_ops.Options()
    options.experimental_autotune = True
    options.experimental_optimization.apply_default_optimizations = False
    dataset = dataset.with_options(options)

    iterator = dataset_ops.make_one_shot_iterator(dataset)
    get_next = iterator.get_next()

    with self.cached_session() as sess:
      self.assertEqual(0, self.evaluate(get_next))
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(get_next)


if __name__ == "__main__":
  test.main()
