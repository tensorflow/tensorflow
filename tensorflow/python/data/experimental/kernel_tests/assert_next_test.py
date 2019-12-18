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
"""Tests for `tf.data.experimental.assert_next()`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.experimental.ops import testing
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class AssertNextTest(test_base.DatasetTestBase):

  def testAssertNext(self):
    dataset = dataset_ops.Dataset.from_tensors(0).apply(
        testing.assert_next(["Map"])).map(lambda x: x)
    options = dataset_ops.Options()
    options.experimental_optimization.apply_default_optimizations = False
    dataset = dataset.with_options(options)
    self.assertDatasetProduces(dataset, expected_output=[0])

  def testAssertNextInvalid(self):
    dataset = dataset_ops.Dataset.from_tensors(0).apply(
        testing.assert_next(["Whoops"])).map(lambda x: x)
    options = dataset_ops.Options()
    options.experimental_optimization.apply_default_optimizations = False
    dataset = dataset.with_options(options)
    self.assertDatasetProduces(
        dataset,
        expected_error=(
            errors.InvalidArgumentError,
            "Asserted Whoops transformation at offset 0 but encountered "
            "Map transformation instead."))

  def testAssertNextShort(self):
    dataset = dataset_ops.Dataset.from_tensors(0).apply(
        testing.assert_next(["Map", "Whoops"])).map(lambda x: x)
    options = dataset_ops.Options()
    options.experimental_optimization.apply_default_optimizations = False
    options.experimental_optimization.autotune = False
    dataset = dataset.with_options(options)
    self.assertDatasetProduces(
        dataset,
        expected_error=(
            errors.InvalidArgumentError,
            "Asserted next 2 transformations but encountered only 1."))


if __name__ == "__main__":
  test.main()
