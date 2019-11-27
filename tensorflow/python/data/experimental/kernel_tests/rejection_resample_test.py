# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for `tf.data.experimental.rejection_resample()`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.experimental.ops import resampling
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test
from tensorflow.python.util import compat


class RejectionResampleTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(initial_known=[True, False])))
  def testDistribution(self, initial_known):
    classes = np.random.randint(5, size=(20000,))  # Uniformly sampled
    target_dist = [0.9, 0.05, 0.05, 0.0, 0.0]
    initial_dist = [0.2] * 5 if initial_known else None
    classes = math_ops.cast(classes, dtypes.int64)  # needed for Windows build.
    dataset = dataset_ops.Dataset.from_tensor_slices(classes).shuffle(
        200, seed=21, reshuffle_each_iteration=False).map(
            lambda c: (c, string_ops.as_string(c))).repeat()

    get_next = self.getNext(
        dataset.apply(
            resampling.rejection_resample(
                target_dist=target_dist,
                initial_dist=initial_dist,
                class_func=lambda c, _: c,
                seed=27)))

    returned = []
    while len(returned) < 4000:
      returned.append(self.evaluate(get_next()))

    returned_classes, returned_classes_and_data = zip(*returned)
    _, returned_data = zip(*returned_classes_and_data)
    self.assertAllEqual([compat.as_bytes(str(c))
                         for c in returned_classes], returned_data)
    total_returned = len(returned_classes)
    class_counts = np.array([
        len([True for v in returned_classes if v == c])
        for c in range(5)])
    returned_dist = class_counts / total_returned
    self.assertAllClose(target_dist, returned_dist, atol=1e-2)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(only_initial_dist=[True, False])))
  def testEdgeCasesSampleFromInitialDataset(self, only_initial_dist):
    init_dist = [0.5, 0.5]
    target_dist = [0.5, 0.5] if only_initial_dist else [0.0, 1.0]
    num_classes = len(init_dist)
    # We don't need many samples to test that this works.
    num_samples = 100
    data_np = np.random.choice(num_classes, num_samples, p=init_dist)

    dataset = dataset_ops.Dataset.from_tensor_slices(data_np)

    # Reshape distribution.
    dataset = dataset.apply(
        resampling.rejection_resample(
            class_func=lambda x: x,
            target_dist=target_dist,
            initial_dist=init_dist))

    get_next = self.getNext(dataset)

    returned = []
    with self.assertRaises(errors.OutOfRangeError):
      while True:
        returned.append(self.evaluate(get_next()))

  @combinations.generate(test_base.default_test_combinations())
  def testRandomClasses(self):
    init_dist = [0.25, 0.25, 0.25, 0.25]
    target_dist = [0.0, 0.0, 0.0, 1.0]
    num_classes = len(init_dist)
    # We don't need many samples to test a dirac-delta target distribution.
    num_samples = 100
    data_np = np.random.choice(num_classes, num_samples, p=init_dist)

    dataset = dataset_ops.Dataset.from_tensor_slices(data_np)

    # Apply a random mapping that preserves the data distribution.
    def _remap_fn(_):
      return math_ops.cast(random_ops.random_uniform([1]) * num_classes,
                           dtypes.int32)[0]
    dataset = dataset.map(_remap_fn)

    # Reshape distribution.
    dataset = dataset.apply(
        resampling.rejection_resample(
            class_func=lambda x: x,
            target_dist=target_dist,
            initial_dist=init_dist))

    get_next = self.getNext(dataset)

    returned = []
    with self.assertRaises(errors.OutOfRangeError):
      while True:
        returned.append(self.evaluate(get_next()))

    classes, _ = zip(*returned)
    bincount = np.bincount(
        np.array(classes),
        minlength=num_classes).astype(np.float32) / len(classes)

    self.assertAllClose(target_dist, bincount, atol=1e-2)


if __name__ == "__main__":
  test.main()
