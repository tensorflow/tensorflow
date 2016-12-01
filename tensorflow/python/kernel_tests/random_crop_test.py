# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for random_crop."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class RandomCropTest(tf.test.TestCase):

  def testNoOp(self):
    # No random cropping is performed since the size is value.shape.
    for shape in (2, 1, 1), (2, 1, 3), (4, 5, 3):
      value = np.arange(0, np.prod(shape), dtype=np.int32).reshape(shape)
      with self.test_session():
        crop = tf.random_crop(value, shape).eval()
        self.assertAllEqual(crop, value)

  def testContains(self):
    with self.test_session():
      shape = (3, 5, 7)
      target = (2, 3, 4)
      value = np.random.randint(1000000, size=shape)
      value_set = set(tuple(value[i:i + 2, j:j + 3, k:k + 4].ravel())
                      for i in range(2) for j in range(3) for k in range(4))
      crop = tf.random_crop(value, size=target)
      for _ in range(20):
        y = crop.eval()
        self.assertAllEqual(y.shape, target)
        self.assertTrue(tuple(y.ravel()) in value_set)

  def testRandomization(self):
    # Run 1x1 crop num_samples times in an image and ensure that one finds each
    # pixel 1/size of the time.
    num_samples = 1000
    shape = [5, 4, 1]
    size = np.prod(shape)
    single = [1, 1, 1]
    value = np.arange(size).reshape(shape)

    with self.test_session():
      crop = tf.random_crop(value, single, seed=7)
      counts = np.zeros(size, dtype=np.int32)
      for _ in range(num_samples):
        y = crop.eval()
        self.assertAllEqual(y.shape, single)
        counts[y] += 1

    # Calculate the mean and 4 * standard deviation.
    mean = np.repeat(num_samples / size, size)
    four_stddev = 4.0 * np.sqrt(mean)

    # Ensure that each entry is observed in 1/size of the samples
    # within 4 standard deviations.
    self.assertAllClose(counts, mean, atol=four_stddev)


if __name__ == '__main__':
  tf.test.main()
