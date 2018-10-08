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
"""Tests for `tf.data.experimental.RandomDataset()`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.data.experimental.ops import random_ops
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import errors


class RandomDatasetTest(test_base.DatasetTestBase, parameterized.TestCase):

  @parameterized.named_parameters(
      ("NoSeed", None),
      ("WithSeed", 42),
  )
  def testZipRandomDataset(self, seed):
    dataset = random_ops.RandomDataset(seed=seed).take(30)
    dataset = dataset_ops.Dataset.zip((dataset, dataset))
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    with self.cached_session() as sess:
      for _ in range(30):
        x, y = sess.run(next_element)
        self.assertEqual(x, y)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)
