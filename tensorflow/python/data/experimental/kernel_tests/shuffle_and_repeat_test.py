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
"""Tests for `tf.data.experimental.shuffle_and_repeat()`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.data.experimental.ops import shuffle_ops
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class ShuffleAndRepeatTest(test_base.DatasetTestBase):

  def _build_ds(self, seed, count=5, num_elements=20):
    return dataset_ops.Dataset.range(num_elements).apply(
        shuffle_ops.shuffle_and_repeat(buffer_size=5, count=count, seed=seed))

  def _gen_outputs(self, ds_fn, num_outputs, verify_exhausted=True):
    get_next = self.getNext(ds_fn())
    outputs = []
    for _ in range(num_outputs):
      outputs.append(self.evaluate(get_next()))
    if verify_exhausted:
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(get_next())
    return outputs

  def testCorrectOutput(self):
    output = self._gen_outputs(lambda: self._build_ds(10), 100)
    self.assertSequenceEqual(
        sorted(output), sorted(
            np.array([range(20) for _ in range(5)]).flatten()))
    for i in range(5):
      self.assertSequenceEqual(sorted(output[i * 20:(i + 1) * 20]), range(20))

  def testReshuffling(self):
    # Check that the output orders of different epochs are indeed different.
    output = self._gen_outputs(lambda: self._build_ds(10), 100)
    for i in range(4):
      epoch1 = output[i * 20:(i + 1) * 20]
      epoch2 = output[(i + 1) * 20:(i + 2) * 20]
      self.assertNotEqual(epoch1, epoch2)

  def testSameOrderForSameSeeds(self):
    output1 = self._gen_outputs(lambda: self._build_ds(10), 100)
    output2 = self._gen_outputs(lambda: self._build_ds(10), 100)
    self.assertEqual(output1, output2)

  def testDifferentOrderForDifferentSeeds(self):
    output1 = self._gen_outputs(lambda: self._build_ds(10), 100)
    output2 = self._gen_outputs(lambda: self._build_ds(20), 100)
    self.assertNotEqual(output1, output2)
    self.assertEqual(sorted(output1), sorted(output2))

  def testCountNone(self):
    output1 = self._gen_outputs(
        lambda: self._build_ds(10, count=None), 100, verify_exhausted=False)
    output2 = self._gen_outputs(
        lambda: self._build_ds(20, count=None), 100, verify_exhausted=False)
    self.assertNotEqual(output1, output2)
    self.assertEqual(sorted(output1), sorted(output2))

  def testCountMinusOne(self):
    output1 = self._gen_outputs(
        lambda: self._build_ds(10, count=-1), 100, verify_exhausted=False)
    output2 = self._gen_outputs(
        lambda: self._build_ds(20, count=-1), 100, verify_exhausted=False)
    self.assertNotEqual(output1, output2)
    self.assertEqual(sorted(output1), sorted(output2))

  def testInfiniteOutputs(self):
    # Asserting the iterator is exhausted after producing 100 items should fail.
    with self.assertRaises(AssertionError):
      self._gen_outputs(lambda: self._build_ds(10, count=None), 100)
    with self.assertRaises(AssertionError):
      self._gen_outputs(lambda: self._build_ds(10, count=-1), 100)

  def testInfiniteEmpty(self):
    with self.assertRaises(errors.OutOfRangeError):
      self._gen_outputs(lambda: self._build_ds(10, count=None, num_elements=0),
                        100)
    with self.assertRaises(errors.OutOfRangeError):
      self._gen_outputs(lambda: self._build_ds(10, count=-1, num_elements=0),
                        100)

  def testLargeBufferSize(self):
    ds = dataset_ops.Dataset.range(20).apply(
        shuffle_ops.shuffle_and_repeat(buffer_size=21))
    get_next = self.getNext(ds)
    self.evaluate(get_next())

  def testVeryLargeBufferSize(self):
    num_epochs = 1000 * 1000
    # Each element being shuffled and repeated has shape (100,). This will OOM
    # or timeout if we actually load everything into the buffer.
    ds = dataset_ops.Dataset.range(500).batch(100).apply(
        shuffle_ops.shuffle_and_repeat(
            buffer_size=5 * num_epochs, count=num_epochs))
    # Verify two epochs worth of output.
    output = self._gen_outputs(lambda: ds, 2 * 5, verify_exhausted=False)
    for i in range(2):
      sorted_epoch = sorted(
          output[i * 5:(i + 1) * 5], key=lambda batch: batch[0])
      self.assertAllEqual(sorted_epoch, np.arange(500).reshape([5, 100]))


if __name__ == "__main__":
  test.main()
