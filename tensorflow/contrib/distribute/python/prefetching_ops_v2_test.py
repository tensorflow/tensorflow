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
"""Tests for prefetching_ops_v2."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.distribute.python import prefetching_ops_v2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


class PrefetchingOpsV2Test(test.TestCase):

  def testPrefetchToOneDevice(self):
    if not test_util.is_gpu_available():
      self.skipTest("No GPU available")

    host_dataset = dataset_ops.Dataset.range(10)
    device_dataset = host_dataset.apply(
        prefetching_ops_v2.prefetch_to_devices("/gpu:0"))

    iterator = device_dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    with self.cached_session() as sess:
      for i in range(10):
        self.assertEqual(i, sess.run(next_element))
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)

  def testPrefetchToTwoDevicesInAList(self):
    if not test_util.is_gpu_available():
      self.skipTest("No GPU available")

    host_dataset = dataset_ops.Dataset.range(10)
    device_dataset = host_dataset.apply(
        prefetching_ops_v2.prefetch_to_devices(["/cpu:0", "/gpu:0"]))

    iterator = device_dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    output = []
    # TODO(rohanj): Modify test to go till the end of the dataset when we
    # switch to MultiDeviceIterator.
    with self.cached_session() as sess:
      for _ in range(4):
        result = sess.run(next_element)
        self.assertEqual(2, len(result))
        output.extend(result)
      self.assertEquals(set(range(8)), set(output))

  def testPrefetchToTwoDevicesWithReinit(self):
    if not test_util.is_gpu_available():
      self.skipTest("No GPU available")

    host_dataset = dataset_ops.Dataset.range(10)
    device_dataset = host_dataset.apply(
        prefetching_ops_v2.prefetch_to_devices(["/cpu:0", "/gpu:0"]))

    iterator = device_dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    # TODO(rohanj): Modify test to go till the end of the dataset when we
    # switch to MultiDeviceIterator.
    with self.cached_session() as sess:
      sess.run(iterator.initializer)
      for _ in range(4):
        sess.run(next_element)
      sess.run(iterator.initializer)
      for _ in range(4):
        sess.run(next_element)


if __name__ == "__main__":
  test.main()
