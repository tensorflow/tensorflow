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
"""Checkpoint tests for `tf.data.Iterator`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os

from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
from tensorflow.python.training import checkpoint_management
from tensorflow.python.training.checkpointable import util as checkpointable_utils


@test_util.run_all_in_graph_and_eager_modes
class IteratorCheckpointingTest(test_base.DatasetTestBase):

  def testSaveRestoreOneShotIterator(self):
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    dataset = dataset_ops.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6]).map(
        math_ops.square).batch(2)
    iterator = iter(dataset) if context.executing_eagerly(
    ) else dataset_ops.make_one_shot_iterator(dataset)
    get_next = iterator.get_next if context.executing_eagerly(
    ) else functools.partial(self.evaluate, iterator.get_next())
    checkpoint = checkpointable_utils.Checkpoint(iterator=iterator)
    self.assertAllEqual([1, 4], get_next())
    save_path = checkpoint.save(checkpoint_prefix)
    self.assertAllEqual([9, 16], get_next())
    self.assertAllEqual([25, 36], get_next())
    checkpoint.restore(save_path).run_restore_ops()
    self.assertAllEqual([9, 16], get_next())
    self.assertAllEqual([25, 36], get_next())
    with self.assertRaises(errors.OutOfRangeError):
      get_next()

  def testSaveRestoreMultipleIterator(self):
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    dataset = dataset_ops.Dataset.from_tensor_slices(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    dataset = dataset.map(math_ops.square).batch(2)
    iterator_1 = iter(dataset) if context.executing_eagerly(
    ) else dataset_ops.make_one_shot_iterator(dataset)
    get_next_1 = iterator_1.get_next if context.executing_eagerly(
    ) else functools.partial(self.evaluate, iterator_1.get_next())
    iterator_2 = iter(dataset) if context.executing_eagerly(
    ) else dataset_ops.make_one_shot_iterator(dataset)
    get_next_2 = iterator_2.get_next if context.executing_eagerly(
    ) else functools.partial(self.evaluate, iterator_2.get_next())
    dataset_2 = dataset_ops.Dataset.range(10)
    iterator_3 = iter(dataset_2) if context.executing_eagerly(
    ) else dataset_ops.make_one_shot_iterator(dataset_2)
    get_next_3 = iterator_3.get_next if context.executing_eagerly(
    ) else functools.partial(self.evaluate, iterator_3.get_next())
    checkpoint = checkpointable_utils.Checkpoint(
        iterator_1=iterator_1, iterator_2=iterator_2, iterator_3=iterator_3)
    self.assertAllEqual([1, 4], get_next_1())
    self.assertAllEqual(0, get_next_3())
    self.assertAllEqual(1, get_next_3())
    self.assertAllEqual(2, get_next_3())
    save_path = checkpoint.save(checkpoint_prefix)
    self.assertAllEqual([1, 4], get_next_2())
    self.assertAllEqual([9, 16], get_next_2())
    self.assertAllEqual(3, get_next_3())
    checkpoint.restore(save_path).run_restore_ops()
    self.assertAllEqual([9, 16], get_next_1())
    self.assertAllEqual([1, 4], get_next_2())
    self.assertAllEqual(3, get_next_3())

  def testRestoreExhaustedIterator(self):
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    dataset = dataset_ops.Dataset.range(3)
    iterator = iter(dataset) if context.executing_eagerly(
    ) else dataset_ops.make_one_shot_iterator(dataset)
    get_next = iterator.get_next if context.executing_eagerly(
    ) else functools.partial(self.evaluate, iterator.get_next())
    checkpoint = checkpointable_utils.Checkpoint(iterator=iterator)
    self.assertAllEqual(0, get_next())
    self.assertAllEqual(1, get_next())
    save_path = checkpoint.save(checkpoint_prefix)
    self.assertAllEqual(2, get_next())
    checkpoint.restore(save_path).run_restore_ops()
    self.assertAllEqual(2, get_next())
    save_path = checkpoint.save(checkpoint_prefix)
    checkpoint.restore(save_path).run_restore_ops()
    with self.assertRaises(errors.OutOfRangeError):
      get_next()

  def testRestoreInReconstructedIteratorInitializable(self):
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    dataset = dataset_ops.Dataset.range(10)
    iterator = iter(dataset) if context.executing_eagerly(
    ) else dataset.make_initializable_iterator()
    get_next = iterator.get_next
    checkpoint = checkpointable_utils.Checkpoint(iterator=iterator)
    for i in range(5):
      checkpoint.restore(
          checkpoint_management.latest_checkpoint(
              checkpoint_directory)).initialize_or_restore()
      for j in range(2):
        self.assertEqual(i * 2 + j, self.evaluate(get_next()))
      checkpoint.save(file_prefix=checkpoint_prefix)


if __name__ == "__main__":
  test.main()
