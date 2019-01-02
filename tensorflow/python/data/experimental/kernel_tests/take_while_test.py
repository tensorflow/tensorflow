# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for `tf.data.experimental.Dataset.take_while()`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.experimental.ops import take_while_ops
from tensorflow.python.framework import errors
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class TakeWhileTest(test_base.DatasetTestBase):

  def testTakeWhileDataset(self):
    def do_test(num_elements, window_size):
      def predicate_func(elem):
        return array_ops.shape(elem)[0] > (window_size - 1)

      flatten_func = lambda x: dataset_ops.Dataset.from_tensor_slices(x)
      take_while = take_while_ops.take_while(predicate_func)

      dataset = dataset_ops.Dataset.range(num_elements).batch(window_size)
      dataset = dataset.apply(take_while).flat_map(flatten_func)
      
      self.assertDatasetProduces(dataset, 
              np.arange(int(num_elements / window_size) * window_size))

    do_test(14, 2)
    do_test(15, 2)
    do_test(100, 3)

  def testTakeWhileDatasetRange(self):
    def get_dataset(num_elemets, upper_bound):
      return dataset_ops.Dataset.range(num_elemets).apply(
              take_while_ops.take_while(lambda x: x < upper_bound))
        
    def do_test(num_elemets, upper_bound):  
      self.assertDatasetProduces(get_dataset(num_elemets, upper_bound), 
              np.arange(upper_bound))
    
    def out_of_bounds(num_elemets, upper_bound):
      with self.assertRaises(errors.OutOfRangeError):
        self.assertDatasetProduces(get_dataset(num_elemets, upper_bound), 
                np.arange(upper_bound))

    do_test(10, 2)
    do_test(16, 7)
    do_test(100, 99)
    out_of_bounds(100, 101)
    out_of_bounds(0, 1)

  def testTakeWhileDatasetString(self):
    def stringNotEquals(string):
        return lambda x: math_ops.not_equal(x, constant_op.constant(string))

    string = ["this", "is", "the", "test", "for", "strings"]
    dataset = dataset_ops.Dataset.from_tensor_slices(string).apply(
            take_while_ops.take_while(stringNotEquals("test")))
    
    next_element = self.getNext(dataset)
    self.assertEqual(b"this", self.evaluate(next_element()))
    self.assertEqual(b"is", self.evaluate(next_element()))
    self.assertEqual(b"the", self.evaluate(next_element()))

    with self.assertRaises(errors.OutOfRangeError):
        self.assertEqual(b"test", self.evaluate(next_element()))


if __name__ == "__main__":
  test.main()
