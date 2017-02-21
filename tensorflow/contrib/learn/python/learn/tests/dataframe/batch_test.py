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
"""Tests for learn.dataframe.transforms.batch."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

# TODO: #6568 Remove this hack that makes dlopen() not crash.
if hasattr(sys, "getdlopenflags") and hasattr(sys, "setdlopenflags"):
  import ctypes
  sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)

import numpy as np

from tensorflow.contrib.learn.python.learn.dataframe.transforms import batch
from tensorflow.contrib.learn.python.learn.dataframe.transforms import in_memory_source
from tensorflow.python.platform import test
from tensorflow.python.training import coordinator
from tensorflow.python.training import queue_runner_impl


class BatchTestCase(test.TestCase):
  """Test class for Batch transform."""

  def testBatch(self):
    initial_batch_size = 7
    final_batch_size = 13
    iterations = 50
    numpy_cols = in_memory_source.NumpySource(
        np.arange(1000, 2000), batch_size=initial_batch_size)()
    index_column = numpy_cols.index
    value_column = numpy_cols.value
    batcher = batch.Batch(
        batch_size=final_batch_size, output_names=["index", "value"])
    batched = batcher([index_column, value_column])
    cache = {}
    index_tensor = batched.index.build(cache)
    value_tensor = batched.value.build(cache)
    with self.test_session() as sess:
      coord = coordinator.Coordinator()
      threads = queue_runner_impl.start_queue_runners(sess=sess, coord=coord)
      for i in range(iterations):
        expected_index = range(i * final_batch_size, (i + 1) * final_batch_size)
        expected_value = range(1000 + i * final_batch_size,
                               1000 + (i + 1) * final_batch_size)
        actual_index, actual_value = sess.run([index_tensor, value_tensor])
        np.testing.assert_array_equal(expected_index, actual_index)
        np.testing.assert_array_equal(expected_value, actual_value)
      coord.request_stop()
      coord.join(threads)


if __name__ == "__main__":
  test.main()
