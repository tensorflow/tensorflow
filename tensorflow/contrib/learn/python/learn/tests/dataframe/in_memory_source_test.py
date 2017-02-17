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
"""Tests NumpySource and PandasSource."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

# TODO: #6568 Remove this hack that makes dlopen() not crash.
if hasattr(sys, "getdlopenflags") and hasattr(sys, "setdlopenflags"):
  import ctypes
  sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)

import numpy as np

from tensorflow.contrib.learn.python.learn.dataframe.transforms import in_memory_source
from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.python.platform import test
from tensorflow.python.training import coordinator
from tensorflow.python.training import queue_runner_impl

# pylint: disable=g-import-not-at-top
try:
  import pandas as pd
  HAS_PANDAS = True
except ImportError:
  HAS_PANDAS = False


def get_rows(array, row_indices):
  rows = [array[i] for i in row_indices]
  return np.vstack(rows)


class NumpySourceTestCase(test.TestCase):

  def testNumpySource(self):
    batch_size = 3
    iterations = 1000
    array = np.arange(32).reshape([16, 2])
    numpy_source = in_memory_source.NumpySource(array, batch_size=batch_size)
    index_column = numpy_source().index
    value_column = numpy_source().value
    cache = {}
    with ops.Graph().as_default():
      value_tensor = value_column.build(cache)
      index_tensor = index_column.build(cache)
      with session.Session() as sess:
        coord = coordinator.Coordinator()
        threads = queue_runner_impl.start_queue_runners(sess=sess, coord=coord)
        for i in range(iterations):
          expected_index = [
              j % array.shape[0]
              for j in range(batch_size * i, batch_size * (i + 1))
          ]
          expected_value = get_rows(array, expected_index)
          actual_index, actual_value = sess.run([index_tensor, value_tensor])
          np.testing.assert_array_equal(expected_index, actual_index)
          np.testing.assert_array_equal(expected_value, actual_value)
        coord.request_stop()
        coord.join(threads)


class PandasSourceTestCase(test.TestCase):

  def testPandasFeeding(self):
    if not HAS_PANDAS:
      return
    batch_size = 3
    iterations = 1000
    index = np.arange(100, 132)
    a = np.arange(32)
    b = np.arange(32, 64)
    dataframe = pd.DataFrame({"a": a, "b": b}, index=index)
    pandas_source = in_memory_source.PandasSource(
        dataframe, batch_size=batch_size)
    pandas_columns = pandas_source()
    cache = {}
    with ops.Graph().as_default():
      pandas_tensors = [col.build(cache) for col in pandas_columns]
      with session.Session() as sess:
        coord = coordinator.Coordinator()
        threads = queue_runner_impl.start_queue_runners(sess=sess, coord=coord)
        for i in range(iterations):
          indices = [
              j % dataframe.shape[0]
              for j in range(batch_size * i, batch_size * (i + 1))
          ]
          expected_df_indices = dataframe.index[indices]
          expected_rows = dataframe.iloc[indices]
          actual_value = sess.run(pandas_tensors)
          np.testing.assert_array_equal(expected_df_indices, actual_value[0])
          for col_num, col in enumerate(dataframe.columns):
            np.testing.assert_array_equal(expected_rows[col].values,
                                          actual_value[col_num + 1])
        coord.request_stop()
        coord.join(threads)


if __name__ == "__main__":
  test.main()
