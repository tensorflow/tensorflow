# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for learn.dataframe.transforms.sparsify and densify."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.dataframe.transforms import densify
from tensorflow.contrib.learn.python.learn.dataframe.transforms import in_memory_source
from tensorflow.contrib.learn.python.learn.dataframe.transforms import sparsify


def _test_sparsify_densify(self, x, default_value):
  """Test roundtrip via Sparsify and Densify."""

  numpy_source = in_memory_source.NumpySource(x, batch_size=len(x))()

  (sparse_series,) = sparsify.Sparsify(default_value)(numpy_source[1])
  (dense_series,) = densify.Densify(default_value)(sparse_series)

  cache = {}
  sparse_tensor = sparse_series.build(cache)
  dense_tensor = dense_series.build(cache)

  with self.test_session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    sparse_val, dense_val = sess.run([sparse_tensor, dense_tensor])

    coord.request_stop()
    coord.join(threads)

  if x.dtype.kind not in ["S", "U"] and np.isnan(default_value):
    x_values = x[~np.isnan(x)]
    x_indexes = np.arange(len(x))[~np.isnan(x)].T.reshape(-1, 1)
  else:
    x_values = x[x != default_value]
    x_indexes = np.arange(len(x))[x != default_value].T.reshape(-1, 1)

  if x.dtype.kind in ["S", "U"]:
    # Python 2/3 compatibility
    # TensorFlow always returns bytes, so we just convert the unicode
    # expectations to bytes also before comparing.
    expected_x = [item.encode("utf-8") for item in x]
    expected_x_values = [item.encode("utf-8") for item in x_values]
  else:
    expected_x = x
    expected_x_values = x_values

  np.testing.assert_array_equal(len(x), sparse_val.shape[0])
  np.testing.assert_array_equal(expected_x_values, sparse_val.values)
  np.testing.assert_array_equal(x_indexes, sparse_val.indices)
  np.testing.assert_array_equal(expected_x, dense_val)


class SparsifyDensifyTestCase(tf.test.TestCase):
  """Test class for Sparsify and Densify transforms."""

  def testSparsifyDensifyIntNan(self):
    x = np.array([0, np.nan, 2, 4, np.nan])
    default_value = np.nan
    _test_sparsify_densify(self, x, default_value)

  def testSparsifyDensifyIntZero(self):
    x = np.array([0, 0, 2, 4, 0])
    default_value = 0
    _test_sparsify_densify(self, x, default_value)

  def testSparsifyDensifyFloatNan(self):
    x = np.array([0.0, np.nan, 2.1, 4.1, np.nan])
    default_value = np.nan
    _test_sparsify_densify(self, x, default_value)

  def testSparsifyDensifyFloatZero(self):
    x = np.array([0.0, 0.0, 2, 4, 0.0])
    default_value = 0.0
    _test_sparsify_densify(self, x, default_value)

  def testSparsifyDensifyStringEmpty(self):
    x = np.array(["zero", "", "two", "four", ""])
    default_value = ""
    _test_sparsify_densify(self, x, default_value)


if __name__ == "__main__":
  tf.test.main()
