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
"""Tests for bincount_ops.bincount."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import bincount_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import googletest


class BincountTest(test_util.TensorFlowTestCase):

  def test_empty(self):
    with self.session():
      self.assertAllEqual(
          self.evaluate(bincount_ops.bincount([], minlength=5)),
          [0, 0, 0, 0, 0])
      self.assertAllEqual(
          self.evaluate(bincount_ops.bincount([], minlength=1)), [0])
      self.assertAllEqual(
          self.evaluate(bincount_ops.bincount([], minlength=0)), [])
      self.assertEqual(
          self.evaluate(
              bincount_ops.bincount([], minlength=0, dtype=np.float32)).dtype,
          np.float32)
      self.assertEqual(
          self.evaluate(
              bincount_ops.bincount([], minlength=3, dtype=np.float64)).dtype,
          np.float64)

  def test_values(self):
    with self.session():
      self.assertAllEqual(
          self.evaluate(bincount_ops.bincount([1, 1, 1, 2, 2, 3])),
          [0, 3, 2, 1])
      arr = [1, 1, 2, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 4, 5]
      self.assertAllEqual(
          self.evaluate(bincount_ops.bincount(arr)), [0, 5, 4, 3, 2, 1])
      arr += [0, 0, 0, 0, 0, 0]
      self.assertAllEqual(
          self.evaluate(bincount_ops.bincount(arr)), [6, 5, 4, 3, 2, 1])

      self.assertAllEqual(self.evaluate(bincount_ops.bincount([])), [])
      self.assertAllEqual(self.evaluate(bincount_ops.bincount([0, 0, 0])), [3])
      self.assertAllEqual(
          self.evaluate(bincount_ops.bincount([5])), [0, 0, 0, 0, 0, 1])
      self.assertAllEqual(
          self.evaluate(bincount_ops.bincount(np.arange(10000))),
          np.ones(10000))

  def test_maxlength(self):
    with self.session():
      self.assertAllEqual(
          self.evaluate(bincount_ops.bincount([5], maxlength=3)), [0, 0, 0])
      self.assertAllEqual(
          self.evaluate(bincount_ops.bincount([1], maxlength=3)), [0, 1])
      self.assertAllEqual(
          self.evaluate(bincount_ops.bincount([], maxlength=3)), [])

  def test_random_with_weights(self):
    num_samples = 10000
    with self.session():
      np.random.seed(42)
      for dtype in [dtypes.int32, dtypes.int64, dtypes.float32, dtypes.float64]:
        arr = np.random.randint(0, 1000, num_samples)
        if dtype == dtypes.int32 or dtype == dtypes.int64:
          weights = np.random.randint(-100, 100, num_samples)
        else:
          weights = np.random.random(num_samples)
        self.assertAllClose(
            self.evaluate(bincount_ops.bincount(arr, weights)),
            np.bincount(arr, weights))

  def test_random_without_weights(self):
    num_samples = 10000
    with self.session():
      np.random.seed(42)
      for dtype in [np.int32, np.float32]:
        arr = np.random.randint(0, 1000, num_samples)
        weights = np.ones(num_samples).astype(dtype)
        self.assertAllClose(
            self.evaluate(bincount_ops.bincount(arr, None)),
            np.bincount(arr, weights))

  def test_zero_weights(self):
    with self.session():
      self.assertAllEqual(
          self.evaluate(bincount_ops.bincount(np.arange(1000), np.zeros(1000))),
          np.zeros(1000))

  def test_negative(self):
    # unsorted_segment_sum will only report InvalidArgumentError on CPU
    with self.cached_session(), ops.device("/CPU:0"):
      with self.assertRaises(errors.InvalidArgumentError):
        self.evaluate(bincount_ops.bincount([1, 2, 3, -1, 6, 8]))

  def test_shape_function(self):
    # size must be scalar.
    with self.assertRaisesRegex(
        (ValueError, errors.InvalidArgumentError),
        "Shape must be rank 0 but is rank 1 .*Bincount"):
      gen_math_ops.bincount([1, 2, 3, 1, 6, 8], [1], [])
    # size must be positive.
    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "must be non-negative"):
      gen_math_ops.bincount([1, 2, 3, 1, 6, 8], -5, [])
    # if size is a constant then the shape is known.
    v1 = gen_math_ops.bincount([1, 2, 3, 1, 6, 8], 5, [])
    self.assertAllEqual(v1.get_shape().as_list(), [5])
    # if size is a placeholder then the shape is unknown.
    with ops.Graph().as_default():
      s = array_ops.placeholder(dtype=dtypes.int32)
      v2 = gen_math_ops.bincount([1, 2, 3, 1, 6, 8], s, [])
      self.assertAllEqual(v2.get_shape().as_list(), [None])


class BincountOpTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  @parameterized.parameters([{
      "dtype": np.int32,
  }, {
      "dtype": np.int64,
  }])
  def test_bincount_all_count(self, dtype):
    np.random.seed(42)
    size = 1000
    inp = np.random.randint(0, size, (4096), dtype=dtype)
    np_out = np.bincount(inp, minlength=size)
    with test_util.use_gpu():
      self.assertAllEqual(
          np_out,
          self.evaluate(
              gen_math_ops.dense_bincount(input=inp, weights=[], size=size)))

  @parameterized.parameters([{
      "dtype": np.int32,
  }, {
      "dtype": np.int64,
  }])
  def test_bincount_all_count_with_weights(self, dtype):
    np.random.seed(42)
    size = 1000
    inp = np.random.randint(0, size, (4096,), dtype=dtype)
    np_weight = np.random.random((4096,))
    np_out = np.bincount(inp, minlength=size, weights=np_weight)
    with test_util.use_gpu():
      self.assertAllEqual(
          np_out,
          self.evaluate(
              gen_math_ops.dense_bincount(
                  input=inp, weights=np_weight, size=size)))

  @parameterized.parameters([{
      "dtype": np.int32,
  }, {
      "dtype": np.int64,
  }])
  def test_bincount_all_binary(self, dtype):
    np.random.seed(42)
    size = 10
    inp = np.random.randint(0, size, (4096), dtype=dtype)
    np_out = np.ones((size,))
    with test_util.use_gpu():
      self.assertAllEqual(
          np_out,
          self.evaluate(
              gen_math_ops.dense_bincount(
                  input=inp, weights=[], size=size, binary_output=True)))

  @parameterized.parameters([{
      "dtype": np.int32,
  }, {
      "dtype": np.int64,
  }])
  def test_bincount_all_binary_with_weights(self, dtype):
    np.random.seed(42)
    size = 10
    inp = np.random.randint(0, size, (4096,), dtype=dtype)
    np_weight = np.random.random((4096,))
    np_out = np.ones((size,))
    with test_util.use_gpu():
      self.assertAllEqual(
          np_out,
          self.evaluate(
              gen_math_ops.dense_bincount(
                  input=inp, weights=np_weight, size=size, binary_output=True)))

  def _test_bincount_col_count(self, num_rows, num_cols, size, dtype):
    np.random.seed(42)
    inp = np.random.randint(0, size, (num_rows, num_cols), dtype=dtype)
    np_out = np.reshape(
        np.concatenate(
            [np.bincount(inp[j, :], minlength=size) for j in range(num_rows)],
            axis=0), (num_rows, size))
    with test_util.use_gpu():
      self.assertAllEqual(
          np_out,
          self.evaluate(
              gen_math_ops.dense_bincount(input=inp, weights=[], size=size)))

  def _test_bincount_col_binary(self, num_rows, num_cols, size, dtype):
    np.random.seed(42)
    inp = np.random.randint(0, size, (num_rows, num_cols), dtype=dtype)
    np_out = np.reshape(
        np.concatenate([
            np.where(np.bincount(inp[j, :], minlength=size) > 0, 1, 0)
            for j in range(num_rows)
        ],
                       axis=0), (num_rows, size))
    with test_util.use_gpu():
      self.assertAllEqual(
          np_out,
          self.evaluate(
              gen_math_ops.dense_bincount(
                  input=inp, weights=[], size=size, binary_output=True)))

  def _test_bincount_col_count_with_weights(self, num_rows, num_cols, size,
                                            dtype):
    np.random.seed(42)
    inp = np.random.randint(0, size, (num_rows, num_cols), dtype=dtype)
    np_weight = np.random.random((num_rows, num_cols))
    np_out = np.reshape(
        np.concatenate([
            np.bincount(inp[j, :], weights=np_weight[j, :], minlength=size)
            for j in range(num_rows)
        ],
                       axis=0), (num_rows, size))
    with test_util.use_gpu():
      self.assertAllEqual(
          np_out,
          self.evaluate(
              gen_math_ops.dense_bincount(
                  input=inp, weights=np_weight, size=size)))

  def test_col_reduce_basic(self):
    with test_util.use_gpu():
      v = self.evaluate(
          gen_math_ops.dense_bincount(
              input=[[1, 2, 3], [0, 3, 2]], weights=[], size=4))
    expected_out = [[0., 1., 1., 1.], [1., 0., 1., 1.]]
    self.assertAllEqual(expected_out, v)

  @parameterized.parameters([{
      "dtype": np.int32,
  }, {
      "dtype": np.int64,
  }])
  def test_col_reduce_shared_memory(self, dtype):
    # num_rows * num_bins less than half of max shared memory.
    num_rows = 128
    num_cols = 27
    size = 10
    self._test_bincount_col_count(num_rows, num_cols, size, dtype)

  @parameterized.parameters([{
      "dtype": np.int32,
  }, {
      "dtype": np.int64,
  }])
  def test_col_reduce_global_memory(self, dtype):
    # num_rows * num_bins more than half of max shared memory.
    num_rows = 128
    num_cols = 27
    size = 1024
    self._test_bincount_col_count(num_rows, num_cols, size, dtype)

  @parameterized.parameters([{
      "dtype": np.int32,
  }, {
      "dtype": np.int64,
  }])
  def test_col_reduce_shared_memory_with_weights(self, dtype):
    # num_rows * num_bins less than half of max shared memory.
    num_rows = 128
    num_cols = 27
    size = 100
    self._test_bincount_col_count_with_weights(num_rows, num_cols, size, dtype)

  @parameterized.parameters([{
      "dtype": np.int32,
  }, {
      "dtype": np.int64,
  }])
  def test_col_reduce_global_memory_with_weights(self, dtype):
    # num_rows * num_bins more than half of max shared memory.
    num_rows = 128
    num_cols = 27
    size = 1024
    self._test_bincount_col_count_with_weights(num_rows, num_cols, size, dtype)

  @parameterized.parameters([{
      "dtype": np.int32,
  }, {
      "dtype": np.int64,
  }])
  def test_col_reduce_binary(self, dtype):
    num_rows = 128
    num_cols = 7
    size = 10
    self._test_bincount_col_binary(num_rows, num_cols, size, dtype)

  def test_invalid_rank(self):
    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "at most rank 2"):
      with test_util.use_gpu():
        self.evaluate(
            gen_math_ops.dense_bincount(
                input=[[[1, 2, 3], [0, 3, 2]]], weights=[], size=10))


class SparseBincountOpTest(test_util.TensorFlowTestCase,
                           parameterized.TestCase):

  @parameterized.parameters([{
      "dtype": np.int32,
  }, {
      "dtype": np.int64,
  }])
  def test_sparse_bincount_all_count(self, dtype):
    np.random.seed(42)
    num_rows = 128
    size = 1000
    n_elems = 4096
    inp_indices = np.random.randint(0, num_rows, (n_elems,))
    inp_vals = np.random.randint(0, size, (n_elems,), dtype=dtype)

    np_out = np.bincount(inp_vals, minlength=size)
    self.assertAllEqual(
        np_out,
        self.evaluate(
            gen_math_ops.sparse_bincount(
                indices=inp_indices,
                values=inp_vals,
                dense_shape=[num_rows],
                size=size,
                weights=[])))

  @parameterized.parameters([{
      "dtype": np.int32,
  }, {
      "dtype": np.int64,
  }])
  def test_sparse_bincount_all_count_with_weights(self, dtype):
    np.random.seed(42)
    num_rows = 128
    size = 1000
    n_elems = 4096
    inp_indices = np.random.randint(0, num_rows, (n_elems,))
    inp_vals = np.random.randint(0, size, (n_elems,), dtype=dtype)
    inp_weight = np.random.random((n_elems,))

    np_out = np.bincount(inp_vals, minlength=size, weights=inp_weight)
    self.assertAllEqual(
        np_out,
        self.evaluate(
            gen_math_ops.sparse_bincount(
                indices=inp_indices,
                values=inp_vals,
                dense_shape=[num_rows],
                size=size,
                weights=inp_weight)))

  @parameterized.parameters([{
      "dtype": np.int32,
  }, {
      "dtype": np.int64,
  }])
  def test_sparse_bincount_all_binary(self, dtype):
    np.random.seed(42)
    num_rows = 128
    size = 10
    n_elems = 4096
    inp_indices = np.random.randint(0, num_rows, (n_elems,))
    inp_vals = np.random.randint(0, size, (n_elems,), dtype=dtype)

    np_out = np.ones((size,))
    self.assertAllEqual(
        np_out,
        self.evaluate(
            gen_math_ops.sparse_bincount(
                indices=inp_indices,
                values=inp_vals,
                dense_shape=[num_rows],
                size=size,
                weights=[],
                binary_output=True)))

  @parameterized.parameters([{
      "dtype": np.int32,
  }, {
      "dtype": np.int64,
  }])
  def test_sparse_bincount_all_binary_weights(self, dtype):
    np.random.seed(42)
    num_rows = 128
    size = 10
    n_elems = 4096
    inp_indices = np.random.randint(0, num_rows, (n_elems,))
    inp_vals = np.random.randint(0, size, (n_elems,), dtype=dtype)
    inp_weight = np.random.random((n_elems,))

    np_out = np.ones((size,))
    self.assertAllEqual(
        np_out,
        self.evaluate(
            gen_math_ops.sparse_bincount(
                indices=inp_indices,
                values=inp_vals,
                dense_shape=[num_rows],
                size=size,
                weights=inp_weight,
                binary_output=True)))

  @parameterized.parameters([{
      "dtype": np.int32,
  }, {
      "dtype": np.int64,
  }])
  def test_sparse_bincount_col_reduce_count(self, dtype):
    num_rows = 128
    num_cols = 27
    size = 100
    np.random.seed(42)
    inp = np.random.randint(0, size, (num_rows, num_cols), dtype=dtype)
    np_out = np.reshape(
        np.concatenate(
            [np.bincount(inp[j, :], minlength=size) for j in range(num_rows)],
            axis=0), (num_rows, size))
    # from_dense will filter out 0s.
    inp = inp + 1
    # from_dense will cause OOM in GPU.
    with ops.device("/CPU:0"):
      inp_sparse = sparse_ops.from_dense(inp)
    self.assertAllEqual(
        np_out,
        self.evaluate(
            gen_math_ops.sparse_bincount(
                indices=inp_sparse.indices,
                values=inp_sparse.values - 1,
                dense_shape=inp_sparse.dense_shape,
                size=size,
                weights=[])))

  @parameterized.parameters([{
      "dtype": np.int32,
  }, {
      "dtype": np.int64,
  }])
  def test_sparse_bincount_col_reduce_binary(self, dtype):
    num_rows = 128
    num_cols = 27
    size = 100
    np.random.seed(42)
    inp = np.random.randint(0, size, (num_rows, num_cols), dtype=dtype)
    np_out = np.reshape(
        np.concatenate([
            np.where(np.bincount(inp[j, :], minlength=size) > 0, 1, 0)
            for j in range(num_rows)
        ],
                       axis=0), (num_rows, size))
    # from_dense will filter out 0s.
    inp = inp + 1
    # from_dense will cause OOM in GPU.
    with ops.device("/CPU:0"):
      inp_sparse = sparse_ops.from_dense(inp)
    self.assertAllEqual(
        np_out,
        self.evaluate(
            gen_math_ops.sparse_bincount(
                indices=inp_sparse.indices,
                values=inp_sparse.values - 1,
                dense_shape=inp_sparse.dense_shape,
                size=size,
                weights=[],
                binary_output=True)))


class RaggedBincountOpTest(test_util.TensorFlowTestCase,
                           parameterized.TestCase):

  @parameterized.parameters([{
      "dtype": np.int32,
  }, {
      "dtype": np.int64,
  }])
  def test_ragged_bincount_count(self, dtype):
    x = ragged_factory_ops.constant([[], [], [3, 0, 1], [], [5, 0, 4, 4]])
    expected_output = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0,
                                            0], [1, 1, 0, 1, 0, 0],
                       [0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 2, 1]]
    self.assertAllEqual(
        expected_output,
        self.evaluate(
            gen_math_ops.ragged_bincount(
                splits=x.row_splits, values=x.values, weights=[], size=6)))

  @parameterized.parameters([{
      "dtype": np.int32,
  }, {
      "dtype": np.int64,
  }])
  def test_ragged_bincount_binary(self, dtype):
    x = ragged_factory_ops.constant([[], [], [3, 0, 1], [], [5, 0, 4, 4]])
    expected_output = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0,
                                            0], [1, 1, 0, 1, 0, 0],
                       [0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 1, 1]]
    self.assertAllEqual(
        expected_output,
        self.evaluate(
            gen_math_ops.ragged_bincount(
                splits=x.row_splits,
                values=x.values,
                weights=[],
                size=6,
                binary_output=True)))

  @parameterized.parameters([{
      "dtype": np.int32,
  }, {
      "dtype": np.int64,
  }])
  def test_ragged_bincount_count_with_weights(self, dtype):
    x = ragged_factory_ops.constant([[], [], [3, 0, 1], [], [5, 0, 4, 4]])
    weights = ragged_factory_ops.constant([[], [], [.1, .2, .3], [],
                                           [.2, .5, .6, .3]])
    expected_output = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                       [.2, .3, 0, .1, 0, 0], [0, 0, 0, 0, 0, 0],
                       [.5, 0, 0, 0, .9, .2]]
    self.assertAllClose(
        expected_output,
        self.evaluate(
            gen_math_ops.ragged_bincount(
                splits=x.row_splits,
                values=x.values,
                weights=weights.values,
                size=6)))

  @parameterized.parameters([{
      "dtype": np.int32,
  }, {
      "dtype": np.int64,
  }])
  def test_ragged_bincount_count_np(self, dtype):
    np.random.seed(42)
    num_rows = 128
    num_cols = 27
    size = 1000
    inp = np.random.randint(0, size, (num_rows, num_cols), dtype=dtype)
    np_out = np.reshape(
        np.concatenate(
            [np.bincount(inp[j, :], minlength=size) for j in range(num_rows)],
            axis=0), (num_rows, size))
    x = ragged_tensor.RaggedTensor.from_tensor(inp)
    self.assertAllEqual(
        np_out,
        self.evaluate(
            gen_math_ops.ragged_bincount(
                splits=x.row_splits, values=x.values, weights=[], size=size)))

  @parameterized.parameters([{
      "dtype": np.int32,
  }, {
      "dtype": np.int64,
  }])
  def test_ragged_bincount_count_np_with_weights(self, dtype):
    np.random.seed(42)
    num_rows = 128
    num_cols = 27
    size = 1000
    inp = np.random.randint(0, size, (num_rows, num_cols), dtype=dtype)
    np_weight = np.random.random((num_rows, num_cols))
    np_out = np.reshape(
        np.concatenate([
            np.bincount(inp[j, :], weights=np_weight[j, :], minlength=size)
            for j in range(num_rows)
        ],
                       axis=0), (num_rows, size))
    x = ragged_tensor.RaggedTensor.from_tensor(inp)
    self.assertAllEqual(
        np_out,
        self.evaluate(
            gen_math_ops.ragged_bincount(
                splits=x.row_splits,
                values=x.values,
                weights=np_weight,
                size=size)))

  @parameterized.parameters([{
      "dtype": np.int32,
  }, {
      "dtype": np.int64,
  }])
  def test_ragged_bincount_binary_np_with_weights(self, dtype):
    np.random.seed(42)
    num_rows = 128
    num_cols = 27
    size = 1000
    inp = np.random.randint(0, size, (num_rows, num_cols), dtype=dtype)
    np_out = np.reshape(
        np.concatenate([
            np.where(np.bincount(inp[j, :], minlength=size) > 0, 1, 0)
            for j in range(num_rows)
        ],
                       axis=0), (num_rows, size))
    x = ragged_tensor.RaggedTensor.from_tensor(inp)
    self.assertAllEqual(
        np_out,
        self.evaluate(
            gen_math_ops.ragged_bincount(
                splits=x.row_splits,
                values=x.values,
                weights=[],
                size=size,
                binary_output=True)))


if __name__ == "__main__":
  googletest.main()
