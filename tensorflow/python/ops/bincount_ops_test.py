# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# maxlengthations under the License.
# ==============================================================================
"""Tests for bincount ops."""

from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import config as tf_config
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import bincount_ops
from tensorflow.python.ops import gen_count_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.platform import test


def _adjust_expected_rank1(x, minlength, maxlength):
  """Trim or pad an expected result based on minlength and maxlength."""
  n = len(x)
  if (minlength is not None) and (n < minlength):
    x = x + [0] * (minlength - n)
  if (maxlength is not None) and (n > maxlength):
    x = x[:maxlength]
  return x


def _adjust_expected_rank2(x, minlength, maxlength):
  return [_adjust_expected_rank1(i, minlength, maxlength) for i in x]


class TestDenseBincount(test.TestCase, parameterized.TestCase):

  @parameterized.parameters([{
      "dtype": np.int32,
  }, {
      "dtype": np.int64,
  }])
  def test_sparse_input_all_count(self, dtype):
    np.random.seed(42)
    num_rows = 4096
    size = 1000
    n_elems = 128
    inp_indices = np.random.randint(0, num_rows, (n_elems, 1))
    inp_indices = np.concatenate([inp_indices, np.zeros((n_elems, 1))], axis=1)
    inp_vals = np.random.randint(0, size, (n_elems,), dtype=dtype)
    sparse_inp = sparse_tensor.SparseTensor(inp_indices, inp_vals,
                                            [num_rows, 1])

    # Note that the default for sparse tensors is to not count implicit zeros.
    np_out = np.bincount(inp_vals, minlength=size)
    self.assertAllEqual(
        np_out,
        self.evaluate(
            bincount_ops.bincount(sparse_inp, axis=0, minlength=size)
        ),
    )

  @parameterized.parameters([{
      "dtype": np.int32,
  }, {
      "dtype": np.int64,
  }])
  def test_sparse_input_all_count_with_weights(self, dtype):
    np.random.seed(42)
    num_rows = 4096
    size = 1000
    n_elems = 128
    inp_indices = np.random.randint(0, num_rows, (n_elems, 1))
    inp_indices = np.concatenate([inp_indices, np.zeros((n_elems, 1))], axis=1)
    inp_vals = np.random.randint(0, size, (n_elems-1,), dtype=dtype)
    # Add an element with value `size-1` to input so bincount output has `size`
    # elements.
    inp_vals = np.concatenate([inp_vals, [size-1]], axis=0)
    sparse_inp = sparse_tensor.SparseTensor(inp_indices, inp_vals,
                                            [num_rows, 1])
    weight_vals = np.random.random((n_elems,))
    sparse_weights = sparse_tensor.SparseTensor(inp_indices, weight_vals,
                                                [num_rows, 1])

    np_out = np.bincount(inp_vals, minlength=size, weights=weight_vals)
    self.assertAllEqual(
        np_out,
        self.evaluate(bincount_ops.bincount(
            sparse_inp, sparse_weights, axis=0)))

  @parameterized.parameters([{
      "dtype": np.int32,
  }, {
      "dtype": np.int64,
  }])
  def test_sparse_input_all_binary(self, dtype):
    np.random.seed(42)
    num_rows = 4096
    size = 10
    n_elems = 128
    inp_indices = np.random.randint(0, num_rows, (n_elems, 1))
    inp_indices = np.concatenate([inp_indices, np.zeros((n_elems, 1))], axis=1)
    inp_vals = np.random.randint(0, size, (n_elems,), dtype=dtype)
    sparse_inp = sparse_tensor.SparseTensor(inp_indices, inp_vals,
                                            [num_rows, 1])

    np_out = np.ones((size,))
    self.assertAllEqual(
        np_out,
        self.evaluate(bincount_ops.bincount(sparse_inp, binary_output=True)))

  @parameterized.parameters([{
      "dtype": np.int32,
  }, {
      "dtype": np.int64,
  }])
  def test_sparse_input_col_reduce_count(self, dtype):
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
      inp_sparse = sparse_tensor.SparseTensor(inp_sparse.indices,
                                              inp_sparse.values - 1,
                                              inp_sparse.dense_shape)
    self.assertAllEqual(
        np_out, self.evaluate(bincount_ops.bincount(arr=inp_sparse, axis=-1)))

  @parameterized.parameters([{
      "dtype": np.int32,
  }, {
      "dtype": np.int64,
  }])
  def test_sparse_input_col_reduce_binary(self, dtype):
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
      inp_sparse = sparse_tensor.SparseTensor(inp_sparse.indices,
                                              inp_sparse.values - 1,
                                              inp_sparse.dense_shape)
    self.assertAllEqual(
        np_out,
        self.evaluate(
            bincount_ops.bincount(arr=inp_sparse, axis=-1, binary_output=True)))

  @parameterized.product(
      (
          dict(
              tid="_d1",
              x=[1, 2, 2, 3, 3, 3],
              expected=[0, 1, 2, 3],
          ),
          dict(
              tid="_d2",
              x=[[0, 0, 0], [0, 1, 0], [2, 0, 2], [3, 3, 3]],
              expected=[6, 1, 2, 3],
          ),
          dict(
              tid="_d3",
              x=[[[0, 0, 0], [0, 1, 0]], [[2, 0, 2], [3, 3, 3]]],
              expected=[6, 1, 2, 3],
          ),
      ),
      (
          dict(minlength=None, maxlength=None),
          dict(minlength=3, maxlength=None),
          dict(minlength=5, maxlength=None),
          dict(minlength=None, maxlength=3),
          dict(minlength=None, maxlength=5),
          dict(minlength=2, maxlength=3),
          dict(minlength=3, maxlength=5),
          dict(minlength=5, maxlength=10),
          dict(minlength=None, maxlength=0),
      ),
  )
  def test_default(
      self,
      x,
      minlength,
      maxlength,
      expected,
      tid=None,
  ):
    expected = _adjust_expected_rank1(expected, minlength, maxlength)
    self.assertAllEqual(
        expected,
        self.evaluate(
            bincount_ops.bincount(x, minlength=minlength, maxlength=maxlength)
        ),
    )
    self.assertAllEqual(
        expected,
        self.evaluate(
            bincount_ops.bincount(
                x, minlength=minlength, maxlength=maxlength, axis=0
            )
        ),
    )

  @parameterized.product(
      (
          dict(
              tid="_d2",
              x=[[0, 0, 0], [0, 1, 0], [2, 0, 2], [3, 3, 3]],
              expected=[[3, 0, 0, 0], [2, 1, 0, 0], [1, 0, 2, 0], [0, 0, 0, 3]],
          ),
      ),
      (
          dict(minlength=None, maxlength=None),
          dict(minlength=3, maxlength=None),
          dict(minlength=5, maxlength=None),
          dict(minlength=None, maxlength=3),
          dict(minlength=None, maxlength=5),
          dict(minlength=2, maxlength=3),
          dict(minlength=3, maxlength=5),
          dict(minlength=5, maxlength=10),
          dict(minlength=None, maxlength=0),
      ),
  )
  def test_axis_neg_1(
      self, tid, x, minlength, maxlength, expected
  ):
    expected = _adjust_expected_rank2(expected, minlength, maxlength)
    self.assertAllEqual(
        expected,
        self.evaluate(
            bincount_ops.bincount(
                x, minlength=minlength, maxlength=maxlength, axis=-1
            )
        ),
    )

  @parameterized.product(
      (
          dict(
              tid="_d1",
              x=[1, 2, 2, 3, 3, 3],
              weights=[1, 2, 3, 4, 5, 6],
              axis=None,
              expected=[0, 1, 5, 15],
          ),
          dict(
              tid="_d2",
              x=[[0, 0, 0], [0, 1, 0], [2, 0, 2], [3, 3, 3]],
              weights=[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
              axis=None,
              expected=[24, 5, 16, 33],
          ),
          dict(
              tid="_d3",
              x=[[[0, 0, 0], [0, 1, 0]], [[2, 0, 2], [3, 3, 3]]],
              weights=[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
              axis=None,
              expected=[24, 5, 16, 33],
          ),
          dict(
              tid="_d2_axis_neg_1",
              x=[[0, 0, 0], [0, 1, 0], [2, 0, 2], [3, 3, 3]],
              weights=[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
              axis=-1,
              expected=[
                  [6, 0, 0, 0],
                  [10, 5, 0, 0],
                  [8, 0, 16, 0],
                  [0, 0, 0, 33],
              ],
          ),
      ),
      (
          dict(minlength=None, maxlength=None),
          dict(minlength=3, maxlength=None),
          dict(minlength=5, maxlength=None),
          dict(minlength=None, maxlength=3),
          dict(minlength=None, maxlength=5),
          dict(minlength=2, maxlength=3),
          dict(minlength=3, maxlength=5),
          dict(minlength=5, maxlength=10),
          dict(minlength=None, maxlength=0),
      ),
  )
  def test_weights(
      self,
      tid,
      x,
      weights,
      minlength,
      maxlength,
      expected,
      axis=None,
  ):
    device_set = set([d.device_type for d in tf_config.list_physical_devices()])
    if "GPU" in device_set and not test_util.is_xla_enabled():
      self.skipTest(
          "b/263004039 The DenseBincount GPU kernel does not support weights."
          " unsorted_segment_sum should be used instead on GPU."
      )
    if axis == -1:
      expected = _adjust_expected_rank2(expected, minlength, maxlength)
    else:
      expected = _adjust_expected_rank1(expected, minlength, maxlength)
    self.assertAllEqual(
        expected,
        self.evaluate(
            bincount_ops.bincount(
                x,
                weights=weights,
                minlength=minlength,
                maxlength=maxlength,
                axis=axis,
            )
        ),
    )

  @parameterized.product(
      (
          dict(
              tid="_d1",
              x=[1, 2, 2, 3, 3, 3],
              expected=[0, 1, 1, 1],
              axis=None,
          ),
          dict(
              tid="_d2",
              x=[[0, 0, 0], [0, 1, 0], [2, 0, 2], [3, 3, 3]],
              expected=[1, 1, 1, 1],
              axis=None,
          ),
          dict(
              tid="_d3",
              x=[[[0, 0, 0], [0, 1, 0]], [[2, 0, 2], [3, 3, 3]]],
              expected=[1, 1, 1, 1],
              axis=None,
          ),
          dict(
              tid="_d2_axis_neg_1",
              x=[[0, 0, 0], [0, 1, 0], [2, 0, 2], [3, 3, 3]],
              expected=[[1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [0, 0, 0, 1]],
              axis=-1,
          ),
      ),
      (
          dict(minlength=None, maxlength=None),
          dict(minlength=3, maxlength=None),
          dict(minlength=5, maxlength=None),
          dict(minlength=None, maxlength=3),
          dict(minlength=None, maxlength=5),
          dict(minlength=2, maxlength=3),
          dict(minlength=3, maxlength=5),
          dict(minlength=5, maxlength=10),
          dict(minlength=None, maxlength=0),
      ),
  )
  def test_binary_output(
      self,
      tid,
      x,
      minlength,
      maxlength,
      expected,
      axis=None,
  ):
    if axis == -1:
      expected = _adjust_expected_rank2(expected, minlength, maxlength)
    else:
      expected = _adjust_expected_rank1(expected, minlength, maxlength)
    self.assertAllEqual(
        expected,
        self.evaluate(
            bincount_ops.bincount(
                x,
                minlength=minlength,
                maxlength=maxlength,
                binary_output=True,
                axis=axis,
            )
        ),
    )


class RawOpsHeapOobTest(test.TestCase, parameterized.TestCase):

  @test_util.run_v1_only("Test security error")
  def testSparseCountSparseOutputBadIndicesShapeTooSmall(self):
    indices = [1]
    values = [[1]]
    weights = []
    dense_shape = [10]
    with self.assertRaisesRegex(ValueError,
                                "Shape must be rank 2 but is rank 1 for"):
      self.evaluate(
          gen_count_ops.SparseCountSparseOutput(
              indices=indices,
              values=values,
              dense_shape=dense_shape,
              weights=weights,
              binary_output=True))


@test_util.run_all_in_graph_and_eager_modes
@test_util.disable_tfrt
class RawOpsTest(test.TestCase, parameterized.TestCase):

  def testSparseCountSparseOutputBadIndicesShape(self):
    indices = [[[0], [0]], [[0], [1]], [[1], [0]], [[1], [2]]]
    values = [1, 1, 1, 10]
    weights = [1, 2, 4, 6]
    dense_shape = [2, 3]
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "Input indices must be a 2-dimensional tensor"):
      self.evaluate(
          gen_count_ops.SparseCountSparseOutput(
              indices=indices,
              values=values,
              dense_shape=dense_shape,
              weights=weights,
              binary_output=False))

  def testSparseCountSparseOutputBadWeightsShape(self):
    indices = [[0, 0], [0, 1], [1, 0], [1, 2]]
    values = [1, 1, 1, 10]
    weights = [1, 2, 4]
    dense_shape = [2, 3]
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "Weights and values must have the same shape"):
      self.evaluate(
          gen_count_ops.SparseCountSparseOutput(
              indices=indices,
              values=values,
              dense_shape=dense_shape,
              weights=weights,
              binary_output=False))

  def testSparseCountSparseOutputBadNumberOfValues(self):
    indices = [[0, 0], [0, 1], [1, 0]]
    values = [1, 1, 1, 10]
    weights = [1, 2, 4, 6]
    dense_shape = [2, 3]
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "Number of values must match first dimension of indices"):
      self.evaluate(
          gen_count_ops.SparseCountSparseOutput(
              indices=indices,
              values=values,
              dense_shape=dense_shape,
              weights=weights,
              binary_output=False))

  def testSparseCountSparseOutputNegativeValue(self):
    indices = [[0, 0], [0, 1], [1, 0], [1, 2]]
    values = [1, 1, -1, 10]
    dense_shape = [2, 3]
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "Input values must all be non-negative"):
      self.evaluate(
          gen_count_ops.SparseCountSparseOutput(
              indices=indices,
              values=values,
              dense_shape=dense_shape,
              binary_output=False))

  def testRaggedCountSparseOutput(self):
    splits = [0, 4, 7]
    values = [1, 1, 2, 1, 2, 10, 5]
    weights = [1, 2, 3, 4, 5, 6, 7]
    output_indices, output_values, output_shape = self.evaluate(
        gen_count_ops.RaggedCountSparseOutput(
            splits=splits, values=values, weights=weights, binary_output=False))
    self.assertAllEqual([[0, 1], [0, 2], [1, 2], [1, 5], [1, 10]],
                        output_indices)
    self.assertAllEqual([7, 3, 5, 7, 6], output_values)
    self.assertAllEqual([2, 11], output_shape)

  def testRaggedCountSparseOutputBadWeightsShape(self):
    splits = [0, 4, 7]
    values = [1, 1, 2, 1, 2, 10, 5]
    weights = [1, 2, 3, 4, 5, 6]
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "Weights and values must have the same shape"):
      self.evaluate(
          gen_count_ops.RaggedCountSparseOutput(
              splits=splits,
              values=values,
              weights=weights,
              binary_output=False))

  def testRaggedCountSparseOutputEmptySplits(self):
    splits = []
    values = [1, 1, 2, 1, 2, 10, 5]
    weights = [1, 2, 3, 4, 5, 6, 7]
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "Must provide at least 2 elements for the splits argument"):
      self.evaluate(
          gen_count_ops.RaggedCountSparseOutput(
              splits=splits,
              values=values,
              weights=weights,
              binary_output=False))

  def testRaggedCountSparseOutputBadSplitsStart(self):
    splits = [1, 7]
    values = [1, 1, 2, 1, 2, 10, 5]
    weights = [1, 2, 3, 4, 5, 6, 7]
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "Splits must start with 0"):
      self.evaluate(
          gen_count_ops.RaggedCountSparseOutput(
              splits=splits,
              values=values,
              weights=weights,
              binary_output=False))

  def testRaggedCountSparseOutputBadSplitsEnd(self):
    splits = [0, 5]
    values = [1, 1, 2, 1, 2, 10, 5]
    weights = [1, 2, 3, 4, 5, 6, 7]
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "Splits must end with the number of values"):
      self.evaluate(
          gen_count_ops.RaggedCountSparseOutput(
              splits=splits,
              values=values,
              weights=weights,
              binary_output=False))

  def testRaggedCountSparseOutputNegativeValue(self):
    splits = [0, 4, 7]
    values = [1, 1, 2, 1, -2, 10, 5]
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "Input values must all be non-negative"):
      self.evaluate(
          gen_count_ops.RaggedCountSparseOutput(
              splits=splits, values=values, binary_output=False))


if __name__ == "__main__":
  test.main()
