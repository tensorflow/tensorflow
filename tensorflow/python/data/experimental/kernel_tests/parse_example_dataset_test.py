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
"""Tests for `tf.data.experimental.parse_example_dataset()."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import numpy as np

from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2
from tensorflow.python.data.experimental.ops import parsing_ops as contrib_parsing_ops
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import parsing_ops
from tensorflow.python.platform import test

# Helpers for creating Example objects
example = example_pb2.Example
feature = feature_pb2.Feature
features = lambda d: feature_pb2.Features(feature=d)
bytes_feature = lambda v: feature(bytes_list=feature_pb2.BytesList(value=v))
int64_feature = lambda v: feature(int64_list=feature_pb2.Int64List(value=v))
float_feature = lambda v: feature(float_list=feature_pb2.FloatList(value=v))
# Helpers for creating SequenceExample objects
feature_list = lambda l: feature_pb2.FeatureList(feature=l)
feature_lists = lambda d: feature_pb2.FeatureLists(feature_list=d)
sequence_example = example_pb2.SequenceExample


@test_util.run_all_in_graph_and_eager_modes
class ParseExampleDatasetTest(test_base.DatasetTestBase):

  def _compare_output_to_expected(self, dict_tensors, expected_tensors):
    self.assertEqual(set(dict_tensors.keys()), set(expected_tensors.keys()))

    for k, v in sorted(dict_tensors.items()):
      expected_v = expected_tensors[k]
      if sparse_tensor.is_sparse(v):
        self.assertSparseValuesEqual(expected_v, v)
      else:
        # One output for standard Tensor.
        self.assertAllEqual(expected_v, v)

  def _test(self,
            input_tensor,
            feature_val,
            expected_values=None,
            expected_err=None,
            create_iterator_twice=False):

    if expected_err:
      with self.assertRaisesWithPredicateMatch(expected_err[0],
                                               expected_err[1]):
        dataset = dataset_ops.Dataset.from_tensors(input_tensor).apply(
            contrib_parsing_ops.parse_example_dataset(feature_val))
        get_next = self.getNext(dataset)
        self.evaluate(get_next())
      return
    else:
      # Returns dict w/ Tensors and SparseTensors.
      # Check values.
      dataset = dataset_ops.Dataset.from_tensors(input_tensor).apply(
          contrib_parsing_ops.parse_example_dataset(feature_val))
      get_next = self.getNext(dataset)
      result = self.evaluate(get_next())
      self._compare_output_to_expected(result, expected_values)
      with self.assertRaises(errors_impl.OutOfRangeError):
        self.evaluate(get_next())
      with self.assertRaises(errors_impl.OutOfRangeError):
        self.evaluate(get_next())
      if create_iterator_twice:
        get_next = self.getNext(dataset)
        result = self.evaluate(get_next())
        self._compare_output_to_expected(result, expected_values)
        with self.assertRaises(errors_impl.OutOfRangeError):
          self.evaluate(get_next())
    # Check shapes; if serialized is a Tensor we need its size to
    # properly check.
    batch_size = (
        self.evaluate(input_tensor).size if isinstance(input_tensor, ops.Tensor)
        else np.asarray(input_tensor).size)
    for k, f in feature_val.items():
      if isinstance(f, parsing_ops.FixedLenFeature) and f.shape is not None:
        self.assertEqual(dataset.output_shapes[k].as_list()[0], batch_size)
      elif isinstance(f, parsing_ops.VarLenFeature):
        self.assertEqual(dataset.output_shapes[k].as_list()[1], None)

  def testEmptySerializedWithAllDefaults(self):
    sparse_name = "st_a"
    a_name = "a"
    b_name = "b"
    c_name = "c:has_a_tricky_name"
    a_default = [0, 42, 0]
    b_default = np.random.rand(3, 3).astype(bytes)
    c_default = np.random.rand(2).astype(np.float32)

    expected_st_a = sparse_tensor.SparseTensorValue(  # indices, values, shape
        np.empty((0, 2), dtype=np.int64),  # indices
        np.empty((0,), dtype=np.int64),  # sp_a is DT_INT64
        np.array([2, 0], dtype=np.int64))  # batch == 2, max_elems = 0

    expected_output = {
        sparse_name: expected_st_a,
        a_name: np.array(2 * [[a_default]]),
        b_name: np.array(2 * [b_default]),
        c_name: np.array(2 * [c_default]),
    }

    self._test(
        ops.convert_to_tensor(["", ""]), {
            sparse_name:
                parsing_ops.VarLenFeature(dtypes.int64),
            a_name:
                parsing_ops.FixedLenFeature(
                    (1, 3), dtypes.int64, default_value=a_default),
            b_name:
                parsing_ops.FixedLenFeature(
                    (3, 3), dtypes.string, default_value=b_default),
            c_name:
                parsing_ops.FixedLenFeature(
                    (2,), dtypes.float32, default_value=c_default),
        },
        expected_values=expected_output,
        create_iterator_twice=True)

  @test_util.run_deprecated_v1
  def testEmptySerializedWithoutDefaultsShouldFail(self):
    input_features = {
        "st_a":
            parsing_ops.VarLenFeature(dtypes.int64),
        "a":
            parsing_ops.FixedLenFeature(
                (1, 3), dtypes.int64, default_value=[0, 42, 0]),
        "b":
            parsing_ops.FixedLenFeature(
                (3, 3),
                dtypes.string,
                default_value=np.random.rand(3, 3).astype(bytes)),
        # Feature "c" is missing a default, this gap will cause failure.
        "c":
            parsing_ops.FixedLenFeature(
                (2,), dtype=dtypes.float32),
    }

    # Edge case where the key is there but the feature value is empty
    original = example(features=features({"c": feature()}))
    self._test(
        [original.SerializeToString()],
        input_features,
        expected_err=(errors_impl.InvalidArgumentError,
                      "Feature: c \\(data type: float\\) is required"))

    # Standard case of missing key and value.
    self._test(
        ["", ""],
        input_features,
        expected_err=(errors_impl.InvalidArgumentError,
                      "Feature: c \\(data type: float\\) is required"))

  @test_util.run_deprecated_v1
  def testDenseNotMatchingShapeShouldFail(self):
    original = [
        example(features=features({
            "a": float_feature([1, 1, 3]),
        })), example(features=features({
            "a": float_feature([-1, -1]),
        }))
    ]

    serialized = [m.SerializeToString() for m in original]

    self._test(
        ops.convert_to_tensor(serialized),
        {"a": parsing_ops.FixedLenFeature((1, 3), dtypes.float32)},
        expected_err=(errors_impl.InvalidArgumentError,
                      "Key: a, Index: 1.  Number of float values"))

  def testDenseDefaultNoShapeShouldFail(self):
    original = [example(features=features({"a": float_feature([1, 1, 3]),})),]

    serialized = [m.SerializeToString() for m in original]

    self._test(
        ops.convert_to_tensor(serialized),
        {"a": parsing_ops.FixedLenFeature(None, dtypes.float32)},
        expected_err=(ValueError, "Missing shape for feature a"))

  def testSerializedContainingSparse(self):
    original = [
        example(features=features({
            "st_c": float_feature([3, 4])
        })),
        example(features=features({
            "st_c": float_feature([]),  # empty float list
        })),
        example(features=features({
            "st_d": feature(),  # feature with nothing in it
        })),
        example(features=features({
            "st_c": float_feature([1, 2, -1]),
            "st_d": bytes_feature([b"hi"])
        }))
    ]

    serialized = [m.SerializeToString() for m in original]

    expected_st_c = sparse_tensor.SparseTensorValue(  # indices, values, shape
        np.array([[0, 0], [0, 1], [3, 0], [3, 1], [3, 2]], dtype=np.int64),
        np.array([3.0, 4.0, 1.0, 2.0, -1.0], dtype=np.float32),
        np.array([4, 3], dtype=np.int64))  # batch == 2, max_elems = 3

    expected_st_d = sparse_tensor.SparseTensorValue(  # indices, values, shape
        np.array([[3, 0]], dtype=np.int64), np.array(["hi"], dtype=bytes),
        np.array([4, 1], dtype=np.int64))  # batch == 2, max_elems = 1

    expected_output = {
        "st_c": expected_st_c,
        "st_d": expected_st_d,
    }

    self._test(
        ops.convert_to_tensor(serialized), {
            "st_c": parsing_ops.VarLenFeature(dtypes.float32),
            "st_d": parsing_ops.VarLenFeature(dtypes.string)
        },
        expected_values=expected_output,
        create_iterator_twice=True)

  def testSerializedContainingSparseFeature(self):
    original = [
        example(features=features({
            "val": float_feature([3, 4]),
            "idx": int64_feature([5, 10])
        })),
        example(features=features({
            "val": float_feature([]),  # empty float list
            "idx": int64_feature([])
        })),
        example(features=features({
            "val": feature(),  # feature with nothing in it
            # missing idx feature
        })),
        example(features=features({
            "val": float_feature([1, 2, -1]),
            "idx":
                int64_feature([0, 9, 3])  # unsorted
        }))
    ]

    serialized = [m.SerializeToString() for m in original]

    expected_sp = sparse_tensor.SparseTensorValue(  # indices, values, shape
        np.array([[0, 5], [0, 10], [3, 0], [3, 3], [3, 9]], dtype=np.int64),
        np.array([3.0, 4.0, 1.0, -1.0, 2.0], dtype=np.float32),
        np.array([4, 13], dtype=np.int64))  # batch == 4, max_elems = 13

    expected_output = {"sp": expected_sp,}

    self._test(
        ops.convert_to_tensor(serialized),
        {"sp": parsing_ops.SparseFeature(["idx"], "val", dtypes.float32, [13])},
        expected_values=expected_output,
        create_iterator_twice=True)

  def testSerializedContainingSparseFeatureReuse(self):
    original = [
        example(features=features({
            "val1": float_feature([3, 4]),
            "val2": float_feature([5, 6]),
            "idx": int64_feature([5, 10])
        })),
        example(features=features({
            "val1": float_feature([]),  # empty float list
            "idx": int64_feature([])
        })),
    ]

    serialized = [m.SerializeToString() for m in original]

    expected_sp1 = sparse_tensor.SparseTensorValue(  # indices, values, shape
        np.array([[0, 5], [0, 10]], dtype=np.int64),
        np.array([3.0, 4.0], dtype=np.float32),
        np.array([2, 13], dtype=np.int64))  # batch == 2, max_elems = 13

    expected_sp2 = sparse_tensor.SparseTensorValue(  # indices, values, shape
        np.array([[0, 5], [0, 10]], dtype=np.int64),
        np.array([5.0, 6.0], dtype=np.float32),
        np.array([2, 7], dtype=np.int64))  # batch == 2, max_elems = 13

    expected_output = {
        "sp1": expected_sp1,
        "sp2": expected_sp2,
    }

    self._test(
        ops.convert_to_tensor(serialized), {
            "sp1":
                parsing_ops.SparseFeature("idx", "val1", dtypes.float32, 13),
            "sp2":
                parsing_ops.SparseFeature(
                    "idx", "val2", dtypes.float32, size=7, already_sorted=True)
        },
        expected_values=expected_output,
        create_iterator_twice=True)

  def testSerializedContaining3DSparseFeature(self):
    original = [
        example(features=features({
            "val": float_feature([3, 4]),
            "idx0": int64_feature([5, 10]),
            "idx1": int64_feature([0, 2]),
        })),
        example(features=features({
            "val": float_feature([]),  # empty float list
            "idx0": int64_feature([]),
            "idx1": int64_feature([]),
        })),
        example(features=features({
            "val": feature(),  # feature with nothing in it
            # missing idx feature
        })),
        example(features=features({
            "val": float_feature([1, 2, -1]),
            "idx0": int64_feature([0, 9, 3]),  # unsorted
            "idx1": int64_feature([1, 0, 2]),
        }))
    ]

    serialized = [m.SerializeToString() for m in original]

    expected_sp = sparse_tensor.SparseTensorValue(
        # indices
        np.array([[0, 5, 0], [0, 10, 2], [3, 0, 1], [3, 3, 2], [3, 9, 0]],
                 dtype=np.int64),
        # values
        np.array([3.0, 4.0, 1.0, -1.0, 2.0], dtype=np.float32),
        # shape batch == 4, max_elems = 13
        np.array([4, 13, 3], dtype=np.int64))

    expected_output = {"sp": expected_sp,}

    self._test(
        ops.convert_to_tensor(serialized), {
            "sp":
                parsing_ops.SparseFeature(["idx0", "idx1"], "val",
                                          dtypes.float32, [13, 3])
        },
        expected_values=expected_output,
        create_iterator_twice=True)

  def testSerializedContainingDense(self):
    aname = "a"
    bname = "b*has+a:tricky_name"
    original = [
        example(features=features({
            aname: float_feature([1, 1]),
            bname: bytes_feature([b"b0_str"]),
        })), example(features=features({
            aname: float_feature([-1, -1]),
            bname: bytes_feature([b""]),
        }))
    ]

    serialized = [m.SerializeToString() for m in original]

    expected_output = {
        aname:
            np.array(
                [[1, 1], [-1, -1]], dtype=np.float32).reshape(2, 1, 2, 1),
        bname:
            np.array(
                ["b0_str", ""], dtype=bytes).reshape(2, 1, 1, 1, 1),
    }

    # No defaults, values required
    self._test(
        ops.convert_to_tensor(serialized), {
            aname:
                parsing_ops.FixedLenFeature((1, 2, 1), dtype=dtypes.float32),
            bname:
                parsing_ops.FixedLenFeature((1, 1, 1, 1), dtype=dtypes.string),
        },
        expected_values=expected_output,
        create_iterator_twice=True)

  # This test is identical as the previous one except
  # for the creation of 'serialized'.
  def testSerializedContainingDenseWithConcat(self):
    aname = "a"
    bname = "b*has+a:tricky_name"
    # TODO(lew): Feature appearing twice should be an error in future.
    original = [
        (example(features=features({
            aname: float_feature([10, 10]),
        })), example(features=features({
            aname: float_feature([1, 1]),
            bname: bytes_feature([b"b0_str"]),
        }))),
        (
            example(features=features({
                bname: bytes_feature([b"b100"]),
            })),
            example(features=features({
                aname: float_feature([-1, -1]),
                bname: bytes_feature([b"b1"]),
            })),),
    ]

    serialized = [
        m.SerializeToString() + n.SerializeToString() for (m, n) in original
    ]

    expected_output = {
        aname:
            np.array(
                [[1, 1], [-1, -1]], dtype=np.float32).reshape(2, 1, 2, 1),
        bname:
            np.array(
                ["b0_str", "b1"], dtype=bytes).reshape(2, 1, 1, 1, 1),
    }

    # No defaults, values required
    self._test(
        ops.convert_to_tensor(serialized), {
            aname:
                parsing_ops.FixedLenFeature((1, 2, 1), dtype=dtypes.float32),
            bname:
                parsing_ops.FixedLenFeature((1, 1, 1, 1), dtype=dtypes.string),
        },
        expected_values=expected_output,
        create_iterator_twice=True)

  def testSerializedContainingDenseScalar(self):
    original = [
        example(features=features({
            "a": float_feature([1]),
        })), example(features=features({}))
    ]

    serialized = [m.SerializeToString() for m in original]

    expected_output = {
        "a":
            np.array(
                [[1], [-1]], dtype=np.float32)  # 2x1 (column vector)
    }

    self._test(
        ops.convert_to_tensor(serialized), {
            "a":
                parsing_ops.FixedLenFeature(
                    (1,), dtype=dtypes.float32, default_value=-1),
        },
        expected_values=expected_output,
        create_iterator_twice=True)

  def testSerializedContainingDenseWithDefaults(self):
    original = [
        example(features=features({
            "a": float_feature([1, 1]),
        })),
        example(features=features({
            "b": bytes_feature([b"b1"]),
        })),
        example(features=features({
            "b": feature()
        })),
    ]

    serialized = [m.SerializeToString() for m in original]

    expected_output = {
        "a":
            np.array(
                [[1, 1], [3, -3], [3, -3]], dtype=np.float32).reshape(3, 1, 2,
                                                                      1),
        "b":
            np.array(
                ["tmp_str", "b1", "tmp_str"], dtype=bytes).reshape(3, 1, 1, 1,
                                                                   1),
    }

    self._test(
        ops.convert_to_tensor(serialized), {
            "a":
                parsing_ops.FixedLenFeature(
                    (1, 2, 1), dtype=dtypes.float32, default_value=[3.0, -3.0]),
            "b":
                parsing_ops.FixedLenFeature(
                    (1, 1, 1, 1), dtype=dtypes.string, default_value="tmp_str"),
        },
        expected_values=expected_output,
        create_iterator_twice=True)

  def testSerializedSparseAndSparseFeatureAndDenseWithNoDefault(self):
    expected_st_a = sparse_tensor.SparseTensorValue(  # indices, values, shape
        np.empty((0, 2), dtype=np.int64),  # indices
        np.empty((0,), dtype=np.int64),  # sp_a is DT_INT64
        np.array([2, 0], dtype=np.int64))  # batch == 2, max_elems = 0
    expected_sp = sparse_tensor.SparseTensorValue(  # indices, values, shape
        np.array([[0, 0], [0, 3], [1, 7]], dtype=np.int64),
        np.array(["a", "b", "c"], dtype="|S"),
        np.array([2, 13], dtype=np.int64))  # batch == 4, max_elems = 13

    original = [
        example(features=features({
            "c": float_feature([3, 4]),
            "val": bytes_feature([b"a", b"b"]),
            "idx": int64_feature([0, 3])
        })), example(features=features({
            "c": float_feature([1, 2]),
            "val": bytes_feature([b"c"]),
            "idx": int64_feature([7])
        }))
    ]

    serialized = [m.SerializeToString() for m in original]

    a_default = [1, 2, 3]
    b_default = np.random.rand(3, 3).astype(bytes)
    expected_output = {
        "st_a": expected_st_a,
        "sp": expected_sp,
        "a": np.array(2 * [[a_default]]),
        "b": np.array(2 * [b_default]),
        "c": np.array(
            [[3, 4], [1, 2]], dtype=np.float32),
    }

    self._test(
        ops.convert_to_tensor(serialized),
        {
            "st_a":
                parsing_ops.VarLenFeature(dtypes.int64),
            "sp":
                parsing_ops.SparseFeature("idx", "val", dtypes.string, 13),
            "a":
                parsing_ops.FixedLenFeature(
                    (1, 3), dtypes.int64, default_value=a_default),
            "b":
                parsing_ops.FixedLenFeature(
                    (3, 3), dtypes.string, default_value=b_default),
            # Feature "c" must be provided, since it has no default_value.
            "c":
                parsing_ops.FixedLenFeature((2,), dtypes.float32),
        },
        expected_values=expected_output,
        create_iterator_twice=True)

  def testerializedContainingSparseAndSparseFeatureWithReuse(self):
    expected_idx = sparse_tensor.SparseTensorValue(  # indices, values, shape
        np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64),
        np.array([0, 3, 7, 1]),
        np.array([2, 2], dtype=np.int64))  # batch == 4, max_elems = 2

    expected_sp = sparse_tensor.SparseTensorValue(  # indices, values, shape
        np.array([[0, 0], [0, 3], [1, 1], [1, 7]], dtype=np.int64),
        np.array(["a", "b", "d", "c"], dtype="|S"),
        np.array([2, 13], dtype=np.int64))  # batch == 4, max_elems = 13

    original = [
        example(features=features({
            "val": bytes_feature([b"a", b"b"]),
            "idx": int64_feature([0, 3])
        })), example(features=features({
            "val": bytes_feature([b"c", b"d"]),
            "idx": int64_feature([7, 1])
        }))
    ]

    serialized = [m.SerializeToString() for m in original]

    expected_output = {
        "idx": expected_idx,
        "sp": expected_sp,
    }

    self._test(
        ops.convert_to_tensor(serialized), {
            "idx":
                parsing_ops.VarLenFeature(dtypes.int64),
            "sp":
                parsing_ops.SparseFeature(["idx"], "val", dtypes.string, [13]),
        },
        expected_values=expected_output,
        create_iterator_twice=True)

  def _testSerializedContainingVarLenDenseLargerBatch(self, batch_size):
    # During parsing, data read from the serialized proto is stored in buffers.
    # For small batch sizes, a buffer will contain one minibatch entry.
    # For larger batch sizes, a buffer may contain several minibatch
    # entries.  This test identified a bug where the code that copied
    # data out of the buffers and into the output tensors assumed each
    # buffer only contained one minibatch entry.  The bug has since been fixed.
    truth_int = [i for i in range(batch_size)]
    truth_str = [[("foo%d" % i).encode(), ("bar%d" % i).encode()]
                 for i in range(batch_size)]

    expected_str = copy.deepcopy(truth_str)

    # Delete some intermediate entries
    for i in range(batch_size):
      col = 1
      if np.random.rand() < 0.25:
        # w.p. 25%, drop out the second entry
        expected_str[i][col] = b"default"
        col -= 1
        truth_str[i].pop()
      if np.random.rand() < 0.25:
        # w.p. 25%, drop out the second entry (possibly again)
        expected_str[i][col] = b"default"
        truth_str[i].pop()

    expected_output = {
        # Batch size batch_size, 1 time step.
        "a": np.array(truth_int, dtype=np.int64).reshape(batch_size, 1),
        # Batch size batch_size, 2 time steps.
        "b": np.array(expected_str, dtype="|S").reshape(batch_size, 2),
    }

    original = [
        example(features=features(
            {"a": int64_feature([truth_int[i]]),
             "b": bytes_feature(truth_str[i])}))
        for i in range(batch_size)
    ]

    serialized = [m.SerializeToString() for m in original]

    self._test(
        ops.convert_to_tensor(serialized, dtype=dtypes.string), {
            "a":
                parsing_ops.FixedLenSequenceFeature(
                    shape=(),
                    dtype=dtypes.int64,
                    allow_missing=True,
                    default_value=-1),
            "b":
                parsing_ops.FixedLenSequenceFeature(
                    shape=[],
                    dtype=dtypes.string,
                    allow_missing=True,
                    default_value="default"),
        },
        expected_values=expected_output,
        create_iterator_twice=True)

  def testSerializedContainingVarLenDenseLargerBatch(self):
    np.random.seed(3456)
    for batch_size in (1, 10, 20, 100, 256):
      self._testSerializedContainingVarLenDenseLargerBatch(batch_size)

  def testSerializedShapeMismatch(self):
    aname = "a"
    bname = "b"
    cname = "c"
    original = [
        example(features=features({
            cname: int64_feature([2]),
        })),
        example(features=features({
            aname: float_feature([1, 1]),
            bname: bytes_feature([b"b0_str", b"b1_str"]),
        })),
        example(features=features({
            aname: float_feature([-1, -1, 2, 2]),
            bname: bytes_feature([b"b1"]),
        })),
        example(features=features({
            aname: float_feature([]),
            cname: int64_feature([3]),
        })),
    ]

    serialized = [m.SerializeToString() for m in original]
    if context.executing_eagerly():
      self._test(
          ops.convert_to_tensor(serialized), {
              aname:
                  parsing_ops.FixedLenSequenceFeature((2, 1),
                                                      dtype=dtypes.float32,
                                                      allow_missing=True,
                                                      default_value=[]),
              bname:
                  parsing_ops.FixedLenSequenceFeature(
                      (2, 1, 1), dtype=dtypes.string, allow_missing=True),
          },
          expected_err=(errors_impl.InvalidArgumentError,
                        "Input to reshape is a tensor with 0 values"))
    else:
      self._test(
          ops.convert_to_tensor(serialized), {
              aname:
                  parsing_ops.FixedLenSequenceFeature((2, 1),
                                                      dtype=dtypes.float32,
                                                      allow_missing=True,
                                                      default_value=[]),
              bname:
                  parsing_ops.FixedLenSequenceFeature(
                      (2, 1, 1), dtype=dtypes.string, allow_missing=True),
          },
          expected_err=(ValueError,
                        "Cannot reshape a tensor with 0 elements to shape"))

  @test_util.run_deprecated_v1
  def testSerializedContainingVarLenDense(self):
    aname = "a"
    bname = "b"
    cname = "c"
    dname = "d"
    original = [
        example(features=features({
            cname: int64_feature([2]),
        })),
        example(
            features=features({
                aname: float_feature([1, 1]),
                bname: bytes_feature([b"b0_str", b"b1_str"]),
            })),
        example(
            features=features({
                aname: float_feature([-1, -1, 2, 2]),
                bname: bytes_feature([b"b1"]),
            })),
        example(
            features=features({
                aname: float_feature([]),
                cname: int64_feature([3]),
            })),
    ]

    serialized = [m.SerializeToString() for m in original]

    expected_output = {
        aname:
            np.array(
                [
                    [0, 0, 0, 0],
                    [1, 1, 0, 0],
                    [-1, -1, 2, 2],
                    [0, 0, 0, 0],
                ],
                dtype=np.float32).reshape(4, 2, 2, 1),
        bname:
            np.array(
                [["", ""], ["b0_str", "b1_str"], ["b1", ""], ["", ""]],
                dtype=bytes).reshape(4, 2, 1, 1, 1),
        cname:
            np.array([2, 0, 0, 3], dtype=np.int64).reshape(4, 1),
        dname:
            np.empty(shape=(4, 0), dtype=bytes),
    }

    self._test(
        ops.convert_to_tensor(serialized), {
            aname:
                parsing_ops.FixedLenSequenceFeature(
                    (2, 1), dtype=dtypes.float32, allow_missing=True),
            bname:
                parsing_ops.FixedLenSequenceFeature(
                    (1, 1, 1), dtype=dtypes.string, allow_missing=True),
            cname:
                parsing_ops.FixedLenSequenceFeature(
                    shape=[], dtype=dtypes.int64, allow_missing=True),
            dname:
                parsing_ops.FixedLenSequenceFeature(
                    shape=[], dtype=dtypes.string, allow_missing=True),
        },
        expected_values=expected_output,
        create_iterator_twice=True)

    # Test with padding values.
    expected_output_custom_padding = dict(expected_output)
    expected_output_custom_padding[aname] = np.array(
        [
            [-2, -2, -2, -2],
            [1, 1, -2, -2],
            [-1, -1, 2, 2],
            [-2, -2, -2, -2],
        ],
        dtype=np.float32).reshape(4, 2, 2, 1)

    self._test(
        ops.convert_to_tensor(serialized), {
            aname:
                parsing_ops.FixedLenSequenceFeature(
                    (2, 1),
                    dtype=dtypes.float32,
                    allow_missing=True,
                    default_value=-2.0),
            bname:
                parsing_ops.FixedLenSequenceFeature(
                    (1, 1, 1), dtype=dtypes.string, allow_missing=True),
            cname:
                parsing_ops.FixedLenSequenceFeature(
                    shape=[], dtype=dtypes.int64, allow_missing=True),
            dname:
                parsing_ops.FixedLenSequenceFeature(
                    shape=[], dtype=dtypes.string, allow_missing=True),
        }, expected_output_custom_padding)

    # Change number of required values so the inputs are not a
    # multiple of this size.
    self._test(
        ops.convert_to_tensor(serialized), {
            aname:
                parsing_ops.FixedLenSequenceFeature(
                    (2, 1), dtype=dtypes.float32, allow_missing=True),
            bname:
                parsing_ops.FixedLenSequenceFeature(
                    (2, 1, 1), dtype=dtypes.string, allow_missing=True),
        },
        expected_err=(
            errors_impl.OpError, "Key: b, Index: 2.  "
            "Number of bytes values is not a multiple of stride length."))

    self._test(
        ops.convert_to_tensor(serialized), {
            aname:
                parsing_ops.FixedLenFeature((None, 2, 1), dtype=dtypes.float32),
            bname:
                parsing_ops.FixedLenSequenceFeature(
                    (2, 1, 1), dtype=dtypes.string, allow_missing=True),
        },
        expected_err=(ValueError,
                      "First dimension of shape for feature a unknown. "
                      "Consider using FixedLenSequenceFeature."))

    self._test(
        ops.convert_to_tensor(serialized), {
            cname:
                parsing_ops.FixedLenFeature(
                    (1, None), dtype=dtypes.int64, default_value=[[1]]),
        },
        expected_err=(ValueError,
                      "All dimensions of shape for feature c need to be known "
                      r"but received \(1, None\)."))

    self._test(
        ops.convert_to_tensor(serialized), {
            aname:
                parsing_ops.FixedLenSequenceFeature(
                    (2, 1), dtype=dtypes.float32, allow_missing=True),
            bname:
                parsing_ops.FixedLenSequenceFeature(
                    (1, 1, 1), dtype=dtypes.string, allow_missing=True),
            cname:
                parsing_ops.FixedLenSequenceFeature(
                    shape=[], dtype=dtypes.int64, allow_missing=False),
            dname:
                parsing_ops.FixedLenSequenceFeature(
                    shape=[], dtype=dtypes.string, allow_missing=True),
        },
        expected_err=(ValueError,
                      "Unsupported: FixedLenSequenceFeature requires "
                      "allow_missing to be True."))


if __name__ == "__main__":
  test.main()
