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
"""Tests for tensorflow.ops.parsing_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import itertools

import numpy as np

from google.protobuf import json_format

from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging

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


def flatten(list_of_lists):
  """Flatten one level of nesting."""
  return itertools.chain.from_iterable(list_of_lists)


def flatten_values_tensors_or_sparse(tensors_list):
  """Flatten each SparseTensor object into 3 Tensors for session.run()."""
  return list(
      flatten([[v.indices, v.values, v.dense_shape]
               if isinstance(v, sparse_tensor.SparseTensor) else [v]
               for v in tensors_list]))


def _compare_output_to_expected(tester, dict_tensors, expected_tensors,
                                flat_output):
  tester.assertEqual(set(dict_tensors.keys()), set(expected_tensors.keys()))

  i = 0  # Index into the flattened output of session.run()
  for k, v in dict_tensors.items():
    expected_v = expected_tensors[k]
    tf_logging.info("Comparing key: %s", k)
    if isinstance(v, sparse_tensor.SparseTensor):
      # Three outputs for SparseTensor : indices, values, shape.
      tester.assertEqual([k, len(expected_v)], [k, 3])
      tester.assertAllEqual(expected_v[0], flat_output[i])
      tester.assertAllEqual(expected_v[1], flat_output[i + 1])
      tester.assertAllEqual(expected_v[2], flat_output[i + 2])
      i += 3
    else:
      # One output for standard Tensor.
      tester.assertAllEqual(expected_v, flat_output[i])
      i += 1


class ParseExampleTest(test.TestCase):

  def _test(self, kwargs, expected_values=None, expected_err=None):
    with self.cached_session() as sess:
      if expected_err:
        with self.assertRaisesWithPredicateMatch(expected_err[0],
                                                 expected_err[1]):
          out = parsing_ops.parse_example(**kwargs)
          sess.run(flatten_values_tensors_or_sparse(out.values()))
        return
      else:
        # Returns dict w/ Tensors and SparseTensors.
        out = parsing_ops.parse_example(**kwargs)
        result = flatten_values_tensors_or_sparse(out.values())
        # Check values.
        tf_result = sess.run(result)
        _compare_output_to_expected(self, out, expected_values, tf_result)

      # Check shapes; if serialized is a Tensor we need its size to
      # properly check.
      serialized = kwargs["serialized"]
      batch_size = (
          serialized.eval().size if isinstance(serialized, ops.Tensor) else
          np.asarray(serialized).size)
      for k, f in kwargs["features"].items():
        if isinstance(f, parsing_ops.FixedLenFeature) and f.shape is not None:
          self.assertEqual(
              tuple(out[k].get_shape().as_list()), (batch_size,) + f.shape)
        elif isinstance(f, parsing_ops.VarLenFeature):
          self.assertEqual(
              tuple(out[k].indices.get_shape().as_list()), (None, 2))
          self.assertEqual(tuple(out[k].values.get_shape().as_list()), (None,))
          self.assertEqual(
              tuple(out[k].dense_shape.get_shape().as_list()), (2,))

  def testEmptySerializedWithAllDefaults(self):
    sparse_name = "st_a"
    a_name = "a"
    b_name = "b"
    c_name = "c:has_a_tricky_name"
    a_default = [0, 42, 0]
    b_default = np.random.rand(3, 3).astype(bytes)
    c_default = np.random.rand(2).astype(np.float32)

    expected_st_a = (  # indices, values, shape
        np.empty((0, 2), dtype=np.int64),  # indices
        np.empty((0,), dtype=np.int64),  # sp_a is DT_INT64
        np.array([2, 0], dtype=np.int64))  # batch == 2, max_elems = 0

    expected_output = {
        sparse_name: expected_st_a,
        a_name: np.array(2 * [[a_default]]),
        b_name: np.array(2 * [b_default]),
        c_name: np.array(2 * [c_default]),
    }

    self._test({
        "example_names": np.empty((0,), dtype=bytes),
        "serialized": ops.convert_to_tensor(["", ""]),
        "features": {
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
        }
    }, expected_output)

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
            parsing_ops.FixedLenFeature((2,), dtype=dtypes.float32),
    }

    # Edge case where the key is there but the feature value is empty
    original = example(features=features({"c": feature()}))
    self._test(
        {
            "example_names": ["in1"],
            "serialized": [original.SerializeToString()],
            "features": input_features,
        },
        expected_err=(
            errors_impl.OpError,
            "Name: in1, Feature: c \\(data type: float\\) is required"))

    # Standard case of missing key and value.
    self._test(
        {
            "example_names": ["in1", "in2"],
            "serialized": ["", ""],
            "features": input_features,
        },
        expected_err=(
            errors_impl.OpError,
            "Name: in1, Feature: c \\(data type: float\\) is required"))

  def testDenseNotMatchingShapeShouldFail(self):
    original = [
        example(features=features({
            "a": float_feature([1, 1, 3]),
        })),
        example(features=features({
            "a": float_feature([-1, -1]),
        }))
    ]

    names = ["passing", "failing"]
    serialized = [m.SerializeToString() for m in original]

    self._test(
        {
            "example_names": names,
            "serialized": ops.convert_to_tensor(serialized),
            "features": {
                "a": parsing_ops.FixedLenFeature((1, 3), dtypes.float32)
            }
        },
        expected_err=(errors_impl.OpError,
                      "Name: failing, Key: a, Index: 1.  Number of float val"))

  def testDenseDefaultNoShapeShouldFail(self):
    original = [
        example(features=features({
            "a": float_feature([1, 1, 3]),
        })),
    ]

    serialized = [m.SerializeToString() for m in original]

    self._test(
        {
            "example_names": ["failing"],
            "serialized": ops.convert_to_tensor(serialized),
            "features": {
                "a": parsing_ops.FixedLenFeature(None, dtypes.float32)
            }
        },
        expected_err=(ValueError, "Missing shape for feature a"))

  def testSerializedContainingSparse(self):
    original = [
        example(features=features({
            "st_c": float_feature([3, 4])
        })),
        example(
            features=features({
                "st_c": float_feature([]),  # empty float list
            })),
        example(
            features=features({
                "st_d": feature(),  # feature with nothing in it
            })),
        example(
            features=features({
                "st_c": float_feature([1, 2, -1]),
                "st_d": bytes_feature([b"hi"])
            }))
    ]

    serialized = [m.SerializeToString() for m in original]

    expected_st_c = (  # indices, values, shape
        np.array([[0, 0], [0, 1], [3, 0], [3, 1], [3, 2]], dtype=np.int64),
        np.array([3.0, 4.0, 1.0, 2.0, -1.0], dtype=np.float32),
        np.array([4, 3], dtype=np.int64))  # batch == 2, max_elems = 3

    expected_st_d = (  # indices, values, shape
        np.array([[3, 0]], dtype=np.int64), np.array(["hi"], dtype=bytes),
        np.array([4, 1], dtype=np.int64))  # batch == 2, max_elems = 1

    expected_output = {
        "st_c": expected_st_c,
        "st_d": expected_st_d,
    }

    self._test({
        "serialized": ops.convert_to_tensor(serialized),
        "features": {
            "st_c": parsing_ops.VarLenFeature(dtypes.float32),
            "st_d": parsing_ops.VarLenFeature(dtypes.string)
        }
    }, expected_output)

  def testSerializedContainingSparseFeature(self):
    original = [
        example(
            features=features({
                "val": float_feature([3, 4]),
                "idx": int64_feature([5, 10])
            })),
        example(
            features=features({
                "val": float_feature([]),  # empty float list
                "idx": int64_feature([])
            })),
        example(
            features=features({
                "val": feature(),  # feature with nothing in it
                # missing idx feature
            })),
        example(
            features=features({
                "val": float_feature([1, 2, -1]),
                "idx":
                    int64_feature([0, 9, 3])  # unsorted
            }))
    ]

    serialized = [m.SerializeToString() for m in original]

    expected_sp = (  # indices, values, shape
        np.array([[0, 5], [0, 10], [3, 0], [3, 3], [3, 9]], dtype=np.int64),
        np.array([3.0, 4.0, 1.0, -1.0, 2.0], dtype=np.float32),
        np.array([4, 13], dtype=np.int64))  # batch == 4, max_elems = 13

    expected_output = {
        "sp": expected_sp,
    }

    self._test({
        "serialized": ops.convert_to_tensor(serialized),
        "features": {
            "sp":
                parsing_ops.SparseFeature(["idx"], "val", dtypes.float32, [13])
        }
    }, expected_output)

  def testSerializedContainingSparseFeatureReuse(self):
    original = [
        example(
            features=features({
                "val1": float_feature([3, 4]),
                "val2": float_feature([5, 6]),
                "idx": int64_feature([5, 10])
            })),
        example(
            features=features({
                "val1": float_feature([]),  # empty float list
                "idx": int64_feature([])
            })),
    ]

    serialized = [m.SerializeToString() for m in original]

    expected_sp1 = (  # indices, values, shape
        np.array([[0, 5], [0, 10]], dtype=np.int64),
        np.array([3.0, 4.0], dtype=np.float32), np.array(
            [2, 13], dtype=np.int64))  # batch == 2, max_elems = 13

    expected_sp2 = (  # indices, values, shape
        np.array([[0, 5], [0, 10]], dtype=np.int64),
        np.array([5.0, 6.0], dtype=np.float32), np.array(
            [2, 7], dtype=np.int64))  # batch == 2, max_elems = 13

    expected_output = {
        "sp1": expected_sp1,
        "sp2": expected_sp2,
    }

    self._test({
        "serialized": ops.convert_to_tensor(serialized),
        "features": {
            "sp1":
                parsing_ops.SparseFeature("idx", "val1", dtypes.float32, 13),
            "sp2":
                parsing_ops.SparseFeature(
                    "idx", "val2", dtypes.float32, size=7, already_sorted=True)
        }
    }, expected_output)

  def testSerializedContaining3DSparseFeature(self):
    original = [
        example(
            features=features({
                "val": float_feature([3, 4]),
                "idx0": int64_feature([5, 10]),
                "idx1": int64_feature([0, 2]),
            })),
        example(
            features=features({
                "val": float_feature([]),  # empty float list
                "idx0": int64_feature([]),
                "idx1": int64_feature([]),
            })),
        example(
            features=features({
                "val": feature(),  # feature with nothing in it
                # missing idx feature
            })),
        example(
            features=features({
                "val": float_feature([1, 2, -1]),
                "idx0": int64_feature([0, 9, 3]),  # unsorted
                "idx1": int64_feature([1, 0, 2]),
            }))
    ]

    serialized = [m.SerializeToString() for m in original]

    expected_sp = (
        # indices
        np.array(
            [[0, 5, 0], [0, 10, 2], [3, 0, 1], [3, 3, 2], [3, 9, 0]],
            dtype=np.int64),
        # values
        np.array([3.0, 4.0, 1.0, -1.0, 2.0], dtype=np.float32),
        # shape batch == 4, max_elems = 13
        np.array([4, 13, 3], dtype=np.int64))

    expected_output = {
        "sp": expected_sp,
    }

    self._test({
        "serialized": ops.convert_to_tensor(serialized),
        "features": {
            "sp":
                parsing_ops.SparseFeature(["idx0", "idx1"], "val",
                                          dtypes.float32, [13, 3])
        }
    }, expected_output)

  def testSerializedContainingDense(self):
    aname = "a"
    bname = "b*has+a:tricky_name"
    original = [
        example(
            features=features({
                aname: float_feature([1, 1]),
                bname: bytes_feature([b"b0_str"]),
            })),
        example(
            features=features({
                aname: float_feature([-1, -1]),
                bname: bytes_feature([b""]),
            }))
    ]

    serialized = [m.SerializeToString() for m in original]

    expected_output = {
        aname:
            np.array([[1, 1], [-1, -1]], dtype=np.float32).reshape(2, 1, 2, 1),
        bname:
            np.array(["b0_str", ""], dtype=bytes).reshape(2, 1, 1, 1, 1),
    }

    # No defaults, values required
    self._test({
        "serialized": ops.convert_to_tensor(serialized),
        "features": {
            aname:
                parsing_ops.FixedLenFeature((1, 2, 1), dtype=dtypes.float32),
            bname:
                parsing_ops.FixedLenFeature((1, 1, 1, 1), dtype=dtypes.string),
        }
    }, expected_output)

  # This test is identical as the previous one except
  # for the creation of 'serialized'.
  def testSerializedContainingDenseWithConcat(self):
    aname = "a"
    bname = "b*has+a:tricky_name"
    # TODO(lew): Feature appearing twice should be an error in future.
    original = [
        (example(features=features({
            aname: float_feature([10, 10]),
        })),
         example(
             features=features({
                 aname: float_feature([1, 1]),
                 bname: bytes_feature([b"b0_str"]),
             }))),
        (
            example(features=features({
                bname: bytes_feature([b"b100"]),
            })),
            example(
                features=features({
                    aname: float_feature([-1, -1]),
                    bname: bytes_feature([b"b1"]),
                })),
        ),
    ]

    serialized = [
        m.SerializeToString() + n.SerializeToString() for (m, n) in original
    ]

    expected_output = {
        aname:
            np.array([[1, 1], [-1, -1]], dtype=np.float32).reshape(2, 1, 2, 1),
        bname:
            np.array(["b0_str", "b1"], dtype=bytes).reshape(2, 1, 1, 1, 1),
    }

    # No defaults, values required
    self._test({
        "serialized": ops.convert_to_tensor(serialized),
        "features": {
            aname:
                parsing_ops.FixedLenFeature((1, 2, 1), dtype=dtypes.float32),
            bname:
                parsing_ops.FixedLenFeature((1, 1, 1, 1), dtype=dtypes.string),
        }
    }, expected_output)

  def testSerializedContainingDenseScalar(self):
    original = [
        example(features=features({
            "a": float_feature([1]),
        })),
        example(features=features({}))
    ]

    serialized = [m.SerializeToString() for m in original]

    expected_output = {
        "a":
            np.array([[1], [-1]], dtype=np.float32)  # 2x1 (column vector)
    }

    self._test({
        "serialized": ops.convert_to_tensor(serialized),
        "features": {
            "a":
                parsing_ops.FixedLenFeature(
                    (1,), dtype=dtypes.float32, default_value=-1),
        }
    }, expected_output)

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
            np.array([[1, 1], [3, -3], [3, -3]], dtype=np.float32).reshape(
                3, 1, 2, 1),
        "b":
            np.array(["tmp_str", "b1", "tmp_str"], dtype=bytes).reshape(
                3, 1, 1, 1, 1),
    }

    self._test({
        "serialized": ops.convert_to_tensor(serialized),
        "features": {
            "a":
                parsing_ops.FixedLenFeature(
                    (1, 2, 1), dtype=dtypes.float32, default_value=[3.0, -3.0]),
            "b":
                parsing_ops.FixedLenFeature(
                    (1, 1, 1, 1), dtype=dtypes.string, default_value="tmp_str"),
        }
    }, expected_output)

  def testSerializedContainingSparseAndSparseFeatureAndDenseWithNoDefault(self):
    expected_st_a = (  # indices, values, shape
        np.empty((0, 2), dtype=np.int64),  # indices
        np.empty((0,), dtype=np.int64),  # sp_a is DT_INT64
        np.array([2, 0], dtype=np.int64))  # batch == 2, max_elems = 0
    expected_sp = (  # indices, values, shape
        np.array([[0, 0], [0, 3], [1, 7]], dtype=np.int64),
        np.array(["a", "b", "c"], dtype="|S"), np.array(
            [2, 13], dtype=np.int64))  # batch == 4, max_elems = 13

    original = [
        example(
            features=features({
                "c": float_feature([3, 4]),
                "val": bytes_feature([b"a", b"b"]),
                "idx": int64_feature([0, 3])
            })),
        example(
            features=features({
                "c": float_feature([1, 2]),
                "val": bytes_feature([b"c"]),
                "idx": int64_feature([7])
            }))
    ]

    names = ["in1", "in2"]
    serialized = [m.SerializeToString() for m in original]

    a_default = [1, 2, 3]
    b_default = np.random.rand(3, 3).astype(bytes)
    expected_output = {
        "st_a": expected_st_a,
        "sp": expected_sp,
        "a": np.array(2 * [[a_default]]),
        "b": np.array(2 * [b_default]),
        "c": np.array([[3, 4], [1, 2]], dtype=np.float32),
    }

    self._test(
        {
            "example_names": names,
            "serialized": ops.convert_to_tensor(serialized),
            "features": {
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
            }
        },
        expected_output)

  def testSerializedContainingSparseAndSparseFeatureWithReuse(self):
    expected_idx = (  # indices, values, shape
        np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64),
        np.array([0, 3, 7, 1]),
        np.array([2, 2], dtype=np.int64))  # batch == 4, max_elems = 2

    expected_sp = (  # indices, values, shape
        np.array([[0, 0], [0, 3], [1, 1], [1, 7]], dtype=np.int64),
        np.array(["a", "b", "d", "c"], dtype="|S"),
        np.array([2, 13], dtype=np.int64))  # batch == 4, max_elems = 13

    original = [
        example(
            features=features({
                "val": bytes_feature([b"a", b"b"]),
                "idx": int64_feature([0, 3])
            })),
        example(
            features=features({
                "val": bytes_feature([b"c", b"d"]),
                "idx": int64_feature([7, 1])
            }))
    ]

    names = ["in1", "in2"]
    serialized = [m.SerializeToString() for m in original]

    expected_output = {
        "idx": expected_idx,
        "sp": expected_sp,
    }

    self._test({
        "example_names": names,
        "serialized": ops.convert_to_tensor(serialized),
        "features": {
            "idx":
                parsing_ops.VarLenFeature(dtypes.int64),
            "sp":
                parsing_ops.SparseFeature(["idx"], "val", dtypes.string, [13]),
        }
    }, expected_output)

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
        example(
            features=features({
                "a": int64_feature([truth_int[i]]),
                "b": bytes_feature(truth_str[i])
            })) for i in range(batch_size)
    ]

    serialized = [m.SerializeToString() for m in original]

    self._test({
        "serialized": ops.convert_to_tensor(serialized, dtype=dtypes.string),
        "features": {
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
        }
    }, expected_output)

  def testSerializedContainingVarLenDenseLargerBatch(self):
    np.random.seed(3456)
    for batch_size in (1, 10, 20, 100, 256):
      self._testSerializedContainingVarLenDenseLargerBatch(batch_size)

  def testSerializedContainingVarLenDense(self):
    aname = "a"
    bname = "b"
    cname = "c"
    dname = "d"
    example_names = ["in1", "in2", "in3", "in4"]
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

    self._test({
        "example_names": example_names,
        "serialized": ops.convert_to_tensor(serialized),
        "features": {
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
        }
    }, expected_output)

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

    self._test({
        "example_names": example_names,
        "serialized": ops.convert_to_tensor(serialized),
        "features": {
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
        }
    }, expected_output_custom_padding)

    # Change number of required values so the inputs are not a
    # multiple of this size.
    self._test(
        {
            "example_names": example_names,
            "serialized": ops.convert_to_tensor(serialized),
            "features": {
                aname:
                    parsing_ops.FixedLenSequenceFeature(
                        (2, 1), dtype=dtypes.float32, allow_missing=True),
                bname:
                    parsing_ops.FixedLenSequenceFeature(
                        (2, 1, 1), dtype=dtypes.string, allow_missing=True),
            }
        },
        expected_err=(
            errors_impl.OpError, "Name: in3, Key: b, Index: 2.  "
            "Number of bytes values is not a multiple of stride length."))

    self._test(
        {
            "example_names": example_names,
            "serialized": ops.convert_to_tensor(serialized),
            "features": {
                aname:
                    parsing_ops.FixedLenSequenceFeature(
                        (2, 1),
                        dtype=dtypes.float32,
                        allow_missing=True,
                        default_value=[]),
                bname:
                    parsing_ops.FixedLenSequenceFeature(
                        (2, 1, 1), dtype=dtypes.string, allow_missing=True),
            }
        },
        expected_err=(ValueError,
                      "Cannot reshape a tensor with 0 elements to shape"))

    self._test(
        {
            "example_names": example_names,
            "serialized": ops.convert_to_tensor(serialized),
            "features": {
                aname:
                    parsing_ops.FixedLenFeature(
                        (None, 2, 1), dtype=dtypes.float32),
                bname:
                    parsing_ops.FixedLenSequenceFeature(
                        (2, 1, 1), dtype=dtypes.string, allow_missing=True),
            }
        },
        expected_err=(ValueError,
                      "First dimension of shape for feature a unknown. "
                      "Consider using FixedLenSequenceFeature."))

    self._test(
        {
            "example_names": example_names,
            "serialized": ops.convert_to_tensor(serialized),
            "features": {
                cname:
                    parsing_ops.FixedLenFeature(
                        (1, None), dtype=dtypes.int64, default_value=[[1]]),
            }
        },
        expected_err=(ValueError,
                      "All dimensions of shape for feature c need to be known "
                      r"but received \(1, None\)."))

    self._test(
        {
            "example_names": example_names,
            "serialized": ops.convert_to_tensor(serialized),
            "features": {
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
            }
        },
        expected_err=(ValueError,
                      "Unsupported: FixedLenSequenceFeature requires "
                      "allow_missing to be True."))


class ParseSingleExampleTest(test.TestCase):

  def _test(self, kwargs, expected_values=None, expected_err=None):
    with self.cached_session() as sess:
      if expected_err:
        with self.assertRaisesWithPredicateMatch(expected_err[0],
                                                 expected_err[1]):
          out = parsing_ops.parse_single_example(**kwargs)
          sess.run(flatten_values_tensors_or_sparse(out.values()))
      else:
        # Returns dict w/ Tensors and SparseTensors.
        out = parsing_ops.parse_single_example(**kwargs)
        # Check values.
        tf_result = sess.run(flatten_values_tensors_or_sparse(out.values()))
        _compare_output_to_expected(self, out, expected_values, tf_result)

      # Check shapes.
      for k, f in kwargs["features"].items():
        if isinstance(f, parsing_ops.FixedLenFeature) and f.shape is not None:
          self.assertEqual(
              tuple(out[k].get_shape()), tensor_shape.as_shape(f.shape))
        elif isinstance(f, parsing_ops.VarLenFeature):
          self.assertEqual(
              tuple(out[k].indices.get_shape().as_list()), (None, 1))
          self.assertEqual(tuple(out[k].values.get_shape().as_list()), (None,))
          self.assertEqual(
              tuple(out[k].dense_shape.get_shape().as_list()), (1,))

  def testSingleExampleWithSparseAndSparseFeatureAndDense(self):
    original = example(
        features=features({
            "c": float_feature([3, 4]),
            "d": float_feature([0.0, 1.0]),
            "val": bytes_feature([b"a", b"b"]),
            "idx": int64_feature([0, 3]),
            "st_a": float_feature([3.0, 4.0])
        }))

    serialized = original.SerializeToString()

    expected_st_a = (
        np.array([[0], [1]], dtype=np.int64),  # indices
        np.array([3.0, 4.0], dtype=np.float32),  # values
        np.array([2], dtype=np.int64))  # shape: max_values = 2

    expected_sp = (  # indices, values, shape
        np.array([[0], [3]], dtype=np.int64), np.array(["a", "b"], dtype="|S"),
        np.array([13], dtype=np.int64))  # max_values = 13

    a_default = [1, 2, 3]
    b_default = np.random.rand(3, 3).astype(bytes)
    expected_output = {
        "st_a": expected_st_a,
        "sp": expected_sp,
        "a": [a_default],
        "b": b_default,
        "c": np.array([3, 4], dtype=np.float32),
        "d": np.array([0.0, 1.0], dtype=np.float32),
    }

    self._test(
        {
            "example_names": ops.convert_to_tensor("in1"),
            "serialized": ops.convert_to_tensor(serialized),
            "features": {
                "st_a":
                    parsing_ops.VarLenFeature(dtypes.float32),
                "sp":
                    parsing_ops.SparseFeature(["idx"], "val", dtypes.string,
                                              [13]),
                "a":
                    parsing_ops.FixedLenFeature(
                        (1, 3), dtypes.int64, default_value=a_default),
                "b":
                    parsing_ops.FixedLenFeature(
                        (3, 3), dtypes.string, default_value=b_default),
                # Feature "c" must be provided, since it has no default_value.
                "c":
                    parsing_ops.FixedLenFeature(2, dtypes.float32),
                "d":
                    parsing_ops.FixedLenSequenceFeature(
                        [], dtypes.float32, allow_missing=True)
            }
        },
        expected_output)


class ParseSequenceExampleTest(test.TestCase):

  def testCreateSequenceExample(self):
    value = sequence_example(
        context=features({
            "global_feature": float_feature([1, 2, 3]),
        }),
        feature_lists=feature_lists({
            "repeated_feature_2_frames":
                feature_list([
                    bytes_feature([b"a", b"b", b"c"]),
                    bytes_feature([b"a", b"d", b"e"])
                ]),
            "repeated_feature_3_frames":
                feature_list([
                    int64_feature([3, 4, 5, 6, 7]),
                    int64_feature([-1, 0, 0, 0, 0]),
                    int64_feature([1, 2, 3, 4, 5])
                ])
        }))
    value.SerializeToString()  # Smoke test

  def _test(self,
            kwargs,
            expected_context_values=None,
            expected_feat_list_values=None,
            expected_length_values=None,
            expected_err=None,
            batch=False):
    expected_context_values = expected_context_values or {}
    expected_feat_list_values = expected_feat_list_values or {}
    expected_length_values = expected_length_values or {}

    with self.cached_session() as sess:
      if expected_err:
        with self.assertRaisesWithPredicateMatch(expected_err[0],
                                                 expected_err[1]):
          if batch:
            c_out, fl_out, _ = parsing_ops.parse_sequence_example(**kwargs)
          else:
            c_out, fl_out = parsing_ops.parse_single_sequence_example(**kwargs)
          if c_out:
            sess.run(flatten_values_tensors_or_sparse(c_out.values()))
          if fl_out:
            sess.run(flatten_values_tensors_or_sparse(fl_out.values()))
      else:
        # Returns dicts w/ Tensors and SparseTensors.
        if batch:
          (context_out, feat_list_out,
           lengths_out) = parsing_ops.parse_sequence_example(**kwargs)
        else:
          (context_out,
           feat_list_out) = parsing_ops.parse_single_sequence_example(**kwargs)
          lengths_out = {}

        context_result = sess.run(
            flatten_values_tensors_or_sparse(
                context_out.values())) if context_out else []
        feat_list_result = sess.run(
            flatten_values_tensors_or_sparse(
                feat_list_out.values())) if feat_list_out else []
        lengths_result = sess.run(
            flatten_values_tensors_or_sparse(
                lengths_out.values())) if lengths_out else []
        # Check values.
        _compare_output_to_expected(self, context_out, expected_context_values,
                                    context_result)
        _compare_output_to_expected(self, feat_list_out,
                                    expected_feat_list_values, feat_list_result)
        _compare_output_to_expected(self, lengths_out, expected_length_values,
                                    lengths_result)

      # Check shapes; if serialized is a Tensor we need its size to
      # properly check.
      if "context_features" in kwargs:
        for k, f in kwargs["context_features"].items():
          if isinstance(f, parsing_ops.FixedLenFeature) and f.shape is not None:
            if batch:
              self.assertEqual(
                  tuple(context_out[k].get_shape().as_list()[1:]), f.shape)
            else:
              self.assertEqual(
                  tuple(context_out[k].get_shape().as_list()), f.shape)
          elif isinstance(f, parsing_ops.VarLenFeature) and batch:
            self.assertEqual(
                tuple(context_out[k].indices.get_shape().as_list()), (None, 2))
            self.assertEqual(
                tuple(context_out[k].values.get_shape().as_list()), (None,))
            self.assertEqual(
                tuple(context_out[k].dense_shape.get_shape().as_list()), (2,))
          elif isinstance(f, parsing_ops.VarLenFeature) and not batch:
            self.assertEqual(
                tuple(context_out[k].indices.get_shape().as_list()), (None, 1))
            self.assertEqual(
                tuple(context_out[k].values.get_shape().as_list()), (None,))
            self.assertEqual(
                tuple(context_out[k].dense_shape.get_shape().as_list()), (1,))

  def _testBoth(self,
                kwargs,
                expected_context_values=None,
                expected_feat_list_values=None,
                expected_err=None):
    # Test using tf.parse_single_sequence_example
    self._test(
        kwargs,
        expected_context_values=expected_context_values,
        expected_feat_list_values=expected_feat_list_values,
        expected_err=expected_err,
        batch=False)

    # Convert the input to a batch of size 1, and test using
    # tf.parse_sequence_example.

    # Some replacements are needed for the batch version.
    kwargs["serialized"] = [kwargs.pop("serialized")]
    kwargs["example_names"] = [kwargs.pop("example_name")
                              ] if "example_name" in kwargs else None
    # Disable error string matching; it's not consistent for batch mode.
    if expected_err:
      expected_err = (expected_err[0], "")

    # Add a batch dimension to expected output
    if expected_context_values:
      new_values = {}
      for k in expected_context_values:
        v = expected_context_values[k]
        if isinstance(kwargs["context_features"][k],
                      parsing_ops.FixedLenFeature):
          new_values[k] = np.expand_dims(v, axis=0)
        else:
          # Sparse tensor.
          new_values[k] = (np.insert(v[0], 0, 0, axis=1), v[1],
                           np.insert(v[2], 0, 1))
      expected_context_values = new_values

    expected_length_values = {}
    if expected_feat_list_values:
      new_values = {}
      for k in expected_feat_list_values:
        v = expected_feat_list_values[k]
        if isinstance(kwargs["sequence_features"][k],
                      parsing_ops.FixedLenSequenceFeature):
          expected_length_values[k] = [np.shape(v)[0]]
          new_values[k] = np.expand_dims(v, axis=0)
        else:
          # Sparse tensor.
          new_values[k] = (np.insert(v[0], 0, 0, axis=1), v[1],
                           np.insert(v[2], 0, 1))
      expected_feat_list_values = new_values

    self._test(
        kwargs,
        expected_context_values=expected_context_values,
        expected_feat_list_values=expected_feat_list_values,
        expected_length_values=expected_length_values,
        expected_err=expected_err,
        batch=True)

  def testSequenceExampleWithSparseAndDenseContext(self):
    original = sequence_example(
        context=features({
            "c": float_feature([3, 4]),
            "st_a": float_feature([3.0, 4.0])
        }))

    serialized = original.SerializeToString()

    expected_st_a = (
        np.array([[0], [1]], dtype=np.int64),  # indices
        np.array([3.0, 4.0], dtype=np.float32),  # values
        np.array([2], dtype=np.int64))  # shape: num_features = 2

    a_default = [[1, 2, 3]]
    b_default = np.random.rand(3, 3).astype(bytes)
    expected_context_output = {
        "st_a": expected_st_a,
        "a": a_default,
        "b": b_default,
        "c": np.array([3, 4], dtype=np.float32),
    }

    self._testBoth(
        {
            "example_name": "in1",
            "serialized": ops.convert_to_tensor(serialized),
            "context_features": {
                "st_a":
                    parsing_ops.VarLenFeature(dtypes.float32),
                "a":
                    parsing_ops.FixedLenFeature(
                        (1, 3), dtypes.int64, default_value=a_default),
                "b":
                    parsing_ops.FixedLenFeature(
                        (3, 3), dtypes.string, default_value=b_default),
                # Feature "c" must be provided, since it has no default_value.
                "c":
                    parsing_ops.FixedLenFeature((2,), dtypes.float32),
            }
        },
        expected_context_values=expected_context_output)

  def testSequenceExampleWithMultipleSizeFeatureLists(self):
    original = sequence_example(
        feature_lists=feature_lists({
            "a":
                feature_list([
                    int64_feature([-1, 0, 1]),
                    int64_feature([2, 3, 4]),
                    int64_feature([5, 6, 7]),
                    int64_feature([8, 9, 10]),
                ]),
            "b":
                feature_list([bytes_feature([b"r00", b"r01", b"r10", b"r11"])]),
            "c":
                feature_list([float_feature([3, 4]),
                              float_feature([-1, 2])]),
        }))

    serialized = original.SerializeToString()

    expected_feature_list_output = {
        "a":
            np.array(
                [  # outer dimension is time.
                    [[-1, 0, 1]],  # inside are 1x3 matrices
                    [[2, 3, 4]],
                    [[5, 6, 7]],
                    [[8, 9, 10]]
                ],
                dtype=np.int64),
        "b":
            np.array(
                [  # outer dimension is time, inside are 2x2 matrices
                    [[b"r00", b"r01"], [b"r10", b"r11"]]
                ],
                dtype=bytes),
        "c":
            np.array(
                [  # outer dimension is time, inside are 2-vectors
                    [3, 4], [-1, 2]
                ],
                dtype=np.float32),
        "d":
            np.empty(shape=(0, 5), dtype=np.float32),  # empty_allowed_missing
    }

    self._testBoth(
        {
            "example_name": "in1",
            "serialized": ops.convert_to_tensor(serialized),
            "sequence_features": {
                "a":
                    parsing_ops.FixedLenSequenceFeature((1, 3), dtypes.int64),
                "b":
                    parsing_ops.FixedLenSequenceFeature((2, 2), dtypes.string),
                "c":
                    parsing_ops.FixedLenSequenceFeature(2, dtypes.float32),
                "d":
                    parsing_ops.FixedLenSequenceFeature(
                        (5,), dtypes.float32, allow_missing=True),
            }
        },
        expected_feat_list_values=expected_feature_list_output)

  def testSequenceExampleWithoutDebugName(self):
    original = sequence_example(
        feature_lists=feature_lists({
            "a":
                feature_list([int64_feature([3, 4]),
                              int64_feature([1, 0])]),
            "st_a":
                feature_list([
                    float_feature([3.0, 4.0]),
                    float_feature([5.0]),
                    float_feature([])
                ]),
            "st_b":
                feature_list([
                    bytes_feature([b"a"]),
                    bytes_feature([]),
                    bytes_feature([]),
                    bytes_feature([b"b", b"c"])
                ])
        }))

    serialized = original.SerializeToString()

    expected_st_a = (
        np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int64),  # indices
        np.array([3.0, 4.0, 5.0], dtype=np.float32),  # values
        np.array([3, 2], dtype=np.int64))  # shape: num_time = 3, max_feat = 2

    expected_st_b = (
        np.array([[0, 0], [3, 0], [3, 1]], dtype=np.int64),  # indices
        np.array(["a", "b", "c"], dtype="|S"),  # values
        np.array([4, 2], dtype=np.int64))  # shape: num_time = 4, max_feat = 2

    expected_st_c = (
        np.empty((0, 2), dtype=np.int64),  # indices
        np.empty((0,), dtype=np.int64),  # values
        np.array([0, 0], dtype=np.int64))  # shape: num_time = 0, max_feat = 0

    expected_feature_list_output = {
        "a": np.array([[3, 4], [1, 0]], dtype=np.int64),
        "st_a": expected_st_a,
        "st_b": expected_st_b,
        "st_c": expected_st_c,
    }

    self._testBoth(
        {
            "serialized": ops.convert_to_tensor(serialized),
            "sequence_features": {
                "st_a": parsing_ops.VarLenFeature(dtypes.float32),
                "st_b": parsing_ops.VarLenFeature(dtypes.string),
                "st_c": parsing_ops.VarLenFeature(dtypes.int64),
                "a": parsing_ops.FixedLenSequenceFeature((2,), dtypes.int64),
            }
        },
        expected_feat_list_values=expected_feature_list_output)

  def testSequenceExampleWithSparseAndDenseFeatureLists(self):
    original = sequence_example(
        feature_lists=feature_lists({
            "a":
                feature_list([int64_feature([3, 4]),
                              int64_feature([1, 0])]),
            "st_a":
                feature_list([
                    float_feature([3.0, 4.0]),
                    float_feature([5.0]),
                    float_feature([])
                ]),
            "st_b":
                feature_list([
                    bytes_feature([b"a"]),
                    bytes_feature([]),
                    bytes_feature([]),
                    bytes_feature([b"b", b"c"])
                ])
        }))

    serialized = original.SerializeToString()

    expected_st_a = (
        np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int64),  # indices
        np.array([3.0, 4.0, 5.0], dtype=np.float32),  # values
        np.array([3, 2], dtype=np.int64))  # shape: num_time = 3, max_feat = 2

    expected_st_b = (
        np.array([[0, 0], [3, 0], [3, 1]], dtype=np.int64),  # indices
        np.array(["a", "b", "c"], dtype="|S"),  # values
        np.array([4, 2], dtype=np.int64))  # shape: num_time = 4, max_feat = 2

    expected_st_c = (
        np.empty((0, 2), dtype=np.int64),  # indices
        np.empty((0,), dtype=np.int64),  # values
        np.array([0, 0], dtype=np.int64))  # shape: num_time = 0, max_feat = 0

    expected_feature_list_output = {
        "a": np.array([[3, 4], [1, 0]], dtype=np.int64),
        "st_a": expected_st_a,
        "st_b": expected_st_b,
        "st_c": expected_st_c,
    }

    self._testBoth(
        {
            "example_name": "in1",
            "serialized": ops.convert_to_tensor(serialized),
            "sequence_features": {
                "st_a": parsing_ops.VarLenFeature(dtypes.float32),
                "st_b": parsing_ops.VarLenFeature(dtypes.string),
                "st_c": parsing_ops.VarLenFeature(dtypes.int64),
                "a": parsing_ops.FixedLenSequenceFeature((2,), dtypes.int64),
            }
        },
        expected_feat_list_values=expected_feature_list_output)

  def testSequenceExampleWithEmptyFeatureInFeatureLists(self):
    original = sequence_example(
        feature_lists=feature_lists({
            "st_a":
                feature_list([
                    float_feature([3.0, 4.0]),
                    feature(),
                    float_feature([5.0]),
                ]),
        }))

    serialized = original.SerializeToString()

    expected_st_a = (
        np.array([[0, 0], [0, 1], [2, 0]], dtype=np.int64),  # indices
        np.array([3.0, 4.0, 5.0], dtype=np.float32),  # values
        np.array([3, 2], dtype=np.int64))  # shape: num_time = 3, max_feat = 2

    expected_feature_list_output = {
        "st_a": expected_st_a,
    }

    self._testBoth(
        {
            "example_name": "in1",
            "serialized": ops.convert_to_tensor(serialized),
            "sequence_features": {
                "st_a": parsing_ops.VarLenFeature(dtypes.float32),
            }
        },
        expected_feat_list_values=expected_feature_list_output)

  def testSequenceExampleListWithInconsistentDataFails(self):
    original = sequence_example(
        feature_lists=feature_lists({
            "a": feature_list([int64_feature([-1, 0]),
                               float_feature([2, 3])])
        }))

    serialized = original.SerializeToString()

    self._testBoth(
        {
            "example_name": "in1",
            "serialized": ops.convert_to_tensor(serialized),
            "sequence_features": {
                "a": parsing_ops.FixedLenSequenceFeature((2,), dtypes.int64)
            }
        },
        expected_err=(errors_impl.OpError, "Feature list: a, Index: 1."
                      "  Data types don't match. Expected type: int64"))

  def testSequenceExampleListWithWrongDataTypeFails(self):
    original = sequence_example(
        feature_lists=feature_lists({
            "a": feature_list([float_feature([2, 3])])
        }))

    serialized = original.SerializeToString()

    self._testBoth(
        {
            "example_name": "in1",
            "serialized": ops.convert_to_tensor(serialized),
            "sequence_features": {
                "a": parsing_ops.FixedLenSequenceFeature((2,), dtypes.int64)
            }
        },
        expected_err=(errors_impl.OpError,
                      "Feature list: a, Index: 0.  Data types don't match."
                      " Expected type: int64"))

  def testSequenceExampleListWithWrongSparseDataTypeFails(self):
    original = sequence_example(
        feature_lists=feature_lists({
            "a":
                feature_list([
                    int64_feature([3, 4]),
                    int64_feature([1, 2]),
                    float_feature([2.0, 3.0])
                ])
        }))

    serialized = original.SerializeToString()

    self._testBoth(
        {
            "example_name": "in1",
            "serialized": ops.convert_to_tensor(serialized),
            "sequence_features": {
                "a": parsing_ops.FixedLenSequenceFeature((2,), dtypes.int64)
            }
        },
        expected_err=(errors_impl.OpError,
                      "Name: in1, Feature list: a, Index: 2."
                      "  Data types don't match. Expected type: int64"
                      "  Feature is: float_list"))

  def testSequenceExampleListWithWrongShapeFails(self):
    original = sequence_example(
        feature_lists=feature_lists({
            "a":
                feature_list([int64_feature([2, 3]),
                              int64_feature([2, 3, 4])]),
        }))

    serialized = original.SerializeToString()

    self._testBoth(
        {
            "example_name": "in1",
            "serialized": ops.convert_to_tensor(serialized),
            "sequence_features": {
                "a": parsing_ops.FixedLenSequenceFeature((2,), dtypes.int64)
            }
        },
        expected_err=(errors_impl.OpError, r"Name: in1, Key: a, Index: 1."
                      r"  Number of int64 values != expected."
                      r"  values size: 3 but output shape: \[2\]"))

  def testSequenceExampleWithMissingFeatureListFails(self):
    original = sequence_example(feature_lists=feature_lists({}))

    # Test fails because we didn't add:
    #  feature_list_dense_defaults = {"a": None}
    self._testBoth(
        {
            "example_name": "in1",
            "serialized": ops.convert_to_tensor(original.SerializeToString()),
            "sequence_features": {
                "a": parsing_ops.FixedLenSequenceFeature((2,), dtypes.int64)
            }
        },
        expected_err=(
            errors_impl.OpError,
            "Name: in1, Feature list 'a' is required but could not be found."
            "  Did you mean to include it in"
            " feature_list_dense_missing_assumed_empty or"
            " feature_list_dense_defaults?"))

  def testSequenceExampleBatch(self):
    first = sequence_example(
        feature_lists=feature_lists({
            "a":
                feature_list([
                    int64_feature([-1, 0, 1]),
                    int64_feature([2, 3, 4]),
                    int64_feature([5, 6, 7]),
                    int64_feature([8, 9, 10]),
                ])
        }))
    second = sequence_example(
        feature_lists=feature_lists({
            "a": feature_list([
                int64_feature([21, 2, 11]),
            ])
        }))

    serialized = [first.SerializeToString(), second.SerializeToString()]

    expected_feature_list_output = {
        "a":
            np.array(
                [  # outermost dimension is example id
                    [  # middle dimension is time.
                        [[-1, 0, 1]],  # inside are 1x3 matrices
                        [[2, 3, 4]],
                        [[5, 6, 7]],
                        [[8, 9, 10]]
                    ],
                    [  # middle dimension is time.
                        [[21, 2, 11]],  # inside are 1x3 matrices
                        [[0, 0, 0]],  # additional entries are padded with 0
                        [[0, 0, 0]],
                        [[0, 0, 0]]
                    ]
                ],
                dtype=np.int64),
        "d":
            np.empty(shape=(2, 0, 5), dtype=np.float32),  # allowed_missing
    }

    self._test(
        {
            "example_names": ops.convert_to_tensor(["in1", "in2"]),
            "serialized": ops.convert_to_tensor(serialized),
            "sequence_features": {
                "a":
                    parsing_ops.FixedLenSequenceFeature((1, 3), dtypes.int64),
                "d":
                    parsing_ops.FixedLenSequenceFeature(
                        (5,), dtypes.float32, allow_missing=True),
            }
        },
        expected_feat_list_values=expected_feature_list_output,
        expected_length_values={
            "a": [4, 1],
            "d": [0, 0]
        },
        batch=True)


class DecodeJSONExampleTest(test.TestCase):

  def _testRoundTrip(self, examples):
    with self.cached_session() as sess:
      examples = np.array(examples, dtype=np.object)

      json_tensor = constant_op.constant(
          [json_format.MessageToJson(m) for m in examples.flatten()],
          shape=examples.shape,
          dtype=dtypes.string)
      binary_tensor = parsing_ops.decode_json_example(json_tensor)
      binary_val = sess.run(binary_tensor)

      if examples.shape:
        self.assertShapeEqual(binary_val, json_tensor)
        for input_example, output_binary in zip(
            np.array(examples).flatten(), binary_val.flatten()):
          output_example = example_pb2.Example()
          output_example.ParseFromString(output_binary)
          self.assertProtoEquals(input_example, output_example)
      else:
        output_example = example_pb2.Example()
        output_example.ParseFromString(binary_val)
        self.assertProtoEquals(examples.item(), output_example)

  def testEmptyTensor(self):
    self._testRoundTrip([])
    self._testRoundTrip([[], [], []])

  def testEmptyExamples(self):
    self._testRoundTrip([example(), example(), example()])

  def testDenseFeaturesScalar(self):
    self._testRoundTrip(
        example(features=features({
            "a": float_feature([1, 1, 3])
        })))

  def testDenseFeaturesVector(self):
    self._testRoundTrip([
        example(features=features({
            "a": float_feature([1, 1, 3])
        })),
        example(features=features({
            "a": float_feature([-1, -1, 2])
        })),
    ])

  def testDenseFeaturesMatrix(self):
    self._testRoundTrip([
        [example(features=features({
            "a": float_feature([1, 1, 3])
        }))],
        [example(features=features({
            "a": float_feature([-1, -1, 2])
        }))],
    ])

  def testSparseFeatures(self):
    self._testRoundTrip([
        example(features=features({
            "st_c": float_feature([3, 4])
        })),
        example(features=features({
            "st_c": float_feature([])
        })),
        example(features=features({
            "st_d": feature()
        })),
        example(
            features=features({
                "st_c": float_feature([1, 2, -1]),
                "st_d": bytes_feature([b"hi"])
            })),
    ])

  def testSerializedContainingBytes(self):
    aname = "a"
    bname = "b*has+a:tricky_name"
    self._testRoundTrip([
        example(
            features=features({
                aname: float_feature([1, 1]),
                bname: bytes_feature([b"b0_str"])
            })),
        example(
            features=features({
                aname: float_feature([-1, -1]),
                bname: bytes_feature([b"b1"])
            })),
    ])

  def testInvalidSyntax(self):
    with self.cached_session() as sess:
      json_tensor = constant_op.constant(["{]"])
      binary_tensor = parsing_ops.decode_json_example(json_tensor)
      with self.assertRaisesOpError("Error while parsing JSON"):
        sess.run(binary_tensor)


class ParseTensorOpTest(test.TestCase):

  def testToFloat32(self):
    with self.cached_session():
      expected = np.random.rand(3, 4, 5).astype(np.float32)
      tensor_proto = tensor_util.make_tensor_proto(expected)

      serialized = array_ops.placeholder(dtypes.string)
      tensor = parsing_ops.parse_tensor(serialized, dtypes.float32)

      result = tensor.eval(
          feed_dict={serialized: tensor_proto.SerializeToString()})

      self.assertAllEqual(expected, result)

  def testToUint8(self):
    with self.cached_session():
      expected = np.random.rand(3, 4, 5).astype(np.uint8)
      tensor_proto = tensor_util.make_tensor_proto(expected)

      serialized = array_ops.placeholder(dtypes.string)
      tensor = parsing_ops.parse_tensor(serialized, dtypes.uint8)

      result = tensor.eval(
          feed_dict={serialized: tensor_proto.SerializeToString()})

      self.assertAllEqual(expected, result)

  def testTypeMismatch(self):
    with self.cached_session():
      expected = np.random.rand(3, 4, 5).astype(np.uint8)
      tensor_proto = tensor_util.make_tensor_proto(expected)

      serialized = array_ops.placeholder(dtypes.string)
      tensor = parsing_ops.parse_tensor(serialized, dtypes.uint16)

      with self.assertRaisesOpError(
          r"Type mismatch between parsed tensor \(uint8\) and dtype "
          r"\(uint16\)"):
        tensor.eval(feed_dict={serialized: tensor_proto.SerializeToString()})

  def testInvalidInput(self):
    with self.cached_session():
      serialized = array_ops.placeholder(dtypes.string)
      tensor = parsing_ops.parse_tensor(serialized, dtypes.uint16)

      with self.assertRaisesOpError(
          "Could not parse `serialized` as TensorProto: 'bogus'"):
        tensor.eval(feed_dict={serialized: "bogus"})

      with self.assertRaisesOpError(
          r"Expected `serialized` to be a scalar, got shape: \[1\]"):
        tensor.eval(feed_dict={serialized: ["bogus"]})


if __name__ == "__main__":
  test.main()
