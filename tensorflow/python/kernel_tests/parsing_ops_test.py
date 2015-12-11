# Copyright 2015 Google Inc. All Rights Reserved.
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

import itertools

import tensorflow.python.platform

import numpy as np
import tensorflow as tf

# Helpers for creating Example objects
example = tf.train.Example
feature = tf.train.Feature
features = lambda d: tf.train.Features(feature=d)
bytes_feature = lambda v: feature(bytes_list=tf.train.BytesList(value=v))
int64_feature = lambda v: feature(int64_list=tf.train.Int64List(value=v))
float_feature = lambda v: feature(float_list=tf.train.FloatList(value=v))
# Helpers for creating SequenceExample objects
feature_list = lambda l: tf.train.FeatureList(feature=l)
feature_lists = lambda d: tf.train.FeatureLists(feature_list=d)
sequence_example = tf.train.SequenceExample


def flatten(list_of_lists):
  """Flatten one level of nesting."""
  return itertools.chain.from_iterable(list_of_lists)


def flatten_values_tensors_or_sparse(tensors_list):
  """Flatten each SparseTensor object into 3 Tensors for session.run()."""
  return list(flatten([[v.indices, v.values, v.shape]
                       if isinstance(v, tf.SparseTensor) else [v]
                       for v in tensors_list]))


def _compare_output_to_expected(
    tester, dict_tensors, expected_tensors, flat_output):
  tester.assertEqual(set(dict_tensors.keys()), set(expected_tensors.keys()))

  i = 0  # Index into the flattened output of session.run()
  for k, v in dict_tensors.items():
    expected_v = expected_tensors[k]
    tf.logging.info("Comparing key: %s", k)
    if isinstance(v, tf.SparseTensor):
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


class ParseExampleTest(tf.test.TestCase):

  def _test(self, kwargs, expected_values=None, expected_err_re=None):
    with self.test_session() as sess:
      # Pull out some keys to check shape inference
      serialized = kwargs["serialized"]
      dense_keys = kwargs["dense_keys"] if "dense_keys" in kwargs else []
      sparse_keys = kwargs["sparse_keys"] if "sparse_keys" in kwargs else []
      dense_shapes = kwargs["dense_shapes"] if "dense_shapes" in kwargs else []

      # Returns dict w/ Tensors and SparseTensors
      out = tf.parse_example(**kwargs)

      # Check shapes; if serialized is a Tensor we need its size to
      # properly check.
      batch_size = (
          serialized.eval().size if isinstance(serialized, tf.Tensor)
          else np.asarray(serialized).size)
      if dense_shapes:
        self.assertEqual(len(dense_keys), len(dense_shapes))
        for (k, s) in zip(dense_keys, dense_shapes):
          self.assertEqual(
              tuple(out[k].get_shape().as_list()), (batch_size,) + s)
      for k in sparse_keys:
        self.assertEqual(
            tuple(out[k].indices.get_shape().as_list()), (None, 2))
        self.assertEqual(tuple(out[k].values.get_shape().as_list()), (None,))
        self.assertEqual(tuple(out[k].shape.get_shape().as_list()), (2,))

      # Check values
      result = flatten_values_tensors_or_sparse(out.values())  # flatten values
      if expected_err_re is None:
        tf_result = sess.run(result)
        _compare_output_to_expected(self, out, expected_values, tf_result)
      else:
        with self.assertRaisesOpError(expected_err_re):
          sess.run(result)

  def testEmptySerializedWithAllDefaults(self):
    cname = "c:has_a_tricky_name"
    dense_keys = ["a", "b", cname]
    dense_shapes = [(1, 3), (3, 3), (2,)]
    dense_types = [tf.int64, tf.string, tf.float32]
    dense_defaults = {
        "a": [0, 42, 0],
        "b": np.random.rand(3, 3).astype(bytes),
        cname: np.random.rand(2).astype(np.float32),
    }

    expected_st_a = (  # indices, values, shape
        np.empty((0, 2), dtype=np.int64),  # indices
        np.empty((0,), dtype=np.int64),  # sp_a is DT_INT64
        np.array([2, 0], dtype=np.int64))  # batch == 2, max_elems = 0

    expected_output = {
        "st_a": expected_st_a,
        "a": np.array(2 * [[dense_defaults["a"]]]),
        "b": np.array(2 * [dense_defaults["b"]]),
        cname: np.array(2 * [dense_defaults[cname]]),
    }

    self._test(
        {
            "names": np.empty((0,), dtype=bytes),
            # empty serialized input Examples
            "serialized": tf.convert_to_tensor(["", ""]),
            "dense_defaults": dense_defaults,
            "sparse_keys": ["st_a"],
            "sparse_types": [tf.int64],
            "dense_keys": dense_keys,
            "dense_types": dense_types,
            "dense_shapes": dense_shapes
        }, expected_output)

  def testEmptySerializedWithoutDefaultsShouldFail(self):
    dense_shapes = [(1, 3), (3, 3), (2,)]
    dense_defaults = {
        "a": [0, 42, 0],
        "b": np.random.rand(3, 3).astype(bytes),
        # Feature "c" is missing, since there's gaps it will cause failure.
    }
    self._test(
        {
            "serialized": ["", ""],  # empty serialized input Examples
            "names": ["in1", "in2"],
            "dense_defaults": dense_defaults,
            "sparse_keys": ["st_a"],
            "sparse_types": [tf.int64],
            "dense_keys": ["a", "b", "c"],
            "dense_types": [tf.int64, tf.string, tf.float32],
            "dense_shapes": dense_shapes
        },
        expected_err_re="Name: in1, Feature: c is required")

  def testDenseNotMatchingShapeShouldFail(self):
    dense_shapes = [(1, 3)]
    dense_defaults = {
        # no default!
    }

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
            "serialized": tf.convert_to_tensor(serialized),
            "names": names,
            "dense_defaults": dense_defaults,
            "dense_keys": ["a"],
            "dense_types": [tf.float32],
            "dense_shapes": dense_shapes,
        },
        expected_err_re="Name: failing, Key: a, Index: 1.  Number of float val")

  def testDenseDefaultNoShapeShouldFail(self):
    original = [
        example(features=features({
            "a": float_feature([1, 1, 3]),
        })),
    ]

    serialized = [m.SerializeToString() for m in original]

    self._test(
        {
            "serialized": tf.convert_to_tensor(serialized),
            "names": ["failing"],
            "dense_keys": ["a"],
            "dense_types": [tf.float32],
        },
        expected_err_re="Name: failing, Key: a, Index: 0.  Number of float val")

  def testDenseDefaultNoShapeOk(self):
    original = [
        example(features=features({
            "a": float_feature([1]),
        })),
        example(features=features({
            "a": float_feature([1]),
        }))
    ]

    serialized = [m.SerializeToString() for m in original]

    self._test(
        {
            "serialized": tf.convert_to_tensor(serialized),
            "names": ["passing", "passing"],
            "dense_keys": ["a"],
            "dense_types": [tf.float32],
        },
        {
            "a": np.array([1, 1], dtype=np.float32)
        })

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

    expected_st_c = (  # indices, values, shape
        np.array([[0, 0], [0, 1], [3, 0], [3, 1], [3, 2]], dtype=np.int64),
        np.array([3.0, 4.0, 1.0, 2.0, -1.0], dtype=np.float32),
        np.array([4, 3], dtype=np.int64))  # batch == 2, max_elems = 3

    expected_st_d = (  # indices, values, shape
        np.array([[3, 0]], dtype=np.int64),
        np.array(["hi"], dtype=bytes),
        np.array([4, 1], dtype=np.int64))  # batch == 2, max_elems = 1

    expected_output = {
        "st_c": expected_st_c,
        "st_d": expected_st_d,
    }

    self._test(
        {
            "serialized": tf.convert_to_tensor(serialized),
            "sparse_keys": ["st_c", "st_d"],
            "sparse_types": [tf.float32, tf.string],
        }, expected_output)

  def testSerializedContainingDense(self):
    bname = "b*has+a:tricky_name"
    original = [
        example(features=features({
            "a": float_feature([1, 1]),
            bname: bytes_feature([b"b0_str"]),
        })),
        example(features=features({
            "a": float_feature([-1, -1]),
            bname: bytes_feature([b"b1"]),
        }))
    ]

    serialized = [m.SerializeToString() for m in original]

    dense_shapes = [(1, 2, 1), (1, 1, 1, 1)]

    expected_output = {
        "a": np.array([[1, 1], [-1, -1]], dtype=np.float32).reshape(2, 1, 2, 1),
        bname: np.array(["b0_str", "b1"], dtype=bytes).reshape(2, 1, 1, 1, 1),
    }

    # No defaults, values required
    self._test(
        {
            "serialized": tf.convert_to_tensor(serialized),
            "dense_keys": ["a", bname],
            "dense_types": [tf.float32, tf.string],
            "dense_shapes": dense_shapes,
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
        "a": np.array([[1], [-1]], dtype=np.float32)  # 2x1 (column vector)
    }

    self._test(
        {
            "serialized": tf.convert_to_tensor(serialized),
            "dense_defaults": {"a": -1},
            "dense_shapes": [(1,)],
            "dense_keys": ["a"],
            "dense_types": [tf.float32],
        }, expected_output)

  def testSerializedContainingDenseWithDefaults(self):
    original = [
        example(features=features({
            "a": float_feature([1, 1]),
        })),
        example(features=features({
            "b": bytes_feature([b"b1"]),
        }))
    ]

    serialized = [m.SerializeToString() for m in original]

    dense_shapes = [(1, 2, 1), (1, 1, 1, 1)]
    dense_types = [tf.float32, tf.string]
    dense_defaults = {
        "a": [3.0, -3.0],
        "b": "tmp_str",
    }

    expected_output = {
        "a": np.array([[1, 1], [3, -3]], dtype=np.float32).reshape(2, 1, 2, 1),
        "b": np.array(["tmp_str", "b1"], dtype=bytes).reshape(2, 1, 1, 1, 1),
    }

    self._test(
        {
            "serialized": tf.convert_to_tensor(serialized),
            "dense_defaults": dense_defaults,
            "dense_keys": ["a", "b"],
            "dense_types": dense_types,
            "dense_shapes": dense_shapes,
        }, expected_output)

  def testSerializedContainingSparseAndDenseWithNoDefault(self):
    dense_defaults = {
        "a": [1, 2, 3],
        "b": np.random.rand(3, 3).astype(bytes),
        # Feature "c" must be provided
    }
    dense_shapes = [(1, 3), (3, 3), (2,)]

    expected_st_a = (  # indices, values, shape
        np.empty((0, 2), dtype=np.int64),  # indices
        np.empty((0,), dtype=np.int64),  # sp_a is DT_INT64
        np.array([2, 0], dtype=np.int64))  # batch == 2, max_elems = 0

    original = [
        example(features=features({
            "c": float_feature([3, 4])
        })),
        example(features=features({
            "c": float_feature([1, 2])
        }))
    ]

    names = ["in1", "in2"]
    serialized = [m.SerializeToString() for m in original]

    expected_output = {
        "st_a": expected_st_a,
        "a": np.array(2 * [[dense_defaults["a"]]]),
        "b": np.array(2 * [dense_defaults["b"]]),
        "c": np.array([[3, 4], [1, 2]], dtype=np.float32),
    }

    self._test(
        {
            "names": names,
            "serialized": tf.convert_to_tensor(serialized),
            "dense_defaults": dense_defaults,
            "sparse_keys": ["st_a"],
            "sparse_types": [tf.int64],
            "dense_keys": ["a", "b", "c"],
            "dense_types": [tf.int64, tf.string, tf.float32],
            "dense_shapes": dense_shapes
        }, expected_output)


class ParseSingleExampleTest(tf.test.TestCase):

  def _test(self, kwargs, expected_values=None, expected_err_re=None):
    with self.test_session() as sess:
      # Pull out some keys to check shape inference
      dense_keys = kwargs["dense_keys"] if "dense_keys" in kwargs else []
      sparse_keys = kwargs["sparse_keys"] if "sparse_keys" in kwargs else []
      dense_shapes = kwargs["dense_shapes"] if "dense_shapes" in kwargs else []

      # Returns dict w/ Tensors and SparseTensors
      out = tf.parse_single_example(**kwargs)

      # Check shapes
      self.assertEqual(len(dense_keys), len(dense_shapes))
      for (k, s) in zip(dense_keys, dense_shapes):
        self.assertEqual(tuple(out[k].get_shape()), s)
      for k in sparse_keys:
        self.assertEqual(tuple(out[k].indices.get_shape().as_list()), (None, 1))
        self.assertEqual(tuple(out[k].values.get_shape().as_list()), (None,))
        self.assertEqual(tuple(out[k].shape.get_shape().as_list()), (1,))

      # Check values
      result = flatten_values_tensors_or_sparse(out.values())  # flatten values
      if expected_err_re is None:
        tf_result = sess.run(result)
        _compare_output_to_expected(self, out, expected_values, tf_result)
      else:
        with self.assertRaisesOpError(expected_err_re):
          sess.run(result)

  def testSingleExampleWithSparseAndDense(self):
    dense_types = [tf.int64, tf.string, tf.float32]
    dense_shapes = [(1, 3), (3, 3), (2,)]
    dense_defaults = {
        "a": [1, 2, 3],
        "b": np.random.rand(3, 3).astype(bytes),
        # Feature "c" must be provided
    }

    original = example(features=features(
        {"c": float_feature([3, 4]),
         "st_a": float_feature([3.0, 4.0])}))

    serialized = original.SerializeToString()

    expected_st_a = (
        np.array([[0], [1]], dtype=np.int64),  # indices
        np.array([3.0, 4.0], dtype=np.float32),  # values
        np.array([2], dtype=np.int64))  # shape: max_values = 2

    expected_output = {
        "st_a": expected_st_a,
        "a": [dense_defaults["a"]],
        "b": dense_defaults["b"],
        "c": np.array([3, 4], dtype=np.float32),
    }

    self._test(
        {
            "names": tf.convert_to_tensor("in1"),
            "serialized": tf.convert_to_tensor(serialized),
            "dense_defaults": dense_defaults,
            "dense_types": dense_types,
            "sparse_keys": ["st_a"],
            "sparse_types": [tf.float32],
            "dense_keys": ["a", "b", "c"],
            "dense_shapes": dense_shapes
        }, expected_output)


class ParseSequenceExampleTest(tf.test.TestCase):

  def testCreateSequenceExample(self):
    value = sequence_example(
        context=features({
            "global_feature": float_feature([1, 2, 3]),
            }),
        feature_lists=feature_lists({
            "repeated_feature_2_frames": feature_list([
                bytes_feature([b"a", b"b", b"c"]),
                bytes_feature([b"a", b"d", b"e"])]),
            "repeated_feature_3_frames": feature_list([
                int64_feature([3, 4, 5, 6, 7]),
                int64_feature([-1, 0, 0, 0, 0]),
                int64_feature([1, 2, 3, 4, 5])])
            }))
    value.SerializeToString()  # Smoke test

  def _test(self, kwargs, expected_context_values=None,
            expected_feat_list_values=None, expected_err_re=None):
    expected_context_values = expected_context_values or {}
    expected_feat_list_values = expected_feat_list_values or {}
    with self.test_session() as sess:
      # Pull out some keys to check shape inference
      context_dense_keys = (
          kwargs["context_dense_keys"]
          if "context_dense_keys" in kwargs else [])
      context_sparse_keys = (
          kwargs["context_sparse_keys"]
          if "context_sparse_keys" in kwargs else [])
      context_dense_shapes = (
          kwargs["context_dense_shapes"]
          if "context_dense_shapes" in kwargs else [])
      feature_list_dense_keys = (
          kwargs["feature_list_dense_keys"]
          if "feature_list_dense_keys" in kwargs else [])
      feature_list_dense_shapes = (
          kwargs["feature_list_dense_shapes"]
          if "feature_list_dense_shapes" in kwargs else [])

      # Returns dict w/ Tensors and SparseTensors
      (context_out, feat_list_out) = tf.parse_single_sequence_example(**kwargs)

      # Check shapes; if serialized is a Tensor we need its size to
      # properly check.
      if context_dense_shapes:
        self.assertEqual(len(context_dense_keys), len(context_dense_shapes))
        for (k, s) in zip(context_dense_keys, context_dense_shapes):
          self.assertEqual(
              tuple(context_out[k].get_shape().as_list()), s)
      for k in context_sparse_keys:
        self.assertEqual(
            tuple(context_out[k].indices.get_shape().as_list()), (None, 1))
        self.assertEqual(
            tuple(context_out[k].values.get_shape().as_list()), (None,))
        self.assertEqual(
            tuple(context_out[k].shape.get_shape().as_list()), (1,))
      if feature_list_dense_shapes:
        self.assertEqual(
            len(feature_list_dense_keys), len(feature_list_dense_shapes))
        for (k, s) in zip(feature_list_dense_keys, feature_list_dense_shapes):
          self.assertEqual(
              tuple(feat_list_out[k].get_shape().as_list()), (None,) + s)

      # Check values
      context_result = flatten_values_tensors_or_sparse(
          context_out.values())  # flatten values
      feature_list_result = flatten_values_tensors_or_sparse(
          feat_list_out.values())
      if expected_err_re is None:
        tf_context_result = sess.run(context_result)
        tf_feat_list_result = sess.run(feature_list_result)
        _compare_output_to_expected(
            self, context_out, expected_context_values, tf_context_result)
        _compare_output_to_expected(
            self, feat_list_out, expected_feat_list_values, tf_feat_list_result)
      else:
        with self.assertRaisesOpError(expected_err_re):
          sess.run(context_result)

  def testSequenceExampleWithSparseAndDenseContext(self):
    context_dense_types = [tf.int64, tf.string, tf.float32]
    context_dense_shapes = [(1, 3), (3, 3), (2,)]
    context_dense_defaults = {
        "a": [1, 2, 3],
        "b": np.random.rand(3, 3).astype(bytes),
        # Feature "c" must be provided
    }

    original = sequence_example(context=features(
        {"c": float_feature([3, 4]),
         "st_a": float_feature([3.0, 4.0])}))

    serialized = original.SerializeToString()

    expected_st_a = (
        np.array([[0], [1]], dtype=np.int64),  # indices
        np.array([3.0, 4.0], dtype=np.float32),  # values
        np.array([2], dtype=np.int64))  # shape: num_features = 2

    expected_context_output = {
        "st_a": expected_st_a,
        "a": [context_dense_defaults["a"]],
        "b": context_dense_defaults["b"],
        "c": np.array([3, 4], dtype=np.float32),
    }

    self._test(
        {
            "debug_name": "in1",
            "serialized": tf.convert_to_tensor(serialized),
            "context_dense_defaults": context_dense_defaults,
            "context_dense_types": context_dense_types,
            "context_sparse_keys": ["st_a"],
            "context_sparse_types": [tf.float32],
            "context_dense_keys": ["a", "b", "c"],
            "context_dense_shapes": context_dense_shapes
        }, expected_context_values=expected_context_output)

  def testSequenceExampleWithMultipleSizeFeatureLists(self):
    feature_list_dense_keys = ["a", "b", "c", "d"]
    feature_list_dense_types = [tf.int64, tf.string, tf.float32, tf.float32]
    feature_list_dense_shapes = [(1, 3), (2, 2), (2,), (5,)]

    original = sequence_example(feature_lists=feature_lists({
        "a": feature_list([
            int64_feature([-1, 0, 1]),
            int64_feature([2, 3, 4]),
            int64_feature([5, 6, 7]),
            int64_feature([8, 9, 10]),]),
        "b": feature_list([
            bytes_feature([b"r00", b"r01", b"r10", b"r11"])]),
        "c": feature_list([
            float_feature([3, 4]),
            float_feature([-1, 2])]),
        }))

    serialized = original.SerializeToString()

    expected_feature_list_output = {
        "a": np.array([  # outer dimension is time.
            [[-1, 0, 1]],  # inside are 1x3 matrices
            [[2, 3, 4]],
            [[5, 6, 7]],
            [[8, 9, 10]]], dtype=np.int64),
        "b": np.array([  # outer dimension is time, inside are 2x2 matrices
            [[b"r00", b"r01"], [b"r10", b"r11"]]], dtype=np.str),
        "c": np.array([  # outer dimension is time, inside are 2-vectors
            [3, 4],
            [-1, 2]], dtype=np.float32),
        "d": np.empty(shape=(0, 5), dtype=np.float32),  # empty_allowed_missing
        }

    self._test(
        {
            "debug_name": "in1",
            "serialized": tf.convert_to_tensor(serialized),
            "feature_list_dense_types": feature_list_dense_types,
            "feature_list_dense_keys": feature_list_dense_keys,
            "feature_list_dense_shapes": feature_list_dense_shapes,
            "feature_list_dense_defaults": {"d": None},
        }, expected_feat_list_values=expected_feature_list_output)

  def testSequenceExampleListWithInconsistentDataFails(self):
    feature_list_dense_types = [tf.int64]
    feature_list_dense_shapes = [(2,)]

    original = sequence_example(feature_lists=feature_lists({
        "a": feature_list([
            int64_feature([-1, 0]),
            float_feature([2, 3])])
        }))

    serialized = original.SerializeToString()

    self._test(
        {
            "debug_name": "in1",
            "serialized": tf.convert_to_tensor(serialized),
            "feature_list_dense_types": feature_list_dense_types,
            "feature_list_dense_keys": ["a"],
            "feature_list_dense_shapes": feature_list_dense_shapes
        },
        expected_err_re=("Feature list: a, Index: 1.  Data types don't match. "
                         "Expected type: int64"))

  def testSequenceExampleListWithWrongDataTypeFails(self):
    feature_list_dense_types = [tf.int64]
    feature_list_dense_shapes = [(2,)]

    original = sequence_example(feature_lists=feature_lists({
        "a": feature_list([
            float_feature([2, 3])])
        }))

    serialized = original.SerializeToString()

    self._test(
        {
            "debug_name": "in1",
            "serialized": tf.convert_to_tensor(serialized),
            "feature_list_dense_types": feature_list_dense_types,
            "feature_list_dense_keys": ["a"],
            "feature_list_dense_shapes": feature_list_dense_shapes
        },
        expected_err_re=("Feature list: a, Index: 0.  Data types don't match. "
                         "Expected type: int64"))

  def testSequenceExampleListWithWrongShapeFails(self):
    feature_list_dense_types = [tf.int64]
    feature_list_dense_shapes = [(2,)]

    original = sequence_example(feature_lists=feature_lists({
        "a": feature_list([
            int64_feature([2, 3]),
            int64_feature([2, 3, 4])]),
        }))

    serialized = original.SerializeToString()

    self._test(
        {
            "debug_name": "in1",
            "serialized": tf.convert_to_tensor(serialized),
            "feature_list_dense_types": feature_list_dense_types,
            "feature_list_dense_keys": ["a"],
            "feature_list_dense_shapes": feature_list_dense_shapes
        },
        expected_err_re=(r"Name: in1, Key: a, Index: 1.  "
                         r"Number of int64 values != expected.  "
                         r"values size: 3 but output shape: \[2\]"))

  def testSequenceExampleWithMissingFeatureListFails(self):
    feature_list_dense_types = [tf.int64]
    feature_list_dense_shapes = [(2,)]

    original = sequence_example(feature_lists=feature_lists({}))

    serialized = original.SerializeToString()

    # Test fails because we didn't add:
    #  feature_list_dense_defaults = {"a": None}
    self._test(
        {
            "debug_name": "in1",
            "serialized": tf.convert_to_tensor(serialized),
            "feature_list_dense_types": feature_list_dense_types,
            "feature_list_dense_keys": ["a"],
            "feature_list_dense_shapes": feature_list_dense_shapes
        },
        expected_err_re=(
            "Name: in1, Feature list 'a' is required but could not be found.  "
            "Did you mean to include it in "
            "feature_list_dense_missing_assumed_empty or "
            "feature_list_dense_defaults?"))


if __name__ == "__main__":
  tf.test.main()
