"""Tests for tensorflow.ops.parsing_ops."""

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
  for k, v in dict_tensors.iteritems():
    expected_v = expected_tensors[k]
    tf.logging.info("Comparing key: %s", k)
    if isinstance(v, tf.SparseTensor):
      # Three outputs for SparseTensor : indices, values, shape.
      tester.assertEqual([k, 3], [k, len(expected_v)])
      tester.assertAllEqual(flat_output[i], expected_v[0])
      tester.assertAllEqual(flat_output[i + 1], expected_v[1])
      tester.assertAllEqual(flat_output[i + 2], expected_v[2])
      i += 3
    else:
      # One output for standard Tensor.
      tester.assertAllEqual(flat_output[i], expected_v)
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
      self.assertEqual(len(dense_keys), len(dense_shapes))
      for (k, s) in zip(dense_keys, dense_shapes):
        self.assertEqual(tuple(out[k].get_shape().as_list()), (batch_size,) + s)
      for k in sparse_keys:
        self.assertEqual(tuple(out[k].indices.get_shape().as_list()), (None, 2))
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
    dense_keys = ["a", "b", "c"]
    dense_shapes = [(1, 3), (3, 3), (2,)]
    dense_types = [tf.int64, tf.string, tf.float32]
    dense_defaults = {
        "a": [0, 42, 0],
        "b": np.random.rand(3, 3).astype(np.str),
        "c": np.random.rand(2).astype(np.float32),
    }

    expected_st_a = (  # indices, values, shape
        np.empty((0, 2), dtype=np.int64),  # indices
        np.empty((0,), dtype=np.int64),  # sp_a is DT_INT64
        np.array([2, 0], dtype=np.int64))  # batch == 2, max_elems = 0

    expected_output = {
        "st_a": expected_st_a,
        "a": np.array(2 * [[dense_defaults["a"]]]),
        "b": np.array(2 * [dense_defaults["b"]]),
        "c": np.array(2 * [dense_defaults["c"]]),
    }

    self._test(
        {
            "names": np.empty((0,), dtype=np.str),
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
        "b": np.random.rand(3, 3).astype(np.str),
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
        expected_err_re="Name: failing, Key: a.  Number of float values")

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
            "st_d": bytes_feature(["hi"])
        }))
    ]

    serialized = [m.SerializeToString() for m in original]

    expected_st_c = (  # indices, values, shape
        np.array([[0, 0], [0, 1], [3, 0], [3, 1], [3, 2]], dtype=np.int64),
        np.array([3.0, 4.0, 1.0, 2.0, -1.0], dtype=np.float32),
        np.array([4, 3], dtype=np.int64))  # batch == 2, max_elems = 3

    expected_st_d = (  # indices, values, shape
        np.array([[3, 0]], dtype=np.int64),
        np.array(["hi"], dtype=np.str),
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
    original = [
        example(features=features({
            "a": float_feature([1, 1]),
            "b": bytes_feature(["b0_str"]),
        })),
        example(features=features({
            "a": float_feature([-1, -1]),
            "b": bytes_feature(["b1"]),
        }))
    ]

    serialized = [m.SerializeToString() for m in original]

    dense_shapes = [(1, 2, 1), (1, 1, 1, 1)]

    expected_output = {
        "a": np.array([[1, 1], [-1, -1]], dtype=np.float32).reshape(2, 1, 2, 1),
        "b": np.array(["b0_str", "b1"], dtype=np.str).reshape(2, 1, 1, 1, 1),
    }

    # No defaults, values required
    self._test(
        {
            "serialized": tf.convert_to_tensor(serialized),
            "dense_keys": ["a", "b"],
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
            "b": bytes_feature(["b1"]),
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
        "b": np.array(["tmp_str", "b1"], dtype=np.str).reshape(2, 1, 1, 1, 1),
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
        "b": np.random.rand(3, 3).astype(np.str),
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
        "b": np.random.rand(3, 3).astype(np.str),
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
            "names": "in1",
            "serialized": tf.convert_to_tensor(serialized),
            "dense_defaults": dense_defaults,
            "dense_types": dense_types,
            "sparse_keys": ["st_a"],
            "sparse_types": [tf.float32],
            "dense_keys": ["a", "b", "c"],
            "dense_shapes": dense_shapes
        }, expected_output)


if __name__ == "__main__":
  tf.test.main()
