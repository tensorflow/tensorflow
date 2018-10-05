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

"""Parsing Ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import re

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_parsing_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
# go/tf-wildcard-import
# pylint: disable=wildcard-import,undefined-variable
from tensorflow.python.ops.gen_parsing_ops import *
# pylint: enable=wildcard-import,undefined-variable
from tensorflow.python.platform import tf_logging
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export


ops.NotDifferentiable("DecodeRaw")
ops.NotDifferentiable("ParseTensor")
ops.NotDifferentiable("SerializeTensor")
ops.NotDifferentiable("StringToNumber")


@tf_export("io.VarLenFeature", "VarLenFeature")
class VarLenFeature(collections.namedtuple("VarLenFeature", ["dtype"])):
  """Configuration for parsing a variable-length input feature.

  Fields:
    dtype: Data type of input.
  """
  pass


@tf_export("io.SparseFeature", "SparseFeature")
class SparseFeature(
    collections.namedtuple(
        "SparseFeature",
        ["index_key", "value_key", "dtype", "size", "already_sorted"])):
  """Configuration for parsing a sparse input feature from an `Example`.

  Note, preferably use `VarLenFeature` (possibly in combination with a
  `SequenceExample`) in order to parse out `SparseTensor`s instead of
  `SparseFeature` due to its simplicity.

  Closely mimicking the `SparseTensor` that will be obtained by parsing an
  `Example` with a `SparseFeature` config, a `SparseFeature` contains a

  * `value_key`: The name of key for a `Feature` in the `Example` whose parsed
    `Tensor` will be the resulting `SparseTensor.values`.

  * `index_key`: A list of names - one for each dimension in the resulting
    `SparseTensor` whose `indices[i][dim]` indicating the position of
    the `i`-th value in the `dim` dimension will be equal to the `i`-th value in
    the Feature with key named `index_key[dim]` in the `Example`.

  * `size`: A list of ints for the resulting `SparseTensor.dense_shape`.

  For example, we can represent the following 2D `SparseTensor`

  ```python
  SparseTensor(indices=[[3, 1], [20, 0]],
               values=[0.5, -1.0]
               dense_shape=[100, 3])
  ```

  with an `Example` input proto

  ```python
  features {
    feature { key: "val" value { float_list { value: [ 0.5, -1.0 ] } } }
    feature { key: "ix0" value { int64_list { value: [ 3, 20 ] } } }
    feature { key: "ix1" value { int64_list { value: [ 1, 0 ] } } }
  }
  ```

  and `SparseFeature` config with 2 `index_key`s

  ```python
  SparseFeature(index_key=["ix0", "ix1"],
                value_key="val",
                dtype=tf.float32,
                size=[100, 3])
  ```

  Fields:
    index_key: A single string name or a list of string names of index features.
      For each key the underlying feature's type must be `int64` and its length
      must always match that of the `value_key` feature.
      To represent `SparseTensor`s with a `dense_shape` of `rank` higher than 1
      a list of length `rank` should be used.
    value_key: Name of value feature.  The underlying feature's type must
      be `dtype` and its length must always match that of all the `index_key`s'
      features.
    dtype: Data type of the `value_key` feature.
    size: A Python int or list thereof specifying the dense shape. Should be a
      list if and only if `index_key` is a list. In that case the list must be
      equal to the length of `index_key`. Each for each entry `i` all values in
      the `index_key`[i] feature must be in `[0, size[i])`.
    already_sorted: A Python boolean to specify whether the values in
      `value_key` are already sorted by their index position. If so skip
      sorting. False by default (optional).
  """

  def __new__(cls, index_key, value_key, dtype, size, already_sorted=False):
    return super(SparseFeature, cls).__new__(
        cls, index_key, value_key, dtype, size, already_sorted)


@tf_export("io.FixedLenFeature", "FixedLenFeature")
class FixedLenFeature(collections.namedtuple(
    "FixedLenFeature", ["shape", "dtype", "default_value"])):
  """Configuration for parsing a fixed-length input feature.

  To treat sparse input as dense, provide a `default_value`; otherwise,
  the parse functions will fail on any examples missing this feature.

  Fields:
    shape: Shape of input data.
    dtype: Data type of input.
    default_value: Value to be used if an example is missing this feature. It
        must be compatible with `dtype` and of the specified `shape`.
  """

  def __new__(cls, shape, dtype, default_value=None):
    return super(FixedLenFeature, cls).__new__(
        cls, shape, dtype, default_value)


@tf_export("io.FixedLenSequenceFeature", "FixedLenSequenceFeature")
class FixedLenSequenceFeature(collections.namedtuple(
    "FixedLenSequenceFeature",
    ["shape", "dtype", "allow_missing", "default_value"])):
  """Configuration for parsing a variable-length input feature into a `Tensor`.

  The resulting `Tensor` of parsing a single `SequenceExample` or `Example` has
  a static `shape` of `[None] + shape` and the specified `dtype`.
  The resulting `Tensor` of parsing a `batch_size` many `Example`s has
  a static `shape` of `[batch_size, None] + shape` and the specified `dtype`.
  The entries in the `batch` from different `Examples` will be padded with
  `default_value` to the maximum length present in the `batch`.

  To treat a sparse input as dense, provide `allow_missing=True`; otherwise,
  the parse functions will fail on any examples missing this feature.

  Fields:
    shape: Shape of input data for dimension 2 and higher. First dimension is
      of variable length `None`.
    dtype: Data type of input.
    allow_missing: Whether to allow this feature to be missing from a feature
      list item. Is available only for parsing `SequenceExample` not for
      parsing `Examples`.
    default_value: Scalar value to be used to pad multiple `Example`s to their
      maximum length. Irrelevant for parsing a single `Example` or
      `SequenceExample`. Defaults to "" for dtype string and 0 otherwise
      (optional).
  """

  def __new__(cls, shape, dtype, allow_missing=False, default_value=None):
    return super(FixedLenSequenceFeature, cls).__new__(
        cls, shape, dtype, allow_missing, default_value)


def _features_to_raw_params(features, types):
  """Split feature tuples into raw params used by `gen_parsing_ops`.

  Args:
    features: A `dict` mapping feature keys to objects of a type in `types`.
    types: Type of features to allow, among `FixedLenFeature`, `VarLenFeature`,
      `SparseFeature`, and `FixedLenSequenceFeature`.

  Returns:
    Tuple of `sparse_keys`, `sparse_types`, `dense_keys`, `dense_types`,
      `dense_defaults`, `dense_shapes`.

  Raises:
    ValueError: if `features` contains an item not in `types`, or an invalid
        feature.
  """
  sparse_keys = []
  sparse_types = []
  dense_keys = []
  dense_types = []
  # When the graph is built twice, multiple dense_defaults in a normal dict
  # could come out in different orders. This will fail the _e2e_test which
  # expects exactly the same graph.
  # OrderedDict which preserves the order can solve the problem.
  dense_defaults = collections.OrderedDict()
  dense_shapes = []
  if features:
    # NOTE: We iterate over sorted keys to keep things deterministic.
    for key in sorted(features.keys()):
      feature = features[key]
      if isinstance(feature, VarLenFeature):
        if VarLenFeature not in types:
          raise ValueError("Unsupported VarLenFeature %s." % feature)
        if not feature.dtype:
          raise ValueError("Missing type for feature %s." % key)
        sparse_keys.append(key)
        sparse_types.append(feature.dtype)
      elif isinstance(feature, SparseFeature):
        if SparseFeature not in types:
          raise ValueError("Unsupported SparseFeature %s." % feature)

        if not feature.index_key:
          raise ValueError(
              "Missing index_key for SparseFeature %s." % feature)
        if not feature.value_key:
          raise ValueError(
              "Missing value_key for SparseFeature %s." % feature)
        if not feature.dtype:
          raise ValueError("Missing type for feature %s." % key)
        index_keys = feature.index_key
        if isinstance(index_keys, str):
          index_keys = [index_keys]
        elif len(index_keys) > 1:
          tf_logging.warning("SparseFeature is a complicated feature config "
                             "and should only be used after careful "
                             "consideration of VarLenFeature.")
        for index_key in sorted(index_keys):
          if index_key in sparse_keys:
            dtype = sparse_types[sparse_keys.index(index_key)]
            if dtype != dtypes.int64:
              raise ValueError("Conflicting type %s vs int64 for feature %s." %
                               (dtype, index_key))
          else:
            sparse_keys.append(index_key)
            sparse_types.append(dtypes.int64)
        if feature.value_key in sparse_keys:
          dtype = sparse_types[sparse_keys.index(feature.value_key)]
          if dtype != feature.dtype:
            raise ValueError("Conflicting type %s vs %s for feature %s." % (
                dtype, feature.dtype, feature.value_key))
        else:
          sparse_keys.append(feature.value_key)
          sparse_types.append(feature.dtype)
      elif isinstance(feature, FixedLenFeature):
        if FixedLenFeature not in types:
          raise ValueError("Unsupported FixedLenFeature %s." % feature)
        if not feature.dtype:
          raise ValueError("Missing type for feature %s." % key)
        if feature.shape is None:
          raise ValueError("Missing shape for feature %s." % key)
        feature_tensor_shape = tensor_shape.as_shape(feature.shape)
        if (feature.shape and feature_tensor_shape.ndims and
            feature_tensor_shape.dims[0].value is None):
          raise ValueError("First dimension of shape for feature %s unknown. "
                           "Consider using FixedLenSequenceFeature." % key)
        if (feature.shape is not None and
            not feature_tensor_shape.is_fully_defined()):
          raise ValueError("All dimensions of shape for feature %s need to be "
                           "known but received %s." % (key, str(feature.shape)))
        dense_keys.append(key)
        dense_shapes.append(feature.shape)
        dense_types.append(feature.dtype)
        if feature.default_value is not None:
          dense_defaults[key] = feature.default_value
      elif isinstance(feature, FixedLenSequenceFeature):
        if FixedLenSequenceFeature not in types:
          raise ValueError("Unsupported FixedLenSequenceFeature %s." % feature)
        if not feature.dtype:
          raise ValueError("Missing type for feature %s." % key)
        if feature.shape is None:
          raise ValueError("Missing shape for feature %s." % key)
        dense_keys.append(key)
        dense_shapes.append(feature.shape)
        dense_types.append(feature.dtype)
        if feature.allow_missing:
          dense_defaults[key] = None
        if feature.default_value is not None:
          dense_defaults[key] = feature.default_value
      else:
        raise ValueError("Invalid feature %s:%s." % (key, feature))
  return (
      sparse_keys, sparse_types, dense_keys, dense_types, dense_defaults,
      dense_shapes)


def _construct_sparse_tensors_for_sparse_features(features, tensor_dict):
  """Merges SparseTensors of indices and values of SparseFeatures.

  Constructs new dict based on `tensor_dict`. For `SparseFeatures` in the values
  of `features` expects their `index_key`s and `index_value`s to be present in
  `tensor_dict` mapping to `SparseTensor`s. Constructs a single `SparseTensor`
  from them, and adds it to the result with the key from `features`.
  Copies other keys and values from `tensor_dict` with keys present in
  `features`.

  Args:
    features: A `dict` mapping feature keys to `SparseFeature` values.
      Values of other types will be ignored.
    tensor_dict: A `dict` mapping feature keys to `Tensor` and `SparseTensor`
      values. Expected to contain keys of the `SparseFeature`s' `index_key`s and
      `value_key`s and mapping them to `SparseTensor`s.
  Returns:
    A `dict` mapping feature keys to `Tensor` and `SparseTensor` values. Similar
    to `tensor_dict` except each `SparseFeature`s in `features` results in a
    single `SparseTensor`.
  """
  tensor_dict = dict(tensor_dict)  # Do not modify argument passed in.
  # Construct SparseTensors for SparseFeatures.
  for key in sorted(features.keys()):
    feature = features[key]
    if isinstance(feature, SparseFeature):
      if isinstance(feature.index_key, str):
        sp_ids = tensor_dict[feature.index_key]
      else:
        sp_ids = [tensor_dict[index_key] for index_key in feature.index_key]
      sp_values = tensor_dict[feature.value_key]
      tensor_dict[key] = sparse_ops.sparse_merge(
          sp_ids,
          sp_values,
          vocab_size=feature.size,
          already_sorted=feature.already_sorted)
  # Remove tensors from dictionary that were only used to construct
  # SparseTensors for SparseFeature.
  for key in set(tensor_dict) - set(features):
    del tensor_dict[key]
  return tensor_dict


def _prepend_none_dimension(features):
  if features:
    modified_features = dict(features)  # Create a copy to modify
    for key, feature in features.items():
      if isinstance(feature, FixedLenSequenceFeature):
        if not feature.allow_missing:
          raise ValueError("Unsupported: FixedLenSequenceFeature requires "
                           "allow_missing to be True.")
        modified_features[key] = FixedLenSequenceFeature(
            [None] + list(feature.shape),
            feature.dtype,
            feature.allow_missing,
            feature.default_value)
    return modified_features
  else:
    return features


@tf_export("io.parse_example", "parse_example")
def parse_example(serialized, features, name=None, example_names=None):
  # pylint: disable=line-too-long
  """Parses `Example` protos into a `dict` of tensors.

  Parses a number of serialized [`Example`](https://www.tensorflow.org/code/tensorflow/core/example/example.proto)
  protos given in `serialized`. We refer to `serialized` as a batch with
  `batch_size` many entries of individual `Example` protos.

  `example_names` may contain descriptive names for the corresponding serialized
  protos. These may be useful for debugging purposes, but they have no effect on
  the output. If not `None`, `example_names` must be the same length as
  `serialized`.

  This op parses serialized examples into a dictionary mapping keys to `Tensor`
  and `SparseTensor` objects. `features` is a dict from keys to `VarLenFeature`,
  `SparseFeature`, and `FixedLenFeature` objects. Each `VarLenFeature`
  and `SparseFeature` is mapped to a `SparseTensor`, and each
  `FixedLenFeature` is mapped to a `Tensor`.

  Each `VarLenFeature` maps to a `SparseTensor` of the specified type
  representing a ragged matrix. Its indices are `[batch, index]` where `batch`
  identifies the example in `serialized`, and `index` is the value's index in
  the list of values associated with that feature and example.

  Each `SparseFeature` maps to a `SparseTensor` of the specified type
  representing a Tensor of `dense_shape` `[batch_size] + SparseFeature.size`.
  Its `values` come from the feature in the examples with key `value_key`.
  A `values[i]` comes from a position `k` in the feature of an example at batch
  entry `batch`. This positional information is recorded in `indices[i]` as
  `[batch, index_0, index_1, ...]` where `index_j` is the `k-th` value of
  the feature in the example at with key `SparseFeature.index_key[j]`.
  In other words, we split the indices (except the first index indicating the
  batch entry) of a `SparseTensor` by dimension into different features of the
  `Example`. Due to its complexity a `VarLenFeature` should be preferred over a
  `SparseFeature` whenever possible.

  Each `FixedLenFeature` `df` maps to a `Tensor` of the specified type (or
  `tf.float32` if not specified) and shape `(serialized.size(),) + df.shape`.

  `FixedLenFeature` entries with a `default_value` are optional. With no default
  value, we will fail if that `Feature` is missing from any example in
  `serialized`.

  Each `FixedLenSequenceFeature` `df` maps to a `Tensor` of the specified type
  (or `tf.float32` if not specified) and shape
  `(serialized.size(), None) + df.shape`.
  All examples in `serialized` will be padded with `default_value` along the
  second dimension.

  Examples:

  For example, if one expects a `tf.float32` `VarLenFeature` `ft` and three
  serialized `Example`s are provided:

  ```
  serialized = [
    features
      { feature { key: "ft" value { float_list { value: [1.0, 2.0] } } } },
    features
      { feature []},
    features
      { feature { key: "ft" value { float_list { value: [3.0] } } }
  ]
  ```

  then the output will look like:

  ```python
  {"ft": SparseTensor(indices=[[0, 0], [0, 1], [2, 0]],
                      values=[1.0, 2.0, 3.0],
                      dense_shape=(3, 2)) }
  ```

  If instead a `FixedLenSequenceFeature` with `default_value = -1.0` and
  `shape=[]` is used then the output will look like:

  ```python
  {"ft": [[1.0, 2.0], [3.0, -1.0]]}
  ```

  Given two `Example` input protos in `serialized`:

  ```
  [
    features {
      feature { key: "kw" value { bytes_list { value: [ "knit", "big" ] } } }
      feature { key: "gps" value { float_list { value: [] } } }
    },
    features {
      feature { key: "kw" value { bytes_list { value: [ "emmy" ] } } }
      feature { key: "dank" value { int64_list { value: [ 42 ] } } }
      feature { key: "gps" value { } }
    }
  ]
  ```

  And arguments

  ```
  example_names: ["input0", "input1"],
  features: {
      "kw": VarLenFeature(tf.string),
      "dank": VarLenFeature(tf.int64),
      "gps": VarLenFeature(tf.float32),
  }
  ```

  Then the output is a dictionary:

  ```python
  {
    "kw": SparseTensor(
        indices=[[0, 0], [0, 1], [1, 0]],
        values=["knit", "big", "emmy"]
        dense_shape=[2, 2]),
    "dank": SparseTensor(
        indices=[[1, 0]],
        values=[42],
        dense_shape=[2, 1]),
    "gps": SparseTensor(
        indices=[],
        values=[],
        dense_shape=[2, 0]),
  }
  ```

  For dense results in two serialized `Example`s:

  ```
  [
    features {
      feature { key: "age" value { int64_list { value: [ 0 ] } } }
      feature { key: "gender" value { bytes_list { value: [ "f" ] } } }
     },
     features {
      feature { key: "age" value { int64_list { value: [] } } }
      feature { key: "gender" value { bytes_list { value: [ "f" ] } } }
    }
  ]
  ```

  We can use arguments:

  ```
  example_names: ["input0", "input1"],
  features: {
      "age": FixedLenFeature([], dtype=tf.int64, default_value=-1),
      "gender": FixedLenFeature([], dtype=tf.string),
  }
  ```

  And the expected output is:

  ```python
  {
    "age": [[0], [-1]],
    "gender": [["f"], ["f"]],
  }
  ```

  An alternative to `VarLenFeature` to obtain a `SparseTensor` is
  `SparseFeature`. For example, given two `Example` input protos in
  `serialized`:

  ```
  [
    features {
      feature { key: "val" value { float_list { value: [ 0.5, -1.0 ] } } }
      feature { key: "ix" value { int64_list { value: [ 3, 20 ] } } }
    },
    features {
      feature { key: "val" value { float_list { value: [ 0.0 ] } } }
      feature { key: "ix" value { int64_list { value: [ 42 ] } } }
    }
  ]
  ```

  And arguments

  ```
  example_names: ["input0", "input1"],
  features: {
      "sparse": SparseFeature(
          index_key="ix", value_key="val", dtype=tf.float32, size=100),
  }
  ```

  Then the output is a dictionary:

  ```python
  {
    "sparse": SparseTensor(
        indices=[[0, 3], [0, 20], [1, 42]],
        values=[0.5, -1.0, 0.0]
        dense_shape=[2, 100]),
  }
  ```

  Args:
    serialized: A vector (1-D Tensor) of strings, a batch of binary
      serialized `Example` protos.
    features: A `dict` mapping feature keys to `FixedLenFeature`,
      `VarLenFeature`, and `SparseFeature` values.
    name: A name for this operation (optional).
    example_names: A vector (1-D Tensor) of strings (optional), the names of
      the serialized protos in the batch.

  Returns:
    A `dict` mapping feature keys to `Tensor` and `SparseTensor` values.

  Raises:
    ValueError: if any feature is invalid.
  """
  if not features:
    raise ValueError("Missing: features was %s." % features)
  features = _prepend_none_dimension(features)
  (sparse_keys, sparse_types, dense_keys, dense_types, dense_defaults,
   dense_shapes) = _features_to_raw_params(
       features,
       [VarLenFeature, SparseFeature, FixedLenFeature, FixedLenSequenceFeature])
  outputs = _parse_example_raw(
      serialized, example_names, sparse_keys, sparse_types, dense_keys,
      dense_types, dense_defaults, dense_shapes, name)
  return _construct_sparse_tensors_for_sparse_features(features, outputs)


def _parse_example_raw(serialized,
                       names=None,
                       sparse_keys=None,
                       sparse_types=None,
                       dense_keys=None,
                       dense_types=None,
                       dense_defaults=None,
                       dense_shapes=None,
                       name=None):
  """Parses `Example` protos.

  Args:
    serialized: A vector (1-D Tensor) of strings, a batch of binary
      serialized `Example` protos.
    names: A vector (1-D Tensor) of strings (optional), the names of
      the serialized protos.
    sparse_keys: A list of string keys in the examples' features.
      The results for these keys will be returned as `SparseTensor` objects.
    sparse_types: A list of `DTypes` of the same length as `sparse_keys`.
      Only `tf.float32` (`FloatList`), `tf.int64` (`Int64List`),
      and `tf.string` (`BytesList`) are supported.
    dense_keys: A list of string keys in the examples' features.
      The results for these keys will be returned as `Tensor`s
    dense_types: A list of DTypes of the same length as `dense_keys`.
      Only `tf.float32` (`FloatList`), `tf.int64` (`Int64List`),
      and `tf.string` (`BytesList`) are supported.
    dense_defaults: A dict mapping string keys to `Tensor`s.
      The keys of the dict must match the dense_keys of the feature.
    dense_shapes: A list of tuples with the same length as `dense_keys`.
      The shape of the data for each dense feature referenced by `dense_keys`.
      Required for any input tensors identified by `dense_keys`.  Must be
      either fully defined, or may contain an unknown first dimension.
      An unknown first dimension means the feature is treated as having
      a variable number of blocks, and the output shape along this dimension
      is considered unknown at graph build time.  Padding is applied for
      minibatch elements smaller than the maximum number of blocks for the
      given feature along this dimension.
    name: A name for this operation (optional).

  Returns:
    A `dict` mapping keys to `Tensor`s and `SparseTensor`s.

  """
  with ops.name_scope(name, "ParseExample", [serialized, names]):
    (names, dense_defaults_vec, sparse_keys, sparse_types,
     dense_keys, dense_shapes, _) = _process_raw_parameters(
         names, dense_defaults, sparse_keys, sparse_types, dense_keys,
         dense_types, dense_shapes)

    outputs = gen_parsing_ops.parse_example(
        serialized=serialized,
        names=names,
        dense_defaults=dense_defaults_vec,
        sparse_keys=sparse_keys,
        sparse_types=sparse_types,
        dense_keys=dense_keys,
        dense_shapes=dense_shapes,
        name=name)

    (sparse_indices, sparse_values, sparse_shapes, dense_values) = outputs

    sparse_tensors = [
        sparse_tensor.SparseTensor(ix, val, shape) for (ix, val, shape)
        in zip(sparse_indices, sparse_values, sparse_shapes)]

    return dict(zip(sparse_keys + dense_keys, sparse_tensors + dense_values))


def _process_raw_parameters(names, dense_defaults, sparse_keys, sparse_types,
                            dense_keys, dense_types, dense_shapes):
  """Process raw parameters to params used by `gen_parsing_ops`.

  Args:
    names: A vector (1-D Tensor) of strings (optional), the names of
      the serialized protos.
    dense_defaults: A dict mapping string keys to `Tensor`s.
      The keys of the dict must match the dense_keys of the feature.
    sparse_keys: A list of string keys in the examples' features.
      The results for these keys will be returned as `SparseTensor` objects.
    sparse_types: A list of `DTypes` of the same length as `sparse_keys`.
      Only `tf.float32` (`FloatList`), `tf.int64` (`Int64List`),
      and `tf.string` (`BytesList`) are supported.
    dense_keys: A list of string keys in the examples' features.
      The results for these keys will be returned as `Tensor`s
    dense_types: A list of DTypes of the same length as `dense_keys`.
      Only `tf.float32` (`FloatList`), `tf.int64` (`Int64List`),
      and `tf.string` (`BytesList`) are supported.
    dense_shapes: A list of tuples with the same length as `dense_keys`.
      The shape of the data for each dense feature referenced by `dense_keys`.
      Required for any input tensors identified by `dense_keys`.  Must be
      either fully defined, or may contain an unknown first dimension.
      An unknown first dimension means the feature is treated as having
      a variable number of blocks, and the output shape along this dimension
      is considered unknown at graph build time.  Padding is applied for
      minibatch elements smaller than the maximum number of blocks for the
      given feature along this dimension.

  Returns:
    Tuple of `names`, `dense_defaults_vec`, `sparse_keys`, `sparse_types`,
    `dense_keys`, `dense_shapes`.

  Raises:
    ValueError: If sparse and dense key sets intersect, or input lengths do not
      match up.
  """
  names = [] if names is None else names
  dense_defaults = collections.OrderedDict(
  ) if dense_defaults is None else dense_defaults
  sparse_keys = [] if sparse_keys is None else sparse_keys
  sparse_types = [] if sparse_types is None else sparse_types
  dense_keys = [] if dense_keys is None else dense_keys
  dense_types = [] if dense_types is None else dense_types
  dense_shapes = ([[]] * len(dense_keys)
                  if dense_shapes is None else dense_shapes)

  num_dense = len(dense_keys)
  num_sparse = len(sparse_keys)

  if len(dense_shapes) != num_dense:
    raise ValueError("len(dense_shapes) != len(dense_keys): %d vs. %d" %
                     (len(dense_shapes), num_dense))
  if len(dense_types) != num_dense:
    raise ValueError("len(dense_types) != len(num_dense): %d vs. %d" %
                     (len(dense_types), num_dense))
  if len(sparse_types) != num_sparse:
    raise ValueError("len(sparse_types) != len(sparse_keys): %d vs. %d" %
                     (len(sparse_types), num_sparse))
  if num_dense + num_sparse == 0:
    raise ValueError("Must provide at least one sparse key or dense key")
  if not set(dense_keys).isdisjoint(set(sparse_keys)):
    raise ValueError(
        "Dense and sparse keys must not intersect; intersection: %s" %
        set(dense_keys).intersection(set(sparse_keys)))

  # Convert dense_shapes to TensorShape object.
  dense_shapes = [tensor_shape.as_shape(shape) for shape in dense_shapes]

  dense_defaults_vec = []
  for i, key in enumerate(dense_keys):
    default_value = dense_defaults.get(key)
    dense_shape = dense_shapes[i]
    if (dense_shape.ndims is not None and dense_shape.ndims > 0 and
        dense_shape[0].value is None):
      # Variable stride dense shape, the default value should be a
      # scalar padding value
      if default_value is None:
        default_value = ops.convert_to_tensor(
            "" if dense_types[i] == dtypes.string else 0, dtype=dense_types[i])
      else:
        # Reshape to a scalar to ensure user gets an error if they
        # provide a tensor that's not intended to be a padding value
        # (0 or 2+ elements).
        key_name = "padding_" + re.sub("[^A-Za-z0-9_.\\-/]", "_", key)
        default_value = ops.convert_to_tensor(
            default_value, dtype=dense_types[i], name=key_name)
        default_value = array_ops.reshape(default_value, [])
    else:
      if default_value is None:
        default_value = constant_op.constant([], dtype=dense_types[i])
      elif not isinstance(default_value, ops.Tensor):
        key_name = "key_" + re.sub("[^A-Za-z0-9_.\\-/]", "_", key)
        default_value = ops.convert_to_tensor(
            default_value, dtype=dense_types[i], name=key_name)
        default_value = array_ops.reshape(default_value, dense_shape)

    dense_defaults_vec.append(default_value)

  # Finally, convert dense_shapes to TensorShapeProto
  dense_shapes_as_proto = [shape.as_proto() for shape in dense_shapes]

  return (names, dense_defaults_vec, sparse_keys, sparse_types, dense_keys,
          dense_shapes_as_proto, dense_shapes)


@tf_export("io.parse_single_example", "parse_single_example")
def parse_single_example(serialized, features, name=None, example_names=None):
  """Parses a single `Example` proto.

  Similar to `parse_example`, except:

  For dense tensors, the returned `Tensor` is identical to the output of
  `parse_example`, except there is no batch dimension, the output shape is the
  same as the shape given in `dense_shape`.

  For `SparseTensor`s, the first (batch) column of the indices matrix is removed
  (the indices matrix is a column vector), the values vector is unchanged, and
  the first (`batch_size`) entry of the shape vector is removed (it is now a
  single element vector).

  One might see performance advantages by batching `Example` protos with
  `parse_example` instead of using this function directly.

  Args:
    serialized: A scalar string Tensor, a single serialized Example.
      See `_parse_single_example_raw` documentation for more details.
    features: A `dict` mapping feature keys to `FixedLenFeature` or
      `VarLenFeature` values.
    name: A name for this operation (optional).
    example_names: (Optional) A scalar string Tensor, the associated name.
      See `_parse_single_example_raw` documentation for more details.

  Returns:
    A `dict` mapping feature keys to `Tensor` and `SparseTensor` values.

  Raises:
    ValueError: if any feature is invalid.
  """
  if not features:
    raise ValueError("Missing features.")
  if example_names is None:
    return parse_single_example_v2(serialized, features, name)
  features = _prepend_none_dimension(features)
  (sparse_keys, sparse_types, dense_keys, dense_types, dense_defaults,
   dense_shapes) = _features_to_raw_params(
       features,
       [VarLenFeature, FixedLenFeature, FixedLenSequenceFeature, SparseFeature])
  outputs = _parse_single_example_raw(
      serialized, example_names, sparse_keys, sparse_types, dense_keys,
      dense_types, dense_defaults, dense_shapes, name)
  return _construct_sparse_tensors_for_sparse_features(features, outputs)


def _parse_single_example_raw(serialized,
                              names=None,
                              sparse_keys=None,
                              sparse_types=None,
                              dense_keys=None,
                              dense_types=None,
                              dense_defaults=None,
                              dense_shapes=None,
                              name=None):
  """Parses a single `Example` proto.

  Args:
    serialized: A scalar string Tensor, a single serialized Example.
      See `_parse_example_raw` documentation for more details.
    names: (Optional) A scalar string Tensor, the associated name.
      See `_parse_example_raw` documentation for more details.
    sparse_keys: See `_parse_example_raw` documentation for more details.
    sparse_types: See `_parse_example_raw` documentation for more details.
    dense_keys: See `_parse_example_raw` documentation for more details.
    dense_types: See `_parse_example_raw` documentation for more details.
    dense_defaults: See `_parse_example_raw` documentation for more details.
    dense_shapes: See `_parse_example_raw` documentation for more details.
    name: A name for this operation (optional).

  Returns:
    A `dict` mapping feature keys to `Tensor` and `SparseTensor` values.

  Raises:
    ValueError: if any feature is invalid.
  """
  with ops.name_scope(name, "ParseSingleExample", [serialized, names]):
    serialized = ops.convert_to_tensor(serialized)
    serialized_shape = serialized.get_shape()
    if serialized_shape.ndims is not None:
      if serialized_shape.ndims != 0:
        raise ValueError("Input serialized must be a scalar")
    else:
      serialized = control_flow_ops.with_dependencies(
          [control_flow_ops.Assert(
              math_ops.equal(array_ops.rank(serialized), 0),
              ["Input serialized must be a scalar"],
              name="SerializedIsScalar")],
          serialized,
          name="SerializedDependencies")
    serialized = array_ops.expand_dims(serialized, 0)
    if names is not None:
      names = ops.convert_to_tensor(names)
      names_shape = names.get_shape()
      if names_shape.ndims is not None:
        if names_shape.ndims != 0:
          raise ValueError("Input names must be a scalar")
      else:
        names = control_flow_ops.with_dependencies(
            [control_flow_ops.Assert(
                math_ops.equal(array_ops.rank(names), 0),
                ["Input names must be a scalar"],
                name="NamesIsScalar")],
            names,
            name="NamesDependencies")
      names = array_ops.expand_dims(names, 0)

    outputs = _parse_example_raw(
        serialized,
        names=names,
        sparse_keys=sparse_keys,
        sparse_types=sparse_types,
        dense_keys=dense_keys,
        dense_types=dense_types,
        dense_defaults=dense_defaults,
        dense_shapes=dense_shapes,
        name=name)
    if dense_keys is not None:
      for d in dense_keys:
        d_name = re.sub("[^A-Za-z0-9_.\\-/]", "_", d)
        outputs[d] = array_ops.squeeze(
            outputs[d], [0], name="Squeeze_%s" % d_name)
    if sparse_keys is not None:
      for s in sparse_keys:
        s_name = re.sub("[^A-Za-z0-9_.\\-/]", "_", s)
        outputs[s] = sparse_tensor.SparseTensor(
            array_ops.slice(outputs[s].indices,
                            [0, 1], [-1, -1], name="Slice_Indices_%s" % s_name),
            outputs[s].values,
            array_ops.slice(outputs[s].dense_shape,
                            [1], [-1], name="Squeeze_Shape_%s" % s_name))
    return outputs


@tf_export("io.parse_sequence_example")
def parse_sequence_example(serialized,
                           context_features=None,
                           sequence_features=None,
                           example_names=None,
                           name=None):
  # pylint: disable=line-too-long
  """Parses a batch of `SequenceExample` protos.

  Parses a vector of serialized
  [`SequenceExample`](https://www.tensorflow.org/code/tensorflow/core/example/example.proto)
  protos given in `serialized`.

  This op parses serialized sequence examples into a tuple of dictionaries
  mapping keys to `Tensor` and `SparseTensor` objects respectively.
  The first dictionary contains mappings for keys appearing in
  `context_features`, and the second dictionary contains mappings for keys
  appearing in `sequence_features`.

  At least one of `context_features` and `sequence_features` must be provided
  and non-empty.

  The `context_features` keys are associated with a `SequenceExample` as a
  whole, independent of time / frame.  In contrast, the `sequence_features` keys
  provide a way to access variable-length data within the `FeatureList` section
  of the `SequenceExample` proto.  While the shapes of `context_features` values
  are fixed with respect to frame, the frame dimension (the first dimension)
  of `sequence_features` values may vary between `SequenceExample` protos,
  and even between `feature_list` keys within the same `SequenceExample`.

  `context_features` contains `VarLenFeature` and `FixedLenFeature` objects.
  Each `VarLenFeature` is mapped to a `SparseTensor`, and each `FixedLenFeature`
  is mapped to a `Tensor`, of the specified type, shape, and default value.

  `sequence_features` contains `VarLenFeature` and `FixedLenSequenceFeature`
  objects. Each `VarLenFeature` is mapped to a `SparseTensor`, and each
  `FixedLenSequenceFeature` is mapped to a `Tensor`, each of the specified type.
  The shape will be `(B,T,) + df.dense_shape` for `FixedLenSequenceFeature`
  `df`, where `B` is the batch size, and `T` is the length of the associated
  `FeatureList` in the `SequenceExample`. For instance,
  `FixedLenSequenceFeature([])` yields a scalar 2-D `Tensor` of static shape
  `[None, None]` and dynamic shape `[B, T]`, while
  `FixedLenSequenceFeature([k])` (for `int k >= 1`) yields a 3-D matrix `Tensor`
  of static shape `[None, None, k]` and dynamic shape `[B, T, k]`.

  Like the input, the resulting output tensors have a batch dimension. This
  means that the original per-example shapes of `VarLenFeature`s and
  `FixedLenSequenceFeature`s can be lost. To handle that situation, this op also
  provides dicts of shape tensors as part of the output. There is one dict for
  the context features, and one for the feature_list features. Context features
  of type `FixedLenFeature`s will not be present, since their shapes are already
  known by the caller. In situations where the input 'FixedLenFeature`s are of
  different lengths across examples, the shorter examples will be padded with
  default datatype values: 0 for numeric types, and the empty string for string
  types.

  Each `SparseTensor` corresponding to `sequence_features` represents a ragged
  vector.  Its indices are `[time, index]`, where `time` is the `FeatureList`
  entry and `index` is the value's index in the list of values associated with
  that time.

  `FixedLenFeature` entries with a `default_value` and `FixedLenSequenceFeature`
  entries with `allow_missing=True` are optional; otherwise, we will fail if
  that `Feature` or `FeatureList` is missing from any example in `serialized`.

  `example_name` may contain a descriptive name for the corresponding serialized
  proto. This may be useful for debugging purposes, but it has no effect on the
  output. If not `None`, `example_name` must be a scalar.

  Args:
    serialized: A vector (1-D Tensor) of type string containing binary
      serialized `SequenceExample` protos.
    context_features: A `dict` mapping feature keys to `FixedLenFeature` or
      `VarLenFeature` values. These features are associated with a
      `SequenceExample` as a whole.
    sequence_features: A `dict` mapping feature keys to
      `FixedLenSequenceFeature` or `VarLenFeature` values. These features are
      associated with data within the `FeatureList` section of the
      `SequenceExample` proto.
    example_names: A vector (1-D Tensor) of strings (optional), the name of the
      serialized protos.
    name: A name for this operation (optional).

  Returns:
    A tuple of three `dict`s, each mapping keys to `Tensor`s and
    `SparseTensor`s. The first dict contains the context key/values,
    the second dict contains the feature_list key/values, and the final dict
    contains the lengths of any dense feature_list features.

  Raises:
    ValueError: if any feature is invalid.
  """
  if not (context_features or sequence_features):
    raise ValueError("Missing features.")
  (context_sparse_keys, context_sparse_types, context_dense_keys,
   context_dense_types,
   context_dense_defaults, context_dense_shapes) = _features_to_raw_params(
       context_features, [VarLenFeature, FixedLenFeature])
  (feature_list_sparse_keys, feature_list_sparse_types, feature_list_dense_keys,
   feature_list_dense_types, feature_list_dense_defaults,
   feature_list_dense_shapes) = _features_to_raw_params(
       sequence_features, [VarLenFeature, FixedLenSequenceFeature])
  return _parse_sequence_example_raw(
      serialized, example_names, context_sparse_keys, context_sparse_types,
      context_dense_keys, context_dense_types, context_dense_defaults,
      context_dense_shapes, feature_list_sparse_keys, feature_list_sparse_types,
      feature_list_dense_keys, feature_list_dense_types,
      feature_list_dense_shapes, feature_list_dense_defaults, name)


def _parse_sequence_example_raw(serialized,
                                debug_name=None,
                                context_sparse_keys=None,
                                context_sparse_types=None,
                                context_dense_keys=None,
                                context_dense_types=None,
                                context_dense_defaults=None,
                                context_dense_shapes=None,
                                feature_list_sparse_keys=None,
                                feature_list_sparse_types=None,
                                feature_list_dense_keys=None,
                                feature_list_dense_types=None,
                                feature_list_dense_shapes=None,
                                feature_list_dense_defaults=None,
                                name=None):
  """Parses a vector of `SequenceExample` protos.

  Args:
    serialized: A vector (1-D Tensor) of type string, containing binary
      serialized `SequenceExample` protos.
    debug_name: A vector (1-D Tensor) of strings (optional), the names of the
      serialized protos.
    context_sparse_keys: A list of string keys in the `SequenceExample`'s
      features.  The results for these keys will be returned as `SparseTensor`
      objects.
    context_sparse_types: A list of `DTypes`, the same length as `sparse_keys`.
      Only `tf.float32` (`FloatList`), `tf.int64` (`Int64List`), and `tf.string`
      (`BytesList`) are supported.
    context_dense_keys: A list of string keys in the examples' features. The
      results for these keys will be returned as `Tensor`s
    context_dense_types: A list of DTypes, same length as `context_dense_keys`.
      Only `tf.float32` (`FloatList`), `tf.int64` (`Int64List`), and `tf.string`
      (`BytesList`) are supported.
    context_dense_defaults: A dict mapping string keys to `Tensor`s. The keys of
      the dict must match the context_dense_keys of the feature.
    context_dense_shapes: A list of tuples, same length as `context_dense_keys`.
      The shape of the data for each context_dense feature referenced by
      `context_dense_keys`.  Required for any input tensors identified by
      `context_dense_keys` whose shapes are anything other than `[]` or `[1]`.
    feature_list_sparse_keys: A list of string keys in the `SequenceExample`'s
      feature_lists.  The results for these keys will be returned as
      `SparseTensor` objects.
    feature_list_sparse_types: A list of `DTypes`, same length as `sparse_keys`.
      Only `tf.float32` (`FloatList`), `tf.int64` (`Int64List`), and `tf.string`
      (`BytesList`) are supported.
    feature_list_dense_keys: A list of string keys in the `SequenceExample`'s
      features_lists. The results for these keys will be returned as `Tensor`s.
    feature_list_dense_types: A list of `DTypes`, same length as
      `feature_list_dense_keys`.  Only `tf.float32` (`FloatList`), `tf.int64`
      (`Int64List`), and `tf.string` (`BytesList`) are supported.
    feature_list_dense_shapes: A list of tuples, same length as
      `feature_list_dense_keys`.  The shape of the data for each `FeatureList`
      feature referenced by `feature_list_dense_keys`.
    feature_list_dense_defaults: A dict mapping key strings to values. The only
      currently allowed value is `None`.  Any key appearing in this dict with
      value `None` is allowed to be missing from the `SequenceExample`.  If
      missing, the key is treated as zero-length.
    name: A name for this operation (optional).

  Returns:
    A tuple of three `dict`s, each mapping keys to `Tensor`s and
    `SparseTensor`s. The first dict contains the context key/values,
    the second dict contains the feature_list key/values, and the final dict
    contains the lengths of any dense feature_list features.

  Raises:
    ValueError: If context_sparse and context_dense key sets intersect,
      if feature_list_sparse and feature_list_dense key sets intersect,
      if input lengths do not match up, or if a value in
      feature_list_dense_defaults is not None.
    TypeError: if feature_list_dense_defaults is not either None or a dict.
  """
  with ops.name_scope(name, "ParseSequenceExample", [serialized]):
    context_dense_defaults = ({} if context_dense_defaults is None else
                              context_dense_defaults)
    context_sparse_keys = ([] if context_sparse_keys is None else
                           context_sparse_keys)
    context_sparse_types = ([] if context_sparse_types is None else
                            context_sparse_types)
    context_dense_keys = ([]
                          if context_dense_keys is None else context_dense_keys)
    context_dense_types = ([] if context_dense_types is None else
                           context_dense_types)
    context_dense_shapes = ([[]] * len(context_dense_keys)
                            if context_dense_shapes is None else
                            context_dense_shapes)
    feature_list_sparse_keys = ([] if feature_list_sparse_keys is None else
                                feature_list_sparse_keys)
    feature_list_sparse_types = ([] if feature_list_sparse_types is None else
                                 feature_list_sparse_types)
    feature_list_dense_keys = ([] if feature_list_dense_keys is None else
                               feature_list_dense_keys)
    feature_list_dense_types = ([] if feature_list_dense_types is None else
                                feature_list_dense_types)
    feature_list_dense_shapes = ([[]] * len(feature_list_dense_keys)
                                 if feature_list_dense_shapes is None else
                                 feature_list_dense_shapes)
    feature_list_dense_defaults = (
        dict()
        if feature_list_dense_defaults is None else feature_list_dense_defaults)
    debug_name = [] if debug_name is None else debug_name

    # Internal
    feature_list_dense_missing_assumed_empty = []

    num_context_dense = len(context_dense_keys)
    num_feature_list_dense = len(feature_list_dense_keys)
    num_context_sparse = len(context_sparse_keys)
    num_feature_list_sparse = len(feature_list_sparse_keys)

    if len(context_dense_shapes) != num_context_dense:
      raise ValueError(
          "len(context_dense_shapes) != len(context_dense_keys): %d vs. %d" %
          (len(context_dense_shapes), num_context_dense))
    if len(context_dense_types) != num_context_dense:
      raise ValueError(
          "len(context_dense_types) != len(num_context_dense): %d vs. %d" %
          (len(context_dense_types), num_context_dense))
    if len(feature_list_dense_shapes) != num_feature_list_dense:
      raise ValueError(
          "len(feature_list_dense_shapes) != len(feature_list_dense_keys): "
          "%d vs. %d" % (len(feature_list_dense_shapes),
                         num_feature_list_dense))
    if len(feature_list_dense_types) != num_feature_list_dense:
      raise ValueError(
          "len(feature_list_dense_types) != len(num_feature_list_dense):"
          "%d vs. %d" % (len(feature_list_dense_types), num_feature_list_dense))
    if len(context_sparse_types) != num_context_sparse:
      raise ValueError(
          "len(context_sparse_types) != len(context_sparse_keys): %d vs. %d" %
          (len(context_sparse_types), num_context_sparse))
    if len(feature_list_sparse_types) != num_feature_list_sparse:
      raise ValueError(
          "len(feature_list_sparse_types) != len(feature_list_sparse_keys): "
          "%d vs. %d" % (len(feature_list_sparse_types),
                         num_feature_list_sparse))
    if (num_context_dense + num_context_sparse + num_feature_list_dense +
        num_feature_list_sparse) == 0:
      raise ValueError(
          "Must provide at least one context_sparse key, context_dense key, "
          ", feature_list_sparse key, or feature_list_dense key")
    if not set(context_dense_keys).isdisjoint(set(context_sparse_keys)):
      raise ValueError(
          "context_dense and context_sparse keys must not intersect; "
          "intersection: %s" % set(context_dense_keys).intersection(
              set(context_sparse_keys)))
    if not set(feature_list_dense_keys).isdisjoint(
        set(feature_list_sparse_keys)):
      raise ValueError(
          "feature_list_dense and feature_list_sparse keys must not intersect; "
          "intersection: %s" % set(feature_list_dense_keys).intersection(
              set(feature_list_sparse_keys)))
    if not isinstance(feature_list_dense_defaults, dict):
      raise TypeError("feature_list_dense_defaults must be a dict")
    for k, v in feature_list_dense_defaults.items():
      if v is not None:
        raise ValueError(
            "Value feature_list_dense_defaults[%s] must be None" % k)
      feature_list_dense_missing_assumed_empty.append(k)

    context_dense_defaults_vec = []
    for i, key in enumerate(context_dense_keys):
      default_value = context_dense_defaults.get(key)
      if default_value is None:
        default_value = constant_op.constant([], dtype=context_dense_types[i])
      elif not isinstance(default_value, ops.Tensor):
        key_name = "key_" + re.sub("[^A-Za-z0-9_.\\-/]", "_", key)
        default_value = ops.convert_to_tensor(
            default_value, dtype=context_dense_types[i], name=key_name)

      context_dense_defaults_vec.append(default_value)

    context_dense_shapes = [
        tensor_shape.as_shape(shape).as_proto()
        for shape in context_dense_shapes
    ]
    feature_list_dense_shapes = [
        tensor_shape.as_shape(shape).as_proto()
        for shape in feature_list_dense_shapes
    ]

    # pylint: disable=protected-access
    outputs = gen_parsing_ops.parse_sequence_example(
        serialized=serialized,
        debug_name=debug_name,
        Ncontext_sparse=num_context_sparse,
        Ncontext_dense=num_context_dense,
        Nfeature_list_sparse=num_feature_list_sparse,
        Nfeature_list_dense=num_feature_list_dense,
        context_dense_defaults=context_dense_defaults_vec,
        context_sparse_keys=context_sparse_keys,
        context_sparse_types=context_sparse_types,
        context_dense_keys=context_dense_keys,
        context_dense_shapes=context_dense_shapes,
        feature_list_sparse_keys=feature_list_sparse_keys,
        feature_list_sparse_types=feature_list_sparse_types,
        feature_list_dense_keys=feature_list_dense_keys,
        feature_list_dense_types=feature_list_dense_types,
        feature_list_dense_shapes=feature_list_dense_shapes,
        feature_list_dense_missing_assumed_empty=(
            feature_list_dense_missing_assumed_empty),
        name=name)
    # pylint: enable=protected-access

    (context_sparse_indices, context_sparse_values, context_sparse_shapes,
     context_dense_values, feature_list_sparse_indices,
     feature_list_sparse_values, feature_list_sparse_shapes,
     feature_list_dense_values, feature_list_dense_lengths) = outputs

    context_sparse_tensors = [
        sparse_tensor.SparseTensor(ix, val, shape)
        for (ix, val,
             shape) in zip(context_sparse_indices, context_sparse_values,
                           context_sparse_shapes)
    ]

    feature_list_sparse_tensors = [
        sparse_tensor.SparseTensor(ix, val, shape)
        for (ix, val, shape
            ) in zip(feature_list_sparse_indices, feature_list_sparse_values,
                     feature_list_sparse_shapes)
    ]

    context_output = dict(
        zip(context_sparse_keys + context_dense_keys,
            context_sparse_tensors + context_dense_values))
    feature_list_output = dict(
        zip(feature_list_sparse_keys + feature_list_dense_keys,
            feature_list_sparse_tensors + feature_list_dense_values))
    feature_list_lengths = dict(
        zip(feature_list_dense_keys, feature_list_dense_lengths))

    return (context_output, feature_list_output, feature_list_lengths)


# TODO(sundberg): rewrite this method to call the batch version, which is more
# efficient especially for large inputs.
@tf_export("io.parse_single_sequence_example", "parse_single_sequence_example")
def parse_single_sequence_example(
    serialized, context_features=None, sequence_features=None,
    example_name=None, name=None):
  # pylint: disable=line-too-long
  """Parses a single `SequenceExample` proto.

  Parses a single serialized [`SequenceExample`](https://www.tensorflow.org/code/tensorflow/core/example/example.proto)
  proto given in `serialized`.

  This op parses a serialized sequence example into a tuple of dictionaries
  mapping keys to `Tensor` and `SparseTensor` objects respectively.
  The first dictionary contains mappings for keys appearing in
  `context_features`, and the second dictionary contains mappings for keys
  appearing in `sequence_features`.

  At least one of `context_features` and `sequence_features` must be provided
  and non-empty.

  The `context_features` keys are associated with a `SequenceExample` as a
  whole, independent of time / frame.  In contrast, the `sequence_features` keys
  provide a way to access variable-length data within the `FeatureList` section
  of the `SequenceExample` proto.  While the shapes of `context_features` values
  are fixed with respect to frame, the frame dimension (the first dimension)
  of `sequence_features` values may vary between `SequenceExample` protos,
  and even between `feature_list` keys within the same `SequenceExample`.

  `context_features` contains `VarLenFeature` and `FixedLenFeature` objects.
  Each `VarLenFeature` is mapped to a `SparseTensor`, and each `FixedLenFeature`
  is mapped to a `Tensor`, of the specified type, shape, and default value.

  `sequence_features` contains `VarLenFeature` and `FixedLenSequenceFeature`
  objects. Each `VarLenFeature` is mapped to a `SparseTensor`, and each
  `FixedLenSequenceFeature` is mapped to a `Tensor`, each of the specified type.
  The shape will be `(T,) + df.dense_shape` for `FixedLenSequenceFeature` `df`, where
  `T` is the length of the associated `FeatureList` in the `SequenceExample`.
  For instance, `FixedLenSequenceFeature([])` yields a scalar 1-D `Tensor` of
  static shape `[None]` and dynamic shape `[T]`, while
  `FixedLenSequenceFeature([k])` (for `int k >= 1`) yields a 2-D matrix `Tensor`
  of static shape `[None, k]` and dynamic shape `[T, k]`.

  Each `SparseTensor` corresponding to `sequence_features` represents a ragged
  vector.  Its indices are `[time, index]`, where `time` is the `FeatureList`
  entry and `index` is the value's index in the list of values associated with
  that time.

  `FixedLenFeature` entries with a `default_value` and `FixedLenSequenceFeature`
  entries with `allow_missing=True` are optional; otherwise, we will fail if
  that `Feature` or `FeatureList` is missing from any example in `serialized`.

  `example_name` may contain a descriptive name for the corresponding serialized
  proto. This may be useful for debugging purposes, but it has no effect on the
  output. If not `None`, `example_name` must be a scalar.

  Args:
    serialized: A scalar (0-D Tensor) of type string, a single binary
      serialized `SequenceExample` proto.
    context_features: A `dict` mapping feature keys to `FixedLenFeature` or
      `VarLenFeature` values. These features are associated with a
      `SequenceExample` as a whole.
    sequence_features: A `dict` mapping feature keys to
      `FixedLenSequenceFeature` or `VarLenFeature` values. These features are
      associated with data within the `FeatureList` section of the
      `SequenceExample` proto.
    example_name: A scalar (0-D Tensor) of strings (optional), the name of
      the serialized proto.
    name: A name for this operation (optional).

  Returns:
    A tuple of two `dict`s, each mapping keys to `Tensor`s and `SparseTensor`s.
    The first dict contains the context key/values.
    The second dict contains the feature_list key/values.

  Raises:
    ValueError: if any feature is invalid.
  """
  # pylint: enable=line-too-long
  if not (context_features or sequence_features):
    raise ValueError("Missing features.")
  (context_sparse_keys, context_sparse_types, context_dense_keys,
   context_dense_types, context_dense_defaults,
   context_dense_shapes) = _features_to_raw_params(
       context_features, [VarLenFeature, FixedLenFeature])
  (feature_list_sparse_keys, feature_list_sparse_types,
   feature_list_dense_keys, feature_list_dense_types,
   feature_list_dense_defaults,
   feature_list_dense_shapes) = _features_to_raw_params(
       sequence_features, [VarLenFeature, FixedLenSequenceFeature])
  return _parse_single_sequence_example_raw(
      serialized, context_sparse_keys, context_sparse_types,
      context_dense_keys, context_dense_types, context_dense_defaults,
      context_dense_shapes, feature_list_sparse_keys,
      feature_list_sparse_types, feature_list_dense_keys,
      feature_list_dense_types, feature_list_dense_shapes,
      feature_list_dense_defaults, example_name, name)


def _parse_single_sequence_example_raw(serialized,
                                       context_sparse_keys=None,
                                       context_sparse_types=None,
                                       context_dense_keys=None,
                                       context_dense_types=None,
                                       context_dense_defaults=None,
                                       context_dense_shapes=None,
                                       feature_list_sparse_keys=None,
                                       feature_list_sparse_types=None,
                                       feature_list_dense_keys=None,
                                       feature_list_dense_types=None,
                                       feature_list_dense_shapes=None,
                                       feature_list_dense_defaults=None,
                                       debug_name=None,
                                       name=None):
  """Parses a single `SequenceExample` proto.

  Args:
    serialized: A scalar (0-D Tensor) of type string, a single binary
      serialized `SequenceExample` proto.
    context_sparse_keys: A list of string keys in the `SequenceExample`'s
      features.  The results for these keys will be returned as
      `SparseTensor` objects.
    context_sparse_types: A list of `DTypes`, the same length as `sparse_keys`.
      Only `tf.float32` (`FloatList`), `tf.int64` (`Int64List`),
      and `tf.string` (`BytesList`) are supported.
    context_dense_keys: A list of string keys in the examples' features.
      The results for these keys will be returned as `Tensor`s
    context_dense_types: A list of DTypes, same length as `context_dense_keys`.
      Only `tf.float32` (`FloatList`), `tf.int64` (`Int64List`),
      and `tf.string` (`BytesList`) are supported.
    context_dense_defaults: A dict mapping string keys to `Tensor`s.
      The keys of the dict must match the context_dense_keys of the feature.
    context_dense_shapes: A list of tuples, same length as `context_dense_keys`.
      The shape of the data for each context_dense feature referenced by
      `context_dense_keys`.  Required for any input tensors identified by
      `context_dense_keys` whose shapes are anything other than `[]` or `[1]`.
    feature_list_sparse_keys: A list of string keys in the `SequenceExample`'s
      feature_lists.  The results for these keys will be returned as
      `SparseTensor` objects.
    feature_list_sparse_types: A list of `DTypes`, same length as `sparse_keys`.
      Only `tf.float32` (`FloatList`), `tf.int64` (`Int64List`),
      and `tf.string` (`BytesList`) are supported.
    feature_list_dense_keys: A list of string keys in the `SequenceExample`'s
      features_lists. The results for these keys will be returned as `Tensor`s.
    feature_list_dense_types: A list of `DTypes`, same length as
      `feature_list_dense_keys`.  Only `tf.float32` (`FloatList`),
      `tf.int64` (`Int64List`), and `tf.string` (`BytesList`) are supported.
    feature_list_dense_shapes: A list of tuples, same length as
      `feature_list_dense_keys`.  The shape of the data for each
      `FeatureList` feature referenced by `feature_list_dense_keys`.
    feature_list_dense_defaults: A dict mapping key strings to values.
      The only currently allowed value is `None`.  Any key appearing
      in this dict with value `None` is allowed to be missing from the
      `SequenceExample`.  If missing, the key is treated as zero-length.
    debug_name: A scalar (0-D Tensor) of strings (optional), the name of
      the serialized proto.
    name: A name for this operation (optional).

  Returns:
    A tuple of two `dict`s, each mapping keys to `Tensor`s and `SparseTensor`s.
    The first dict contains the context key/values.
    The second dict contains the feature_list key/values.

  Raises:
    ValueError: If context_sparse and context_dense key sets intersect,
      if input lengths do not match up, or if a value in
      feature_list_dense_defaults is not None.
    TypeError: if feature_list_dense_defaults is not either None or a dict.
  """
  with ops.name_scope(name, "ParseSingleSequenceExample", [serialized]):
    context_dense_defaults = (
        {} if context_dense_defaults is None else context_dense_defaults)
    context_sparse_keys = (
        [] if context_sparse_keys is None else context_sparse_keys)
    context_sparse_types = (
        [] if context_sparse_types is None else context_sparse_types)
    context_dense_keys = (
        [] if context_dense_keys is None else context_dense_keys)
    context_dense_types = (
        [] if context_dense_types is None else context_dense_types)
    context_dense_shapes = (
        [[]] * len(context_dense_keys)
        if context_dense_shapes is None else context_dense_shapes)
    feature_list_sparse_keys = (
        [] if feature_list_sparse_keys is None else feature_list_sparse_keys)
    feature_list_sparse_types = (
        [] if feature_list_sparse_types is None else feature_list_sparse_types)
    feature_list_dense_keys = (
        [] if feature_list_dense_keys is None else feature_list_dense_keys)
    feature_list_dense_types = (
        [] if feature_list_dense_types is None else feature_list_dense_types)
    feature_list_dense_shapes = (
        [[]] * len(feature_list_dense_keys)
        if feature_list_dense_shapes is None else feature_list_dense_shapes)
    feature_list_dense_defaults = (
        dict() if feature_list_dense_defaults is None
        else feature_list_dense_defaults)
    debug_name = "" if debug_name is None else debug_name

    # Internal
    feature_list_dense_missing_assumed_empty = []

    num_context_dense = len(context_dense_keys)
    num_feature_list_dense = len(feature_list_dense_keys)
    num_context_sparse = len(context_sparse_keys)
    num_feature_list_sparse = len(feature_list_sparse_keys)

    if len(context_dense_shapes) != num_context_dense:
      raise ValueError(
          "len(context_dense_shapes) != len(context_dense_keys): %d vs. %d"
          % (len(context_dense_shapes), num_context_dense))
    if len(context_dense_types) != num_context_dense:
      raise ValueError(
          "len(context_dense_types) != len(num_context_dense): %d vs. %d"
          % (len(context_dense_types), num_context_dense))
    if len(feature_list_dense_shapes) != num_feature_list_dense:
      raise ValueError(
          "len(feature_list_dense_shapes) != len(feature_list_dense_keys): "
          "%d vs. %d" % (len(feature_list_dense_shapes),
                         num_feature_list_dense))
    if len(feature_list_dense_types) != num_feature_list_dense:
      raise ValueError(
          "len(feature_list_dense_types) != len(num_feature_list_dense):"
          "%d vs. %d" % (len(feature_list_dense_types), num_feature_list_dense))
    if len(context_sparse_types) != num_context_sparse:
      raise ValueError(
          "len(context_sparse_types) != len(context_sparse_keys): %d vs. %d"
          % (len(context_sparse_types), num_context_sparse))
    if len(feature_list_sparse_types) != num_feature_list_sparse:
      raise ValueError(
          "len(feature_list_sparse_types) != len(feature_list_sparse_keys): "
          "%d vs. %d"
          % (len(feature_list_sparse_types), num_feature_list_sparse))
    if (num_context_dense + num_context_sparse
        + num_feature_list_dense + num_feature_list_sparse) == 0:
      raise ValueError(
          "Must provide at least one context_sparse key, context_dense key, "
          ", feature_list_sparse key, or feature_list_dense key")
    if not set(context_dense_keys).isdisjoint(set(context_sparse_keys)):
      raise ValueError(
          "context_dense and context_sparse keys must not intersect; "
          "intersection: %s" %
          set(context_dense_keys).intersection(set(context_sparse_keys)))
    if not set(feature_list_dense_keys).isdisjoint(
        set(feature_list_sparse_keys)):
      raise ValueError(
          "feature_list_dense and feature_list_sparse keys must not intersect; "
          "intersection: %s" %
          set(feature_list_dense_keys).intersection(
              set(feature_list_sparse_keys)))
    if not isinstance(feature_list_dense_defaults, dict):
      raise TypeError("feature_list_dense_defaults must be a dict")
    for k, v in feature_list_dense_defaults.items():
      if v is not None:
        raise ValueError("Value feature_list_dense_defaults[%s] must be None"
                         % k)
      feature_list_dense_missing_assumed_empty.append(k)

    context_dense_defaults_vec = []
    for i, key in enumerate(context_dense_keys):
      default_value = context_dense_defaults.get(key)
      if default_value is None:
        default_value = constant_op.constant([], dtype=context_dense_types[i])
      elif not isinstance(default_value, ops.Tensor):
        key_name = "key_" + re.sub("[^A-Za-z0-9_.\\-/]", "_", key)
        default_value = ops.convert_to_tensor(
            default_value, dtype=context_dense_types[i], name=key_name)
        default_value = array_ops.reshape(
            default_value, context_dense_shapes[i])

      context_dense_defaults_vec.append(default_value)

    context_dense_shapes = [tensor_shape.as_shape(shape).as_proto()
                            for shape in context_dense_shapes]
    feature_list_dense_shapes = [tensor_shape.as_shape(shape).as_proto()
                                 for shape in feature_list_dense_shapes]

    outputs = gen_parsing_ops.parse_single_sequence_example(
        serialized=serialized,
        debug_name=debug_name,
        context_dense_defaults=context_dense_defaults_vec,
        context_sparse_keys=context_sparse_keys,
        context_sparse_types=context_sparse_types,
        context_dense_keys=context_dense_keys,
        context_dense_shapes=context_dense_shapes,
        feature_list_sparse_keys=feature_list_sparse_keys,
        feature_list_sparse_types=feature_list_sparse_types,
        feature_list_dense_keys=feature_list_dense_keys,
        feature_list_dense_types=feature_list_dense_types,
        feature_list_dense_shapes=feature_list_dense_shapes,
        feature_list_dense_missing_assumed_empty=(
            feature_list_dense_missing_assumed_empty),
        name=name)

    (context_sparse_indices, context_sparse_values,
     context_sparse_shapes, context_dense_values,
     feature_list_sparse_indices, feature_list_sparse_values,
     feature_list_sparse_shapes, feature_list_dense_values) = outputs

    context_sparse_tensors = [
        sparse_tensor.SparseTensor(ix, val, shape) for (ix, val, shape)
        in zip(context_sparse_indices,
               context_sparse_values,
               context_sparse_shapes)]

    feature_list_sparse_tensors = [
        sparse_tensor.SparseTensor(ix, val, shape) for (ix, val, shape)
        in zip(feature_list_sparse_indices,
               feature_list_sparse_values,
               feature_list_sparse_shapes)]

    context_output = dict(
        zip(context_sparse_keys + context_dense_keys,
            context_sparse_tensors + context_dense_values))
    feature_list_output = dict(
        zip(feature_list_sparse_keys + feature_list_dense_keys,
            feature_list_sparse_tensors + feature_list_dense_values))

    return (context_output, feature_list_output)


# Swap `name` and `na_value` for backward compatibility.
@tf_export("io.decode_csv", "decode_csv")
@deprecation.deprecated_endpoints("decode_csv")
def decode_csv(records,
               record_defaults,
               field_delim=",",
               use_quote_delim=True,
               name=None,
               na_value="",
               select_cols=None):
  """Convert CSV records to tensors. Each column maps to one tensor.

  RFC 4180 format is expected for the CSV records.
  (https://tools.ietf.org/html/rfc4180)
  Note that we allow leading and trailing spaces with int or float field.

  Args:
    records: A `Tensor` of type `string`.
      Each string is a record/row in the csv and all records should have
      the same format.
    record_defaults: A list of `Tensor` objects with specific types.
      Acceptable types are `float32`, `float64`, `int32`, `int64`, `string`.
      One tensor per column of the input record, with either a
      scalar default value for that column or an empty vector if the column is
      required.
    field_delim: An optional `string`. Defaults to `","`.
      char delimiter to separate fields in a record.
    use_quote_delim: An optional `bool`. Defaults to `True`.
      If false, treats double quotation marks as regular
      characters inside of the string fields (ignoring RFC 4180, Section 2,
      Bullet 5).
    name: A name for the operation (optional).
    na_value: Additional string to recognize as NA/NaN.
    select_cols: Optional sorted list of column indices to select. If specified,
      only this subset of columns will be parsed and returned.

  Returns:
    A list of `Tensor` objects. Has the same type as `record_defaults`.
    Each tensor will have the same shape as records.

  Raises:
    ValueError: If any of the arguments is malformed.
  """
  if select_cols is not None and any(select_cols[i] >= select_cols[i + 1]
                                     for i in range(len(select_cols) - 1)):
    raise ValueError("select_cols is not strictly increasing.")
  if select_cols is not None and select_cols[0] < 0:
    raise ValueError("select_cols contains negative values.")
  if select_cols is not None and len(select_cols) != len(record_defaults):
    raise ValueError("Length of select_cols and record_defaults do not match.")
  return gen_parsing_ops.decode_csv(
      records=records,
      record_defaults=record_defaults,
      field_delim=field_delim,
      use_quote_delim=use_quote_delim,
      na_value=na_value,
      name=name,
      select_cols=select_cols,
  )


# TODO(b/70890287): Combine the implementation of this op and
# `parse_single_example()` after 1/10/2018.
def parse_single_example_v2(serialized, features, name=None):
  # pylint: disable=line-too-long
  """Parses an `Example` proto into a `dict` of tensors.

  Parses a serialized
  [`Example`](https://www.tensorflow.org/code/tensorflow/core/example/example.proto)
  proto given in `serialized`.

  This op parses serialized examples into a dictionary mapping keys to `Tensor`
  and `SparseTensor` objects. `features` is a dict from keys to `VarLenFeature`,
  `SparseFeature`, and `FixedLenFeature` objects. Each `VarLenFeature`
  and `SparseFeature` is mapped to a `SparseTensor`, and each
  `FixedLenFeature` is mapped to a `Tensor`.

  Each `VarLenFeature` maps to a `SparseTensor` of the specified type
  representing a ragged matrix. Its indices are `[index]` where
  `index` is the value's index in the list of values associated with
  that feature and example.

  Each `SparseFeature` maps to a `SparseTensor` of the specified type
  representing a Tensor of `dense_shape` `SparseFeature.size`.
  Its `values` come from the feature in the examples with key `value_key`.
  A `values[i]` comes from a position `k` in the feature of an example at batch
  entry `batch`. This positional information is recorded in `indices[i]` as
  `[batch, index_0, index_1, ...]` where `index_j` is the `k-th` value of
  the feature in the example at with key `SparseFeature.index_key[j]`.
  In other words, we split the indices (except the first index indicating the
  batch entry) of a `SparseTensor` by dimension into different features of the
  `Example`. Due to its complexity a `VarLenFeature` should be preferred over a
  `SparseFeature` whenever possible.

  Each `FixedLenFeature` `df` maps to a `Tensor` of the specified type (or
  `tf.float32` if not specified) and shape `df.shape`.

  `FixedLenFeature` entries with a `default_value` are optional. With no default
  value, we will fail if that `Feature` is missing from any example in
  `serialized`.

  Each `FixedLenSequenceFeature` `df` maps to a `Tensor` of the specified type
  (or `tf.float32` if not specified) and shape `(None,) + df.shape`.

  Args:
    serialized: A scalar (0-D Tensor) string, a serialized `Example` proto.
    features: A `dict` mapping feature keys to `FixedLenFeature`,
      `VarLenFeature`, and `SparseFeature` values.
    name: A name for this operation (optional).

  Returns:
    A `dict` mapping feature keys to `Tensor` and `SparseTensor` values.

  Raises:
    ValueError: if any feature is invalid.
  """
  if not features:
    raise ValueError("Missing: features was %s." % features)
  features = _prepend_none_dimension(features)
  (sparse_keys, sparse_types, dense_keys, dense_types,
   dense_defaults, dense_shapes) = _features_to_raw_params(
       features,
       [VarLenFeature, SparseFeature, FixedLenFeature, FixedLenSequenceFeature])
  outputs = _parse_single_example_v2_raw(serialized, sparse_keys, sparse_types,
                                         dense_keys, dense_types,
                                         dense_defaults, dense_shapes, name)
  return _construct_sparse_tensors_for_sparse_features(features, outputs)


def _parse_single_example_v2_raw(serialized, sparse_keys, sparse_types,
                                 dense_keys, dense_types, dense_defaults,
                                 dense_shapes, name):
  """Parses `Example` protos.

  Args:
    serialized: A scalar (0-D Tensor) string, containing a binary
      serialized `Example` proto.
    sparse_keys: A list of string keys in the examples' features.
      The results for these keys will be returned as `SparseTensor` objects.
    sparse_types: A list of `DTypes` of the same length as `sparse_keys`.
      Only `tf.float32` (`FloatList`), `tf.int64` (`Int64List`),
      and `tf.string` (`BytesList`) are supported.
    dense_keys: A list of string keys in the examples' features.
      The results for these keys will be returned as `Tensor`s
    dense_types: A list of DTypes of the same length as `dense_keys`.
      Only `tf.float32` (`FloatList`), `tf.int64` (`Int64List`),
      and `tf.string` (`BytesList`) are supported.
    dense_defaults: A dict mapping string keys to `Tensor`s.
      The keys of the dict must match the dense_keys of the feature.
    dense_shapes: A list of tuples with the same length as `dense_keys`.
      The shape of the data for each dense feature referenced by `dense_keys`.
      Required for any input tensors identified by `dense_keys`.  Must be
      either fully defined, or may contain an unknown first dimension.
      An unknown first dimension means the feature is treated as having
      a variable number of blocks, and the output shape along this dimension
      is considered unknown at graph build time.  Padding is applied for
      minibatch elements smaller than the maximum number of blocks for the
      given feature along this dimension.
    name: A name for this operation (optional).

  Returns:
    A `dict` mapping keys to `Tensor`s and `SparseTensor`s.

  Raises:
    ValueError: If sparse and dense key sets intersect, or input lengths do not
      match up.
  """
  with ops.name_scope(name, "ParseSingleExample", [serialized]):
    serialized = ops.convert_to_tensor(serialized, name="serialized")
    dense_defaults = collections.OrderedDict(
    ) if dense_defaults is None else dense_defaults
    sparse_keys = [] if sparse_keys is None else sparse_keys
    sparse_types = [] if sparse_types is None else sparse_types
    dense_keys = [] if dense_keys is None else dense_keys
    dense_types = [] if dense_types is None else dense_types
    dense_shapes = ([[]] * len(dense_keys)
                    if dense_shapes is None else dense_shapes)

    num_dense = len(dense_keys)
    num_sparse = len(sparse_keys)

    if len(dense_shapes) != num_dense:
      raise ValueError("len(dense_shapes) != len(dense_keys): %d vs. %d" %
                       (len(dense_shapes), num_dense))
    if len(dense_types) != num_dense:
      raise ValueError("len(dense_types) != len(num_dense): %d vs. %d" %
                       (len(dense_types), num_dense))
    if len(sparse_types) != num_sparse:
      raise ValueError("len(sparse_types) != len(sparse_keys): %d vs. %d" %
                       (len(sparse_types), num_sparse))
    if num_dense + num_sparse == 0:
      raise ValueError("Must provide at least one sparse key or dense key")
    if not set(dense_keys).isdisjoint(set(sparse_keys)):
      raise ValueError(
          "Dense and sparse keys must not intersect; intersection: %s" %
          set(dense_keys).intersection(set(sparse_keys)))

    # Convert dense_shapes to TensorShape object.
    dense_shapes = [tensor_shape.as_shape(shape) for shape in dense_shapes]

    dense_defaults_vec = []
    for i, key in enumerate(dense_keys):
      default_value = dense_defaults.get(key)
      dense_shape = dense_shapes[i]
      if (dense_shape.ndims is not None and dense_shape.ndims > 0 and
          dense_shape[0].value is None):
        # Variable stride dense shape, the default value should be a
        # scalar padding value
        if default_value is None:
          default_value = ops.convert_to_tensor(
              "" if dense_types[i] == dtypes.string else 0,
              dtype=dense_types[i])
        else:
          # Reshape to a scalar to ensure user gets an error if they
          # provide a tensor that's not intended to be a padding value
          # (0 or 2+ elements).
          key_name = "padding_" + re.sub("[^A-Za-z0-9_.\\-/]", "_", key)
          default_value = ops.convert_to_tensor(
              default_value, dtype=dense_types[i], name=key_name)
          default_value = array_ops.reshape(default_value, [])
      else:
        if default_value is None:
          default_value = constant_op.constant([], dtype=dense_types[i])
        elif not isinstance(default_value, ops.Tensor):
          key_name = "key_" + re.sub("[^A-Za-z0-9_.\\-/]", "_", key)
          default_value = ops.convert_to_tensor(
              default_value, dtype=dense_types[i], name=key_name)
          default_value = array_ops.reshape(default_value, dense_shape)

      dense_defaults_vec.append(default_value)

    # Finally, convert dense_shapes to TensorShapeProto
    dense_shapes = [shape.as_proto() for shape in dense_shapes]

    outputs = gen_parsing_ops.parse_single_example(
        serialized=serialized,
        dense_defaults=dense_defaults_vec,
        num_sparse=len(sparse_keys),
        sparse_keys=sparse_keys,
        sparse_types=sparse_types,
        dense_keys=dense_keys,
        dense_shapes=dense_shapes,
        name=name)

    (sparse_indices, sparse_values, sparse_shapes, dense_values) = outputs

    sparse_tensors = [
        sparse_tensor.SparseTensor(ix, val, shape)
        for (ix, val,
             shape) in zip(sparse_indices, sparse_values, sparse_shapes)
    ]

    return dict(zip(sparse_keys + dense_keys, sparse_tensors + dense_values))
