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

from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_parsing_ops
from tensorflow.python.ops import math_ops
# go/tf-wildcard-import
# pylint: disable=wildcard-import,undefined-variable
from tensorflow.python.ops.gen_parsing_ops import *
# pylint: enable=wildcard-import,undefined-variable


ops.NotDifferentiable("DecodeRaw")
ops.NotDifferentiable("ParseTensor")
ops.NotDifferentiable("StringToNumber")


class VarLenFeature(collections.namedtuple("VarLenFeature", ["dtype"])):
  """Configuration for parsing a variable-length input feature.

  Fields:
    dtype: Data type of input.
  """
  pass


class FixedLenFeature(collections.namedtuple(
    "FixedLenFeature", ["shape", "dtype", "default_value"])):
  """Configuration for parsing a fixed-length input feature.

  To treat sparse input as dense, provide a `default_value`; otherwise,
  the parse functions will fail on any examples missing this feature.

  Fields:
    shape: Shape of input data.
    dtype: Data type of input.
    default_value: Value to be used if an example is missing this feature. It
        must be compatible with `dtype`.
  """
  pass
FixedLenFeature.__new__.__defaults__ = (None,)


# NOTE: If we ever support a default_value for sequence dense features, we can
# remove this class and use FixedLenFeature in its place.
class FixedLenSequenceFeature(collections.namedtuple(
    "FixedLenSequenceFeature", ["shape", "dtype", "allow_missing"])):
  """Configuration for a dense input feature in a sequence item.

  To treat a sparse input as dense, provide `allow_missing=True`; otherwise,
  the parse functions will fail on any examples missing this feature.

  Fields:
    shape: Shape of input data.
    dtype: Data type of input.
    allow_missing: Whether to allow this feature to be missing from a feature
      list item.
  """
  pass
FixedLenSequenceFeature.__new__.__defaults__ = (False,)


def _features_to_raw_params(features, types):
  """Split feature tuples into raw params used by `gen_parsing_ops`.

  Args:
    features: A `dict` mapping feature keys to objects of a type in `types`.
    types: Type of features to allow, among `FixedLenFeature`, `VarLenFeature`,
      and `FixedLenSequenceFeature`.

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
  dense_defaults = {}
  dense_shapes = []
  if features:
    # NOTE: We iterate over sorted keys to keep things deterministic.
    for key in sorted(features.keys()):
      feature = features[key]
      if isinstance(feature, VarLenFeature):
        if VarLenFeature not in types:
          raise ValueError("Unsupported VarLenFeature %s.", feature)
        if not feature.dtype:
          raise ValueError("Missing type for feature %s." % key)
        sparse_keys.append(key)
        sparse_types.append(feature.dtype)
      elif isinstance(feature, FixedLenFeature):
        if FixedLenFeature not in types:
          raise ValueError("Unsupported FixedLenFeature %s.", feature)
        if not feature.dtype:
          raise ValueError("Missing type for feature %s." % key)
        if feature.shape is None:
          raise ValueError("Missing shape for feature %s." % key)
        dense_keys.append(key)
        dense_shapes.append(feature.shape)
        dense_types.append(feature.dtype)
        if feature.default_value is not None:
          dense_defaults[key] = feature.default_value
      elif isinstance(feature, FixedLenSequenceFeature):
        if FixedLenSequenceFeature not in types:
          raise ValueError("Unsupported FixedLenSequenceFeature %s.", feature)
        if not feature.dtype:
          raise ValueError("Missing type for feature %s." % key)
        if feature.shape is None:
          raise ValueError("Missing shape for feature %s." % key)
        dense_keys.append(key)
        dense_shapes.append(feature.shape)
        dense_types.append(feature.dtype)
        if feature.allow_missing:
          dense_defaults[key] = None
      else:
        raise ValueError("Invalid feature %s:%s." % (key, feature))
  return (
      sparse_keys, sparse_types, dense_keys, dense_types, dense_defaults,
      dense_shapes)


def parse_example(serialized, features, name=None, example_names=None):
  # pylint: disable=line-too-long
  """Parses `Example` protos into a `dict` of tensors.

  Parses a number of serialized [`Example`]
  (https://www.tensorflow.org/code/tensorflow/core/example/example.proto)
  protos given in `serialized`.

  `example_names` may contain descriptive names for the corresponding serialized
  protos. These may be useful for debugging purposes, but they have no effect on
  the output. If not `None`, `example_names` must be the same length as `serialized`.

  This op parses serialized examples into a dictionary mapping keys to `Tensor`
  and `SparseTensor` objects. `features` is a dict from keys to `VarLenFeature`
  and `FixedLenFeature` objects. Each `VarLenFeature` is mapped to a
  `SparseTensor`, and each `FixedLenFeature` is mapped to a `Tensor`.

  Each `VarLenFeature` maps to a `SparseTensor` of the specified type
  representing a ragged matrix. Its indices are `[batch, index]` where `batch`
  is the batch entry the value is from in `serialized`, and `index` is the
  value's index in the list of values associated with that feature and example.

  Each `FixedLenFeature` `df` maps to a `Tensor` of the specified type (or
  `tf.float32` if not specified) and shape `(serialized.size(),) + df.shape`.

  `FixedLenFeature` entries with a `default_value` are optional. With no default
  value, we will fail if that `Feature` is missing from any example in
  `serialized`.

  Examples:

  For example, if one expects a `tf.float32` sparse feature `ft` and three
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

  ```
  {"ft": SparseTensor(indices=[[0, 0], [0, 1], [2, 0]],
                      values=[1.0, 2.0, 3.0],
                      shape=(3, 2)) }
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
        shape=[2, 2]),
    "dank": SparseTensor(
        indices=[[1, 0]],
        values=[42],
        shape=[2, 1]),
    "gps": SparseTensor(
        indices=[],
        values=[],
        shape=[2, 0]),
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

  Args:
    serialized: A vector (1-D Tensor) of strings, a batch of binary
      serialized `Example` protos.
    features: A `dict` mapping feature keys to `FixedLenFeature` or
      `VarLenFeature` values.
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
  (sparse_keys, sparse_types, dense_keys, dense_types, dense_defaults,
   dense_shapes) = _features_to_raw_params(
       features, [VarLenFeature, FixedLenFeature])
  return _parse_example_raw(
      serialized, example_names, sparse_keys, sparse_types, dense_keys,
      dense_types, dense_defaults, dense_shapes, name)


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
      Required for any input tensors identified by `dense_keys` whose shapes are
      anything other than `[]` or `[1]`.
    name: A name for this operation (optional).

  Returns:
    A `dict` mapping keys to `Tensor`s and `SparseTensor`s.

  Raises:
    ValueError: If sparse and dense key sets intersect, or input lengths do not
      match up.
  """
  with ops.name_scope(name, "ParseExample", [serialized, names]):
    names = [] if names is None else names
    dense_defaults = {} if dense_defaults is None else dense_defaults
    sparse_keys = [] if sparse_keys is None else sparse_keys
    sparse_types = [] if sparse_types is None else sparse_types
    dense_keys = [] if dense_keys is None else dense_keys
    dense_types = [] if dense_types is None else dense_types
    dense_shapes = (
        [[]] * len(dense_keys) if dense_shapes is None else dense_shapes)

    num_dense = len(dense_keys)
    num_sparse = len(sparse_keys)

    if len(dense_shapes) != num_dense:
      raise ValueError("len(dense_shapes) != len(dense_keys): %d vs. %d"
                       % (len(dense_shapes), num_dense))
    if len(dense_types) != num_dense:
      raise ValueError("len(dense_types) != len(num_dense): %d vs. %d"
                       % (len(dense_types), num_dense))
    if len(sparse_types) != num_sparse:
      raise ValueError("len(sparse_types) != len(sparse_keys): %d vs. %d"
                       % (len(sparse_types), num_sparse))
    if num_dense + num_sparse == 0:
      raise ValueError("Must provide at least one sparse key or dense key")
    if not set(dense_keys).isdisjoint(set(sparse_keys)):
      raise ValueError(
          "Dense and sparse keys must not intersect; intersection: %s" %
          set(dense_keys).intersection(set(sparse_keys)))

    dense_defaults_vec = []
    for i, key in enumerate(dense_keys):
      default_value = dense_defaults.get(key)
      if default_value is None:
        default_value = constant_op.constant([], dtype=dense_types[i])
      elif not isinstance(default_value, ops.Tensor):
        key_name = "key_" + re.sub("[^A-Za-z0-9_.\\-/]", "_", key)
        default_value = ops.convert_to_tensor(
            default_value, dtype=dense_types[i], name=key_name)
        default_value = array_ops.reshape(default_value, dense_shapes[i])

      dense_defaults_vec.append(default_value)

    dense_shapes = [tensor_shape.as_shape(shape).as_proto()
                    for shape in dense_shapes]

    # pylint: disable=protected-access
    outputs = gen_parsing_ops._parse_example(
        serialized=serialized,
        names=names,
        dense_defaults=dense_defaults_vec,
        sparse_keys=sparse_keys,
        sparse_types=sparse_types,
        dense_keys=dense_keys,
        dense_shapes=dense_shapes,
        name=name)
    # pylint: enable=protected-access

    (sparse_indices, sparse_values, sparse_shapes, dense_values) = outputs

    sparse_tensors = [ops.SparseTensor(ix, val, shape) for (ix, val, shape)
                      in zip(sparse_indices, sparse_values, sparse_shapes)]

    return dict(
        zip(sparse_keys + dense_keys, sparse_tensors + dense_values))


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
  (sparse_keys, sparse_types, dense_keys, dense_types, dense_defaults,
   dense_shapes) = _features_to_raw_params(
       features, [VarLenFeature, FixedLenFeature])
  return _parse_single_example_raw(
      serialized, example_names, sparse_keys, sparse_types, dense_keys,
      dense_types, dense_defaults, dense_shapes, name)


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

    outputs = _parse_example_raw(serialized,
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
        outputs[s] = ops.SparseTensor(
            array_ops.slice(outputs[s].indices,
                            [0, 1], [-1, -1], name="Slice_Indices_%s" % s_name),
            outputs[s].values,
            array_ops.slice(outputs[s].shape,
                            [1], [-1], name="Squeeze_Shape_%s" % s_name))
    return outputs


ops.RegisterShape("ParseExample")(common_shapes.call_cpp_shape_fn)


def parse_single_sequence_example(
    serialized, context_features=None, sequence_features=None,
    example_name=None, name=None):
  # pylint: disable=line-too-long
  """Parses a single `SequenceExample` proto.

  Parses a single serialized [`SequenceExample`]
  (https://www.tensorflow.org/code/tensorflow/core/example/example.proto)
  proto given in `serialized`.

  This op parses a serialize sequence example into a tuple of dictionaries
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
  The shape will be `(T,) + df.shape` for `FixedLenSequenceFeature` `df`, where
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

    # pylint: disable=protected-access
    outputs = gen_parsing_ops._parse_single_sequence_example(
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
    # pylint: enable=protected-access

    (context_sparse_indices, context_sparse_values,
     context_sparse_shapes, context_dense_values,
     feature_list_sparse_indices, feature_list_sparse_values,
     feature_list_sparse_shapes, feature_list_dense_values) = outputs

    context_sparse_tensors = [
        ops.SparseTensor(ix, val, shape) for (ix, val, shape)
        in zip(context_sparse_indices,
               context_sparse_values,
               context_sparse_shapes)]

    feature_list_sparse_tensors = [
        ops.SparseTensor(ix, val, shape) for (ix, val, shape)
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


ops.RegisterShape("ParseSingleSequenceExample")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("ParseTensor")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("DecodeJSONExample")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("StringToNumber")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("DecodeRaw")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("DecodeCSV")(common_shapes.call_cpp_shape_fn)
