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

from tensorflow.python.compat import compat
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_parsing_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import parsing_config
# go/tf-wildcard-import
# pylint: disable=wildcard-import,undefined-variable
from tensorflow.python.ops.gen_parsing_ops import *
# pylint: enable=wildcard-import,undefined-variable
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export


ops.NotDifferentiable("DecodeRaw")
ops.NotDifferentiable("DecodePaddedRaw")
ops.NotDifferentiable("ParseTensor")
ops.NotDifferentiable("SerializeTensor")
ops.NotDifferentiable("StringToNumber")


VarLenFeature = parsing_config.VarLenFeature
RaggedFeature = parsing_config.RaggedFeature
SparseFeature = parsing_config.SparseFeature
FixedLenFeature = parsing_config.FixedLenFeature
FixedLenSequenceFeature = parsing_config.FixedLenSequenceFeature
# pylint: disable=protected-access
_ParseOpParams = parsing_config._ParseOpParams
_construct_tensors_for_composite_features = (
    parsing_config._construct_tensors_for_composite_features)
# pylint: enable=protected-access


# TODO(b/122887740) Switch files that use this private symbol to use new name.
_construct_sparse_tensors_for_sparse_features = \
    _construct_tensors_for_composite_features


def _prepend_none_dimension(features):
  """Returns a copy of features with adjusted FixedLenSequenceFeature shapes."""
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


@tf_export("io.parse_example", v1=[])
def parse_example_v2(serialized, features, example_names=None, name=None):
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
  `SparseTensor`, and `RaggedTensor` objects. `features` is a dict from keys to
  `VarLenFeature`, `SparseFeature`, `RaggedFeature`, and `FixedLenFeature`
  objects. Each `VarLenFeature` and `SparseFeature` is mapped to a
  `SparseTensor`; each `FixedLenFeature` is mapped to a `Tensor`; and each
  `RaggedFeature` is mapped to a `RaggedTensor`.

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

  Each `RaggedFeature` maps to a `RaggedTensor` of the specified type.  It
  is formed by stacking the `RaggedTensor` for each example, where the
  `RaggedTensor` for each individual example is constructed using the tensors
  specified by `RaggedTensor.values_key` and `RaggedTensor.partition`.  See
  the `tf.io.RaggedFeature` documentation for details and examples.

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

  See the `tf.io.RaggedFeature` documentation for examples showing how
  `RaggedFeature` can be used to obtain `RaggedTensor`s.

  Args:
    serialized: A vector (1-D Tensor) of strings, a batch of binary
      serialized `Example` protos.
    features: A `dict` mapping feature keys to `FixedLenFeature`,
      `VarLenFeature`, `SparseFeature`, and `RaggedFeature` values.
    example_names: A vector (1-D Tensor) of strings (optional), the names of
      the serialized protos in the batch.
    name: A name for this operation (optional).

  Returns:
    A `dict` mapping feature keys to `Tensor`, `SparseTensor`, and
    `RaggedTensor` values.

  Raises:
    ValueError: if any feature is invalid.
  """
  if not features:
    raise ValueError("Missing: features was %s." % features)
  features = _prepend_none_dimension(features)
  params = _ParseOpParams.from_features(features, [
      VarLenFeature, SparseFeature, FixedLenFeature, FixedLenSequenceFeature,
      RaggedFeature
  ])

  outputs = _parse_example_raw(serialized, example_names, params, name=name)
  return _construct_tensors_for_composite_features(features, outputs)


@tf_export(v1=["io.parse_example", "parse_example"])
def parse_example(serialized, features, name=None, example_names=None):
  return parse_example_v2(serialized, features, example_names, name)


parse_example.__doc__ = parse_example_v2.__doc__


def _parse_example_raw(serialized, names, params, name):
  """Parses `Example` protos.

  Args:
    serialized: A vector (1-D Tensor) of strings, a batch of binary
      serialized `Example` protos.
    names: A vector (1-D Tensor) of strings (optional), the names of
      the serialized protos.
    params: A `ParseOpParams` containing the parameters for the parse op.
    name: A name for this operation (optional).

  Returns:
    A `dict` mapping keys to `Tensor`s and `SparseTensor`s and `RaggedTensor`s.

  """
  if params.num_features == 0:
    raise ValueError("Must provide at least one feature key")
  with ops.name_scope(name, "ParseExample", [serialized, names]):
    names = [] if names is None else names
    serialized = ops.convert_to_tensor(serialized, name="serialized")
    if params.ragged_keys and serialized.shape.ndims is None:
      raise ValueError("serialized must have statically-known rank to "
                       "parse ragged features.")
    outputs = gen_parsing_ops.parse_example_v2(
        serialized=serialized,
        names=names,
        sparse_keys=params.sparse_keys,
        dense_keys=params.dense_keys,
        ragged_keys=params.ragged_keys,
        dense_defaults=params.dense_defaults_vec,
        num_sparse=len(params.sparse_keys),
        sparse_types=params.sparse_types,
        ragged_value_types=params.ragged_value_types,
        ragged_split_types=params.ragged_split_types,
        dense_shapes=params.dense_shapes_as_proto,
        name=name)
    (sparse_indices, sparse_values, sparse_shapes, dense_values,
     ragged_values, ragged_row_splits) = outputs
    # pylint: disable=protected-access
    ragged_tensors = parsing_config._build_ragged_tensors(
        serialized.shape, ragged_values, ragged_row_splits)

    sparse_tensors = [
        sparse_tensor.SparseTensor(ix, val, shape) for (ix, val, shape)
        in zip(sparse_indices, sparse_values, sparse_shapes)]

    return dict(
        zip(params.sparse_keys + params.dense_keys + params.ragged_keys,
            sparse_tensors + dense_values + ragged_tensors))


@tf_export(v1=["io.parse_single_example", "parse_single_example"])
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
    features: A `dict` mapping feature keys to `FixedLenFeature` or
      `VarLenFeature` values.
    name: A name for this operation (optional).
    example_names: (Optional) A scalar string Tensor, the associated name.

  Returns:
    A `dict` mapping feature keys to `Tensor` and `SparseTensor` values.

  Raises:
    ValueError: if any feature is invalid.
  """
  return parse_single_example_v2(serialized, features, example_names, name)


@tf_export("io.parse_single_example", v1=[])
def parse_single_example_v2(
    serialized, features, example_names=None, name=None
    ):
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
    features: A `dict` mapping feature keys to `FixedLenFeature` or
      `VarLenFeature` values.
    example_names: (Optional) A scalar string Tensor, the associated name.
    name: A name for this operation (optional).

  Returns:
    A `dict` mapping feature keys to `Tensor` and `SparseTensor` values.

  Raises:
    ValueError: if any feature is invalid.
  """
  if not features:
    raise ValueError("Missing features.")
  with ops.name_scope(name, "ParseSingleExample", [serialized, example_names]):
    serialized = ops.convert_to_tensor(serialized, name="serialized")
    serialized = _assert_scalar(serialized, "serialized")
    return parse_example_v2(serialized, features, example_names, name)


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

  This op parses serialized sequence examples into a tuple of dictionaries,
  each mapping keys to `Tensor` and `SparseTensor` objects.
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

  `context_features` contains `VarLenFeature`, `RaggedFeature`, and
  `FixedLenFeature`  objects. Each `VarLenFeature` is mapped to a
  `SparseTensor`; each `RaggedFeature` is  mapped to a `RaggedTensor`; and each
  `FixedLenFeature` is mapped to a `Tensor`, of the specified type, shape, and
  default value.

  `sequence_features` contains `VarLenFeature`, `RaggedFeature`, and
  `FixedLenSequenceFeature` objects. Each `VarLenFeature` is mapped to a
  `SparseTensor`; each `RaggedFeature` is mapped to a `RaggedTensor; and
  each `FixedLenSequenceFeature` is mapped to a `Tensor`, each of the specified
  type. The shape will be `(B,T,) + df.dense_shape` for
  `FixedLenSequenceFeature` `df`, where `B` is the batch size, and `T` is the
  length of the associated `FeatureList` in the `SequenceExample`. For instance,
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
      `VarLenFeature` or `RaggedFeature` values. These features are associated
      with a `SequenceExample` as a whole.
    sequence_features: A `dict` mapping feature keys to
      `FixedLenSequenceFeature` or `VarLenFeature` or `RaggedFeature` values.
      These features are associated with data within the `FeatureList` section
      of the `SequenceExample` proto.
    example_names: A vector (1-D Tensor) of strings (optional), the name of the
      serialized protos.
    name: A name for this operation (optional).

  Returns:
    A tuple of three `dict`s, each mapping keys to `Tensor`s,
    `SparseTensor`s, and `RaggedTensor`. The first dict contains the context
    key/values, the second dict contains the feature_list key/values, and the
    final dict contains the lengths of any dense feature_list features.

  Raises:
    ValueError: if any feature is invalid.
  """
  if not (context_features or sequence_features):
    raise ValueError("Missing features.")
  context_params = _ParseOpParams.from_features(
      context_features, [VarLenFeature, FixedLenFeature, RaggedFeature])
  feature_list_params = _ParseOpParams.from_features(
      sequence_features,
      [VarLenFeature, FixedLenSequenceFeature, RaggedFeature])

  with ops.name_scope(name, "ParseSequenceExample",
                      [serialized, example_names]):
    outputs = _parse_sequence_example_raw(serialized, example_names,
                                          context_params, feature_list_params,
                                          name)
    context_output, feature_list_output, feature_list_lengths = outputs

    if context_params.ragged_keys:
      context_output = _construct_tensors_for_composite_features(
          context_features, context_output)
    if feature_list_params.ragged_keys:
      feature_list_output = _construct_tensors_for_composite_features(
          sequence_features, feature_list_output)

    return context_output, feature_list_output, feature_list_lengths


def _parse_sequence_example_raw(serialized,
                                debug_name,
                                context,
                                feature_list,
                                name=None):
  """Parses a vector of `SequenceExample` protos.

  Args:
    serialized: A vector (1-D Tensor) of type string, containing binary
      serialized `SequenceExample` protos.
    debug_name: A vector (1-D Tensor) of strings (optional), the names of the
      serialized protos.
    context: A `ParseOpParams` containing the parameters for the parse
      op for the context features.
    feature_list: A `ParseOpParams` containing the parameters for the
      parse op for the feature_list features.
    name: A name for this operation (optional).

  Returns:
    A tuple of three `dict`s, each mapping keys to `Tensor`s, `SparseTensor`s,
    and `RaggedTensor`s. The first dict contains the context key/values, the
    second dict contains the feature_list key/values, and the final dict
    contains the lengths of any dense feature_list features.

  Raises:
    TypeError: if feature_list.dense_defaults is not either None or a dict.
  """
  if context.num_features + feature_list.num_features == 0:
    raise ValueError("Must provide at least one feature key")
  with ops.name_scope(name, "ParseSequenceExample", [serialized]):
    debug_name = [] if debug_name is None else debug_name

    # Internal
    feature_list_dense_missing_assumed_empty = []
    for k, v in feature_list.dense_defaults.items():
      if v is not None:
        raise ValueError("Value feature_list.dense_defaults[%s] must be None" %
                         k)
      feature_list_dense_missing_assumed_empty.append(k)

    has_ragged = context.ragged_keys or feature_list.ragged_keys
    if compat.forward_compatible(2019, 10, 26) or has_ragged:
      serialized = ops.convert_to_tensor(serialized, name="serialized")
      if has_ragged and serialized.shape.ndims is None:
        raise ValueError("serialized must have statically-known rank to "
                         "parse ragged features.")
      feature_list_dense_missing_assumed_empty_vector = [
          key in feature_list_dense_missing_assumed_empty
          for key in feature_list.dense_keys
      ]
      outputs = gen_parsing_ops.parse_sequence_example_v2(
          # Inputs
          serialized=serialized,
          debug_name=debug_name,
          context_sparse_keys=context.sparse_keys,
          context_dense_keys=context.dense_keys,
          context_ragged_keys=context.ragged_keys,
          feature_list_sparse_keys=feature_list.sparse_keys,
          feature_list_dense_keys=feature_list.dense_keys,
          feature_list_ragged_keys=feature_list.ragged_keys,
          feature_list_dense_missing_assumed_empty=(
              feature_list_dense_missing_assumed_empty_vector),
          context_dense_defaults=context.dense_defaults_vec,
          # Attrs
          Ncontext_sparse=len(context.sparse_keys),
          Nfeature_list_sparse=len(feature_list.sparse_keys),
          Nfeature_list_dense=len(feature_list.dense_keys),
          context_sparse_types=context.sparse_types,
          context_ragged_value_types=context.ragged_value_types,
          context_ragged_split_types=context.ragged_split_types,
          feature_list_dense_types=feature_list.dense_types,
          feature_list_sparse_types=feature_list.sparse_types,
          feature_list_ragged_value_types=feature_list.ragged_value_types,
          feature_list_ragged_split_types=feature_list.ragged_split_types,
          context_dense_shapes=context.dense_shapes_as_proto,
          feature_list_dense_shapes=feature_list.dense_shapes,
          name=name)
      (context_sparse_indices, context_sparse_values, context_sparse_shapes,
       context_dense_values, context_ragged_values, context_ragged_row_splits,
       feature_list_sparse_indices, feature_list_sparse_values,
       feature_list_sparse_shapes, feature_list_dense_values,
       feature_list_dense_lengths, feature_list_ragged_values,
       feature_list_ragged_outer_splits,
       feature_list_ragged_inner_splits) = outputs
      # pylint: disable=protected-access
      context_ragged_tensors = parsing_config._build_ragged_tensors(
          serialized.shape, context_ragged_values, context_ragged_row_splits)
      feature_list_ragged_tensors = parsing_config._build_ragged_tensors(
          serialized.shape, feature_list_ragged_values,
          feature_list_ragged_outer_splits, feature_list_ragged_inner_splits)
    else:
      outputs = gen_parsing_ops.parse_sequence_example(
          serialized=serialized,
          debug_name=debug_name,
          Ncontext_sparse=len(context.sparse_keys),
          Ncontext_dense=len(context.dense_keys),
          Nfeature_list_sparse=len(feature_list.sparse_keys),
          Nfeature_list_dense=len(feature_list.dense_keys),
          context_dense_defaults=context.dense_defaults_vec,
          context_sparse_keys=context.sparse_keys,
          context_sparse_types=context.sparse_types,
          context_dense_keys=context.dense_keys,
          context_dense_shapes=context.dense_shapes_as_proto,
          feature_list_sparse_keys=feature_list.sparse_keys,
          feature_list_sparse_types=feature_list.sparse_types,
          feature_list_dense_keys=feature_list.dense_keys,
          feature_list_dense_types=feature_list.dense_types,
          feature_list_dense_shapes=feature_list.dense_shapes,
          feature_list_dense_missing_assumed_empty=(
              feature_list_dense_missing_assumed_empty),
          name=name)

      (context_sparse_indices, context_sparse_values, context_sparse_shapes,
       context_dense_values, feature_list_sparse_indices,
       feature_list_sparse_values, feature_list_sparse_shapes,
       feature_list_dense_values, feature_list_dense_lengths) = outputs
      context_ragged_tensors = []
      feature_list_ragged_tensors = []

    # pylint: disable=g-complex-comprehension
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
    # pylint: enable=g-complex-comprehension

    context_output = dict(
        zip(
            context.sparse_keys + context.dense_keys + context.ragged_keys,
            context_sparse_tensors + context_dense_values +
            context_ragged_tensors))
    feature_list_output = dict(
        zip(
            feature_list.sparse_keys + feature_list.dense_keys +
            feature_list.ragged_keys, feature_list_sparse_tensors +
            feature_list_dense_values + feature_list_ragged_tensors))
    feature_list_lengths = dict(
        zip(feature_list.dense_keys, feature_list_dense_lengths))

    return (context_output, feature_list_output, feature_list_lengths)


@tf_export("io.parse_single_sequence_example",
           v1=["io.parse_single_sequence_example",
               "parse_single_sequence_example"])
def parse_single_sequence_example(
    serialized, context_features=None, sequence_features=None,
    example_name=None, name=None):
  # pylint: disable=line-too-long
  """Parses a single `SequenceExample` proto.

  Parses a single serialized [`SequenceExample`](https://www.tensorflow.org/code/tensorflow/core/example/example.proto)
  proto given in `serialized`.

  This op parses a serialized sequence example into a tuple of dictionaries,
  each mapping keys to `Tensor` and `SparseTensor` objects.
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

  `context_features` contains `VarLenFeature`, `RaggedFeature`, and
  `FixedLenFeature` objects. Each `VarLenFeature` is mapped to a `SparseTensor`;
  each `RaggedFeature` is mapped to a `RaggedTensor`; and each `FixedLenFeature`
  is mapped to a `Tensor`, of the specified type, shape, and default value.

  `sequence_features` contains `VarLenFeature`, `RaggedFeature`, and
  `FixedLenSequenceFeature` objects. Each `VarLenFeature` is mapped to a
  `SparseTensor`; each `RaggedFeature` is mapped to a `RaggedTensor`; and each
  `FixedLenSequenceFeature` is mapped to a `Tensor`, each of the specified type.
  The shape will be `(T,) + df.dense_shape` for `FixedLenSequenceFeature` `df`,
  where `T` is the length of the associated `FeatureList` in the
  `SequenceExample`. For instance, `FixedLenSequenceFeature([])` yields a scalar
  1-D `Tensor` of static shape `[None]` and dynamic shape `[T]`, while
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

  Note that the batch version of this function, `tf.parse_sequence_example`,
  is written for better memory efficiency and will be faster on large
  `SequenceExample`s.

  Args:
    serialized: A scalar (0-D Tensor) of type string, a single binary
      serialized `SequenceExample` proto.
    context_features: A `dict` mapping feature keys to `FixedLenFeature` or
      `VarLenFeature` or `RaggedFeature` values. These features are associated
      with a `SequenceExample` as a whole.
    sequence_features: A `dict` mapping feature keys to
      `FixedLenSequenceFeature` or `VarLenFeature` or `RaggedFeature` values.
      These features are associated with data within the `FeatureList` section
      of the `SequenceExample` proto.
    example_name: A scalar (0-D Tensor) of strings (optional), the name of
      the serialized proto.
    name: A name for this operation (optional).

  Returns:
    A tuple of two `dict`s, each mapping keys to `Tensor`s and `SparseTensor`s
    and `RaggedTensor`s.

    * The first dict contains the context key/values.
    * The second dict contains the feature_list key/values.

  Raises:
    ValueError: if any feature is invalid.
  """
  # pylint: enable=line-too-long
  if not (context_features or sequence_features):
    raise ValueError("Missing features.")
  context_params = _ParseOpParams.from_features(
      context_features, [VarLenFeature, FixedLenFeature, RaggedFeature])
  feature_list_params = _ParseOpParams.from_features(
      sequence_features,
      [VarLenFeature, FixedLenSequenceFeature, RaggedFeature])

  with ops.name_scope(name, "ParseSingleSequenceExample",
                      [serialized, example_name]):
    context_output, feature_list_output = (
        _parse_single_sequence_example_raw(serialized, context_params,
                                           feature_list_params, example_name,
                                           name))

    if context_params.ragged_keys:
      context_output = _construct_tensors_for_composite_features(
          context_features, context_output)
    if feature_list_params.ragged_keys:
      feature_list_output = _construct_tensors_for_composite_features(
          sequence_features, feature_list_output)

    return context_output, feature_list_output


def _parse_single_sequence_example_raw(serialized,
                                       context,
                                       feature_list,
                                       debug_name,
                                       name=None):
  """Parses a single `SequenceExample` proto.

  Args:
    serialized: A scalar (0-D Tensor) of type string, a single binary serialized
      `SequenceExample` proto.
    context: A `ParseOpParams` containing the parameters for the parse op for
      the context features.
    feature_list: A `ParseOpParams` containing the parameters for the parse op
      for the feature_list features.
    debug_name: A scalar (0-D Tensor) of strings (optional), the name of the
      serialized proto.
    name: A name for this operation (optional).

  Returns:
    A tuple of two `dict`s, each mapping keys to `Tensor`s and `SparseTensor`s.
    The first dict contains the context key/values.
    The second dict contains the feature_list key/values.

  Raises:
    TypeError: if feature_list.dense_defaults is not either None or a dict.
  """
  has_ragged = context.ragged_keys or feature_list.ragged_keys
  if compat.forward_compatible(2019, 10, 26) or has_ragged:
    with ops.name_scope(name, "ParseSingleExample", [serialized, debug_name]):
      serialized = ops.convert_to_tensor(serialized, name="serialized")
      serialized = _assert_scalar(serialized, "serialized")
    return _parse_sequence_example_raw(serialized, debug_name, context,
                                       feature_list, name)[:2]

  if context.num_features + feature_list.num_features == 0:
    raise ValueError("Must provide at least one feature key")
  with ops.name_scope(name, "ParseSingleSequenceExample", [serialized]):
    debug_name = "" if debug_name is None else debug_name

    # Internal
    feature_list_dense_missing_assumed_empty = []
    for k, v in feature_list.dense_defaults.items():
      if v is not None:
        raise ValueError("Value feature_list.dense_defaults[%s] must be None" %
                         k)
      feature_list_dense_missing_assumed_empty.append(k)

    outputs = gen_parsing_ops.parse_single_sequence_example(
        serialized=serialized,
        debug_name=debug_name,
        context_dense_defaults=context.dense_defaults_vec,
        context_sparse_keys=context.sparse_keys,
        context_sparse_types=context.sparse_types,
        context_dense_keys=context.dense_keys,
        context_dense_shapes=context.dense_shapes,
        feature_list_sparse_keys=feature_list.sparse_keys,
        feature_list_sparse_types=feature_list.sparse_types,
        feature_list_dense_keys=feature_list.dense_keys,
        feature_list_dense_types=feature_list.dense_types,
        feature_list_dense_shapes=feature_list.dense_shapes,
        feature_list_dense_missing_assumed_empty=(
            feature_list_dense_missing_assumed_empty),
        name=name)

    (context_sparse_indices, context_sparse_values,
     context_sparse_shapes, context_dense_values,
     feature_list_sparse_indices, feature_list_sparse_values,
     feature_list_sparse_shapes, feature_list_dense_values) = outputs

    # pylint: disable=g-complex-comprehension
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
    # pylint: enable=g-complex-comprehension

    context_output = dict(
        zip(context.sparse_keys + context.dense_keys,
            context_sparse_tensors + context_dense_values))
    feature_list_output = dict(
        zip(feature_list.sparse_keys + feature_list.dense_keys,
            feature_list_sparse_tensors + feature_list_dense_values))

    return (context_output, feature_list_output)


@tf_export("io.decode_raw", v1=[])
def decode_raw(input_bytes,
               out_type,
               little_endian=True,
               fixed_length=None,
               name=None):
  """Convert raw byte strings into tensors.

  Args:
    input_bytes:
      Each element of the input Tensor is converted to an array of bytes.
    out_type:
      `DType` of the output. Acceptable types are `half`, `float`, `double`,
      `int32`, `uint16`, `uint8`, `int16`, `int8`, `int64`.
    little_endian:
      Whether the `input_bytes` data is in little-endian format. Data will be
      converted into host byte order if necessary.
    fixed_length:
      If set, the first `fixed_length` bytes of each element will be converted.
      Data will be zero-padded or truncated to the specified length.

      `fixed_length` must be a multiple of the size of `out_type`.
      `fixed_length` must be specified if the elements of `input_bytes` are of
      variable length.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` object storing the decoded bytes.

  """
  if fixed_length is not None:
    return gen_parsing_ops.decode_padded_raw(
        input_bytes,
        fixed_length=fixed_length,
        out_type=out_type,
        little_endian=little_endian,
        name=name)
  else:
    return gen_parsing_ops.decode_raw(
        input_bytes, out_type, little_endian=little_endian, name=name)


@tf_export(v1=["decode_raw", "io.decode_raw"])
@deprecation.deprecated_args(None,
                             "bytes is deprecated, use input_bytes instead",
                             "bytes")
def decode_raw_v1(
    input_bytes=None,
    out_type=None,
    little_endian=True,
    name=None,
    bytes=None  # pylint: disable=redefined-builtin
):
  """Convert raw byte strings into tensors.

  Args:
    input_bytes:
      Each element of the input Tensor is converted to an array of bytes.
    out_type:
      `DType` of the output. Acceptable types are `half`, `float`, `double`,
      `int32`, `uint16`, `uint8`, `int16`, `int8`, `int64`.
    little_endian:
      Whether the `input_bytes` data is in little-endian format. Data will be
      converted into host byte order if necessary.
    name: A name for the operation (optional).
    bytes: Deprecated parameter. Use `input_bytes` instead.

  Returns:
    A `Tensor` object storing the decoded bytes.
  """
  input_bytes = deprecation.deprecated_argument_lookup("input_bytes",
                                                       input_bytes, "bytes",
                                                       bytes)

  # out_type is a required positional argument in the original API, and had to
  # be changed to a keyword argument in order to facilitate the transition from
  # the reserved named `bytes` to `input_bytes`. Ensure it's still set.
  if out_type is None:
    raise ValueError(
        "decode_raw_v1() missing 1 positional argument: 'out_type'")

  return gen_parsing_ops.decode_raw(
      input_bytes, out_type, little_endian=little_endian, name=name)


# Swap `name` and `na_value` for backward compatibility.
@tf_export(v1=["io.decode_csv", "decode_csv"])
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
  return decode_csv_v2(
      records, record_defaults,
      field_delim, use_quote_delim,
      na_value, select_cols, name
      )


@tf_export("io.decode_csv", v1=[])
def decode_csv_v2(records,
                  record_defaults,
                  field_delim=",",
                  use_quote_delim=True,
                  na_value="",
                  select_cols=None,
                  name=None):
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
    na_value: Additional string to recognize as NA/NaN.
    select_cols: Optional sorted list of column indices to select. If specified,
      only this subset of columns will be parsed and returned.
    name: A name for the operation (optional).

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


def _assert_scalar(value, name):
  """Asserts that `value` is scalar, and returns `value`."""
  value_rank = value.shape.rank
  if value_rank is None:
    check = control_flow_ops.Assert(
        math_ops.equal(array_ops.rank(value), 0),
        ["Input %s must be a scalar" % name],
        name="%sIsScalar" % name.capitalize())
    result = control_flow_ops.with_dependencies([check],
                                                value,
                                                name="%sDependencies" % name)
    result.set_shape([])
    return result
  elif value_rank == 0:
    return value
  else:
    raise ValueError("Input %s must be a scalar" % name)
