# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Feature configuration for tf.io.parse_example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import re

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.ragged import ragged_math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import tf_logging
from tensorflow.python.util.tf_export import tf_export


# TODO(b/122887740) Refactor code:
#   * Move input verification to feature configuration objects (e.g.,
#     VarLenFeature should check that dtype is a valid dtype).
#   * Add an _add_feature() method to each feature configuration object
#     (rather than using a dispatch table in _ParseOpParams._add_feature).
#   * Update _construct_tensors_for_composite_features() to call a method
#     on the feature object (rather than using dispatch).


@tf_export("io.VarLenFeature", v1=["VarLenFeature", "io.VarLenFeature"])
class VarLenFeature(collections.namedtuple("VarLenFeature", ["dtype"])):
  """Configuration for parsing a variable-length input feature.

  Fields:
    dtype: Data type of input.
  """
  pass


@tf_export("io.RaggedFeature")
class RaggedFeature(
    collections.namedtuple(
        "RaggedFeature",
        ["dtype", "value_key", "partitions", "row_splits_dtype", "validate"])):
  """Configuration for passing a RaggedTensor input feature.

  `value_key` specifies the feature key for a variable-length list of values;
  and `partitions` specifies zero or more feature keys for partitioning those
  values into higher dimensions.  Each element of `partitions` must be one of
  the following:

    * `tf.io.RaggedFeature.RowSplits(key: string)`
    * `tf.io.RaggedFeature.RowLengths(key: string)`
    * `tf.io.RaggedFeature.RowStarts(key: string)`
    * `tf.io.RaggedFeature.RowLimits(key: string)`
    * `tf.io.RaggedFeature.ValueRowIds(key: string)`
    * `tf.io.RaggedFeature.UniformRowLength(length: int)`.

  Where `key` is a feature key whose values are used to partition the values.
  Partitions are listed from outermost to innermost.

  * If `len(partitions) == 0` (the default), then:

    * A feature from a single `tf.Example` is parsed into a 1D `tf.Tensor`.
    * A feature from a batch of `tf.Example`s is parsed into a 2D
      `tf.RaggedTensor`, where the outer dimension is the batch dimension, and
      the inner (ragged) dimension is the feature length in each example.

  * If `len(partitions) == 1`, then:

    * A feature from a single `tf.Example` is parsed into a 2D
      `tf.RaggedTensor`, where the values taken from the `value_key` are
      separated into rows using the partition key.
    * A feature from a batch of `tf.Example`s is parsed into a 3D
      `tf.RaggedTensor`, where the outer dimension is the batch dimension,
      the two inner dimensions are formed by separating the `value_key` values
      from each example into rows using that example's partition key.

  * If `len(partitions) > 1`, then:

    * A feature from a single `tf.Example` is parsed into a `tf.RaggedTensor`
      whose rank is `len(partitions)+1`, and whose ragged_rank is
      `len(partitions)`.

    * A feature from a batch of `tf.Example`s is parsed into a `tf.RaggedTensor`
      whose rank is `len(partitions)+2` and whose ragged_rank is
      `len(partitions)+1`, where the outer dimension is the batch dimension.

  There is one exception: if the final (i.e., innermost) element(s) of
  `partitions` are `UniformRowLength`s, then the values are simply reshaped (as
  a higher-dimensional `tf.Tensor`), rather than being wrapped in a
  `tf.RaggedTensor`.

  #### Examples

  >>> import google.protobuf.text_format as pbtext
  >>> example_batch = [
  ...   pbtext.Merge(r'''
  ...     features {
  ...       feature {key: "v" value {int64_list {value: [3, 1, 4, 1, 5, 9]}}}
  ...       feature {key: "s1" value {int64_list {value: [0, 2, 3, 3, 6]}}}
  ...       feature {key: "s2" value {int64_list {value: [0, 2, 3, 4]}}}
  ...     }''', tf.train.Example()).SerializeToString(),
  ...   pbtext.Merge(r'''
  ...     features {
  ...       feature {key: "v" value {int64_list {value: [2, 7, 1, 8, 2, 8, 1]}}}
  ...       feature {key: "s1" value {int64_list {value: [0, 3, 4, 5, 7]}}}
  ...       feature {key: "s2" value {int64_list {value: [0, 1, 1, 4]}}}
  ...     }''', tf.train.Example()).SerializeToString()]

  >>> features = {
  ...     # Zero partitions: returns 1D tf.Tensor for each Example.
  ...     'f1': tf.io.RaggedFeature(value_key="v", dtype=tf.int64),
  ...     # One partition: returns 2D tf.RaggedTensor for each Example.
  ...     'f2': tf.io.RaggedFeature(value_key="v", dtype=tf.int64, partitions=[
  ...         tf.io.RaggedFeature.RowSplits("s1")]),
  ...     # Two partitions: returns 3D tf.RaggedTensor for each Example.
  ...     'f3': tf.io.RaggedFeature(value_key="v", dtype=tf.int64, partitions=[
  ...         tf.io.RaggedFeature.RowSplits("s2"),
  ...         tf.io.RaggedFeature.RowSplits("s1")])
  ... }

  >>> feature_dict = tf.io.parse_single_example(example_batch[0], features)
  >>> for (name, val) in sorted(feature_dict.items()):
  ...   print('%s: %s' % (name, val))
  f1: tf.Tensor([3 1 4 1 5 9], shape=(6,), dtype=int64)
  f2: <tf.RaggedTensor [[3, 1], [4], [], [1, 5, 9]]>
  f3: <tf.RaggedTensor [[[3, 1], [4]], [[]], [[1, 5, 9]]]>

  >>> feature_dict = tf.io.parse_example(example_batch, features)
  >>> for (name, val) in sorted(feature_dict.items()):
  ...   print('%s: %s' % (name, val))
  f1: <tf.RaggedTensor [[3, 1, 4, 1, 5, 9],
                        [2, 7, 1, 8, 2, 8, 1]]>
  f2: <tf.RaggedTensor [[[3, 1], [4], [], [1, 5, 9]],
                        [[2, 7, 1], [8], [2], [8, 1]]]>
  f3: <tf.RaggedTensor [[[[3, 1], [4]], [[]], [[1, 5, 9]]],
                        [[[2, 7, 1]], [], [[8], [2], [8, 1]]]]>

  Fields:
    dtype: Data type of the `RaggedTensor`.  Must be one of:
      `tf.dtypes.int64`, `tf.dtypes.float32`, `tf.dtypes.string`.
    value_key: (Optional.) Key for a `Feature` in the input `Example`, whose
      parsed `Tensor` will be the resulting `RaggedTensor.flat_values`.  If
      not specified, then it defaults to the key for this `RaggedFeature`.
    partitions: (Optional.) A list of objects specifying the row-partitioning
      tensors (from outermost to innermost).  Each entry in this list must be
      one of:
        * `tf.io.RaggedFeature.RowSplits(key: string)`
        * `tf.io.RaggedFeature.RowLengths(key: string)`
        * `tf.io.RaggedFeature.RowStarts(key: string)`
        * `tf.io.RaggedFeature.RowLimits(key: string)`
        * `tf.io.RaggedFeature.ValueRowIds(key: string)`
        * `tf.io.RaggedFeature.UniformRowLength(length: int)`.
      Where `key` is a key for a `Feature` in the input `Example`, whose parsed
      `Tensor` will be the resulting row-partitioning tensor.
    row_splits_dtype: (Optional.) Data type for the row-partitioning tensor(s).
      One of `int32` or `int64`.  Defaults to `int32`.
    validate: (Optional.) Boolean indicating whether or not to validate that
      the input values form a valid RaggedTensor.  Defaults to `False`.
  """

  # pylint: disable=invalid-name
  RowSplits = collections.namedtuple("RowSplits", ["key"])
  RowLengths = collections.namedtuple("RowLengths", ["key"])
  RowStarts = collections.namedtuple("RowStarts", ["key"])
  RowLimits = collections.namedtuple("RowLimits", ["key"])
  ValueRowIds = collections.namedtuple("ValueRowIds", ["key"])
  UniformRowLength = collections.namedtuple("UniformRowLength", ["length"])
  # pylint: enable=invalid-name

  _PARTITION_TYPES = (RowSplits, RowLengths, RowStarts, RowLimits, ValueRowIds,
                      UniformRowLength)

  def __new__(cls,
              dtype,
              value_key=None,
              partitions=(),
              row_splits_dtype=dtypes.int32,
              validate=False):
    if value_key is not None:
      if not isinstance(value_key, str):
        raise ValueError("value_key must be a string; got %r" % value_key)
      if not value_key:
        raise ValueError("value_key may not be empty")
    dtype = dtypes.as_dtype(dtype)
    if dtype not in (dtypes.int64, dtypes.float32, dtypes.string):
      raise ValueError("dtypes must be int64, float32, or bytes; got %r" %
                       dtype)
    row_splits_dtype = dtypes.as_dtype(row_splits_dtype)
    if row_splits_dtype not in (dtypes.int32, dtypes.int64):
      raise ValueError("row_splits_dtype must be int32 or int64; got %r" %
                       row_splits_dtype)
    if not isinstance(partitions, (list, tuple)):
      raise TypeError("partitions must be a list or tuple")
    for partition in partitions:
      if not isinstance(partition, cls._PARTITION_TYPES):
        raise TypeError("partitions must be a list of partition objects %s;"
                        " got: %r" % (cls._PARTITION_TYPES, partition))
    if not isinstance(validate, bool):
      raise TypeError("validate must be a bool; got %r" % validate)
    return super(RaggedFeature, cls).__new__(cls, dtype, value_key, partitions,
                                             row_splits_dtype, validate)


@tf_export("io.SparseFeature", v1=["io.SparseFeature", "SparseFeature"])
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


@tf_export("io.FixedLenFeature", v1=["io.FixedLenFeature", "FixedLenFeature"])
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


@tf_export("io.FixedLenSequenceFeature",
           v1=["io.FixedLenSequenceFeature", "FixedLenSequenceFeature"])
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


class _ParseOpParams(object):
  """Raw parameters used by `gen_parsing_ops`.

  Attributes:
    sparse_keys: A list of string keys in the examples' features. The results
      for these keys will be returned as `SparseTensor` objects.
    sparse_types: A list of `DTypes` of the same length as `sparse_keys`. Only
      `tf.float32` (`FloatList`), `tf.int64` (`Int64List`), and `tf.string`
      (`BytesList`) are supported.
    dense_keys: A list of string keys in the examples' features. The results for
      these keys will be returned as `Tensor`s
    dense_types: A list of DTypes of the same length as `dense_keys`. Only
      `tf.float32` (`FloatList`), `tf.int64` (`Int64List`), and `tf.string`
      (`BytesList`) are supported.
    dense_defaults: A dict mapping string keys to `Tensor`s. The keys of the
      dict must match the dense_keys of the feature.
    dense_shapes: A list of tuples with the same length as `dense_keys`. The
      shape of the data for each dense feature referenced by `dense_keys`.
      Required for any input tensors identified by `dense_keys`.  Must be either
      fully defined, or may contain an unknown first dimension. An unknown first
      dimension means the feature is treated as having a variable number of
      blocks, and the output shape along this dimension is considered unknown at
      graph build time.  Padding is applied for minibatch elements smaller than
      the maximum number of blocks for the given feature along this dimension.
    ragged_keys: A list of string keys in the examples' features.  The
      results for these keys will be returned as `RaggedTensor` objects.
    ragged_value_types: A list of `DTypes` of the same length as `ragged_keys`,
      specifying the value type for each ragged feature.  Must be one of:
      `tf.float32`, `tf.int64`, `tf.string`.
    ragged_split_types: A list of `DTypes` of the same length as `ragged_keys`,
      specifying the row_splits type for each ragged feature.  Must be one of:
      `tf.int32`, `tf.int64`.
    dense_shapes_as_proto: dense_shapes converted to TensorShapeProto.
    dense_defaults_vec: A vector of `Tensor`s containing the default values,
      corresponding 1:1 with `dense_keys`.
    num_features: The total number of feature keys.
  """

  def __init__(self,
               sparse_keys=None,
               sparse_types=None,
               dense_keys=None,
               dense_types=None,
               dense_defaults=None,
               dense_shapes=None,
               ragged_keys=None,
               ragged_value_types=None,
               ragged_split_types=None):
    # Note: we use an OrderedDict for dense_defaults, to ensure consistent
    # graph construction order for _e2e_test.
    dense_defaults = (
        collections.OrderedDict() if dense_defaults is None else dense_defaults)
    sparse_keys = [] if sparse_keys is None else sparse_keys
    sparse_types = [] if sparse_types is None else sparse_types
    dense_keys = [] if dense_keys is None else dense_keys
    dense_types = [] if dense_types is None else dense_types
    dense_shapes = ([[]] *
                    len(dense_keys) if dense_shapes is None else dense_shapes)
    ragged_keys = [] if ragged_keys is None else ragged_keys
    ragged_value_types = ([]
                          if ragged_value_types is None else ragged_value_types)
    ragged_split_types = ([]
                          if ragged_split_types is None else ragged_split_types)
    self.sparse_keys = sparse_keys
    self.sparse_types = [dtypes.as_dtype(t) for t in sparse_types]
    self.dense_keys = dense_keys
    self.dense_types = [dtypes.as_dtype(t) for t in dense_types]
    self.dense_shapes = [tensor_shape.as_shape(s) for s in dense_shapes]
    self.dense_defaults = dense_defaults
    self.ragged_keys = ragged_keys
    self.ragged_value_types = [dtypes.as_dtype(t) for t in ragged_value_types]
    self.ragged_split_types = [dtypes.as_dtype(t) for t in ragged_split_types]
    self._validate()

  @classmethod
  def from_features(cls, features, types):
    """Builds _ParseOpParams for a given set of features and allowed types.

    Args:
      features: A `dict` mapping feature keys to objects of a type in `types`.
      types: Type of features to allow, among `FixedLenFeature`,
        `VarLenFeature`, `SparseFeature`, and `FixedLenSequenceFeature`.

    Returns:
      A `_ParseOpParams` containing the raw parameters for `gen_parsing_ops`.

    Raises:
      ValueError: if `features` contains an item not in `types`, or an invalid
          feature.
      ValueError: if sparse and dense key sets intersect.
      ValueError: if input lengths do not match up.
    """
    params = cls()
    if features:
      # NOTE: We iterate over sorted keys to keep things deterministic.
      for key in sorted(features.keys()):
        feature = features[key]
        if not isinstance(feature, tuple(types)):
          raise ValueError("Unsupported %s %s for key '%s')." %
                           (type(feature).__name__, feature, key))
        params._add_feature(key, feature)  # pylint: disable=protected-access
    params._validate()  # pylint: disable=protected-access
    return params

  @property
  def dense_shapes_as_proto(self):
    return [shape.as_proto() for shape in self.dense_shapes]

  @property
  def num_features(self):
    return len(self.dense_keys) + len(self.sparse_keys) + len(self.ragged_keys)

  @property
  def dense_defaults_vec(self):
    return [
        self._make_dense_default(k, s, t)
        for k, s, t in zip(self.dense_keys, self.dense_shapes, self.dense_types)
    ]

  def _make_dense_default(self, key, shape, dtype):
    """Construct the default value tensor for a specified dense feature.

    Args:
      key: The key string identifying the dense feature.
      shape: The dense feature's shape.
      dtype: The dense feature's dtype.

    Returns:
      A Tensor.
    """
    default_value = self.dense_defaults.get(key)
    if (shape.ndims is not None and shape.ndims > 0 and
        shape.dims[0].value is None):
      # Variable stride dense shape, the default value should be a
      # scalar padding value.
      if default_value is None:
        default_value = ops.convert_to_tensor(
            "" if dtype == dtypes.string else 0, dtype=dtype)
      else:
        # Reshape to a scalar to ensure user gets an error if they
        # provide a tensor that's not intended to be a padding value
        # (0 or 2+ elements).
        key_name = "padding_" + re.sub("[^A-Za-z0-9_.\\-/]", "_", key)
        default_value = ops.convert_to_tensor(
            default_value, dtype=dtype, name=key_name)
        default_value = array_ops.reshape(default_value, [])
    else:
      if default_value is None:
        default_value = constant_op.constant([], dtype=dtype)
      elif not isinstance(default_value, ops.Tensor):
        key_name = "key_" + re.sub("[^A-Za-z0-9_.\\-/]", "_", key)
        default_value = ops.convert_to_tensor(
            default_value, dtype=dtype, name=key_name)
        default_value = array_ops.reshape(default_value, shape)

    return default_value

  def _add_feature(self, key, feature):
    """Adds the specified feature to this ParseOpParams."""
    if isinstance(feature, VarLenFeature):
      self._add_varlen_feature(key, feature)
    elif isinstance(feature, SparseFeature):
      self._add_sparse_feature(key, feature)
    elif isinstance(feature, FixedLenFeature):
      self._add_fixed_len_feature(key, feature)
    elif isinstance(feature, FixedLenSequenceFeature):
      self._add_fixed_len_sequence_feature(key, feature)
    elif isinstance(feature, RaggedFeature):
      self._add_ragged_feature(key, feature)
    else:
      raise ValueError("Invalid feature %s:%s." % (key, feature))

  def _add_varlen_feature(self, key, feature):
    """Adds a VarLenFeature."""
    if not feature.dtype:
      raise ValueError("Missing type for feature %s." % key)
    self._add_sparse_key(key, feature.dtype)

  def _add_sparse_key(self, key, dtype):
    """Adds a sparse key & dtype, checking for duplicates."""
    if key in self.sparse_keys:
      original_dtype = self.sparse_types[self.sparse_keys.index(key)]
      if original_dtype != dtype:
        raise ValueError("Conflicting type %s vs %s for feature %s." %
                         (original_dtype, dtype, key))
    else:
      self.sparse_keys.append(key)
      self.sparse_types.append(dtype)

  def _add_sparse_feature(self, key, feature):
    """Adds a SparseFeature."""

    if not feature.index_key:
      raise ValueError("Missing index_key for SparseFeature %s." % (feature,))
    if not feature.value_key:
      raise ValueError("Missing value_key for SparseFeature %s." % (feature,))
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
      self._add_sparse_key(index_key, dtypes.int64)
    self._add_sparse_key(feature.value_key, feature.dtype)

  def _add_fixed_len_feature(self, key, feature):
    """Adds a FixedLenFeature."""
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
    self.dense_keys.append(key)
    self.dense_shapes.append(tensor_shape.as_shape(feature.shape))
    self.dense_types.append(feature.dtype)
    if feature.default_value is not None:
      self.dense_defaults[key] = feature.default_value

  def _add_fixed_len_sequence_feature(self, key, feature):
    """Adds a FixedLenSequenceFeature."""
    if not feature.dtype:
      raise ValueError("Missing type for feature %s." % key)
    if feature.shape is None:
      raise ValueError("Missing shape for feature %s." % key)
    self.dense_keys.append(key)
    self.dense_shapes.append(tensor_shape.as_shape(feature.shape))
    self.dense_types.append(feature.dtype)
    if feature.allow_missing:
      self.dense_defaults[key] = None
    if feature.default_value is not None:
      self.dense_defaults[key] = feature.default_value

  def _add_ragged_key(self, key, value_type, split_type):
    """Adds a ragged key & dtype, checking for duplicates."""
    if key in self.ragged_keys:
      original_value_type = self.ragged_value_types[self.ragged_keys.index(key)]
      original_split_type = self.ragged_split_types[self.ragged_keys.index(key)]
      if original_value_type != value_type:
        raise ValueError("Conflicting type %s vs %s for feature %s." %
                         (original_value_type, value_type, key))
      if original_split_type != split_type:
        raise ValueError("Conflicting partition type %s vs %s for feature %s." %
                         (original_split_type, split_type, key))
    else:
      self.ragged_keys.append(key)
      self.ragged_value_types.append(value_type)
      self.ragged_split_types.append(split_type)

  def _add_ragged_feature(self, key, feature):
    """Adds a RaggedFeature."""
    value_key = key if feature.value_key is None else feature.value_key
    self._add_ragged_key(value_key, feature.dtype, feature.row_splits_dtype)
    for partition in feature.partitions:
      if not isinstance(partition, RaggedFeature.UniformRowLength):
        self._add_ragged_key(partition.key, dtypes.int64,
                             feature.row_splits_dtype)

  def _validate(self):
    """Validates the features in this ParseOpParams."""
    if len(self.dense_shapes) != len(self.dense_keys):
      raise ValueError(
          "len(self.dense_shapes) != len(self.dense_keys): %d vs %d" %
          (len(self.dense_shapes), len(self.dense_keys)))
    if len(self.dense_types) != len(self.dense_keys):
      raise ValueError(
          "len(self.dense_types) != len(self.dense_keys): %d vs %d" %
          (len(self.dense_types), len(self.dense_keys)))
    if len(self.sparse_types) != len(self.sparse_keys):
      raise ValueError(
          "len(self.sparse_types) != len(self.sparse_keys): %d vs %d" %
          (len(self.sparse_types), len(self.sparse_keys)))
    if len(self.ragged_value_types) != len(self.ragged_keys):
      raise ValueError(
          "len(self.ragged_value_types) != len(self.ragged_keys): %d vs %d" %
          (len(self.ragged_value_types), len(self.ragged_keys)))
    if len(self.ragged_split_types) != len(self.ragged_keys):
      raise ValueError(
          "len(self.ragged_split_types) != len(self.ragged_keys): %d vs %d" %
          (len(self.ragged_split_types), len(self.ragged_keys)))

    dense_key_set = set(self.dense_keys)
    sparse_key_set = set(self.sparse_keys)
    ragged_key_set = set(self.ragged_keys)
    if not dense_key_set.isdisjoint(sparse_key_set):
      raise ValueError(
          "Dense and sparse keys must not intersect; intersection: %s" %
          dense_key_set.intersection(sparse_key_set))
    if not dense_key_set.isdisjoint(ragged_key_set):
      raise ValueError(
          "Dense and ragged keys must not intersect; intersection: %s" %
          dense_key_set.intersection(ragged_key_set))
    if not ragged_key_set.isdisjoint(sparse_key_set):
      raise ValueError(
          "Ragged and sparse keys must not intersect; intersection: %s" %
          ragged_key_set.intersection(sparse_key_set))


def _construct_tensors_for_composite_features(features, tensor_dict):
  """Creates tensors for SparseFeatures and RaggedFeatures.

  Constructs new dict based on `tensor_dict`.

  For each key in `features` whose value is a `SparseFeature`:

    * Looks up that SparseFeature's value_key and index_keys in tensor_dict.
    * Uses those tensors to construct a single SparseTensor.
    * Stores that SparseTensor in the output dict under the same key.

  For each key in `features` whose value is a `RaggedFeature`:

    * Looks up that RaggedFeature's value_key and partition keys in tensor_dict.
    * Uses those tensors to construct a single RaggedTensor.
    * Stores that RaggedTensor in the output dict under the same key.

  For any other key in `features`:

    * Copies that key and its value from tensor_dict to the output dictionary.

  Args:
    features: A `dict` mapping feature keys to `SparseFeature` or
      `RaggedFeature` values.  Values of other types will be ignored.
    tensor_dict: A `dict` mapping feature keys to `Tensor`, `SparseTensor`, and
      `RaggedTensor` values.  Expected to contain keys of the `SparseFeature`s'
      `index_key`s and `value_key`s and mapping them to `SparseTensor`s.

  Returns:
    A `dict` mapping feature keys to `Tensor`, `SparseTensor`, and
    `RaggedTensor` values. Similar to `tensor_dict` except each `SparseFeature`
    in `features` results in a single `SparseTensor`; and each `RaggedFeature`
    in `features` results in a single `RaggedTensor`.
  """
  tensor_dict = dict(tensor_dict)  # Do not modify argument passed in.
  updates = {}
  for key in sorted(features.keys()):
    feature = features[key]
    if isinstance(feature, SparseFeature):
      # Construct SparseTensors for SparseFeatures
      if isinstance(feature.index_key, str):
        sp_ids = tensor_dict[feature.index_key]
      else:
        sp_ids = [tensor_dict[index_key] for index_key in feature.index_key]
      sp_values = tensor_dict[feature.value_key]
      updates[key] = sparse_ops.sparse_merge(
          sp_ids,
          sp_values,
          vocab_size=feature.size,
          already_sorted=feature.already_sorted)
    elif isinstance(feature, RaggedFeature):
      # Construct RaggedTensors for RaggedFeatures.
      value_key = key if feature.value_key is None else feature.value_key
      rt = tensor_dict[value_key]
      if isinstance(rt, ragged_tensor.RaggedTensor):
        # We processed a batch of tf.Example or tf.SequenceExample, or single
        # tf.SequenceExample.
        if rt.ragged_rank > 1:
          # We're processing a batch of SequenceExample, and we effectively have
          # two batch dimensions.  Cllapse those batch dimensions here, and
          # restore them below (using outer_splits).
          outer_splits = rt.row_splits
          rt = rt.values
        else:
          outer_splits = None
        for partition in reversed(feature.partitions):
          rt = _add_batched_ragged_partition(rt, partition, tensor_dict,
                                             key, feature.validate,
                                             outer_splits)
        if outer_splits is not None:
          rt = ragged_tensor.RaggedTensor.from_row_splits(
              rt, outer_splits, validate=feature.validate)
      else:
        # We processed a single tf.Example.
        for partition in reversed(feature.partitions):
          rt = _add_ragged_partition(rt, partition, tensor_dict,
                                     feature.row_splits_dtype, feature.validate)
      updates[key] = rt

  # Process updates after all composite tensors have been constructed (in case
  # multiple features use the same value_key, and one uses that key as its
  # feature key).
  tensor_dict.update(updates)

  # Remove tensors from dictionary that were only used to construct
  # tensors for SparseFeature or RaggedTensor.
  for key in set(tensor_dict) - set(features):
    del tensor_dict[key]
  return tensor_dict


def _add_ragged_partition(values, partition, tensor_dict, row_splits_dtype,
                          validate):
  """Creates a RaggedTensor from a values tensor and a partition tensor.

  Args:
    values: The values tensor for the new RaggedTensor.
    partition: The partition configuration object.  Specifies the key that
      should be used to look up the partition tensor (unless partition is a
      RaggedFeature.UniformRowLength, in which case there is no partition
      tensor).
    tensor_dict: The dictionary mapping keys to tensors.
    row_splits_dtype: The dtype for the partition tensor.
    validate: Whether to validate that the values form a valid RaggedTensor.

  Returns:
    A new RaggedTensor formed from the values and partition tensors.
  """
  if isinstance(partition, RaggedFeature.UniformRowLength):
    if isinstance(values, ragged_tensor.RaggedTensor):
      length = ops.convert_to_tensor(partition.length, dtype=row_splits_dtype)
      return ragged_tensor.RaggedTensor.from_uniform_row_length(
          values, length, validate=validate)
    else:
      return array_ops.reshape(values, array_ops.concat(
          [[-1, partition.length], array_ops.shape(values)[1:]], axis=0))
  else:
    partition_t = math_ops.cast(tensor_dict[partition.key], row_splits_dtype)
    if isinstance(partition, RaggedFeature.RowSplits):
      return ragged_tensor.RaggedTensor.from_row_splits(
          values, partition_t, validate=validate)
    elif isinstance(partition, RaggedFeature.RowLengths):
      return ragged_tensor.RaggedTensor.from_row_lengths(
          values, partition_t, validate=validate)
    elif isinstance(partition, RaggedFeature.RowStarts):
      return ragged_tensor.RaggedTensor.from_row_starts(
          values, partition_t, validate=validate)
    elif isinstance(partition, RaggedFeature.RowLimits):
      return ragged_tensor.RaggedTensor.from_row_limits(
          values, partition_t, validate=validate)
    elif isinstance(partition, RaggedFeature.ValueRowIds):
      return ragged_tensor.RaggedTensor.from_value_rowids(
          values, partition_t, validate=validate)
    raise ValueError("Unhandled partition type %r" % partition)


def _add_batched_ragged_partition(rt, partition, tensor_dict, feature_key,
                                  validate, outer_splits=None):
  """Adds a batched ragged partition tensor to a batched ragged tensor.

  Args:
    rt: A RaggedTensor with shape [batch_size, ...].
    partition: The partition configuration object.  Specifies the key that
      should be used to look up the partition tensor (unless partition is a
      RaggedFeature.UniformRowLength, in which case there is no partition
      tensor).  The specified tensor must have shape [batch_size, ...].
    tensor_dict: The dictionary mapping keys to tensors.
    feature_key: The name of the feature being parsed (for error messages).
    validate: Whether to validate that the values form a valid RaggedTensor.
    outer_splits: If not None, then we have two batch dimensions, and this
      is the row-splits for the collapsed batch dimension.  Every partition
      tensor must have an outer row_splits that matches this value.

  Returns:
    A new RaggedTensor where each batch item `rt[i]` has been partitioned
    using the `partition_t[i]`.
  """
  if isinstance(partition, RaggedFeature.UniformRowLength):
    if rt.ragged_rank > 1:
      length = ops.convert_to_tensor(partition.length, rt.row_splits.dtype)
      return ragged_tensor.RaggedTensor.from_row_splits(
          ragged_tensor.RaggedTensor.from_uniform_row_length(
              rt.values, length, validate=validate),
          rt.row_splits // length,
          validate=validate)
    else:
      reshaped_vals = array_ops.reshape(rt.values, array_ops.concat(
          [[-1, partition.length], array_ops.shape(rt.values)[1:]], axis=0))
      return ragged_tensor.RaggedTensor.from_row_splits(
          reshaped_vals, rt.row_splits // partition.length, validate=validate)

  partition_t = tensor_dict[partition.key]
  if partition_t.values.dtype != rt.row_splits.dtype:
    partition_t = math_ops.cast(partition_t, rt.row_splits.dtype)

  checks = []
  if outer_splits is not None:
    if validate:
      checks.append(check_ops.assert_equal(
          outer_splits, partition_t.row_splits,
          message="Feature %s: values and partitions are not aligned"
          % feature_key))
    partition_t = partition_t.values

  with ops.control_dependencies(checks):
    if isinstance(partition, (RaggedFeature.RowSplits,
                              RaggedFeature.RowLimits)):
      if isinstance(partition, RaggedFeature.RowSplits):
        partition_t = partition_t[:, 1:]
      adjusted_limits = partition_t.values + array_ops.repeat(
          rt.row_starts(), partition_t.row_lengths())
      return partition_t.with_values(
          ragged_tensor.RaggedTensor.from_row_limits(
              rt.values, adjusted_limits, validate=validate))
    elif isinstance(partition, RaggedFeature.RowStarts):
      adjusted_starts = partition_t.values + array_ops.repeat(
          rt.row_starts(), partition_t.row_lengths())
      return partition_t.with_values(
          ragged_tensor.RaggedTensor.from_row_starts(
              rt.values, adjusted_starts, validate=validate))
    elif isinstance(partition, RaggedFeature.RowLengths):
      return partition_t.with_values(
          ragged_tensor.RaggedTensor.from_row_lengths(
              rt.values, partition_t.values, validate=validate))
    elif isinstance(partition, RaggedFeature.ValueRowIds):
      nrows = math_ops.maximum(  # number of rows in each batch item
          ragged_math_ops.reduce_max(partition_t + 1, axis=1), 0)
      adjusted_rowids = partition_t.values + array_ops.repeat(
          math_ops.cumsum(nrows, exclusive=True), partition_t.row_lengths())
      return ragged_tensor.RaggedTensor.from_row_lengths(
          ragged_tensor.RaggedTensor.from_value_rowids(
              rt.values, adjusted_rowids, validate=validate),
          nrows,
          validate=validate)

    raise ValueError("Unhandled partition type %r" % partition)


def _build_ragged_tensors(serialized_shape,
                          ragged_values,
                          ragged_row_splits,
                          ragged_inner_splits=None):
  """Builds RaggedTensors from the outputs of a parse op."""
  if ragged_inner_splits is not None:
    ragged_values = [
        ragged_tensor.RaggedTensor.from_row_splits(val, split, validate=False)
        for (val, split) in zip(ragged_values, ragged_inner_splits)
    ]
  if serialized_shape.ndims == 0:
    return ragged_values
  else:
    return [
        ragged_tensor.RaggedTensor.from_row_splits(val, split, validate=False)
        for (val, split) in zip(ragged_values, ragged_row_splits)
    ]
