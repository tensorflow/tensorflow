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
"""This API defines FeatureColumn abstraction.

To distinguish the concept of a feature family and a specific binary feature
within a family, we refer to a feature family like "country" as a feature
column. For example "country:US" is a feature which is in "country" feature
column and has a feature value ("US").

Supported feature types are:
 * _SparseColumn: also known as categorical features.
 * _RealValuedColumn: also known as continuous features.

Supported transformations on above features are:
 * Bucketization: also known as binning.
 * Crossing: also known as composition or union.
 * Embedding.

Typical usage example:

  ```python
  # Define features and transformations
  country = sparse_column_with_keys(column_name="native_country",
                                    keys=["US", "BRA", ...])
  country_emb = embedding_column(sparse_id_column=country, dimension=3,
                                 combiner="sum")
  occupation = sparse_column_with_hash_bucket(column_name="occupation",
                                              hash_bucket_size=1000)
  occupation_emb = embedding_column(sparse_id_column=occupation, dimension=16,
                                   combiner="sum")
  occupation_x_country = crossed_column(columns=[occupation, country],
                                        hash_bucket_size=10000)
  age = real_valued_column("age")
  age_buckets = bucketized_column(
      source_column=age,
      boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

  my_features = [occupation_emb, age_buckets, country_emb]
  # Building model via layers
  columns_to_tensor = parse_feature_columns_from_examples(
      serialized=my_data,
      feature_columns=my_features)
  first_layer = input_from_feature_columns(
      columns_to_tensors=columns_to_tensor,
      feature_columns=my_features)
  second_layer = fully_connected(first_layer, ...)

  # Building model via tf.learn.estimators
  estimator = DNNLinearCombinedClassifier(
      linear_feature_columns=my_wide_features,
      dnn_feature_columns=my_deep_features,
      dnn_hidden_units=[500, 250, 50])
  estimator.train(...)

  See feature_column_ops_test for more examples.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import math

from tensorflow.contrib.framework.python.ops import embedding_ops as contrib_embedding_ops
from tensorflow.contrib.layers.python.ops import bucketization_op
from tensorflow.contrib.layers.python.ops import sparse_feature_cross_op
from tensorflow.contrib.lookup import lookup_ops as contrib_lookup_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging


class _FeatureColumn(object):
  """Represents a feature column abstraction.

  To distinguish the concept of a feature family and a specific binary feature
  within a family, we refer to a feature family like "country" as a feature
  column. For example "country:US" is a feature which is in "country" feature
  column and has a feature value ("US").
  This class is an abstract class. User should not create one instance of this.
  Following classes (_SparseColumn, _RealValuedColumn, ...) are concrete
  instances.
  """
  __metaclass__ = abc.ABCMeta

  @abc.abstractproperty
  def name(self):
    """Returns the name of column or transformed column."""
    pass

  @abc.abstractproperty
  def config(self):
    """Returns configuration of the base feature for `tf.parse_example`."""
    pass

  @abc.abstractproperty
  def key(self):
    """Returns a string which will be used as a key when we do sorting."""
    pass

  @abc.abstractmethod
  def insert_transformed_feature(self, columns_to_tensors):
    """Apply transformation and inserts it into columns_to_tensors.

    Args:
      columns_to_tensors: A mapping from feature columns to tensors. 'string'
        key means a base feature (not-transformed). It can have _FeatureColumn
        as a key too. That means that _FeatureColumn is already transformed.
    """
    raise NotImplementedError("Transform is not implemented for {}.".format(
        self))

  @abc.abstractmethod
  def to_dnn_input_layer(self,
                         input_tensor,
                         weight_collection=None,
                         trainable=True):
    """Returns a Tensor as an input to the first layer of neural network."""
    raise ValueError("Calling an abstract method.")

  @abc.abstractmethod
  def to_weighted_sum(self,
                      input_tensor,
                      num_outputs=1,
                      weight_collections=None,
                      trainable=True):
    """Returns a Tensor as linear predictions and a list of created Variable."""
    raise ValueError("Calling an abstract method.")


class _SparseColumn(_FeatureColumn,
                    collections.namedtuple("_SparseColumn",
                                           ["column_name", "is_integerized",
                                            "bucket_size", "lookup_config",
                                            "weight_column", "combiner",
                                            "dtype"])):
  """"Represents a sparse feature column also known as categorical features.

  Instances of this class are immutable. A sparse column means features are
  sparse and dictionary returned by InputBuilder contains a
  ("column_name", SparseTensor) pair.
  One and only one of bucket_size or lookup_config should be set. If
  is_integerized is True then bucket_size should be set.

  Attributes:
    column_name: A string defining sparse column name.
    is_integerized: A bool if True means type of feature is an integer.
      Integerized means we can use the feature itself as id.
    bucket_size: An int that is > 1. The number of buckets.
    lookup_config: A _SparseIdLookupConfig defining feature-to-id lookup
      configuration
    weight_column: A string defining a sparse column name which represents
      weight or value of the corresponding sparse feature. Please check
      weighted_sparse_column for more information.
    combiner: A string specifying how to reduce if the sparse column is
      multivalent. Currently "mean", "sqrtn" and "sum" are supported, with
      "sum" the default:
        * "sum": do not normalize features in the column
        * "mean": do l1 normalization on features in the column
        * "sqrtn": do l2 normalization on features in the column
      For more information: `tf.embedding_lookup_sparse`.
    dtype: Type of features, such as `tf.string` or `tf.int64`.

  Raises:
    TypeError: if lookup_config is not a _SparseIdLookupConfig.
    ValueError: if above expectations about input fails.
  """

  def __new__(cls,
              column_name,
              is_integerized=False,
              bucket_size=None,
              lookup_config=None,
              weight_column=None,
              combiner="sum",
              dtype=dtypes.string):
    if is_integerized and bucket_size is None:
      raise ValueError("bucket_size should be set if is_integerized=True. "
                       "column_name: {}".format(column_name))

    if is_integerized and not dtype.is_integer:
      raise ValueError("dtype should be an integer if is_integerized is True. "
                       "Column {}.".format(column_name))

    if bucket_size is None and lookup_config is None:
      raise ValueError("one of bucket_size or lookup_config should be "
                       "set. column_name: {}".format(column_name))

    if bucket_size is not None and lookup_config:
      raise ValueError("one and only one of bucket_size or lookup_config "
                       "should be set. column_name: {}".format(column_name))

    if bucket_size is not None and bucket_size < 2:
      raise ValueError("bucket_size should be at least 2. "
                       "column_name: {}".format(column_name))

    if ((lookup_config) and
        (not isinstance(lookup_config, _SparseIdLookupConfig))):
      raise TypeError(
          "lookup_config should be an instance of _SparseIdLookupConfig. "
          "Given one is in type {} for column_name {}".format(
              type(lookup_config), column_name))

    if (lookup_config and lookup_config.vocabulary_file and
        lookup_config.vocab_size is None):
      raise ValueError("vocab_size should be defined. "
                       "column_name: {}".format(column_name))

    return super(_SparseColumn, cls).__new__(cls, column_name, is_integerized,
                                             bucket_size, lookup_config,
                                             weight_column, combiner, dtype)

  @property
  def name(self):
    return self.column_name

  @property
  def length(self):
    """Returns vocabulary or hash_bucket size."""
    if self.bucket_size is not None:
      return self.bucket_size
    return self.lookup_config.vocab_size + self.lookup_config.num_oov_buckets

  @property
  def config(self):
    return {self.column_name: parsing_ops.VarLenFeature(self.dtype)}

  @property
  def key(self):
    """Returns a string which will be used as a key when we do sorting."""
    return "{}".format(self)

  # pylint: disable=unused-argument
  def to_dnn_input_layer(self,
                         input_tensor,
                         weight_collections=None,
                         trainable=True):
    raise ValueError("Column {} is not supported in DNN. "
                     "Please use embedding_column.".format(self))

  def to_weighted_sum(self,
                      input_tensor,
                      num_outputs=1,
                      weight_collections=None,
                      trainable=True):
    return _create_embedding_lookup(
        input_tensor, self.length, num_outputs,
        _add_variable_collection(weight_collections), 0., self.combiner,
        trainable, self.name + "_weights")


class _SparseColumnIntegerized(_SparseColumn):
  """See `sparse_column_with_integerized_feature`."""

  def __new__(cls,
              column_name,
              bucket_size,
              combiner="sum",
              dtype=dtypes.int64):
    if not dtype.is_integer:
      raise ValueError("dtype should be an integer. Given {}".format(
          column_name))

    return super(_SparseColumnIntegerized, cls).__new__(cls,
                                                        column_name,
                                                        is_integerized=True,
                                                        bucket_size=bucket_size,
                                                        combiner=combiner,
                                                        dtype=dtype)

  def insert_transformed_feature(self, columns_to_tensors):
    """Handles sparse column to id conversion."""
    sparse_id_values = math_ops.mod(columns_to_tensors[self.name].values,
                                    self.bucket_size)
    columns_to_tensors[self] = ops.SparseTensor(
        columns_to_tensors[self.name].indices, sparse_id_values,
        columns_to_tensors[self.name].shape)


def sparse_column_with_integerized_feature(column_name,
                                           bucket_size,
                                           combiner="sum",
                                           dtype=dtypes.int64):
  """Creates an integerized _SparseColumn.

  Use this when your features are already pre-integerized into int64 IDs.
  output_id = input_feature

  Args:
    column_name: A string defining sparse column name.
    bucket_size: An int that is > 1. The number of buckets. It should be bigger
      than maximum feature. In other words features in this column should be an
      int64 in range [0, bucket_size)
    combiner: A string specifying how to reduce if the sparse column is
      multivalent. Currently "mean", "sqrtn" and "sum" are supported, with
      "sum" the default:
        * "sum": do not normalize features in the column
        * "mean": do l1 normalization on features in the column
        * "sqrtn": do l2 normalization on features in the column
      For more information: `tf.embedding_lookup_sparse`.
    dtype: Type of features. It should be an integer type. Default value is
      dtypes.int64.

  Returns:
    An integerized _SparseColumn definition.

  Raises:
    ValueError: bucket_size is not greater than 1.
    ValueError: dtype is not integer.
  """
  return _SparseColumnIntegerized(column_name,
                                  bucket_size,
                                  combiner=combiner,
                                  dtype=dtype)


class _SparseColumnHashed(_SparseColumn):
  """See `sparse_column_with_hash_bucket`."""

  def __new__(cls, column_name, hash_bucket_size, combiner="sum"):

    return super(_SparseColumnHashed, cls).__new__(cls,
                                                   column_name,
                                                   bucket_size=hash_bucket_size,
                                                   combiner=combiner,
                                                   dtype=dtypes.string)

  def insert_transformed_feature(self, columns_to_tensors):
    """Handles sparse column to id conversion."""
    sparse_id_values = string_ops.string_to_hash_bucket_fast(
        columns_to_tensors[self.name].values,
        self.bucket_size,
        name=self.name + "_lookup")
    columns_to_tensors[self] = ops.SparseTensor(
        columns_to_tensors[self.name].indices, sparse_id_values,
        columns_to_tensors[self.name].shape)


def sparse_column_with_hash_bucket(column_name,
                                   hash_bucket_size,
                                   combiner="sum"):
  """Creates a _SparseColumn with hashed bucket configuration.

  Use this when your sparse features are in string format, but you don't have a
  vocab file that maps each string to an integer ID.
  output_id = Hash(input_feature_string) % bucket_size

  Args:
    column_name: A string defining sparse column name.
    hash_bucket_size: An int that is > 1. The number of buckets.
    combiner: A string specifying how to reduce if the sparse column is
      multivalent. Currently "mean", "sqrtn" and "sum" are supported, with
      "sum" the default:
        * "sum": do not normalize features in the column
        * "mean": do l1 normalization on features in the column
        * "sqrtn": do l2 normalization on features in the column
      For more information: `tf.embedding_lookup_sparse`.

  Returns:
    A _SparseColumn with hashed bucket configuration

  Raises:
    ValueError: hash_bucket_size is not greater than 2.
  """
  return _SparseColumnHashed(column_name, hash_bucket_size, combiner)


class _SparseColumnKeys(_SparseColumn):
  """See `sparse_column_with_keys`."""

  def __new__(cls,
              column_name,
              keys,
              default_value=-1,
              combiner="sum"):
    return super(_SparseColumnKeys, cls).__new__(
        cls,
        column_name,
        combiner=combiner,
        lookup_config=_SparseIdLookupConfig(keys=keys,
                                            vocab_size=len(keys),
                                            default_value=default_value),
        dtype=dtypes.string)

  def insert_transformed_feature(self, columns_to_tensors):
    """Handles sparse column to id conversion."""
    columns_to_tensors[self] = contrib_lookup_ops.string_to_index(
        tensor=columns_to_tensors[self.name],
        mapping=list(self.lookup_config.keys),
        default_value=self.lookup_config.default_value,
        name=self.name + "_lookup")


def sparse_column_with_keys(column_name,
                            keys,
                            default_value=-1,
                            combiner="sum"):
  """Creates a _SparseColumn with keys.

  Look up logic is as follows:
  lookup_id = index_of_feature_in_keys if feature in keys else default_value

  Args:
    column_name: A string defining sparse column name.
    keys: a string list defining vocabulary.
    default_value: The value to use for out-of-vocabulary feature values.
      Default is -1.
    combiner: A string specifying how to reduce if the sparse column is
      multivalent. Currently "mean", "sqrtn" and "sum" are supported, with
      "sum" the default:
        * "sum": do not normalize features in the column
        * "mean": do l1 normalization on features in the column
        * "sqrtn": do l2 normalization on features in the column
      For more information: `tf.embedding_lookup_sparse`.

  Returns:
    A _SparseColumnKeys with keys configuration.
  """
  return _SparseColumnKeys(column_name,
                           tuple(keys),
                           default_value=default_value,
                           combiner=combiner)


class _EmbeddingColumn(_FeatureColumn, collections.namedtuple(
    "_EmbeddingColumn",
    ["sparse_id_column", "dimension", "combiner", "stddev"])):
  """Represents an embedding column.

  Args:
    sparse_id_column: A _SparseColumn which is created by `sparse_column_with_*`
      functions.
    dimension: An integer specifying dimension of the embedding.
    combiner: A string specifying how to reduce if there are multiple entries
      in a single row. Currently "mean", "sqrtn" and "sum" are supported. Each
      of this can be thought as example level normalizations on the column:
        * "sum": do not normalize features in the column
        * "mean": do l1 normalization on features in the column
        * "sqrtn": do l2 normalization on features in the column
      For more information: `tf.embedding_lookup_sparse`.
    stddev: the standard deviation to be used in embedding initialization.
      Default is 1/sqrt(sparse_id_column.length).
  """

  def __new__(cls, sparse_id_column, dimension, combiner="mean", stddev=None):
    if stddev is None:
      stddev = 1 / math.sqrt(sparse_id_column.length)
    return super(_EmbeddingColumn, cls).__new__(cls, sparse_id_column,
                                                dimension, combiner, stddev)

  @property
  def name(self):
    return self.sparse_id_column.name + "_embedding"

  @property
  def length(self):
    """Returns id size."""
    return self.sparse_id_column.length

  @property
  def config(self):
    return _get_feature_config(self.sparse_id_column)

  @property
  def key(self):
    """Returns a string which will be used as a key when we do sorting."""
    return "{}".format(self)

  def insert_transformed_feature(self, columns_to_tensors):
    self.sparse_id_column.insert_transformed_feature(columns_to_tensors)
    columns_to_tensors[self] = columns_to_tensors[self.sparse_id_column]

  def to_dnn_input_layer(self,
                         input_tensor,
                         weight_collections=None,
                         trainable=True):
    output, _ = _create_embedding_lookup(
        input_tensor, self.length, self.dimension,
        _add_variable_collection(weight_collections), self.stddev,
        self.combiner, trainable, self.name + "_weights")
    return output

  # pylint: disable=unused-argument
  def to_weighted_sum(self,
                      input_tensor,
                      num_outputs=1,
                      weight_collections=None,
                      trainable=True):
    raise ValueError("Column {} is not supported in linear models. "
                     "Please use sparse_column.".format(self))


def embedding_column(sparse_id_column, dimension, combiner="mean", stddev=None):
  """Creates an _EmbeddingColumn.

  Args:
    sparse_id_column: A _SparseColumn which is created by `sparse_column_with_*`
      functions. Note that `combiner` defined in `sparse_id_column` is ignored.
    dimension: An integer specifying dimension of the embedding.
    combiner: A string specifying how to reduce if there are multiple entries
      in a single row. Currently "mean", "sqrtn" and "sum" are supported. Each
      of this can be thought as example level normalizations on the column:
        * "sum": do not normalize
        * "mean": do l1 normalization
        * "sqrtn": do l2 normalization
      For more information: `tf.embedding_lookup_sparse`.
    stddev: the standard deviation to be used in embedding initialization.
      Default is 1/sqrt(sparse_id_column.length).

  Returns:
    An _EmbeddingColumn.
  """
  return _EmbeddingColumn(sparse_id_column, dimension, combiner, stddev)


class _RealValuedColumn(_FeatureColumn, collections.namedtuple(
    "_RealValuedColumn",
    ["column_name", "dimension", "default_value", "dtype"])):
  """Represents a real valued feature column also known as continuous features.

  Instances of this class are immutable. A real valued column means features are
  dense. It means dictionary returned by InputBuilder contains a
  ("column_name", Tensor) pair. Tensor shape should be (batch_size, 1).
  """

  def __new__(cls, column_name, dimension, default_value, dtype):
    if default_value is not None:
      default_value = tuple(default_value)
    return super(_RealValuedColumn, cls).__new__(cls, column_name, dimension,
                                                 default_value, dtype)

  @property
  def name(self):
    return self.column_name

  @property
  def config(self):
    default_value = self.default_value
    if default_value is not None:
      default_value = list(default_value)
    return {self.column_name: parsing_ops.FixedLenFeature(
        [self.dimension], self.dtype, default_value)}

  @property
  def key(self):
    """Returns a string which will be used as a key when we do sorting."""
    return "{}".format(self)

  def insert_transformed_feature(self, columns_to_tensors):
    # No transformation is needed for _RealValuedColumn except reshaping.
    input_tensor = columns_to_tensors[self.name]
    batch_size = input_tensor.get_shape().as_list()[0]
    batch_size = int(batch_size) if batch_size else -1
    flattened_shape = [batch_size, self.dimension]
    columns_to_tensors[self] = array_ops.reshape(
        math_ops.to_float(input_tensor), flattened_shape)

  # pylint: disable=unused-argument
  def to_dnn_input_layer(self,
                         input_tensor,
                         weight_collections=None,
                         trainable=True):
    return input_tensor

  def to_weighted_sum(self,
                      input_tensor,
                      num_outputs=1,
                      weight_collections=None,
                      trainable=True):
    """Returns a Tensor as linear predictions and a list of created Variable."""
    weight = variables.Variable(
        array_ops.zeros([self.dimension, num_outputs]),
        collections=_add_variable_collection(weight_collections),
        name=self.name + "_weight")
    # The _RealValuedColumn has the shape of [batch_size, column.dimension].
    feature_by_dim = array_ops.reshape(
        math_ops.to_float(input_tensor), [-1, 1, self.dimension])
    log_odds_by_dim = (array_ops.transpose(weight) * feature_by_dim)
    # Sum over all the dimensions.
    return math_ops.reduce_sum(log_odds_by_dim, 2), [weight]


def real_valued_column(column_name,
                       dimension=None,
                       default_value=None,
                       dtype=dtypes.float32):
  """Creates a _RealValuedColumn.

  Args:
    column_name: A string defining real valued column name.
    dimension: An integer specifying dimension of the real valued column.
      The default is 1. The Tensor representing the _RealValuedColumn
      will have the shape of [batch_size, dimension].
    default_value: A signle value compatible with dtype or a list of values
      compatible with dtype which the column takes on if data is missing. If
      None, then tf.parse_example will fail if an example does not contain
      this column. If a single value is provided, the same value will be
      applied as the default value for every dimension. If a list of values
      is provided, the length of the list should be equal to the value of
      `dimension`.
    dtype: defines the type of values. Default value is tf.float32.
  Returns:
    A _RealValuedColumn.
  Raises:
    TypeError: if default_value is a list but its length is not equal to the
      value of `dimension`.
    TypeError: if default_value is not compatible with dtype.
    ValueError: if dtype is not convertable to tf.float32.
  """
  if dimension is None:
    dimension = 1

  if not (dtype.is_integer or dtype.is_floating):
    raise ValueError("dtype is not convertible to tf.float32. Given {}".format(
        dtype))

  if default_value is None:
    return _RealValuedColumn(column_name, dimension, default_value, dtype)

  if isinstance(default_value, int):
    if dtype.is_integer:
      default_value = [default_value for _ in range(dimension)]
      return _RealValuedColumn(column_name, dimension, default_value, dtype)
    if dtype.is_floating:
      default_value = float(default_value)
      default_value = [default_value for _ in range(dimension)]
      return _RealValuedColumn(column_name, dimension, default_value, dtype)

  if isinstance(default_value, float):
    if dtype.is_floating and (not dtype.is_integer):
      default_value = [default_value for _ in range(dimension)]
      return _RealValuedColumn(column_name, dimension, default_value, dtype)

  if isinstance(default_value, list):
    if len(default_value) != dimension:
      raise ValueError("The length of default_value is not equal to the "
                       "value of dimension. default_value is {}.".format(
                           default_value))
    # Check if the values in the list are all integers or are convertible to
    # floats.
    is_list_all_int = True
    is_list_all_float = True
    for v in default_value:
      if not isinstance(v, int):
        is_list_all_int = False
      if not (isinstance(v, float) or isinstance(v, int)):
        is_list_all_float = False
    if is_list_all_int:
      if dtype.is_integer:
        return _RealValuedColumn(column_name, dimension, default_value, dtype)
      elif dtype.is_floating:
        default_value = [float(v) for v in default_value]
        return _RealValuedColumn(column_name, dimension, default_value, dtype)
    if is_list_all_float:
      if dtype.is_floating and (not dtype.is_integer):
        default_value = [float(v) for v in default_value]
        return _RealValuedColumn(column_name, dimension, default_value, dtype)

  raise TypeError("default_value is not compatible with dtype. "
                  "default_value is {}.".format(default_value))


class _BucketizedColumn(_FeatureColumn, collections.namedtuple(
    "_BucketizedColumn", ["source_column", "boundaries"])):
  """Represents a bucketization transformation also known as binning.

  Instances of this class are immutable. Values in `source_column` will be
  bucketized based on `boundaries`.
  For example, if the inputs are:
      boundaries = [0, 10, 100]
      source_column = [[-5], [150], [10], [0], [4], [19]]

  then the bucketized feature will be:
      output = [[0], [3], [2], [1], [1], [2]]

  Attributes:
    source_column: A _RealValuedColumn defining dense column.
    boundaries: A list of floats specifying the boundaries. It has to be sorted.
      [a, b, c] defines following buckets: (-inf., a), [a, b), [b, c), [c, inf.)
  Raises:
    ValueError: if 'boundaries' is empty or not sorted.
  """

  def __new__(cls, source_column, boundaries):
    if not isinstance(source_column, _RealValuedColumn):
      raise TypeError(
          "source_column should be an instance of _RealValuedColumn.")

    if not isinstance(boundaries, list) or not boundaries:
      raise ValueError("boundaries must be a list and it should not be empty.")

    # We allow bucket boundaries to be monotonically increasing
    # (ie a[i+1] >= a[i]). When two bucket boundaries are the same, we
    # de-duplicate.
    sanitized_boundaries = []
    for i in range(len(boundaries) - 1):
      if boundaries[i] == boundaries[i + 1]:
        continue
      elif boundaries[i] < boundaries[i + 1]:
        sanitized_boundaries.append(boundaries[i])
      else:
        raise ValueError("boundaries must be a sorted list")
    sanitized_boundaries.append(boundaries[len(boundaries) - 1])

    return super(_BucketizedColumn, cls).__new__(cls, source_column,
                                                 tuple(sanitized_boundaries))

  @property
  def name(self):
    return self.source_column.name + "_BUCKETIZED"

  @property
  def length(self):
    """Returns total number of buckets."""
    return len(self.boundaries) + 1

  @property
  def config(self):
    return self.source_column.config

  @property
  def key(self):
    """Returns a string which will be used as a key when we do sorting."""
    return "{}".format(self)

  def insert_transformed_feature(self, columns_to_tensors):
    # Bucketize the source column.
    if self.source_column not in columns_to_tensors:
      self.source_column.insert_transformed_feature(columns_to_tensors)
    columns_to_tensors[self] = bucketization_op.bucketize(
        columns_to_tensors[self.source_column],
        boundaries=list(self.boundaries))

  # pylint: disable=unused-argument
  def to_dnn_input_layer(self,
                         input_tensor,
                         weight_collections=None,
                         trainable=True):
    return array_ops.reshape(
        array_ops.one_hot(
            math_ops.to_int64(input_tensor), self.length, 1., 0.),
        [-1, self.length * self.source_column.dimension])

  def to_weighted_sum(self,
                      input_tensor,
                      num_outputs=1,
                      weight_collections=None,
                      trainable=True):
    """Returns a Tensor as linear predictions and a list of created Variable."""
    dimension = self.source_column.dimension
    bucket_size = len(self.boundaries) + 1
    weight = variables.Variable(
        array_ops.zeros([bucket_size * dimension, num_outputs]),
        collections=_add_variable_collection(weight_collections),
        name=self.name + "_weight")
    # input has the shape of [batch_size, dimension].
    one_hot = array_ops.one_hot(
        math_ops.to_int64(input_tensor), bucket_size, 1, 0)
    # one_hot has the shape of [batch_size, bucket_size * dimension, 1].
    one_hot = array_ops.reshape(one_hot, [-1, bucket_size * dimension, 1])
    # feature_by_dim has the shape of [batch_size, bucket_size * dimension,
    # num_classes].
    feature_by_dim = weight * math_ops.to_float(one_hot)
    return array_ops.reshape(
        math_ops.reduce_sum(feature_by_dim, 1), [-1, num_outputs]), [weight]


def bucketized_column(source_column, boundaries):
  """Creates a _BucketizedColumn.

  Args:
    source_column: A _RealValuedColumn defining dense column.
    boundaries: A list of floats specifying the boundaries. It has to be sorted.

  Returns:
    A _BucketizedColumn.

  Raises:
    ValueError: if 'boundaries' is empty or not sorted.
  """
  return _BucketizedColumn(source_column, boundaries)


class _CrossedColumn(_FeatureColumn, collections.namedtuple(
    "_CrossedColumn", ["columns", "hash_bucket_size", "combiner"])):
  """"Represents a cross transformation also known as composition or union.

  Instances of this class are immutable. It crosses given `columns`. Crossed
  column output will be hashed to hash_bucket_size.
  Conceptually, transformation can be thought as:
    Hash(cartesian product of features in columns) % `hash_bucket_size`

  For example, if the columns are

      SparseTensor referred by first column: shape = [2, 2]
      [0, 0]: "a"
      [1, 0]: "b"
      [1, 1]: "c"

      SparseTensor referred by second column: : shape = [2, 1]
      [0, 0]: "d"
      [1, 1]: "e"

  then crossed feature will look like:

      shape = [2, 2]
      [0, 0]: Hash64("d", Hash64("a")) % hash_bucket_size
      [1, 0]: Hash64("e", Hash64("b")) % hash_bucket_size
      [1, 1]: Hash64("e", Hash64("c")) % hash_bucket_size

  Attributes:
    columns: An iterable of _FeatureColumn. Items can be an instance of
      _SparseColumn, _CrossedColumn, or _BucketizedColumn.
    hash_bucket_size: An int that is > 1. The number of buckets.
    combiner: A string specifying how to reduce if there are multiple entries
      in a single row. Currently "mean", "sqrtn" and "sum" are supported. Each
      of this can be thought as example level normalizations on the column:
        * "sum": do not normalize
        * "mean": do l1 normalization
        * "sqrtn": do l2 normalization
      For more information: `tf.embedding_lookup_sparse`.

  Raises:
    TypeError: if all items in columns are not an instance of _SparseColumn,
      _CrossedColumn, or _BucketizedColumn or
      hash_bucket_size is not an int.
    ValueError: if hash_bucket_size is not > 1 or
      len(columns) is not > 1.
  """

  @staticmethod
  def _is_crossable(column):
    return isinstance(column,
                      (_SparseColumn, _CrossedColumn, _BucketizedColumn))

  def __new__(cls, columns, hash_bucket_size, combiner="sum"):
    for column in columns:
      if not _CrossedColumn._is_crossable(column):
        raise TypeError("columns should be a set of "
                        "_SparseColumn, _CrossedColumn, or _BucketizedColumn. "
                        "Column is {}".format(column))

    if len(columns) < 2:
      raise ValueError("columns should contain at least 2 elements.")

    if not isinstance(hash_bucket_size, int):
      raise TypeError("hash_bucket_size should be an int.")

    if hash_bucket_size < 2:
      raise ValueError("hash_bucket_size should be at least 2.")

    sorted_columns = sorted([column for column in columns],
                            key=lambda column: column.name)
    return super(_CrossedColumn, cls).__new__(cls, tuple(sorted_columns),
                                              hash_bucket_size, combiner)

  @property
  def name(self):
    sorted_names = sorted([column.name for column in self.columns])
    return "_X_".join(sorted_names)

  @property
  def config(self):
    config = {}
    for column in self.columns:
      config.update(_get_feature_config(column))
    return config

  @property
  def length(self):
    """Returns total number of buckets."""
    return self.hash_bucket_size

  @property
  def key(self):
    """Returns a string which will be used as a key when we do sorting."""
    return "{}".format(self)

  def insert_transformed_feature(self, columns_to_tensors):
    """Handles cross transformation."""

    def _collect_leaf_level_columns(cross):
      """Collects base columns contained in the cross."""
      leaf_level_columns = []
      for c in cross.columns:
        if isinstance(c, _CrossedColumn):
          leaf_level_columns.extend(_collect_leaf_level_columns(c))
        else:
          leaf_level_columns.append(c)
      return leaf_level_columns

    feature_tensors = []
    for c in _collect_leaf_level_columns(self):
      if isinstance(c, _SparseColumn):
        feature_tensors.append(columns_to_tensors[c.name])
      else:
        if c not in columns_to_tensors:
          c.insert_transformed_feature(columns_to_tensors)
        feature_tensors.append(columns_to_tensors[c])
    columns_to_tensors[self] = sparse_feature_cross_op.sparse_feature_cross(
        feature_tensors,
        hashed_output=True,
        num_buckets=self.hash_bucket_size)

  # pylint: disable=unused-argument
  def to_dnn_input_layer(self,
                         input_tensor,
                         weight_collections=None,
                         trainable=True):
    raise ValueError("Column {} is not supported in DNN. "
                     "Please use embedding_column.".format(self))

  def to_weighted_sum(self,
                      input_tensor,
                      num_outputs=1,
                      weight_collections=None,
                      trainable=True):
    return _create_embedding_lookup(
        input_tensor, self.length, num_outputs,
        _add_variable_collection(weight_collections), -1, self.combiner,
        trainable, self.name + "_weights")


def crossed_column(columns, hash_bucket_size, combiner="sum"):
  """Creates a _CrossedColumn.

  Args:
    columns: An iterable of _FeatureColumn. Items can be an instance of
      _SparseColumn, _CrossedColumn, or _BucketizedColumn.
    hash_bucket_size: An int that is > 1. The number of buckets.
    combiner: A combiner string, supports sum, mean, sqrtn.

  Returns:
    A _CrossedColumn.

  Raises:
    TypeError: if any item in columns is not an instance of _SparseColumn,
      _CrossedColumn, or _BucketizedColumn, or
      hash_bucket_size is not an int.
    ValueError: if hash_bucket_size is not > 1 or
      len(columns) is not > 1.
  """
  return _CrossedColumn(columns, hash_bucket_size, combiner=combiner)


def _get_feature_config(feature_column):
  """Returns configuration for the base feature defined in feature_column."""
  if not isinstance(feature_column, _FeatureColumn):
    raise TypeError(
        "feature_columns should only contain instances of _FeatureColumn. "
        "Given column is {}".format(feature_column))
  if isinstance(feature_column, (_SparseColumn, _EmbeddingColumn,
                                 _RealValuedColumn, _BucketizedColumn,
                                 _CrossedColumn)):
    return feature_column.config

  raise TypeError("Not supported _FeatureColumn type. "
                  "Given column is {}".format(feature_column))


def create_feature_spec_for_parsing(feature_columns):
  """Helper that prepares features config from input feature_columns.

  The returned feature config can be used as arg 'features' in tf.parse_example.

  Typical usage example:

  ```python
  # Define features and transformations
  country = sparse_column_with_vocabulary_file("country", VOCAB_FILE)
  age = real_valued_column("age")
  click_bucket = bucketized_column(real_valued_column("historical_click_ratio"),
                                   boundaries=[i/10. for i in range(10)])
  country_x_click = crossed_column([country, click_bucket], 10)

  feature_columns = set([age, click_bucket, country_x_click])
  batch_examples = tf.parse_example(
      serialized_examples,
      create_feature_spec_for_parsing(feature_columns))
  ```

  For the above example, create_feature_spec_for_parsing would return the dict:
  {"age": parsing_ops.FixedLenFeature([1], dtype=tf.float32),
   "historical_click_ratio": parsing_ops.FixedLenFeature([1], dtype=tf.float32),
   "country": parsing_ops.VarLenFeature(tf.string)}

  Args:
    feature_columns: An iterable containing all the feature columns. All items
      should be instances of classes derived from _FeatureColumn.
  Returns:
    A dict mapping feature keys to FixedLenFeature or VarLenFeature values.
  """
  features_config = {}
  for column in feature_columns:
    features_config.update(_get_feature_config(column))
  return features_config


def make_place_holder_tensors_for_base_features(feature_columns):
  """Returns placeholder tensors for inference.

  Args:
    feature_columns: An iterable containing all the feature columns. All items
      should be instances of classes derived from _FeatureColumn.
  Returns:
    A dict mapping feature keys to SparseTensors (sparse columns) or
    placeholder Tensors (dense columns).
  """
  # Get dict mapping features to FixedLenFeature or VarLenFeature values.
  dict_for_parse_example = create_feature_spec_for_parsing(feature_columns)
  placeholders = {}
  for column_name, column_type in dict_for_parse_example.items():
    if isinstance(column_type, parsing_ops.VarLenFeature):
      # Sparse placeholder for sparse tensors.
      placeholders[column_name] = array_ops.sparse_placeholder(
          column_type.dtype,
          name="Placeholder_" + column_name)
    else:
      # Simple placeholder for dense tensors.
      placeholders[column_name] = array_ops.placeholder(
          column_type.dtype,
          shape=(None, column_type.shape[0]),
          name="Placeholder_" + column_name)
  return placeholders


class _SparseIdLookupConfig(collections.namedtuple("_SparseIdLookupConfig",
                                                   ["vocabulary_file", "keys",
                                                    "num_oov_buckets",
                                                    "vocab_size",
                                                    "default_value"])):
  """Defines lookup configuration for a sparse feature.

  An immutable object defines lookup table configuration used by
  tf.feature_to_id_v2.

  Attributes:
    vocabulary_file: The vocabulary filename. vocabulary_file cannot be combined
      with keys.
    keys: A 1-D string iterable that specifies the mapping of strings to
      indices. It means a feature in keys will map to it's index in keys.
    num_oov_buckets: The number of out-of-vocabulary buckets. If zero all out of
      vocabulary features will be ignored.
    vocab_size: Number of the elements in the vocabulary.
    default_value: The value to use for out-of-vocabulary feature values.
      Defaults to -1.
  """

  def __new__(cls,
              vocabulary_file=None,
              keys=None,
              num_oov_buckets=0,
              vocab_size=None,
              default_value=-1):

    return super(_SparseIdLookupConfig, cls).__new__(cls, vocabulary_file, keys,
                                                     num_oov_buckets,
                                                     vocab_size, default_value)


def _add_variable_collection(weight_collections):
  if weight_collections:
    weight_collections = list(set(list(weight_collections) +
                                  [ops.GraphKeys.VARIABLES]))
  return weight_collections


def _max_size_embedding_partitioner(max_shard_bytes=(64 << 20) - 1):
  """Partitioner based on max size.

  Args:
    max_shard_bytes: max shard bytes.

  Returns:
    partitioner
  """
  # max_shard_bytes defaults to ~64MB to keep below open sourced proto buffer
  # size limit.
  # TODO(zakaria): b/28274688 might cause low performance if there are too many
  #   partitions. Consider higher size, possily based on ps shards if the bug is
  #   not fixed.
  # TODO(zakaria): Use a better heuristic based on vocab size and upper/lower
  #   bound. Partitioning only at over 16M vicab_size is suboptimal for most
  #   cases.
  def partitioner(vocab_size, embed_dim):
    total_size = 1.0 * vocab_size * embed_dim * 4  # 4 bytes for float32
    shards = total_size / max_shard_bytes
    shards = min(vocab_size, max(1, int(math.ceil(shards))))
    return [shards, 1]

  return partitioner


def _create_embedding_lookup(input_tensor, vocab_size, dimension,
                             weight_collections, stddev, combiner, trainable,
                             name):
  """Creates embedding variable and does a lookup.

  Args:
    input_tensor: A tensor which should contain sparse id to look up.
    vocab_size: An integer specifying the vocabulary size.
    dimension: An integer specifying the embedding vector dimension.
    weight_collections: List of graph collections to which weights are added.
    stddev: the standard deviation to be used in embedding initialization.
    combiner: A string specifying how to reduce if the sparse column is
      multivalent. Currently "mean", "sqrtn" and "sum" are supported:
        * "sum": do not normalize features in the column
        * "mean": do l1 normalization on features in the column
        * "sqrtn": do l2 normalization on features in the column
      For more information: `tf.embedding_lookup_sparse`.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    name: A string specifying the name of the embedding variable.

  Returns:
    A Tensor with shape [batch_size, dimension] and embedding Variable.
  """
  slicing = _max_size_embedding_partitioner()(vocab_size, dimension)
  logging.info("Slicing=%s for name=%s, vocab_size=%d, embed_dim=%d",
               str(slicing), name, vocab_size, dimension)
  if stddev > 0:
    initializer = init_ops.truncated_normal_initializer(stddev=stddev)
  else:
    initializer = init_ops.zeros_initializer
  embeddings = partitioned_variables.create_partitioned_variables(
      shape=[vocab_size, dimension],
      slicing=slicing,
      initializer=initializer,
      dtype=dtypes.float32,
      collections=weight_collections,
      name=name,
      reuse=False,
      trainable=trainable)

  return contrib_embedding_ops.safe_embedding_lookup_sparse(
      embeddings,
      input_tensor,
      default_id=0,
      combiner=combiner,
      name=name), embeddings
