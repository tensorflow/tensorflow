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

from tensorflow.contrib.framework.python.framework import checkpoint_utils
from tensorflow.contrib.framework.python.ops import variables as contrib_variables
from tensorflow.contrib.layers.python.layers import embedding_ops
from tensorflow.contrib.layers.python.ops import bucketization_op
from tensorflow.contrib.layers.python.ops import sparse_feature_cross_op
from tensorflow.contrib.lookup import lookup_ops as contrib_lookup_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables


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


# TODO(b/30410315): Support warm starting in all feature columns.
class _SparseColumn(_FeatureColumn,
                    collections.namedtuple("_SparseColumn",
                                           ["column_name", "is_integerized",
                                            "bucket_size", "lookup_config",
                                            "combiner", "dtype"])):
  """Represents a sparse feature column also known as categorical features.

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
              combiner="sum",
              dtype=dtypes.string):
    if is_integerized and bucket_size is None:
      raise ValueError("bucket_size must be set if is_integerized is True. "
                       "column_name: {}".format(column_name))

    if is_integerized and not dtype.is_integer:
      raise ValueError("dtype must be an integer if is_integerized is True. "
                       "dtype: {}, column_name: {}.".format(dtype, column_name))

    if bucket_size is None and lookup_config is None:
      raise ValueError("one of bucket_size or lookup_config must be set. "
                       "column_name: {}".format(column_name))

    if bucket_size is not None and lookup_config:
      raise ValueError("one and only one of bucket_size or lookup_config "
                       "must be set. column_name: {}".format(column_name))

    if bucket_size is not None and bucket_size < 2:
      raise ValueError("bucket_size must be at least 2. "
                       "bucket_size: {}, column_name: {}".format(bucket_size,
                                                                 column_name))

    if ((lookup_config) and
        (not isinstance(lookup_config, _SparseIdLookupConfig))):
      raise TypeError(
          "lookup_config must be an instance of _SparseIdLookupConfig. "
          "Given one is in type {} for column_name {}".format(
              type(lookup_config), column_name))

    if (lookup_config and lookup_config.vocabulary_file and
        lookup_config.vocab_size is None):
      raise ValueError("vocab_size must be defined. "
                       "column_name: {}".format(column_name))

    return super(_SparseColumn, cls).__new__(cls, column_name, is_integerized,
                                             bucket_size, lookup_config,
                                             combiner, dtype)

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

  def id_tensor(self, input_tensor):
    """Returns the id tensor from the given transformed input_tensor."""
    return input_tensor

  # pylint: disable=unused-argument
  def weight_tensor(self, input_tensor):
    """Returns the weight tensor from the given transformed input_tensor."""
    return None

  # pylint: disable=unused-argument
  def to_dnn_input_layer(self,
                         input_tensor,
                         weight_collections=None,
                         trainable=True):
    raise ValueError("SparseColumn is not supported in DNN. "
                     "Please use embedding_column. column: {}".format(self))

  def to_weighted_sum(self,
                      input_tensor,
                      num_outputs=1,
                      weight_collections=None,
                      trainable=True):
    return _create_embedding_lookup(
        input_tensor=self.id_tensor(input_tensor),
        weight_tensor=self.weight_tensor(input_tensor),
        vocab_size=self.length,
        dimension=num_outputs,
        weight_collections=_add_variable_collection(weight_collections),
        initializer=init_ops.zeros_initializer,
        combiner=self.combiner,
        trainable=trainable,
        name=self.name)


class _SparseColumnIntegerized(_SparseColumn):
  """See `sparse_column_with_integerized_feature`."""

  def __new__(cls,
              column_name,
              bucket_size,
              combiner="sum",
              dtype=dtypes.int64):
    if not dtype.is_integer:
      raise ValueError("dtype must be an integer. "
                       "dtype: {}, column_name: {}".format(dtype, column_name))

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


class _WeightedSparseColumn(_FeatureColumn, collections.namedtuple(
    "_WeightedSparseColumn",
    ["sparse_id_column", "weight_column_name", "dtype"])):
  """See `weighted_sparse_column`."""

  def __new__(cls, sparse_id_column, weight_column_name, dtype):
    return super(_WeightedSparseColumn, cls).__new__(
        cls, sparse_id_column, weight_column_name, dtype)

  @property
  def name(self):
    return (self.sparse_id_column.name + "_weighted_by_" +
            self.weight_column_name)

  @property
  def length(self):
    """Returns id size."""
    return self.sparse_id_column.length

  @property
  def config(self):
    config = _get_feature_config(self.sparse_id_column)
    config.update(
        {self.weight_column_name: parsing_ops.VarLenFeature(self.dtype)})
    return config

  @property
  def key(self):
    """Returns a string which will be used as a key when we do sorting."""
    return "{}".format(self)

  def insert_transformed_feature(self, columns_to_tensors):
    """Inserts a tuple with the id and weight tensors."""
    if self.sparse_id_column not in columns_to_tensors:
      self.sparse_id_column.insert_transformed_feature(columns_to_tensors)
    columns_to_tensors[self] = tuple([
        columns_to_tensors[self.sparse_id_column],
        columns_to_tensors[self.weight_column_name]])

  def id_tensor(self, input_tensor):
    """Returns the id tensor from the given transformed input_tensor."""
    return input_tensor[0]

  def weight_tensor(self, input_tensor):
    """Returns the weight tensor from the given transformed input_tensor."""
    return input_tensor[1]

  # pylint: disable=unused-argument
  def to_dnn_input_layer(self,
                         input_tensor,
                         weight_collections=None,
                         trainable=True):
    raise ValueError("WeightedSparseColumn is not supported in DNN. "
                     "Please use embedding_column. column: {}".format(self))

  def to_weighted_sum(self,
                      input_tensor,
                      num_outputs=1,
                      weight_collections=None,
                      trainable=True):
    return _create_embedding_lookup(
        input_tensor=self.id_tensor(input_tensor),
        weight_tensor=self.weight_tensor(input_tensor),
        vocab_size=self.length,
        dimension=num_outputs,
        weight_collections=_add_variable_collection(weight_collections),
        initializer=init_ops.zeros_initializer,
        combiner=self.sparse_id_column.combiner,
        trainable=trainable,
        name=self.name)


def weighted_sparse_column(sparse_id_column,
                           weight_column_name,
                           dtype=dtypes.float32):
  """Creates a _SparseColumn by combining sparse_id_column with a weight column.

  Args:
    sparse_id_column: A _SparseColumn which is created by `sparse_column_with_*`
      functions.
    weight_column_name: A string defining a sparse column name which represents
      weight or value of the corresponding sparse id feature.
    dtype: Type of weights, such as `tf.float32`
  Returns:
    A _WeightedSparseColumn composed of two sparse features: one represents id,
    the other represents weight (value) of the id feature in that example.
  Raises:
    ValueError: if dtype is not convertible to float.

  An example usage:
    ```python
    words = sparse_column_with_hash_bucket("words", 1000)
    tfidf_weighted_words = weighted_sparse_column(words, "tfidf_score")
    ```

    This configuration assumes that input dictionary of model contains the
    following two items:
      * (key="words", value=word_tensor) where word_tensor is a SparseTensor.
      * (key="tfidf_score", value=tfidf_score_tensor) where tfidf_score_tensor
        is a SparseTensor.
     Following are assumed to be true:
       * word_tensor.indices = tfidf_score_tensor.indices
       * word_tensor.shape = tfidf_score_tensor.shape
  """
  if not (dtype.is_integer or dtype.is_floating):
    raise ValueError("dtype is not convertible to float. Given {}".format(
        dtype))

  return _WeightedSparseColumn(sparse_id_column,
                               weight_column_name,
                               dtype)


class _EmbeddingColumn(_FeatureColumn, collections.namedtuple(
    "_EmbeddingColumn",
    ["sparse_id_column", "dimension", "combiner", "initializer",
     "ckpt_to_load_from", "tensor_name_in_ckpt"])):
  """Represents an embedding column.

  Args:
    sparse_id_column: A _SparseColumn which is created by `sparse_column_with_*`
      or `weighted_sparse_column` functions.
    dimension: An integer specifying dimension of the embedding.
    combiner: A string specifying how to reduce if there are multiple entries
      in a single row. Currently "mean", "sqrtn" and "sum" are supported. Each
      of this can be thought as example level normalizations on the column:
        * "sum": do not normalize features in the column
        * "mean": do l1 normalization on features in the column
        * "sqrtn": do l2 normalization on features in the column
      For more information: `tf.embedding_lookup_sparse`.
    initializer: A variable initializer function to be used in embedding
      variable initialization. If not specified, defaults to
      `tf.truncated_normal_initializer` with mean 0.0 and standard deviation
      1/sqrt(sparse_id_column.length).
    ckpt_to_load_from: (Optional). String representing checkpoint name/pattern
      to restore the column weights. Required if `tensor_name_in_ckpt` is not
      None.
    tensor_name_in_ckpt: (Optional). Name of the `Tensor` in the provided
      checkpoint from which to restore the column weights. Required if
      `ckpt_to_load_from` is not None.

  Raises:
    ValueError: if `initializer` is specified and is not callable. Also,
      if only one of `ckpt_to_load_from` and `tensor_name_in_ckpt` is specified.
  """

  def __new__(cls,
              sparse_id_column,
              dimension,
              combiner="mean",
              initializer=None,
              ckpt_to_load_from=None,
              tensor_name_in_ckpt=None):
    if initializer is not None and not callable(initializer):
      raise ValueError("initializer must be callable if specified. "
                       "Embedding of column_name: {}".format(
                           sparse_id_column.name))

    if (ckpt_to_load_from is None) != (tensor_name_in_ckpt is None):
      raise ValueError("Must specify both `ckpt_to_load_from` and "
                       "`tensor_name_in_ckpt` or none of them.")
    if initializer is None:
      stddev = 1 / math.sqrt(sparse_id_column.length)
      # TODO(b/25671353): Better initial value?
      initializer = init_ops.truncated_normal_initializer(mean=0.0,
                                                          stddev=stddev)
    return super(_EmbeddingColumn, cls).__new__(cls, sparse_id_column,
                                                dimension, combiner,
                                                initializer, ckpt_to_load_from,
                                                tensor_name_in_ckpt)

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
    fields_values = []
    # pylint: disable=protected-access
    for k, v in self._asdict().items():
      if k == "initializer":
        # Excludes initializer from the key since we don't support allowing
        # users to specify different initializers for the same embedding column.
        # Special treatment is needed since the default str form of a
        # function contains its address, which could introduce non-determinism
        # in sorting.
        continue
      fields_values.append("{}={}".format(k, v))
    # pylint: enable=protected-access

    # This is effectively the same format as str(self), except with our special
    # treatment.
    return "%s(%s)" % (type(self).__name__, ", ".join(fields_values))

  def insert_transformed_feature(self, columns_to_tensors):
    self.sparse_id_column.insert_transformed_feature(columns_to_tensors)
    columns_to_tensors[self] = columns_to_tensors[self.sparse_id_column]

  def to_dnn_input_layer(self,
                         input_tensor,
                         weight_collections=None,
                         trainable=True):
    output, embedding_weights = _create_embedding_lookup(
        input_tensor=self.sparse_id_column.id_tensor(input_tensor),
        weight_tensor=self.sparse_id_column.weight_tensor(input_tensor),
        vocab_size=self.length,
        dimension=self.dimension,
        weight_collections=_add_variable_collection(weight_collections),
        initializer=self.initializer,
        combiner=self.combiner,
        trainable=trainable,
        name=self.name)
    if self.ckpt_to_load_from is not None:
      weights_to_restore = embedding_weights
      if len(embedding_weights) == 1:
        weights_to_restore = embedding_weights[0]
      checkpoint_utils.init_from_checkpoint(
          self.ckpt_to_load_from,
          {self.tensor_name_in_ckpt: weights_to_restore})
    return output

  # pylint: disable=unused-argument
  def to_weighted_sum(self,
                      input_tensor,
                      num_outputs=1,
                      weight_collections=None,
                      trainable=True):
    raise ValueError("EmbeddingColumn is not supported in linear models. "
                     "Please use sparse_column. column: {}".format(self))


def embedding_column(sparse_id_column,
                     dimension,
                     combiner="mean",
                     initializer=None,
                     ckpt_to_load_from=None,
                     tensor_name_in_ckpt=None):
  """Creates an _EmbeddingColumn.

  Args:
    sparse_id_column: A _SparseColumn which is created by `sparse_column_with_*`
      or crossed_column functions. Note that `combiner` defined in
      `sparse_id_column` is ignored.
    dimension: An integer specifying dimension of the embedding.
    combiner: A string specifying how to reduce if there are multiple entries
      in a single row. Currently "mean", "sqrtn" and "sum" are supported. Each
      of this can be thought as example level normalizations on the column:
        * "sum": do not normalize
        * "mean": do l1 normalization
        * "sqrtn": do l2 normalization
      For more information: `tf.embedding_lookup_sparse`.
    initializer: A variable initializer function to be used in embedding
      variable initialization. If not specified, defaults to
      `tf.truncated_normal_initializer` with mean 0.0 and standard deviation
      1/sqrt(sparse_id_column.length).
    ckpt_to_load_from: (Optional). String representing checkpoint name/pattern
      to restore the column weights. Required if `tensor_name_in_ckpt` is not
      None.
    tensor_name_in_ckpt: (Optional). Name of the `Tensor` in the provided
      checkpoint from which to restore the column weights. Required if
      `ckpt_to_load_from` is not None.

  Returns:
    An _EmbeddingColumn.
  """
  return _EmbeddingColumn(sparse_id_column, dimension, combiner, initializer,
                          ckpt_to_load_from, tensor_name_in_ckpt)


class _HashedEmbeddingColumn(collections.namedtuple(
    "_HashedEmbeddingColumn", ["column_name", "size", "dimension", "combiner",
                               "initializer"]), _EmbeddingColumn):
  """See `hashed_embedding_column`."""

  def __new__(cls,
              column_name,
              size,
              dimension,
              combiner="mean",
              initializer=None):
    if initializer is not None and not callable(initializer):
      raise ValueError("initializer must be callable if specified. "
                       "column_name: {}".format(column_name))
    if initializer is None:
      stddev = 0.1
      # TODO(b/25671353): Better initial value?
      initializer = init_ops.truncated_normal_initializer(mean=0.0,
                                                          stddev=stddev)
    return super(_HashedEmbeddingColumn, cls).__new__(cls, column_name, size,
                                                      dimension, combiner,
                                                      initializer)

  @property
  def name(self):
    return self.column_name + "_embedding"

  @property
  def config(self):
    return {self.column_name: parsing_ops.VarLenFeature(dtypes.string)}

  def insert_transformed_feature(self, columns_to_tensors):
    columns_to_tensors[self] = columns_to_tensors[self.column_name]

  def to_dnn_input_layer(self,
                         input_tensor,
                         weight_collections=None,
                         trainable=True):
    embeddings = _create_embeddings(
        name=self.name,
        shape=[self.size],
        initializer=self.initializer,
        dtype=dtypes.float32,
        trainable=trainable,
        weight_collections=_add_variable_collection(weight_collections))

    return embedding_ops.hashed_embedding_lookup_sparse(
        embeddings, input_tensor, self.dimension, name=self.name + "_lookup")


def hashed_embedding_column(column_name,
                            size,
                            dimension,
                            combiner="mean",
                            initializer=None):
  """Creates an embedding column of a sparse feature using parameter hashing.

  The i-th embedding component of a value v is found by retrieving an
  embedding weight whose index is a fingerprint of the pair (v,i).

  Args:
    column_name: A string defining sparse column name.
    size: An integer specifying the number of parameters in the embedding layer.
    dimension: An integer specifying dimension of the embedding.
    combiner: A string specifying how to reduce if there are multiple entries
      in a single row. Currently "mean", "sqrtn" and "sum" are supported. Each
      of this can be thought as example level normalizations on the column:
        * "sum": do not normalize features in the column
        * "mean": do l1 normalization on features in the column
        * "sqrtn": do l2 normalization on features in the column
      For more information: `tf.embedding_lookup_sparse`.
    initializer: A variable initializer function to be used in embedding
      variable initialization. If not specified, defaults to
      `tf.truncated_normal_initializer` with mean 0 and standard deviation 0.1.

  Returns:
    A _HashedEmbeddingColumn.

  Raises:
    ValueError: if dimension or size is not a positive integer; or if combiner
      is not supported.

  """
  if (dimension < 1) or (size < 1):
    raise ValueError("Dimension and size must be greater than 0. "
                     "dimension: {}, size: {}, column_name: {}".format(
                         dimension, size, column_name))

  if combiner not in ("mean", "sqrtn", "sum"):
    raise ValueError("Combiner must be one of 'mean', 'sqrtn' or 'sum'. "
                     "combiner: {}, column_name: {}".format(
                         combiner, column_name))

  return _HashedEmbeddingColumn(column_name, size, dimension, combiner,
                                initializer)


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
    def _weight(name):
      return variable_scope.get_variable(
          name,
          shape=[self.dimension, num_outputs],
          initializer=array_ops.zeros_initializer,
          collections=_add_variable_collection(weight_collections))

    if self.name:
      with variable_scope.variable_op_scope([input_tensor], None, self.name):
        weight = _weight("weight")
    else:
      # Old behavior to support a subset of old checkpoints.
      weight = _weight("_weight")

    # The _RealValuedColumn has the shape of [batch_size, column.dimension].
    log_odds_by_dim = math_ops.matmul(input_tensor, weight)
    return log_odds_by_dim, [weight]


def real_valued_column(column_name,
                       dimension=1,
                       default_value=None,
                       dtype=dtypes.float32):
  """Creates a _RealValuedColumn.

  Args:
    column_name: A string defining real valued column name.
    dimension: An integer specifying dimension of the real valued column.
      The default is 1. The Tensor representing the _RealValuedColumn
      will have the shape of [batch_size, dimension].
    default_value: A single value compatible with dtype or a list of values
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
    TypeError: if dimension is not an int
    ValueError: if dimension is not a positive integer
    TypeError: if default_value is a list but its length is not equal to the
      value of `dimension`.
    TypeError: if default_value is not compatible with dtype.
    ValueError: if dtype is not convertable to tf.float32.
  """

  if not isinstance(dimension, int):
    raise TypeError("dimension must be an integer. "
                    "dimension: {}, column_name: {}".format(dimension,
                                                            column_name))

  if dimension < 1:
    raise ValueError("dimension must be greater than 0. "
                     "dimension: {}, column_name: {}".format(dimension,
                                                             column_name))

  if not (dtype.is_integer or dtype.is_floating):
    raise ValueError("dtype must be convertible to float. "
                     "dtype: {}, column_name: {}".format(dtype, column_name))

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
      raise ValueError(
          "The length of default_value must be equal to dimension. "
          "default_value: {}, dimension: {}, column_name: {}".format(
              default_value, dimension, column_name))
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

  raise TypeError("default_value must be compatible with dtype. "
                  "default_value: {}, dtype: {}, column_name: {}".format(
                      default_value, dtype, column_name))


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
          "source_column must be an instance of _RealValuedColumn. "
          "source_column: {}".format(source_column))

    if not isinstance(boundaries, list) or not boundaries:
      raise ValueError("boundaries must be a non-empty list. "
                       "boundaries: {}".format(boundaries))

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
        raise ValueError("boundaries must be a sorted list. "
                         "boundaries: {}".format(boundaries))
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

  def to_sparse_tensor(self, input_tensor):
    """Creates a SparseTensor from the bucketized Tensor."""
    dimension = self.source_column.dimension
    batch_size = array_ops.shape(input_tensor)[0]

    if dimension > 1:
      i1 = array_ops.reshape(array_ops.tile(array_ops.expand_dims(
          math_ops.range(0, batch_size), 1), [1, dimension]), [-1])
      i2 = array_ops.tile(math_ops.range(0, dimension), [batch_size])
      # Flatten the bucket indices and unique them across dimensions
      # E.g. 2nd dimension indices will range from k to 2*k-1 with k buckets
      bucket_indices = array_ops.reshape(input_tensor, [-1]) + self.length * i2
    else:
      # Simpler indices when dimension=1
      i1 = math_ops.range(0, batch_size)
      i2 = array_ops.zeros([batch_size], dtype=dtypes.int32)
      bucket_indices = array_ops.reshape(input_tensor, [-1])

    indices = math_ops.to_int64(array_ops.transpose(array_ops.pack((i1, i2))))
    shape = math_ops.to_int64(array_ops.pack([batch_size, dimension]))
    sparse_id_values = ops.SparseTensor(indices, bucket_indices, shape)

    return sparse_id_values

  def to_weighted_sum(self,
                      input_tensor,
                      num_outputs=1,
                      weight_collections=None,
                      trainable=True):
    """Returns a Tensor as linear predictions and a list of created Variable."""
    return _create_embedding_lookup(
        input_tensor=self.to_sparse_tensor(input_tensor),
        weight_tensor=None,
        vocab_size=self.length * self.source_column.dimension,
        dimension=num_outputs,
        weight_collections=_add_variable_collection(weight_collections),
        initializer=init_ops.zeros_initializer,
        combiner="sum",
        trainable=trainable,
        name=self.name)


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
    "_CrossedColumn", ["columns", "hash_bucket_size", "combiner",
                       "ckpt_to_load_from", "tensor_name_in_ckpt"])):
  """Represents a cross transformation also known as composition or union.

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
    ckpt_to_load_from: (Optional). String representing checkpoint name/pattern
      to restore the column weights. Required if `tensor_name_in_ckpt` is not
      None.
    tensor_name_in_ckpt: (Optional). Name of the `Tensor` in the provided
      checkpoint from which to restore the column weights. Required if
      `ckpt_to_load_from` is not None.

  Raises:
    TypeError: if all items in columns are not an instance of _SparseColumn,
      _CrossedColumn, or _BucketizedColumn or
      hash_bucket_size is not an int.
    ValueError: if hash_bucket_size is not > 1 or len(columns) is not > 1. Also,
      if only one of `ckpt_to_load_from` and `tensor_name_in_ckpt` is specified.
  """

  @staticmethod
  def _is_crossable(column):
    return isinstance(column,
                      (_SparseColumn, _CrossedColumn, _BucketizedColumn))

  def __new__(cls, columns, hash_bucket_size, combiner="sum",
              ckpt_to_load_from=None, tensor_name_in_ckpt=None):
    for column in columns:
      if not _CrossedColumn._is_crossable(column):
        raise TypeError("columns must be a set of _SparseColumn, "
                        "_CrossedColumn, or _BucketizedColumn instances. "
                        "column: {}".format(column))

    if len(columns) < 2:
      raise ValueError("columns must contain at least 2 elements. "
                       "columns: {}".format(columns))

    if not isinstance(hash_bucket_size, int):
      raise TypeError("hash_bucket_size must be an int. "
                      "hash_bucket_size: {}".format(hash_bucket_size))

    if hash_bucket_size < 2:
      raise ValueError("hash_bucket_size must be at least 2. "
                       "hash_bucket_size: {}".format(hash_bucket_size))

    if (ckpt_to_load_from is None) != (tensor_name_in_ckpt is None):
      raise ValueError("Must specify both `ckpt_to_load_from` and "
                       "`tensor_name_in_ckpt` or none of them.")

    sorted_columns = sorted([column for column in columns],
                            key=lambda column: column.name)
    return super(_CrossedColumn, cls).__new__(cls, tuple(sorted_columns),
                                              hash_bucket_size, combiner,
                                              ckpt_to_load_from,
                                              tensor_name_in_ckpt)

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

  def id_tensor(self, input_tensor):
    """Returns the id tensor from the given transformed input_tensor."""
    return input_tensor

  # pylint: disable=unused-argument
  def weight_tensor(self, input_tensor):
    """Returns the weight tensor from the given transformed input_tensor."""
    return None

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
        if isinstance(c, _BucketizedColumn):
          feature_tensors.append(c.to_sparse_tensor(columns_to_tensors[c]))
        else:
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
    raise ValueError("CrossedColumn is not supported in DNN. "
                     "Please use embedding_column. column: {}".format(self))

  def to_weighted_sum(self,
                      input_tensor,
                      num_outputs=1,
                      weight_collections=None,
                      trainable=True):
    output, embedding_weights = _create_embedding_lookup(
        input_tensor=input_tensor,
        weight_tensor=None,
        vocab_size=self.length,
        dimension=num_outputs,
        weight_collections=_add_variable_collection(weight_collections),
        initializer=init_ops.zeros_initializer,
        combiner=self.combiner,
        trainable=trainable,
        name=self.name)
    if self.ckpt_to_load_from is not None:
      weights_to_restore = embedding_weights
      if len(embedding_weights) == 1:
        weights_to_restore = embedding_weights[0]
      checkpoint_utils.init_from_checkpoint(
          self.ckpt_to_load_from,
          {self.tensor_name_in_ckpt: weights_to_restore})
    return output, embedding_weights


def crossed_column(columns, hash_bucket_size, combiner="sum",
                   ckpt_to_load_from=None,
                   tensor_name_in_ckpt=None):
  """Creates a _CrossedColumn.

  Args:
    columns: An iterable of _FeatureColumn. Items can be an instance of
      _SparseColumn, _CrossedColumn, or _BucketizedColumn.
    hash_bucket_size: An int that is > 1. The number of buckets.
    combiner: A combiner string, supports sum, mean, sqrtn.
    ckpt_to_load_from: (Optional). String representing checkpoint name/pattern
      to restore the column weights. Required if `tensor_name_in_ckpt` is not
      None.
    tensor_name_in_ckpt: (Optional). Name of the `Tensor` in the provided
      checkpoint from which to restore the column weights. Required if
      `ckpt_to_load_from` is not None.

  Returns:
    A _CrossedColumn.

  Raises:
    TypeError: if any item in columns is not an instance of _SparseColumn,
      _CrossedColumn, or _BucketizedColumn, or
      hash_bucket_size is not an int.
    ValueError: if hash_bucket_size is not > 1 or
      len(columns) is not > 1.
  """
  return _CrossedColumn(columns, hash_bucket_size, combiner=combiner,
                        ckpt_to_load_from=ckpt_to_load_from,
                        tensor_name_in_ckpt=tensor_name_in_ckpt)


class DataFrameColumn(_FeatureColumn,
                      collections.namedtuple("DataFrameColumn",
                                             ["column_name", "series"])):
  """Represents a feature column produced from a `DataFrame`.

  Instances of this class are immutable.  A `DataFrame` column may be dense or
  sparse, and may have any shape, with the constraint that dimension 0 is
  batch_size.

  Args:
    column_name: a name for this column
    series: a `Series` to be wrapped, which has already had its base features
      substituted with `PredefinedSeries`.
  """

  def __new__(cls, column_name, series):
    return super(DataFrameColumn, cls).__new__(cls, column_name, series)

  @property
  def name(self):
    return self.column_name

  @property
  def config(self):
    return self.series.required_base_features()

  @property
  def key(self):
    """Returns a string which will be used as a key when we do sorting."""
    return self.name

  def insert_transformed_feature(self, columns_to_tensors):
    # The cache must already contain mappings from the expected base feature
    # names to Tensors.

    # Passing columns_to_tensors as the cache here means that multiple outputs
    # of the transform will be cached, keyed by the repr of their associated
    # TransformedSeries.
    # The specific requested output ends up in columns_to_tensors twice: once
    # keyed by the TransformedSeries repr, and once keyed by this
    # DataFrameColumn instance.
    columns_to_tensors[self] = self.series.build(columns_to_tensors)

  # pylint: disable=unused-argument
  def to_dnn_input_layer(self,
                         input_tensor,
                         weight_collections=None,
                         trainable=True):
    # DataFrame typically provides Tensors of shape [batch_size],
    # but Estimator requires shape [batch_size, 1]
    dims = input_tensor.get_shape().ndims
    if dims == 0:
      raise ValueError(
          "Can't build input layer from tensor of shape (): {}".format(
              self.column_name))
    elif dims == 1:
      return array_ops.expand_dims(input_tensor, 1)
    else:
      return input_tensor

  # TODO(soergel): This mirrors RealValuedColumn for now, but should become
  # better abstracted with less code duplication when we add other kinds.
  def to_weighted_sum(self,
                      input_tensor,
                      num_outputs=1,
                      weight_collections=None,
                      trainable=True):
    def _weight(name):
      return variable_scope.get_variable(
          name,
          shape=[self.dimension, num_outputs],
          initializer=array_ops.zeros_initializer,
          collections=_add_variable_collection(weight_collections))

    if self.name:
      with variable_scope.variable_op_scope([input_tensor], None, self.name):
        weight = _weight("weight")
    else:
      # Old behavior to support a subset of old checkpoints.
      weight = _weight("_weight")

    # The _RealValuedColumn has the shape of [batch_size, column.dimension].
    log_odds_by_dim = math_ops.matmul(input_tensor, weight)
    return log_odds_by_dim, [weight]

  def __eq__(self, other):
    if isinstance(other, self.__class__):
      return self.__dict__ == other.__dict__
    else:
      return False

  def __ne__(self, other):
    return not self.__eq__(other)


def _get_feature_config(feature_column):
  """Returns configuration for the base feature defined in feature_column."""
  if not isinstance(feature_column, _FeatureColumn):
    raise TypeError(
        "feature_columns should only contain instances of _FeatureColumn. "
        "Given column is {}".format(feature_column))
  if isinstance(feature_column, (_SparseColumn, _WeightedSparseColumn,
                                 _EmbeddingColumn, _RealValuedColumn,
                                 _BucketizedColumn, _CrossedColumn)):
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


def _create_embeddings(name, shape, dtype, initializer, trainable,
                       weight_collections):
  """Creates embedding variable.

  If called within the scope of a partitioner, will partition the variable and
  return a list of `tf.Variable`. If no partitioner is specified, returns a list
  with just one variable.

  Args:
    name: A string. The name of the embedding variable will be name + _weights.
    shape: shape of the embeddding. Note this is not the shape of partitioned
      variables.
    dtype: type of the embedding. Also the shape of each partitioned variable.
    initializer: A variable initializer function to be used in embedding
      variable initialization.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    weight_collections: List of graph collections to which embedding variables
      are added.

  Returns:
    A list of `tf.Variable` containing the partitioned embeddings.

  Raises:
    ValueError: If initializer is None or not callable.
  """
  if not initializer:
    raise ValueError("initializer must be defined.")
  if not callable(initializer):
    raise ValueError("initializer must be callable.")
  embeddings = contrib_variables.model_variable(name=name,
                                                shape=shape,
                                                dtype=dtype,
                                                initializer=initializer,
                                                trainable=trainable,
                                                collections=weight_collections)
  if isinstance(embeddings, variables.Variable):
    return [embeddings]
  else:  # Else it should be of type `_PartitionedVariable`.
    return embeddings._get_variable_list()  # pylint: disable=protected-access


def _create_embedding_lookup(input_tensor, weight_tensor, vocab_size, dimension,
                             weight_collections, initializer, combiner,
                             trainable, name):
  """Creates embedding variable and does a lookup.

  Args:
    input_tensor: A `SparseTensor` which should contain sparse id to look up.
    weight_tensor: A `SparseTensor` with the same shape and indices as
      `input_tensor`, which contains the float weights corresponding to each
      sparse id, or None if all weights are assumed to be 1.0.
    vocab_size: An integer specifying the vocabulary size.
    dimension: An integer specifying the embedding vector dimension.
    weight_collections: List of graph collections to which weights are added.
    initializer: A variable initializer function to be used in embedding
      variable initialization.
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

  embeddings = _create_embeddings(name=name + "_weights",
                                  shape=[vocab_size, dimension],
                                  dtype=dtypes.float32,
                                  initializer=initializer,
                                  trainable=trainable,
                                  weight_collections=weight_collections)
  return embedding_ops.safe_embedding_lookup_sparse(
      embeddings,
      input_tensor,
      sparse_weights=weight_tensor,
      default_id=0,
      combiner=combiner,
      name=name + "_weights"), embeddings
