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

FeatureColumns provide a high level abstraction for ingesting and representing
features in `Estimator` models.

FeatureColumns are the primary way of encoding features for pre-canned
`Estimator` models.

When using FeatureColumns with `Estimator` models, the type of feature column
you should choose depends on (1) the feature type and (2) the model type.

(1) Feature type:

 * Continuous features can be represented by `real_valued_column`.
 * Categorical features can be represented by any `sparse_column_with_*`
 column (`sparse_column_with_keys`, `sparse_column_with_vocabulary_file`,
 `sparse_column_with_hash_bucket`, `sparse_column_with_integerized_feature`).

(2) Model type:

 * Deep neural network models (`DNNClassifier`, `DNNRegressor`).

   Continuous features can be directly fed into deep neural network models.

     age_column = real_valued_column("age")

   To feed sparse features into DNN models, wrap the column with
   `embedding_column` or `one_hot_column`. `one_hot_column` will create a dense
   boolean tensor with an entry for each possible value, and thus the
   computation cost is linear in the number of possible values versus the number
   of values that occur in the sparse tensor. Thus using a "one_hot_column" is
   only recommended for features with only a few possible values. For features
   with many possible values or for very sparse features, `embedding_column` is
   recommended.

     embedded_dept_column = embedding_column(
       sparse_column_with_keys("department", ["math", "philosophy", ...]),
       dimension=10)

* Wide (aka linear) models (`LinearClassifier`, `LinearRegressor`).

   Sparse features can be fed directly into linear models. When doing so
   an embedding_lookups are used to efficiently perform the sparse matrix
   multiplication.

     dept_column = sparse_column_with_keys("department",
       ["math", "philosophy", "english"])

   It is recommended that continuous features be bucketized before being
   fed into linear models.

     bucketized_age_column = bucketized_column(
      source_column=age_column,
      boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

   Sparse features can be crossed (also known as conjuncted or combined) in
   order to form non-linearities, and then fed into linear models.

    cross_dept_age_column = crossed_column(
      columns=[department_column, bucketized_age_column],
      hash_bucket_size=1000)

Example of building an `Estimator` model using FeatureColumns:

  # Define features and transformations
  deep_feature_columns = [age_column, embedded_dept_column]
  wide_feature_columns = [dept_column, bucketized_age_column,
      cross_dept_age_column]

  # Build deep model
  estimator = DNNClassifier(
      feature_columns=deep_feature_columns,
      hidden_units=[500, 250, 50])
  estimator.train(...)

  # Or build a wide model
  estimator = LinearClassifier(
      feature_columns=wide_feature_columns)
  estimator.train(...)

  # Or build a wide and deep model!
  estimator = DNNLinearCombinedClassifier(
      linear_feature_columns=wide_feature_columns,
      dnn_feature_columns=deep_feature_columns,
      dnn_hidden_units=[500, 250, 50])
  estimator.train(...)


FeatureColumns can also be transformed into a generic input layer for
custom models using `input_from_feature_columns` within
`feature_column_ops.py`.

Example of building a non-`Estimator` model using FeatureColumns:

  # Building model via layers

  deep_feature_columns = [age_column, embedded_dept_column]
  columns_to_tensor = parse_feature_columns_from_examples(
      serialized=my_data,
      feature_columns=deep_feature_columns)
  first_layer = input_from_feature_columns(
      columns_to_tensors=columns_to_tensor,
      feature_columns=deep_feature_columns)
  second_layer = fully_connected(first_layer, ...)

See feature_column_ops_test for more examples.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import math

import six

from tensorflow.contrib import lookup
from tensorflow.contrib.framework.python.framework import checkpoint_utils
from tensorflow.contrib.framework.python.framework import experimental
from tensorflow.contrib.framework.python.ops import variables as contrib_variables
from tensorflow.contrib.layers.python.layers import embedding_ops
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.layers.python.ops import bucketization_op
from tensorflow.contrib.layers.python.ops import sparse_feature_cross_op
from tensorflow.contrib.layers.python.ops import sparse_ops as contrib_sparse_ops
from tensorflow.python.feature_column import feature_column as fc_core
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor as sparse_tensor_py
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest


# Imports the core `InputLayer` symbol in contrib during development.
InputLayer = fc_core.InputLayer  # pylint: disable=invalid-name


class _LinearEmbeddingLookupArguments(
    collections.namedtuple("_LinearEmbeddingLookupArguments",
                           ["input_tensor",
                            "weight_tensor",
                            "vocab_size",
                            "initializer",
                            "combiner"])):
  """Represents the information needed from a column for embedding lookup.

  Used to compute DNN inputs and weighted sum.
  """
  pass


class _DeepEmbeddingLookupArguments(
    collections.namedtuple("_DeepEmbeddingLookupArguments",
                           ["input_tensor",
                            "weight_tensor",
                            "vocab_size",
                            "initializer",
                            "combiner",
                            "dimension",
                            "shared_embedding_name",
                            "hash_key",
                            "max_norm",
                            "trainable"])):
  """Represents the information needed from a column for embedding lookup.

  Used to compute DNN inputs and weighted sum.
  """
  pass


@six.add_metaclass(abc.ABCMeta)
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

  @abc.abstractproperty
  @deprecation.deprecated(
      "2016-09-25",
      "Should be private.")
  def name(self):
    """Returns the name of column or transformed column."""
    pass

  @abc.abstractproperty
  @deprecation.deprecated(
      "2016-09-25",
      "Should be private.")
  def config(self):
    """Returns configuration of the base feature for `tf.parse_example`."""
    pass

  @abc.abstractproperty
  @deprecation.deprecated(
      "2016-09-25",
      "Should be private.")
  def key(self):
    """Returns a string which will be used as a key when we do sorting."""
    pass

  @abc.abstractmethod
  @deprecation.deprecated(
      "2016-09-25",
      "Should be private.")
  def insert_transformed_feature(self, columns_to_tensors):
    """Apply transformation and inserts it into columns_to_tensors.

    Args:
      columns_to_tensors: A mapping from feature columns to tensors. 'string'
        key means a base feature (not-transformed). It can have _FeatureColumn
        as a key too. That means that _FeatureColumn is already transformed.
    """
    raise NotImplementedError("Transform is not implemented for {}.".format(
        self))

  # pylint: disable=unused-argument
  def _to_dnn_input_layer(self,
                          input_tensor,
                          weight_collection=None,
                          trainable=True,
                          output_rank=2):
    """Returns a Tensor as an input to the first layer of neural network."""
    raise ValueError("Calling an abstract method.")

  def _deep_embedding_lookup_arguments(self, input_tensor):
    """Returns arguments to embedding lookup to build an input layer."""
    raise NotImplementedError(
        "No deep embedding lookup arguments for column {}.".format(self))

  # It is expected that classes implement either wide_embedding_lookup_arguments
  # or to_dense_tensor to be used in linear models.
  # pylint: disable=unused-argument
  def _wide_embedding_lookup_arguments(self, input_tensor):
    """Returns arguments to look up embeddings for this column."""
    raise NotImplementedError(
        "No wide embedding lookup arguments for column {}.".format(self))

  # pylint: disable=unused-argument
  def _to_dense_tensor(self, input_tensor):
    """Returns a dense tensor representing this column's values."""
    raise NotImplementedError(
        "No dense tensor representation for column {}.".format(self))

  def _checkpoint_path(self):
    """Returns None, or a (path,tensor_name) to load a checkpoint from."""
    return None

  def _key_without_properties(self, properties):
    """Helper method for self.key() that omits particular properties."""
    fields_values = []
    # pylint: disable=protected-access
    for i, k in enumerate(self._fields):
      if k in properties:
        # Excludes a property from the key.
        # For instance, exclude `initializer` from the key of EmbeddingColumn
        # since we don't support users specifying different initializers for
        # the same embedding column. Ditto for `normalizer` and
        # RealValuedColumn.
        # Special treatment is needed since the default str form of a
        # function contains its address, which could introduce non-determinism
        # in sorting.
        continue
      fields_values.append("{}={}".format(k, self[i]))
    # pylint: enable=protected-access

    # This is effectively the same format as str(self), except with our special
    # treatment.
    return "{}({})".format(type(self).__name__, ", ".join(fields_values))


# TODO(b/30410315): Support warm starting in all feature columns.
class _SparseColumn(
    _FeatureColumn,
    fc_core._CategoricalColumn,  # pylint: disable=protected-access
    collections.namedtuple("_SparseColumn", [
        "column_name", "is_integerized", "bucket_size", "lookup_config",
        "combiner", "dtype"
    ])):
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
    bucket_size: An int that is > 0. The number of buckets.
    lookup_config: A _SparseIdLookupConfig defining feature-to-id lookup
      configuration
    combiner: A string specifying how to reduce if the sparse column is
      multivalent. Currently "mean", "sqrtn" and "sum" are supported, with "sum"
      the default. "sqrtn" often achieves good accuracy, in particular with
      bag-of-words columns.
        * "sum": do not normalize features in the column
        * "mean": do l1 normalization on features in the column
        * "sqrtn": do l2 normalization on features in the column
      For more information: `tf.embedding_lookup_sparse`.
    dtype: Type of features, either `tf.string` or `tf.int64`.

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
    if dtype != dtypes.string and not dtype.is_integer:
      raise ValueError("dtype must be string or integer. "
                       "dtype: {}, column_name: {}".format(dtype, column_name))

    if bucket_size is None and lookup_config is None:
      raise ValueError("one of bucket_size or lookup_config must be set. "
                       "column_name: {}".format(column_name))

    if bucket_size is not None and lookup_config:
      raise ValueError("one and only one of bucket_size or lookup_config "
                       "must be set. column_name: {}".format(column_name))

    if bucket_size is not None and bucket_size < 1:
      raise ValueError("bucket_size must be at least 1. "
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

    return super(_SparseColumn, cls).__new__(
        cls,
        column_name,
        is_integerized=is_integerized,
        bucket_size=bucket_size,
        lookup_config=lookup_config,
        combiner=combiner,
        dtype=dtype)

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
  def _to_dnn_input_layer(self,
                          input_tensor,
                          weight_collections=None,
                          trainable=True,
                          output_rank=2):
    raise ValueError(
        "SparseColumn is not supported in DNN. "
        "Please use embedding_column or one_hot_column. column: {}".format(
            self))

  def _wide_embedding_lookup_arguments(self, input_tensor):
    return _LinearEmbeddingLookupArguments(
        input_tensor=self.id_tensor(input_tensor),
        weight_tensor=self.weight_tensor(input_tensor),
        vocab_size=self.length,
        initializer=init_ops.zeros_initializer(),
        combiner=self.combiner)

  def _get_input_sparse_tensor(self, input_tensor):
    """sparsify input_tensor if dense."""
    if not isinstance(input_tensor, sparse_tensor_py.SparseTensor):
      # To avoid making any assumptions about which values are to be ignored,
      # we set ignore_value to -1 for numeric tensors to avoid excluding valid
      # indices.
      if input_tensor.dtype == dtypes.string:
        ignore_value = ""
      else:
        ignore_value = -1
      input_tensor = _reshape_real_valued_tensor(input_tensor, 2, self.name)
      input_tensor = contrib_sparse_ops.dense_to_sparse_tensor(
          input_tensor, ignore_value=ignore_value)

    return input_tensor

  def is_compatible(self, other_column):
    """Check compatibility of two sparse columns."""
    if self.lookup_config and other_column.lookup_config:
      return self.lookup_config == other_column.lookup_config
    compatible = (self.length == other_column.length and
                  (self.dtype == other_column.dtype or
                   (self.dtype.is_integer and other_column.dtype.is_integer)))
    if compatible:
      logging.warn("Column {} and {} may not have the same vocabulary.".
                   format(self.name, other_column.name))
    return compatible

  @abc.abstractmethod
  def _do_transform(self, input_tensor):
    pass

  def insert_transformed_feature(self, columns_to_tensors):
    """Handles sparse column to id conversion."""
    input_tensor = self._get_input_sparse_tensor(columns_to_tensors[self.name])
    columns_to_tensors[self] = self._do_transform(input_tensor)

  def _transform_feature(self, inputs):
    input_tensor = self._get_input_sparse_tensor(inputs.get(self.name))
    return self._do_transform(input_tensor)

  @property
  def _parse_example_spec(self):
    return self.config

  @property
  def _num_buckets(self):
    return self.length

  def _get_sparse_tensors(self, inputs, weight_collections=None,
                          trainable=None):
    del weight_collections
    del trainable
    input_tensor = inputs.get(self)
    return fc_core._CategoricalColumn.IdWeightPair(  # pylint: disable=protected-access
        self.id_tensor(input_tensor), self.weight_tensor(input_tensor))


class _SparseColumnIntegerized(_SparseColumn):
  """See `sparse_column_with_integerized_feature`."""

  def _do_transform(self, input_tensor):
    sparse_id_values = math_ops.mod(input_tensor.values, self.bucket_size,
                                    name="mod")
    return sparse_tensor_py.SparseTensor(input_tensor.indices, sparse_id_values,
                                         input_tensor.dense_shape)


def sparse_column_with_integerized_feature(column_name,
                                           bucket_size,
                                           combiner="sum",
                                           dtype=dtypes.int64):
  """Creates an integerized _SparseColumn.

  Use this when your features are already pre-integerized into int64 IDs, that
  is, when the set of values to output is already coming in as what's desired in
  the output. Integerized means we can use the feature value itself as id.

  Typically this is used for reading contiguous ranges of integers indexes, but
  it doesn't have to be. The output value is simply copied from the
  input_feature, whatever it is. Just be aware, however, that if you have large
  gaps of unused integers it might affect what you feed those in (for instance,
  if you make up a one-hot tensor from these, the unused integers will appear as
  values in the tensor which are always zero.)

  Args:
    column_name: A string defining sparse column name.
    bucket_size: An int that is >= 1. The number of buckets. It should be bigger
      than maximum feature. In other words features in this column should be an
      int64 in range [0, bucket_size)
    combiner: A string specifying how to reduce if the sparse column is
      multivalent. Currently "mean", "sqrtn" and "sum" are supported, with "sum"
      the default. "sqrtn" often achieves good accuracy, in particular with
      bag-of-words columns.
        * "sum": do not normalize features in the column
        * "mean": do l1 normalization on features in the column
        * "sqrtn": do l2 normalization on features in the column
      For more information: `tf.embedding_lookup_sparse`.
    dtype: Type of features. It should be an integer type. Default value is
      dtypes.int64.

  Returns:
    An integerized _SparseColumn definition.

  Raises:
    ValueError: bucket_size is less than 1.
    ValueError: dtype is not integer.
  """
  return _SparseColumnIntegerized(
      column_name, is_integerized=True, bucket_size=bucket_size,
      combiner=combiner, dtype=dtype)


class _SparseColumnHashed(_SparseColumn):
  """See `sparse_column_with_hash_bucket`."""

  def __new__(cls,
              column_name,
              is_integerized=False,
              bucket_size=None,
              lookup_config=None,
              combiner="sum",
              dtype=dtypes.string,
              hash_keys=None):
    if hash_keys is not None:
      if not isinstance(hash_keys, list) or not hash_keys:
        raise ValueError("hash_keys must be a non-empty list.")
      if (any([not isinstance(key_pair, list) for key_pair in hash_keys]) or
          any([len(key_pair) != 2 for key_pair in hash_keys]) or
          any([not isinstance(key, int) for key in nest.flatten(hash_keys)])):
        raise ValueError(
            "Each element of hash_keys must be a pair of integers.")
    obj = super(_SparseColumnHashed, cls).__new__(
        cls,
        column_name,
        is_integerized=is_integerized,
        bucket_size=bucket_size,
        lookup_config=lookup_config,
        combiner=combiner,
        dtype=dtype)
    obj.hash_keys = hash_keys
    return obj

  def _do_transform(self, input_tensor):
    if self.dtype.is_integer:
      sparse_values = string_ops.as_string(input_tensor.values)
    else:
      sparse_values = input_tensor.values

    if self.hash_keys:
      result = []
      for key in self.hash_keys:
        sparse_id_values = string_ops.string_to_hash_bucket_strong(
            sparse_values, self.bucket_size, key)
        result.append(
            sparse_tensor_py.SparseTensor(input_tensor.indices,
                                          sparse_id_values,
                                          input_tensor.dense_shape))
      return sparse_ops.sparse_concat(axis=1, sp_inputs=result, name="lookup")
    else:
      sparse_id_values = string_ops.string_to_hash_bucket_fast(
          sparse_values, self.bucket_size, name="lookup")
      return sparse_tensor_py.SparseTensor(
          input_tensor.indices, sparse_id_values, input_tensor.dense_shape)


def sparse_column_with_hash_bucket(column_name,
                                   hash_bucket_size,
                                   combiner="sum",
                                   dtype=dtypes.string,
                                   hash_keys=None):
  """Creates a _SparseColumn with hashed bucket configuration.

  Use this when your sparse features are in string or integer format, but you
  don't have a vocab file that maps each value to an integer ID.
  output_id = Hash(input_feature_string) % bucket_size

  When hash_keys is set, multiple integer IDs would be created with each key
  pair in the `hash_keys`. This is useful to reduce the collision of hashed ids.

  Args:
    column_name: A string defining sparse column name.
    hash_bucket_size: An int that is > 1. The number of buckets.
    combiner: A string specifying how to reduce if the sparse column is
      multivalent. Currently "mean", "sqrtn" and "sum" are supported, with "sum"
      the default. "sqrtn" often achieves good accuracy, in particular with
      bag-of-words columns.
        * "sum": do not normalize features in the column
        * "mean": do l1 normalization on features in the column
        * "sqrtn": do l2 normalization on features in the column
      For more information: `tf.embedding_lookup_sparse`.
    dtype: The type of features. Only string and integer types are supported.
    hash_keys: The hash keys to use. It is a list of lists of two uint64s. If
      None, simple and fast hashing algorithm is used. Otherwise, multiple
      strong hash ids would be produced with each two unit64s in this argument.

  Returns:
    A _SparseColumn with hashed bucket configuration

  Raises:
    ValueError: hash_bucket_size is not greater than 2.
    ValueError: dtype is neither string nor integer.
  """
  return _SparseColumnHashed(
      column_name,
      bucket_size=hash_bucket_size,
      combiner=combiner,
      dtype=dtype,
      hash_keys=hash_keys)


class _SparseColumnKeys(_SparseColumn):
  """See `sparse_column_with_keys`."""

  def _do_transform(self, input_tensor):
    table = lookup.index_table_from_tensor(
        mapping=tuple(self.lookup_config.keys),
        default_value=self.lookup_config.default_value,
        dtype=self.dtype,
        name="lookup")
    return table.lookup(input_tensor)


def sparse_column_with_keys(
    column_name, keys, default_value=-1, combiner="sum", dtype=dtypes.string):
  """Creates a _SparseColumn with keys.

  Look up logic is as follows:
  lookup_id = index_of_feature_in_keys if feature in keys else default_value

  Args:
    column_name: A string defining sparse column name.
    keys: A list or tuple defining vocabulary. Must be castable to `dtype`.
    default_value: The value to use for out-of-vocabulary feature values.
      Default is -1.
    combiner: A string specifying how to reduce if the sparse column is
      multivalent. Currently "mean", "sqrtn" and "sum" are supported, with "sum"
      the default. "sqrtn" often achieves good accuracy, in particular with
      bag-of-words columns.
        * "sum": do not normalize features in the column
        * "mean": do l1 normalization on features in the column
        * "sqrtn": do l2 normalization on features in the column
      For more information: `tf.embedding_lookup_sparse`.
    dtype: Type of features. Only integer and string are supported.

  Returns:
    A _SparseColumnKeys with keys configuration.
  """
  keys = tuple(keys)
  return _SparseColumnKeys(
      column_name,
      lookup_config=_SparseIdLookupConfig(
          keys=keys, vocab_size=len(keys), default_value=default_value),
      combiner=combiner,
      dtype=dtype)


class _SparseColumnVocabulary(_SparseColumn):
  """See `sparse_column_with_vocabulary_file`."""

  def _do_transform(self, st):
    if self.dtype.is_integer:
      sparse_string_values = string_ops.as_string(st.values)
      sparse_string_tensor = sparse_tensor_py.SparseTensor(st.indices,
                                                           sparse_string_values,
                                                           st.dense_shape)
    else:
      sparse_string_tensor = st

    table = lookup.index_table_from_file(
        vocabulary_file=self.lookup_config.vocabulary_file,
        num_oov_buckets=self.lookup_config.num_oov_buckets,
        vocab_size=self.lookup_config.vocab_size,
        default_value=self.lookup_config.default_value,
        name=self.name + "_lookup")
    return table.lookup(sparse_string_tensor)


def sparse_column_with_vocabulary_file(column_name,
                                       vocabulary_file,
                                       num_oov_buckets=0,
                                       vocab_size=None,
                                       default_value=-1,
                                       combiner="sum",
                                       dtype=dtypes.string):
  """Creates a _SparseColumn with vocabulary file configuration.

  Use this when your sparse features are in string or integer format, and you
  have a vocab file that maps each value to an integer ID.
  output_id = LookupIdFromVocab(input_feature_string)

  Args:
    column_name: A string defining sparse column name.
    vocabulary_file: The vocabulary filename.
    num_oov_buckets: The number of out-of-vocabulary buckets. If zero all out of
      vocabulary features will be ignored.
    vocab_size: Number of the elements in the vocabulary.
    default_value: The value to use for out-of-vocabulary feature values.
      Defaults to -1.
    combiner: A string specifying how to reduce if the sparse column is
      multivalent. Currently "mean", "sqrtn" and "sum" are supported, with "sum"
      the default. "sqrtn" often achieves good accuracy, in particular with
      bag-of-words columns.
        * "sum": do not normalize features in the column
        * "mean": do l1 normalization on features in the column
        * "sqrtn": do l2 normalization on features in the column
      For more information: `tf.embedding_lookup_sparse`.
    dtype: The type of features. Only string and integer types are supported.

  Returns:
    A _SparseColumn with vocabulary file configuration.

  Raises:
    ValueError: vocab_size is not defined.
    ValueError: dtype is neither string nor integer.
  """
  if vocab_size is None:
    raise ValueError("vocab_size should be defined. "
                     "column_name: {}".format(column_name))

  return _SparseColumnVocabulary(
      column_name,
      lookup_config=_SparseIdLookupConfig(
          vocabulary_file=vocabulary_file,
          num_oov_buckets=num_oov_buckets,
          vocab_size=vocab_size,
          default_value=default_value),
      combiner=combiner,
      dtype=dtype)


class _WeightedSparseColumn(
    _FeatureColumn,
    fc_core._CategoricalColumn,  # pylint: disable=protected-access
    collections.namedtuple("_WeightedSparseColumn",
                           ["sparse_id_column", "weight_column_name",
                            "dtype"])):
  """See `weighted_sparse_column`."""

  def __new__(cls, sparse_id_column, weight_column_name, dtype):
    return super(_WeightedSparseColumn, cls).__new__(cls, sparse_id_column,
                                                     weight_column_name, dtype)

  @property
  def name(self):
    return "{}_weighted_by_{}".format(self.sparse_id_column.name,
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
  def lookup_config(self):
    return self.sparse_id_column.lookup_config

  @property
  def key(self):
    """Returns a string which will be used as a key when we do sorting."""
    return "{}".format(self)

  def id_tensor(self, input_tensor):
    """Returns the id tensor from the given transformed input_tensor."""
    return input_tensor[0]

  def weight_tensor(self, input_tensor):
    """Returns the weight tensor from the given transformed input_tensor."""
    return input_tensor[1]

  # pylint: disable=unused-argument
  def _to_dnn_input_layer(self,
                          input_tensor,
                          weight_collections=None,
                          trainable=True,
                          output_rank=2):
    raise ValueError(
        "WeightedSparseColumn is not supported in DNN. "
        "Please use embedding_column or one_hot_column. column: {}".format(
            self))

  def _wide_embedding_lookup_arguments(self, input_tensor):
    return _LinearEmbeddingLookupArguments(
        input_tensor=self.id_tensor(input_tensor),
        weight_tensor=self.weight_tensor(input_tensor),
        vocab_size=self.length,
        initializer=init_ops.zeros_initializer(),
        combiner=self.sparse_id_column.combiner)

  def _do_transform(self, id_tensor, weight_tensor):
    if not isinstance(weight_tensor, sparse_tensor_py.SparseTensor):
      # The weight tensor can be a regular Tensor. In such case, sparsify it.
      weight_tensor = contrib_sparse_ops.dense_to_sparse_tensor(weight_tensor)
    if not self.dtype.is_floating:
      weight_tensor = math_ops.cast(weight_tensor, dtypes.float32)
    return tuple([id_tensor, weight_tensor])

  def insert_transformed_feature(self, columns_to_tensors):
    """Inserts a tuple with the id and weight tensors."""
    if self.sparse_id_column not in columns_to_tensors:
      self.sparse_id_column.insert_transformed_feature(columns_to_tensors)

    weight_tensor = columns_to_tensors[self.weight_column_name]
    columns_to_tensors[self] = self._do_transform(
        columns_to_tensors[self.sparse_id_column], weight_tensor)

  def _transform_feature(self, inputs):
    return self._do_transform(
        inputs.get(self.sparse_id_column), inputs.get(self.weight_column_name))

  @property
  def _parse_example_spec(self):
    return self.config

  @property
  def _num_buckets(self):
    return self.length

  def _get_sparse_tensors(self, inputs, weight_collections=None,
                          trainable=None):
    del weight_collections
    del trainable
    input_tensor = inputs.get(self)
    return fc_core._CategoricalColumn.IdWeightPair(  # pylint: disable=protected-access
        self.id_tensor(input_tensor), self.weight_tensor(input_tensor))

  def is_compatible(self, other_column):
    """Check compatibility with other sparse column."""
    if isinstance(other_column, _WeightedSparseColumn):
      return self.sparse_id_column.is_compatible(other_column.sparse_id_column)
    return self.sparse_id_column.is_compatible(other_column)


def weighted_sparse_column(sparse_id_column,
                           weight_column_name,
                           dtype=dtypes.float32):
  """Creates a _SparseColumn by combining sparse_id_column with a weight column.

  Example:

    ```python
    sparse_feature = sparse_column_with_hash_bucket(column_name="sparse_col",
                                                    hash_bucket_size=1000)
    weighted_feature = weighted_sparse_column(sparse_id_column=sparse_feature,
                                              weight_column_name="weights_col")
    ```

    This configuration assumes that input dictionary of model contains the
    following two items:
      * (key="sparse_col", value=sparse_tensor) where sparse_tensor is
        a SparseTensor.
      * (key="weights_col", value=weights_tensor) where weights_tensor
        is a SparseTensor.
     Following are assumed to be true:
       * sparse_tensor.indices = weights_tensor.indices
       * sparse_tensor.dense_shape = weights_tensor.dense_shape

  Args:
    sparse_id_column: A `_SparseColumn` which is created by
      `sparse_column_with_*` functions.
    weight_column_name: A string defining a sparse column name which represents
      weight or value of the corresponding sparse id feature.
    dtype: Type of weights, such as `tf.float32`. Only floating and integer
      weights are supported.

  Returns:
    A _WeightedSparseColumn composed of two sparse features: one represents id,
    the other represents weight (value) of the id feature in that example.

  Raises:
    ValueError: if dtype is not convertible to float.
  """
  if not (dtype.is_integer or dtype.is_floating):
    raise ValueError("dtype is not convertible to float. Given {}".format(
        dtype))

  return _WeightedSparseColumn(sparse_id_column, weight_column_name, dtype)


class _OneHotColumn(
    _FeatureColumn,
    fc_core._DenseColumn,  # pylint: disable=protected-access
    collections.namedtuple("_OneHotColumn", ["sparse_id_column"])):
  """Represents a one-hot column for use in deep networks.

  Args:
    sparse_id_column: A _SparseColumn which is created by `sparse_column_with_*`
      function.
  """

  @property
  def name(self):
    return "{}_one_hot".format(self.sparse_id_column.name)

  @property
  def length(self):
    """Returns vocabulary or hash_bucket size."""
    return self.sparse_id_column.length

  @property
  def config(self):
    """Returns the parsing config of the origin column."""
    return _get_feature_config(self.sparse_id_column)

  @property
  def key(self):
    """Returns a string which will be used as a key when we do sorting."""
    return "{}".format(self)

  def insert_transformed_feature(self, columns_to_tensors):
    """Used by the Transformer to prevent double transformations."""
    if self.sparse_id_column not in columns_to_tensors:
      self.sparse_id_column.insert_transformed_feature(columns_to_tensors)
    columns_to_tensors[self] = columns_to_tensors[self.sparse_id_column]

  def _to_dnn_input_layer(self,
                          transformed_input_tensor,
                          unused_weight_collections=None,
                          unused_trainable=False,
                          output_rank=2):
    """Returns a Tensor as an input to the first layer of neural network.

    Args:
      transformed_input_tensor: A tensor that has undergone the transformations
      in `insert_transformed_feature`. Rank should be >= `output_rank`.
      unused_weight_collections: Unused. One hot encodings are not variable.
      unused_trainable: Unused. One hot encodings are not trainable.
      output_rank: the desired rank of the output `Tensor`.

    Returns:
      A multi-hot Tensor to be fed into the first layer of neural network.

    Raises:
      ValueError: When using one_hot_column with weighted_sparse_column.
      This is not yet supported.
    """

    # Reshape ID column to `output_rank`.
    sparse_id_column = self.sparse_id_column.id_tensor(transformed_input_tensor)
    # pylint: disable=protected-access
    sparse_id_column = layers._inner_flatten(sparse_id_column, output_rank)

    weight_tensor = self.sparse_id_column.weight_tensor(
        transformed_input_tensor)
    if weight_tensor is not None:
      weighted_column = sparse_ops.sparse_merge(sp_ids=sparse_id_column,
                                                sp_values=weight_tensor,
                                                vocab_size=self.length)
      # Remove (?, -1) index
      weighted_column = sparse_ops.sparse_slice(
          weighted_column,
          array_ops.zeros_like(weighted_column.dense_shape),
          weighted_column.dense_shape)
      dense_tensor = sparse_ops.sparse_tensor_to_dense(weighted_column)
      batch_shape = array_ops.shape(dense_tensor)[:-1]
      dense_tensor_shape = array_ops.concat(
          [batch_shape, [self.length]], axis=0)
      dense_tensor = array_ops.reshape(dense_tensor, dense_tensor_shape)
      return dense_tensor

    dense_id_tensor = sparse_ops.sparse_tensor_to_dense(sparse_id_column,
                                                        default_value=-1)

    # One hot must be float for tf.concat reasons since all other inputs to
    # input_layer are float32.
    one_hot_id_tensor = array_ops.one_hot(
        dense_id_tensor, depth=self.length, on_value=1.0, off_value=0.0)

    # Reduce to get a multi-hot per example.
    return math_ops.reduce_sum(one_hot_id_tensor, axis=[output_rank - 1])

  @property
  def _variable_shape(self):
    return tensor_shape.TensorShape([self.length])

  def _get_dense_tensor(self, inputs, weight_collections=None, trainable=None):
    del weight_collections
    del trainable
    return inputs.get(self)

  def _transform_feature(self, inputs):
    return self._to_dnn_input_layer(inputs.get(self.sparse_id_column))

  @property
  def _parse_example_spec(self):
    return self.config


class _EmbeddingColumn(
    _FeatureColumn,
    fc_core._DenseColumn,  # pylint: disable=protected-access
    collections.namedtuple("_EmbeddingColumn", [
        "sparse_id_column", "dimension", "combiner", "initializer",
        "ckpt_to_load_from", "tensor_name_in_ckpt", "shared_embedding_name",
        "shared_vocab_size", "max_norm", "trainable"
    ])):
  """Represents an embedding column.

  Args:
    sparse_id_column: A `_SparseColumn` which is created by
      `sparse_column_with_*` or `weighted_sparse_column` functions.
    dimension: An integer specifying dimension of the embedding.
    combiner: A string specifying how to reduce if there are multiple entries
      in a single row. Currently "mean", "sqrtn" and "sum" are supported, with
      "mean" the default. "sqrtn" often achieves good accuracy, in particular
      with bag-of-words columns. Each of this can be thought as example level
      normalizations on the column:
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
    shared_embedding_name: (Optional). The common name for shared embedding.
    shared_vocab_size: (Optional). The common vocab_size used for shared
      embedding space.
    max_norm: (Optional). If not None, embedding values are l2-normalized to
      the value of max_norm.
    trainable: (Optional). Should the embedding be trainable. Default is True.

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
              tensor_name_in_ckpt=None,
              shared_embedding_name=None,
              shared_vocab_size=None,
              max_norm=None,
              trainable=True):
    if initializer is not None and not callable(initializer):
      raise ValueError("initializer must be callable if specified. "
                       "Embedding of column_name: {}".format(
                           sparse_id_column.name))

    if (ckpt_to_load_from is None) != (tensor_name_in_ckpt is None):
      raise ValueError("Must specify both `ckpt_to_load_from` and "
                       "`tensor_name_in_ckpt` or none of them.")
    if initializer is None:
      logging.warn("The default stddev value of initializer was changed from "
                   "\"1/sqrt(vocab_size)\" to \"1/sqrt(dimension)\" in core "
                   "implementation (tf.feature_column.embedding_column).")
      stddev = 1 / math.sqrt(sparse_id_column.length)
      initializer = init_ops.truncated_normal_initializer(
          mean=0.0, stddev=stddev)
    return super(_EmbeddingColumn, cls).__new__(cls, sparse_id_column,
                                                dimension, combiner,
                                                initializer, ckpt_to_load_from,
                                                tensor_name_in_ckpt,
                                                shared_embedding_name,
                                                shared_vocab_size,
                                                max_norm,
                                                trainable)

  @property
  def name(self):
    if self.shared_embedding_name is None:
      return "{}_embedding".format(self.sparse_id_column.name)
    else:
      return "{}_shared_embedding".format(self.sparse_id_column.name)

  @property
  def length(self):
    """Returns id size."""
    if self.shared_vocab_size is None:
      return self.sparse_id_column.length
    else:
      return self.shared_vocab_size

  @property
  def config(self):
    return _get_feature_config(self.sparse_id_column)

  @property
  def key(self):
    """Returns a string which will be used as a key when we do sorting."""
    return self._key_without_properties(["initializer"])

  def insert_transformed_feature(self, columns_to_tensors):
    if self.sparse_id_column not in columns_to_tensors:
      self.sparse_id_column.insert_transformed_feature(columns_to_tensors)
    columns_to_tensors[self] = columns_to_tensors[self.sparse_id_column]

  def _deep_embedding_lookup_arguments(self, input_tensor):
    return _DeepEmbeddingLookupArguments(
        input_tensor=self.sparse_id_column.id_tensor(input_tensor),
        weight_tensor=self.sparse_id_column.weight_tensor(input_tensor),
        vocab_size=self.length,
        dimension=self.dimension,
        initializer=self.initializer,
        combiner=self.combiner,
        shared_embedding_name=self.shared_embedding_name,
        hash_key=None,
        max_norm=self.max_norm,
        trainable=self.trainable)

  def _checkpoint_path(self):
    if self.ckpt_to_load_from is not None:
      return self.ckpt_to_load_from, self.tensor_name_in_ckpt
    return None

  # pylint: disable=unused-argument
  def _wide_embedding_lookup_arguments(self, input_tensor):
    raise ValueError("Column {} is not supported in linear models. "
                     "Please use sparse_column.".format(self))

  @property
  def _variable_shape(self):
    return tensor_shape.TensorShape([self.dimension])

  def _get_dense_tensor(self, inputs, weight_collections=None, trainable=None):
    return _embeddings_from_arguments(
        self,
        self._deep_embedding_lookup_arguments(inputs.get(self)),
        weight_collections, trainable)

  def _transform_feature(self, inputs):
    return inputs.get(self.sparse_id_column)

  @property
  def _parse_example_spec(self):
    return self.config


def _is_variable(v):
  """Returns true if `v` is a variable."""
  return isinstance(v, (variables.Variable,
                        resource_variable_ops.ResourceVariable))


def _embeddings_from_arguments(column,
                               args,
                               weight_collections,
                               trainable,
                               output_rank=2):
  """Returns embeddings for a column based on the computed arguments.

  Args:
   column: the column name.
   args: the _DeepEmbeddingLookupArguments for this column.
   weight_collections: collections to store weights in.
   trainable: whether these embeddings should be trainable.
   output_rank: the desired rank of the returned `Tensor`. Inner dimensions will
     be combined to produce the desired rank.

  Returns:
   the embeddings.

  Raises:
   ValueError: if not possible to create.
  """
  # pylint: disable=protected-access
  input_tensor = layers._inner_flatten(args.input_tensor, output_rank)
  weight_tensor = None
  if args.weight_tensor is not None:
    weight_tensor = layers._inner_flatten(args.weight_tensor, output_rank)
  # pylint: enable=protected-access

  # This option is only enabled for scattered_embedding_column.
  if args.hash_key:
    embeddings = contrib_variables.model_variable(
        name="weights",
        shape=[args.vocab_size],
        dtype=dtypes.float32,
        initializer=args.initializer,
        trainable=(trainable and args.trainable),
        collections=weight_collections)

    return embedding_ops.scattered_embedding_lookup_sparse(
        embeddings,
        input_tensor,
        args.dimension,
        hash_key=args.hash_key,
        combiner=args.combiner,
        name="lookup")

  if args.shared_embedding_name is not None:
    shared_embedding_collection_name = (
        "SHARED_EMBEDDING_COLLECTION_" + args.shared_embedding_name.upper())
    graph = ops.get_default_graph()
    shared_embedding_collection = (
        graph.get_collection_ref(shared_embedding_collection_name))
    shape = [args.vocab_size, args.dimension]
    if shared_embedding_collection:
      if len(shared_embedding_collection) > 1:
        raise ValueError(
            "Collection %s can only contain one "
            "(partitioned) variable." % shared_embedding_collection_name)
      else:
        embeddings = shared_embedding_collection[0]
        if embeddings.get_shape() != shape:
          raise ValueError(
              "The embedding variable with name {} already "
              "exists, but its shape does not match required "
              "embedding shape here. Please make sure to use "
              "different shared_embedding_name for different "
              "shared embeddings.".format(args.shared_embedding_name))
    else:
      embeddings = contrib_variables.model_variable(
          name=args.shared_embedding_name,
          shape=shape,
          dtype=dtypes.float32,
          initializer=args.initializer,
          trainable=(trainable and args.trainable),
          collections=weight_collections)
      graph.add_to_collection(shared_embedding_collection_name, embeddings)
  else:
    embeddings = contrib_variables.model_variable(
        name="weights",
        shape=[args.vocab_size, args.dimension],
        dtype=dtypes.float32,
        initializer=args.initializer,
        trainable=(trainable and args.trainable),
        collections=weight_collections)

  if _is_variable(embeddings):
    embeddings = [embeddings]
  else:
    embeddings = embeddings._get_variable_list()  # pylint: disable=protected-access
  # pylint: disable=protected-access
  _maybe_restore_from_checkpoint(column._checkpoint_path(), embeddings)
  return embedding_ops.safe_embedding_lookup_sparse(
      embeddings,
      input_tensor,
      sparse_weights=weight_tensor,
      combiner=args.combiner,
      name=column.name + "weights",
      max_norm=args.max_norm)


def _maybe_restore_from_checkpoint(checkpoint_path, variable):
  if checkpoint_path is not None:
    path, tensor_name = checkpoint_path
    weights_to_restore = variable
    if len(variable) == 1:
      weights_to_restore = variable[0]
    checkpoint_utils.init_from_checkpoint(path,
                                          {tensor_name: weights_to_restore})


def one_hot_column(sparse_id_column):
  """Creates an `_OneHotColumn` for a one-hot or multi-hot repr in a DNN.

  Args:
      sparse_id_column: A _SparseColumn which is created by
        `sparse_column_with_*`
        or crossed_column functions. Note that `combiner` defined in
        `sparse_id_column` is ignored.

  Returns:
    An _OneHotColumn.
  """
  return _OneHotColumn(sparse_id_column)


def embedding_column(sparse_id_column,
                     dimension,
                     combiner="mean",
                     initializer=None,
                     ckpt_to_load_from=None,
                     tensor_name_in_ckpt=None,
                     max_norm=None,
                     trainable=True):
  """Creates an `_EmbeddingColumn` for feeding sparse data into a DNN.

  Args:
    sparse_id_column: A `_SparseColumn` which is created by for example
      `sparse_column_with_*` or crossed_column functions. Note that `combiner`
      defined in `sparse_id_column` is ignored.
    dimension: An integer specifying dimension of the embedding.
    combiner: A string specifying how to reduce if there are multiple entries
      in a single row. Currently "mean", "sqrtn" and "sum" are supported, with
      "mean" the default. "sqrtn" often achieves good accuracy, in particular
      with bag-of-words columns. Each of this can be thought as example level
      normalizations on the column:
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
    max_norm: (Optional). If not None, embedding values are l2-normalized to
      the value of max_norm.
    trainable: (Optional). Should the embedding be trainable. Default is True

  Returns:
    An `_EmbeddingColumn`.
  """
  return _EmbeddingColumn(sparse_id_column, dimension, combiner, initializer,
                          ckpt_to_load_from, tensor_name_in_ckpt,
                          max_norm=max_norm, trainable=trainable)


def shared_embedding_columns(sparse_id_columns,
                             dimension,
                             combiner="mean",
                             shared_embedding_name=None,
                             initializer=None,
                             ckpt_to_load_from=None,
                             tensor_name_in_ckpt=None,
                             max_norm=None,
                             trainable=True):
  """Creates a list of `_EmbeddingColumn` sharing the same embedding.

  Args:
    sparse_id_columns: An iterable of `_SparseColumn`, such as those created by
      `sparse_column_with_*` or crossed_column functions. Note that `combiner`
      defined in each sparse_id_column is ignored.
    dimension: An integer specifying dimension of the embedding.
    combiner: A string specifying how to reduce if there are multiple entries
      in a single row. Currently "mean", "sqrtn" and "sum" are supported, with
      "mean" the default. "sqrtn" often achieves good accuracy, in particular
      with bag-of-words columns. Each of this can be thought as example level
      normalizations on the column:
        * "sum": do not normalize
        * "mean": do l1 normalization
        * "sqrtn": do l2 normalization
      For more information: `tf.embedding_lookup_sparse`.
    shared_embedding_name: (Optional). A string specifying the name of shared
      embedding weights. This will be needed if you want to reference the shared
      embedding separately from the generated `_EmbeddingColumn`.
    initializer: A variable initializer function to be used in embedding
      variable initialization. If not specified, defaults to
      `tf.truncated_normal_initializer` with mean 0.0 and standard deviation
      1/sqrt(sparse_id_columns[0].length).
    ckpt_to_load_from: (Optional). String representing checkpoint name/pattern
      to restore the column weights. Required if `tensor_name_in_ckpt` is not
      None.
    tensor_name_in_ckpt: (Optional). Name of the `Tensor` in the provided
      checkpoint from which to restore the column weights. Required if
      `ckpt_to_load_from` is not None.
    max_norm: (Optional). If not None, embedding values are l2-normalized to
      the value of max_norm.
    trainable: (Optional). Should the embedding be trainable. Default is True

  Returns:
    A tuple of `_EmbeddingColumn` with shared embedding space.

  Raises:
    ValueError: if sparse_id_columns is empty, or its elements are not
      compatible with each other.
    TypeError: if `sparse_id_columns` is not a sequence or is a string. If at
      least one element of `sparse_id_columns` is not a `SparseColumn` or a
      `WeightedSparseColumn`.
  """
  if (not isinstance(sparse_id_columns, collections.Sequence) or
      isinstance(sparse_id_columns, six.string_types)):
    raise TypeError(
        "sparse_id_columns must be a non-string sequence (ex: list or tuple) "
        "instead of type {}.".format(type(sparse_id_columns)))
  if len(sparse_id_columns) < 1:
    raise ValueError("The input sparse_id_columns should have at least one "
                     "element.")
  for sparse_id_column in sparse_id_columns:
    if not (isinstance(sparse_id_column, _SparseColumn) or
            isinstance(sparse_id_column, _WeightedSparseColumn)):
      raise TypeError("Elements of sparse_id_columns must be _SparseColumn or "
                      "_WeightedSparseColumn, but {} is not."
                      .format(sparse_id_column))

  if len(sparse_id_columns) == 1:
    return [
        _EmbeddingColumn(sparse_id_columns[0], dimension, combiner, initializer,
                         ckpt_to_load_from, tensor_name_in_ckpt,
                         shared_embedding_name, max_norm=max_norm,
                         trainable=trainable)]
  else:
    # Check compatibility of sparse_id_columns
    compatible = True
    for column in sparse_id_columns[1:]:
      if isinstance(sparse_id_columns[0], _WeightedSparseColumn):
        compatible = compatible and sparse_id_columns[0].is_compatible(column)
      else:
        compatible = compatible and column.is_compatible(sparse_id_columns[0])
    if not compatible:
      raise ValueError("The input sparse id columns are not compatible.")
    # Construct the shared name and size for shared embedding space.
    if not shared_embedding_name:
      # Sort the columns so that shared_embedding_name will be deterministic
      # even if users pass in unsorted columns from a dict or something.
      # Since they are different classes, ordering is SparseColumns first,
      # then WeightedSparseColumns.
      sparse_columns = []
      weighted_sparse_columns = []
      for column in sparse_id_columns:
        if isinstance(column, _SparseColumn):
          sparse_columns.append(column)
        else:
          weighted_sparse_columns.append(column)
      sorted_columns = sorted(sparse_columns) + sorted(
          weighted_sparse_columns, key=lambda x: x.name)
      if len(sorted_columns) <= 3:
        shared_embedding_name = "_".join([column.name
                                          for column in sorted_columns])
      else:
        shared_embedding_name = "_".join([column.name
                                          for column in sorted_columns[0:3]])
        shared_embedding_name += (
            "_plus_{}_others".format(len(sorted_columns) - 3))
      shared_embedding_name += "_shared_embedding"
    shared_vocab_size = sparse_id_columns[0].length

    embedded_columns = []
    for column in sparse_id_columns:
      embedded_columns.append(
          _EmbeddingColumn(column, dimension, combiner, initializer,
                           ckpt_to_load_from, tensor_name_in_ckpt,
                           shared_embedding_name, shared_vocab_size,
                           max_norm=max_norm, trainable=trainable))
    return tuple(embedded_columns)


class _ScatteredEmbeddingColumn(
    _FeatureColumn,
    fc_core._DenseColumn,  # pylint: disable=protected-access
    collections.namedtuple("_ScatteredEmbeddingColumn", [
        "column_name", "size", "dimension", "hash_key", "combiner",
        "initializer"
    ])):
  """See `scattered_embedding_column`."""

  def __new__(cls,
              column_name,
              size,
              dimension,
              hash_key,
              combiner="sqrtn",
              initializer=None):
    if initializer is not None and not callable(initializer):
      raise ValueError("initializer must be callable if specified. "
                       "column_name: {}".format(column_name))
    if initializer is None:
      stddev = 0.1
      initializer = init_ops.truncated_normal_initializer(
          mean=0.0, stddev=stddev)
    return super(_ScatteredEmbeddingColumn, cls).__new__(cls, column_name, size,
                                                         dimension, hash_key,
                                                         combiner,
                                                         initializer)

  @property
  def name(self):
    return "{}_scattered_embedding".format(self.column_name)

  @property
  def config(self):
    return {self.column_name: parsing_ops.VarLenFeature(dtypes.string)}

  @property
  def key(self):
    """Returns a string which will be used as a key when we do sorting."""
    return self._key_without_properties(["initializer"])

  def insert_transformed_feature(self, columns_to_tensors):
    columns_to_tensors[self] = columns_to_tensors[self.column_name]

  def _deep_embedding_lookup_arguments(self, input_tensor):
    return _DeepEmbeddingLookupArguments(
        input_tensor=input_tensor,
        weight_tensor=None,
        vocab_size=self.size,
        initializer=self.initializer,
        combiner=self.combiner,
        dimension=self.dimension,
        shared_embedding_name=None,
        hash_key=self.hash_key,
        max_norm=None,
        trainable=True)

  @property
  def _variable_shape(self):
    return tensor_shape.TensorShape([self.dimension])

  def _get_dense_tensor(self, inputs, weight_collections=None, trainable=None):
    return _embeddings_from_arguments(
        self,
        self._deep_embedding_lookup_arguments(inputs.get(self)),
        weight_collections, trainable)

  def _transform_feature(self, inputs):
    return inputs.get(self.column_name)

  @property
  def _parse_example_spec(self):
    return self.config


def scattered_embedding_column(column_name,
                               size,
                               dimension,
                               hash_key,
                               combiner="mean",
                               initializer=None):
  """Creates an embedding column of a sparse feature using parameter hashing.

  This is a useful shorthand when you have a sparse feature you want to use an
  embedding for, but also want to hash the embedding's values in each dimension
  to a variable based on a different hash.

  Specifically, the i-th embedding component of a value v is found by retrieving
  an embedding weight whose index is a fingerprint of the pair (v,i).

  An embedding column with sparse_column_with_hash_bucket such as

      embedding_column(
        sparse_column_with_hash_bucket(column_name, bucket_size),
        dimension)

  could be replaced by

      scattered_embedding_column(
        column_name,
        size=bucket_size * dimension,
        dimension=dimension,
        hash_key=tf.contrib.layers.SPARSE_FEATURE_CROSS_DEFAULT_HASH_KEY)

  for the same number of embedding parameters. This should hopefully reduce the
  impact of collisions, but adds the cost of slowing down training.

  Args:
    column_name: A string defining sparse column name.
    size: An integer specifying the number of parameters in the embedding layer.
    dimension: An integer specifying dimension of the embedding.
    hash_key: Specify the hash_key that will be used by the `FingerprintCat64`
      function to combine the crosses fingerprints on SparseFeatureCrossOp.
    combiner: A string specifying how to reduce if there are multiple entries
      in a single row. Currently "mean", "sqrtn" and "sum" are supported, with
      "mean" the default. "sqrtn" often achieves good accuracy, in particular
      with bag-of-words columns. Each of this can be thought as example level
      normalizations on the column:
        * "sum": do not normalize features in the column
        * "mean": do l1 normalization on features in the column
        * "sqrtn": do l2 normalization on features in the column
      For more information: `tf.embedding_lookup_sparse`.
    initializer: A variable initializer function to be used in embedding
      variable initialization. If not specified, defaults to
      `tf.truncated_normal_initializer` with mean 0 and standard deviation 0.1.

  Returns:
    A _ScatteredEmbeddingColumn.

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
                     "combiner: {}, column_name: {}".format(combiner,
                                                            column_name))

  return _ScatteredEmbeddingColumn(column_name, size, dimension, hash_key,
                                   combiner, initializer)


def _reshape_real_valued_tensor(input_tensor, output_rank, column_name=None):
  """Reshaping logic for dense, numeric `Tensors`.

  Follows the following rules:
    1. If `output_rank > input_rank + 1` raise a `ValueError`.
    2. If `output_rank == input_rank + 1`, expand `input_tensor` by one
       dimension and return
    3. If `output_rank == input_rank`, return `input_tensor`.
    4. If `output_rank < input_rank`, flatten the inner dimensions of
       `input_tensor` and return a `Tensor` with `output_rank`

  Args:
    input_tensor: a dense `Tensor` to be reshaped.
    output_rank: the desired rank of the reshaped `Tensor`.
    column_name: (optional) the name of the associated column. Used for error
      messages.
  Returns:
    A `Tensor` with the same entries as `input_tensor` and rank `output_rank`.
  Raises:
    ValueError: if `output_rank > input_rank + 1`.
  """
  input_rank = input_tensor.get_shape().ndims
  if input_rank is not None:
    if output_rank > input_rank + 1:
      error_string = ("Rank of input Tensor ({}) should be the same as "
                      "output_rank ({}). For example, sequence data should "
                      "typically be 3 dimensional (rank 3) while non-sequence "
                      "data is typically 2 dimensional (rank 2).".format(
                          input_rank, output_rank))
      if column_name is not None:
        error_string = ("Error while processing column {}.".format(column_name)
                        + error_string)
      raise ValueError(error_string)
    if output_rank == input_rank + 1:
      logging.warning(
          "Rank of input Tensor ({}) should be the same as output_rank ({}) "
          "for column. Will attempt to expand dims. It is highly recommended "
          "that you resize your input, as this behavior may change.".format(
              input_rank, output_rank))
      return array_ops.expand_dims(input_tensor, -1, name="expand_dims")
    if output_rank == input_rank:
      return input_tensor
  # Here, either `input_rank` is unknown or it is greater than `output_rank`.
  return layers._inner_flatten(input_tensor, output_rank)  # pylint: disable=protected-access


class _RealValuedVarLenColumn(_FeatureColumn, collections.namedtuple(
    "_RealValuedVarLenColumn",
    ["column_name", "default_value", "dtype", "normalizer", "is_sparse"])):
  """Represents a real valued feature column for variable length Features.

  Instances of this class are immutable.
  If is_sparse=False, the dictionary returned by InputBuilder contains a
  ("column_name", Tensor) pair with a Tensor shape of (batch_size, dimension).
  If is_sparse=True, the dictionary contains a ("column_name", SparseTensor)
  pair instead with shape inferred after parsing.
  """

  @property
  def name(self):
    return self.column_name

  @property
  def config(self):
    if self.is_sparse:
      return {self.column_name: parsing_ops.VarLenFeature(self.dtype)}
    else:
      return {self.column_name: parsing_ops.FixedLenSequenceFeature(
          [], self.dtype, allow_missing=True,
          default_value=self.default_value)}

  @property
  def key(self):
    """Returns a string which will be used as a key when we do sorting."""
    return self._key_without_properties(["normalizer"])

  @property
  def normalizer_fn(self):
    """Returns the function used to normalize the column."""
    return self.normalizer

  def _normalized_input_tensor(self, input_tensor):
    """Returns the input tensor after custom normalization is applied."""
    if self.normalizer is None:
      return input_tensor
    if self.is_sparse:
      return sparse_tensor_py.SparseTensor(
          input_tensor.indices,
          self.normalizer(input_tensor.values),
          input_tensor.dense_shape)
    else:
      return self.normalizer(input_tensor)

  def insert_transformed_feature(self, columns_to_tensors):
    """Apply transformation and inserts it into columns_to_tensors.

    Args:
      columns_to_tensors: A mapping from feature columns to tensors. 'string'
        key means a base feature (not-transformed). It can have _FeatureColumn
        as a key too. That means that _FeatureColumn is already transformed.
    """
    # Transform the input tensor according to the normalizer function.
    input_tensor = self._normalized_input_tensor(columns_to_tensors[self.name])
    columns_to_tensors[self] = math_ops.cast(input_tensor, dtypes.float32)

  # pylint: disable=unused-argument
  def _to_dnn_input_layer(self,
                          input_tensor,
                          weight_collections=None,
                          trainable=True,
                          output_rank=2):
    return _reshape_real_valued_tensor(
        self._to_dense_tensor(input_tensor), output_rank, self.name)

  def _to_dense_tensor(self, input_tensor):
    if not self.is_sparse:
      return input_tensor
    raise ValueError("Set is_sparse to False if you want a dense Tensor for "
                     "column_name: {}".format(self.name))


@experimental
def _real_valued_var_len_column(column_name,
                                default_value=None,
                                dtype=dtypes.float32,
                                normalizer=None,
                                is_sparse=False):
  """Creates a `_RealValuedVarLenColumn` for variable-length numeric data.

  Note, this is not integrated with any of the DNNEstimators, except the RNN
  ones DynamicRNNEstimator and the StateSavingRNNEstimator.

  It can either create a parsing config for a SparseTensor (with is_sparse=True)
  or a padded Tensor.
  The (dense_)shape of the result will be [batch_size, None], which can be used
  with is_sparse=False as input into an RNN (see DynamicRNNEstimator or
  StateSavingRNNEstimator) or with is_sparse=True as input into a tree (see
  gtflow).

  Use real_valued_column if the Feature has a fixed length. Use some
  SparseColumn for columns to be embedded / one-hot-encoded.

  Args:
    column_name: A string defining real valued column name.
    default_value: A scalar value compatible with dtype. Needs to be specified
      if is_sparse=False.
    dtype: Defines the type of values. Default value is tf.float32. Needs to be
      convertible to tf.float32.
    normalizer: If not None, a function that can be used to normalize the value
      of the real valued column after default_value is applied for parsing.
      Normalizer function takes the input tensor as its argument, and returns
      the output tensor. (e.g. lambda x: (x - 3.0) / 4.2). Note that for
      is_sparse=False, the normalizer will be run on the values of the
      `SparseTensor`.
    is_sparse: A boolean defining whether to create a SparseTensor or a Tensor.
  Returns:
    A _RealValuedSparseColumn.
  Raises:
    TypeError: if default_value is not a scalar value compatible with dtype.
    TypeError: if dtype is not convertible to tf.float32.
    ValueError: if default_value is None and is_sparse is False.
  """
  if not (dtype.is_integer or dtype.is_floating):
    raise TypeError("dtype must be convertible to float. "
                    "dtype: {}, column_name: {}".format(dtype, column_name))

  if default_value is None and not is_sparse:
    raise ValueError("default_value must be provided when is_sparse=False to "
                     "parse a padded Tensor. "
                     "column_name: {}".format(column_name))
  if isinstance(default_value, list):
    raise ValueError(
        "Only scalar default value. default_value: {}, column_name: {}".format(
            default_value, column_name))
  if default_value is not None:
    if dtype.is_integer:
      default_value = int(default_value)
    elif dtype.is_floating:
      default_value = float(default_value)

  return _RealValuedVarLenColumn(column_name, default_value, dtype, normalizer,
                                 is_sparse)


class _RealValuedColumn(
    _FeatureColumn,
    fc_core._DenseColumn,  # pylint: disable=protected-access
    collections.namedtuple(
        "_RealValuedColumn",
        ["column_name", "dimension", "default_value", "dtype", "normalizer"])):
  """Represents a real valued feature column also known as continuous features.

  Instances of this class are immutable. The dictionary returned by InputBuilder
  contains a ("column_name", Tensor) pair with a Tensor shape of
  (batch_size, dimension).
  """

  def __new__(cls, column_name, dimension, default_value,
              dtype, normalizer):
    if default_value is not None:
      default_value = tuple(default_value)
    return super(_RealValuedColumn, cls).__new__(cls, column_name, dimension,
                                                 default_value, dtype,
                                                 normalizer)

  @property
  def name(self):
    return self.column_name

  @property
  def config(self):
    default_value = self.default_value
    if default_value is not None:
      default_value = list(default_value)
    return {self.column_name: parsing_ops.FixedLenFeature([self.dimension],
                                                          self.dtype,
                                                          default_value)}

  @property
  def key(self):
    """Returns a string which will be used as a key when we do sorting."""
    return self._key_without_properties(["normalizer"])

  @property
  def normalizer_fn(self):
    """Returns the function used to normalize the column."""
    return self.normalizer

  def _normalized_input_tensor(self, input_tensor):
    """Returns the input tensor after custom normalization is applied."""
    return (self.normalizer(input_tensor) if self.normalizer is not None else
            input_tensor)

  def insert_transformed_feature(self, columns_to_tensors):
    """Apply transformation and inserts it into columns_to_tensors.

    Args:
      columns_to_tensors: A mapping from feature columns to tensors. 'string'
        key means a base feature (not-transformed). It can have _FeatureColumn
        as a key too. That means that _FeatureColumn is already transformed.
    """
    # Transform the input tensor according to the normalizer function.
    input_tensor = self._normalized_input_tensor(columns_to_tensors[self.name])
    columns_to_tensors[self] = math_ops.cast(input_tensor, dtypes.float32)

  # pylint: disable=unused-argument
  def _to_dnn_input_layer(self,
                          input_tensor,
                          weight_collections=None,
                          trainable=True,
                          output_rank=2):
    input_tensor = self._to_dense_tensor(input_tensor)
    if input_tensor.dtype != dtypes.float32:
      input_tensor = math_ops.cast(input_tensor, dtypes.float32)
    return _reshape_real_valued_tensor(input_tensor, output_rank, self.name)

  def _to_dense_tensor(self, input_tensor):
    return input_tensor

  @property
  def _variable_shape(self):
    return tensor_shape.TensorShape([self.dimension])

  def _get_dense_tensor(self, inputs, weight_collections=None, trainable=None):
    del weight_collections
    del trainable
    return inputs.get(self)

  def _transform_feature(self, inputs):
    return math_ops.cast(
        self._normalized_input_tensor(inputs.get(self.name)), dtypes.float32)

  @property
  def _parse_example_spec(self):
    return self.config


def real_valued_column(column_name,
                       dimension=1,
                       default_value=None,
                       dtype=dtypes.float32,
                       normalizer=None):
  """Creates a `_RealValuedColumn` for dense numeric data.

  Args:
    column_name: A string defining real valued column name.
    dimension: An integer specifying dimension of the real valued column.
      The default is 1.
    default_value: A single value compatible with dtype or a list of values
      compatible with dtype which the column takes on during tf.Example parsing
      if data is missing. When dimension is not None, a default value of None
      will cause tf.parse_example to fail if an example does not contain this
      column. If a single value is provided, the same value will be applied as
      the default value for every dimension. If a list of values is provided,
      the length of the list should be equal to the value of `dimension`.
      Only scalar default value is supported in case dimension is not specified.
    dtype: defines the type of values. Default value is tf.float32. Must be a
      non-quantized, real integer or floating point type.
    normalizer: If not None, a function that can be used to normalize the value
      of the real valued column after default_value is applied for parsing.
      Normalizer function takes the input tensor as its argument, and returns
      the output tensor. (e.g. lambda x: (x - 3.0) / 4.2). Note that for
      variable length columns, the normalizer should expect an input_tensor of
      type `SparseTensor`.
  Returns:
    A _RealValuedColumn.
  Raises:
    TypeError: if dimension is not an int
    ValueError: if dimension is not a positive integer
    TypeError: if default_value is a list but its length is not equal to the
      value of `dimension`.
    TypeError: if default_value is not compatible with dtype.
    ValueError: if dtype is not convertible to tf.float32.
  """

  if dimension is None:
    raise TypeError("dimension must be an integer. Use the "
                    "_real_valued_var_len_column for variable length features."
                    "dimension: {}, column_name: {}".format(dimension,
                                                            column_name))
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
    return _RealValuedColumn(column_name, dimension, default_value, dtype,
                             normalizer)

  if isinstance(default_value, int):
    if dtype.is_integer:
      default_value = ([default_value for _ in range(dimension)] if dimension
                       else [default_value])
      return _RealValuedColumn(column_name, dimension, default_value, dtype,
                               normalizer)
    if dtype.is_floating:
      default_value = float(default_value)
      default_value = ([default_value for _ in range(dimension)] if dimension
                       else [default_value])
      return _RealValuedColumn(column_name, dimension, default_value, dtype,
                               normalizer)

  if isinstance(default_value, float):
    if dtype.is_floating and (not dtype.is_integer):
      default_value = ([default_value for _ in range(dimension)] if dimension
                       else [default_value])
      return _RealValuedColumn(column_name, dimension, default_value, dtype,
                               normalizer)

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
        return _RealValuedColumn(column_name, dimension, default_value, dtype,
                                 normalizer)
      elif dtype.is_floating:
        default_value = [float(v) for v in default_value]
        return _RealValuedColumn(column_name, dimension, default_value, dtype,
                                 normalizer)
    if is_list_all_float:
      if dtype.is_floating and (not dtype.is_integer):
        default_value = [float(v) for v in default_value]
        return _RealValuedColumn(column_name, dimension, default_value, dtype,
                                 normalizer)

  raise TypeError("default_value must be compatible with dtype. "
                  "default_value: {}, dtype: {}, column_name: {}".format(
                      default_value, dtype, column_name))


class _BucketizedColumn(
    _FeatureColumn,
    fc_core._CategoricalColumn,  # pylint: disable=protected-access
    fc_core._DenseColumn,  # pylint: disable=protected-access
    collections.namedtuple("_BucketizedColumn", ["source_column",
                                                 "boundaries"])):
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
    boundaries: A list or tuple of floats specifying the boundaries. It has to
      be sorted. [a, b, c] defines following buckets: (-inf., a), [a, b),
      [b, c), [c, inf.)
  Raises:
    ValueError: if 'boundaries' is empty or not sorted.
  """

  def __new__(cls, source_column, boundaries):
    if not isinstance(source_column, _RealValuedColumn):
      raise TypeError("source_column must be an instance of _RealValuedColumn. "
                      "source_column: {}".format(source_column))

    if source_column.dimension is None:
      raise ValueError("source_column must have a defined dimension. "
                       "source_column: {}".format(source_column))

    if (not isinstance(boundaries, list) and
        not isinstance(boundaries, tuple)) or not boundaries:
      raise ValueError("boundaries must be a non-empty list or tuple. "
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
    return "{}_bucketized".format(self.source_column.name)

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

  # pylint: disable=unused-argument
  def _to_dnn_input_layer(self,
                          input_tensor,
                          weight_collections=None,
                          trainable=True,
                          output_rank=2):
    if output_rank != 2:
      raise ValueError("BucketizedColumn currently only supports output_rank=2")
    return array_ops.reshape(
        array_ops.one_hot(
            math_ops.cast(input_tensor, dtypes.int64),
            self.length,
            1.,
            0.,
            name="one_hot"), [-1, self.length * self.source_column.dimension],
        name="reshape")

  def to_sparse_tensor(self, input_tensor):
    """Creates a SparseTensor from the bucketized Tensor."""
    dimension = self.source_column.dimension
    batch_size = array_ops.shape(input_tensor, name="shape")[0]

    if dimension > 1:
      i1 = array_ops.reshape(
          array_ops.tile(
              array_ops.expand_dims(
                  math_ops.range(0, batch_size), 1, name="expand_dims"),
              [1, dimension],
              name="tile"), [-1],
          name="reshape")
      i2 = array_ops.tile(
          math_ops.range(0, dimension), [batch_size], name="tile")
      # Flatten the bucket indices and unique them across dimensions
      # E.g. 2nd dimension indices will range from k to 2*k-1 with k buckets
      bucket_indices = array_ops.reshape(
          input_tensor, [-1], name="reshape") + self.length * i2
    else:
      # Simpler indices when dimension=1
      i1 = math_ops.range(0, batch_size)
      i2 = array_ops.zeros([batch_size], dtype=dtypes.int32, name="zeros")
      bucket_indices = array_ops.reshape(input_tensor, [-1], name="reshape")

    indices = math_ops.cast(array_ops.transpose(array_ops.stack((i1, i2))),
                            dtypes.int64)
    shape = math_ops.cast(array_ops.stack([batch_size, dimension]),
                          dtypes.int64)
    sparse_id_values = sparse_tensor_py.SparseTensor(
        indices, bucket_indices, shape)

    return sparse_id_values

  def _wide_embedding_lookup_arguments(self, input_tensor):
    return _LinearEmbeddingLookupArguments(
        input_tensor=self.to_sparse_tensor(input_tensor),
        weight_tensor=None,
        vocab_size=self.length * self.source_column.dimension,
        initializer=init_ops.zeros_initializer(),
        combiner="sum")

  def _transform_feature(self, inputs):
    """Handles cross transformation."""
    # Bucketize the source column.
    return bucketization_op.bucketize(
        inputs.get(self.source_column),
        boundaries=list(self.boundaries),
        name="bucketize")

  def insert_transformed_feature(self, columns_to_tensors):
    """Handles sparse column to id conversion."""
    columns_to_tensors[self] = self._transform_feature(
        _LazyBuilderByColumnsToTensor(columns_to_tensors))

  @property
  def _parse_example_spec(self):
    return self.config

  @property
  def _num_buckets(self):
    return self.length * self.source_column.dimension

  def _get_sparse_tensors(self, inputs, weight_collections=None,
                          trainable=None):
    del weight_collections
    del trainable
    return fc_core._CategoricalColumn.IdWeightPair(  # pylint: disable=protected-access
        self.to_sparse_tensor(inputs.get(self)), None)

  @property
  def _variable_shape(self):
    return tensor_shape.TensorShape(
        [self.length * self.source_column.dimension])

  def _get_dense_tensor(self, inputs, weight_collections=None, trainable=None):
    return self._to_dnn_input_layer(
        inputs.get(self), weight_collections, trainable)


def bucketized_column(source_column, boundaries):
  """Creates a _BucketizedColumn for discretizing dense input.

  Args:
    source_column: A _RealValuedColumn defining dense column.
    boundaries: A list or tuple of floats specifying the boundaries. It has to
      be sorted.

  Returns:
    A _BucketizedColumn.

  Raises:
    ValueError: if 'boundaries' is empty or not sorted.
  """
  return _BucketizedColumn(source_column, boundaries)


class _CrossedColumn(
    _FeatureColumn,
    fc_core._CategoricalColumn,  # pylint: disable=protected-access
    collections.namedtuple("_CrossedColumn", [
        "columns", "hash_bucket_size", "hash_key", "combiner",
        "ckpt_to_load_from", "tensor_name_in_ckpt"
    ])):
  """Represents a cross transformation also known as conjunction or combination.

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
      [1, 0]: "e"

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
      in a single row. Currently "mean", "sqrtn" and "sum" are supported, with
      "sum" the default. "sqrtn" often achieves good accuracy, in particular
      with bag-of-words columns. Each of this can be thought as example level
      normalizations on the column::
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
      _CrossedColumn, or _BucketizedColumn.
    ValueError: if hash_bucket_size is not > 1 or len(columns) is not > 1. Also,
      if only one of `ckpt_to_load_from` and `tensor_name_in_ckpt` is specified.
  """

  @staticmethod
  def _assert_is_crossable(column):
    if isinstance(column, (_SparseColumn, _CrossedColumn, _BucketizedColumn)):
      return
    raise TypeError("columns must be a set of _SparseColumn, "
                    "_CrossedColumn, or _BucketizedColumn instances. "
                    "(column {} is a {})".format(column,
                                                 column.__class__.__name__))

  def __new__(cls,
              columns,
              hash_bucket_size,
              hash_key,
              combiner="sum",
              ckpt_to_load_from=None,
              tensor_name_in_ckpt=None):
    for column in columns:
      _CrossedColumn._assert_is_crossable(column)

    if len(columns) < 2:
      raise ValueError("columns must contain at least 2 elements. "
                       "columns: {}".format(columns))

    if hash_bucket_size < 2:
      raise ValueError("hash_bucket_size must be at least 2. "
                       "hash_bucket_size: {}".format(hash_bucket_size))

    if (ckpt_to_load_from is None) != (tensor_name_in_ckpt is None):
      raise ValueError("Must specify both `ckpt_to_load_from` and "
                       "`tensor_name_in_ckpt` or none of them.")

    sorted_columns = sorted(
        [column for column in columns], key=lambda column: column.name)
    return super(_CrossedColumn, cls).__new__(cls, tuple(sorted_columns),
                                              hash_bucket_size, hash_key,
                                              combiner,
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

  def weight_tensor(self, input_tensor):
    """Returns the weight tensor from the given transformed input_tensor."""
    del input_tensor
    return None

  def _to_dnn_input_layer(self,
                          input_tensor,
                          weight_collections=None,
                          trainable=True,
                          output_rank=2):
    del input_tensor
    del weight_collections
    del trainable
    del output_rank
    raise ValueError("CrossedColumn is not supported in DNN. "
                     "Please use embedding_column. column: {}".format(self))

  def _checkpoint_path(self):
    if self.ckpt_to_load_from is not None:
      return self.ckpt_to_load_from, self.tensor_name_in_ckpt
    return None

  def _wide_embedding_lookup_arguments(self, input_tensor):
    return _LinearEmbeddingLookupArguments(
        input_tensor=input_tensor,
        weight_tensor=None,
        vocab_size=self.length,
        initializer=init_ops.zeros_initializer(),
        combiner=self.combiner)

  def _transform_feature(self, inputs):
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
        feature_tensors.append(inputs.get(c.name))
      else:
        if isinstance(c, _BucketizedColumn):
          feature_tensors.append(c.to_sparse_tensor(inputs.get(c)))
        else:
          feature_tensors.append(inputs.get(c))
    return sparse_feature_cross_op.sparse_feature_cross(
        feature_tensors,
        hashed_output=True,
        num_buckets=self.hash_bucket_size,
        hash_key=self.hash_key,
        name="cross")

  def insert_transformed_feature(self, columns_to_tensors):
    """Handles sparse column to id conversion."""
    columns_to_tensors[self] = self._transform_feature(
        _LazyBuilderByColumnsToTensor(columns_to_tensors))

  @property
  def _parse_example_spec(self):
    return self.config

  @property
  def _num_buckets(self):
    return self.length

  def _get_sparse_tensors(self, inputs, weight_collections=None,
                          trainable=None):
    del weight_collections
    del trainable
    return fc_core._CategoricalColumn.IdWeightPair(inputs.get(self), None)  # pylint: disable=protected-access


class _LazyBuilderByColumnsToTensor(object):

  def __init__(self, columns_to_tensors):
    self._columns_to_tensors = columns_to_tensors

  def get(self, key):
    """Gets the transformed feature column."""
    if key in self._columns_to_tensors:
      return self._columns_to_tensors[key]
    if isinstance(key, str):
      raise ValueError(
          "features dictionary doesn't contain key ({})".format(key))
    if not isinstance(key, _FeatureColumn):
      raise TypeError('"key" must be either a "str" or "_FeatureColumn". '
                      "Provided: {}".format(key))

    key.insert_transformed_feature(self._columns_to_tensors)
    return self._columns_to_tensors[key]


def crossed_column(columns, hash_bucket_size, combiner="sum",
                   ckpt_to_load_from=None,
                   tensor_name_in_ckpt=None,
                   hash_key=None):
  """Creates a _CrossedColumn for performing feature crosses.

  Args:
    columns: An iterable of _FeatureColumn. Items can be an instance of
      _SparseColumn, _CrossedColumn, or _BucketizedColumn.
    hash_bucket_size: An int that is > 1. The number of buckets.
    combiner: A string specifying how to reduce if there are multiple entries
      in a single row. Currently "mean", "sqrtn" and "sum" are supported, with
      "sum" the default. "sqrtn" often achieves good accuracy, in particular
      with bag-of-words columns. Each of this can be thought as example level
      normalizations on the column::
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
    hash_key: Specify the hash_key that will be used by the `FingerprintCat64`
      function to combine the crosses fingerprints on SparseFeatureCrossOp
      (optional).

  Returns:
    A _CrossedColumn.

  Raises:
    TypeError: if any item in columns is not an instance of _SparseColumn,
      _CrossedColumn, or _BucketizedColumn, or
      hash_bucket_size is not an int.
    ValueError: if hash_bucket_size is not > 1 or
      len(columns) is not > 1.
  """
  return _CrossedColumn(
      columns,
      hash_bucket_size,
      hash_key,
      combiner=combiner,
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
  def _to_dnn_input_layer(self,
                          input_tensor,
                          weight_collections=None,
                          trainable=True,
                          output_rank=2):
    if input_tensor.dtype != dtypes.float32:
      input_tensor = math_ops.cast(input_tensor, dtypes.float32)
    return _reshape_real_valued_tensor(input_tensor, output_rank, self.name)

  def _to_dense_tensor(self, input_tensor):
    return self._to_dnn_input_layer(input_tensor)

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
                                 _RealValuedVarLenColumn,
                                 _BucketizedColumn, _CrossedColumn,
                                 _OneHotColumn, _ScatteredEmbeddingColumn)):
    return feature_column.config

  raise TypeError("Not supported _FeatureColumn type. "
                  "Given column is {}".format(feature_column))


def create_feature_spec_for_parsing(feature_columns):
  """Helper that prepares features config from input feature_columns.

  The returned feature config can be used as arg 'features' in tf.parse_example.

  Typical usage example:

  ```python
  # Define features and transformations
  feature_a = sparse_column_with_vocabulary_file(...)
  feature_b = real_valued_column(...)
  feature_c_bucketized = bucketized_column(real_valued_column("feature_c"), ...)
  feature_a_x_feature_c = crossed_column(
    columns=[feature_a, feature_c_bucketized], ...)

  feature_columns = set(
    [feature_b, feature_c_bucketized, feature_a_x_feature_c])
  batch_examples = tf.parse_example(
      serialized=serialized_examples,
      features=create_feature_spec_for_parsing(feature_columns))
  ```

  For the above example, create_feature_spec_for_parsing would return the dict:
  {
    "feature_a": parsing_ops.VarLenFeature(tf.string),
    "feature_b": parsing_ops.FixedLenFeature([1], dtype=tf.float32),
    "feature_c": parsing_ops.FixedLenFeature([1], dtype=tf.float32)
  }

  Args:
    feature_columns: An iterable containing all the feature columns. All items
      should be instances of classes derived from _FeatureColumn, unless
      feature_columns is a dict -- in which case, this should be true of all
      values in the dict.
  Returns:
    A dict mapping feature keys to FixedLenFeature or VarLenFeature values.
  """
  if isinstance(feature_columns, dict):
    feature_columns = feature_columns.values()

  features_config = {}
  for column in feature_columns:
    features_config.update(_get_feature_config(column))
  return features_config


def _create_sequence_feature_spec_for_parsing(sequence_feature_columns,
                                              allow_missing_by_default=False):
  """Prepares a feature spec for parsing `tf.SequenceExample`s.

  Args:
    sequence_feature_columns: an iterable containing all the feature columns.
      All items should be instances of classes derived from `_FeatureColumn`.
    allow_missing_by_default: whether to set `allow_missing=True` by default for
      `FixedLenSequenceFeature`s.
  Returns:
    A dict mapping feature keys to `FixedLenSequenceFeature` or `VarLenFeature`.
  """
  feature_spec = create_feature_spec_for_parsing(sequence_feature_columns)
  sequence_feature_spec = {}
  for key, feature in feature_spec.items():
    if isinstance(feature, parsing_ops.VarLenFeature):
      sequence_feature = feature
    elif (isinstance(feature, parsing_ops.FixedLenFeature) or
          isinstance(feature, parsing_ops.FixedLenSequenceFeature)):
      default_is_set = feature.default_value is not None
      if default_is_set:
        logging.warning(
            'Found default value {} for feature "{}". Ignoring this value and '
            'setting `allow_missing=True` instead.'.
            format(feature.default_value, key))
      sequence_feature = parsing_ops.FixedLenSequenceFeature(
          shape=feature.shape,
          dtype=feature.dtype,
          allow_missing=(allow_missing_by_default or default_is_set))
    else:
      raise TypeError(
          "Unsupported feature type: {}".format(type(feature).__name__))
    sequence_feature_spec[key] = sequence_feature
  return sequence_feature_spec


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
          column_type.dtype, name="Placeholder_{}".format(column_name))
    else:
      # Simple placeholder for dense tensors.
      placeholders[column_name] = array_ops.placeholder(
          column_type.dtype,
          shape=(None, column_type.shape[0]),
          name="Placeholder_{}".format(column_name))
  return placeholders


class _SparseIdLookupConfig(
    collections.namedtuple("_SparseIdLookupConfig",
                           ["vocabulary_file", "keys", "num_oov_buckets",
                            "vocab_size", "default_value"])):
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
