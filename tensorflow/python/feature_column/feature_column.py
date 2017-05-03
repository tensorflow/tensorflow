# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
features. FeatureColumns are also the primary way of encoding features for
canned ${tf.estimator.Estimator}s.

When using FeatureColumns with `Estimators`, the type of feature column you
should choose depends on (1) the feature type and (2) the model type.

(1) Feature type:

 * Continuous features can be represented by `numeric_column`.
 * Categorical features can be represented by any `categorical_column_with_*`
 column:
  - `categorical_column_with_keys`
  - `categorical_column_with_vocabulary_file`
  - `categorical_column_with_hash_bucket`
  - `categorical_column_with_integerized_feature`

(2) Model type:

 * Deep neural network models (`DNNClassifier`, `DNNRegressor`).

   Continuous features can be directly fed into deep neural network models.

     age_column = numeric_column("age")

   To feed sparse features into DNN models, wrap the column with
   `embedding_column` or `indicator_column`. `indicator_column` is recommended
   for features with only a few possible values. For features with many possible
   values, `embedding_column` is recommended.

     embedded_dept_column = embedding_column(
       categorical_column_with_keys("department", ["math", "philosphy", ...]),
       dimension=10)

* Wide (aka linear) models (`LinearClassifier`, `LinearRegressor`).

   Sparse features can be fed directly into linear models. They behave like an
   indicator column but with an efficient implementation.

     dept_column = categorical_column_with_keys("department",
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

Example of building canned `Estimator`s using FeatureColumns:

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
custom models using `input_from_feature_columns`.

Example of building model using FeatureColumns, this can be used in a
`model_fn` which is given to the {tf.estimator.Estimator}:

  # Building model via layers

  deep_feature_columns = [age_column, embedded_dept_column]
  columns_to_tensor = parse_feature_columns_from_examples(
      serialized=my_data,
      feature_columns=deep_feature_columns)
  first_layer = input_from_feature_columns(
      columns_to_tensors=columns_to_tensor,
      feature_columns=deep_feature_columns)
  second_layer = fully_connected(first_layer, ...)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest


def make_linear_model(features,
                      feature_columns,
                      units=1,
                      sparse_combiner='sum',
                      weight_collections=None,
                      trainable=True):
  """Returns a linear prediction `Tensor` based on given `feature_columns`.

  This function generates a weighted sum for each unitss`. Weighted sum
  refers to logits in classification problems. It refers to the prediction
  itself for linear regression problems.

  Main difference of `make_linear_model` and `make_input_layer` is handling of
  categorical columns. `make_linear_model` treats them as `indicator_column`s
  while `make_input_layer` explicitly requires wrapping each of them with an
  `embedding_column` or an `indicator_column`.

  Args:
    features: A mapping from key to tensors. 'string' key means a base feature.
      It can have `_FeatureColumn` as a key too. That means that FeatureColumn
      is already transformed by the input pipeline.
    feature_columns: An iterable containing all the FeatureColumns. All items
      should be instances of classes derived from FeatureColumn.
    units: units: An integer, dimensionality of the output space. Default
      value is 1.
    sparse_combiner: A string specifying how to reduce if a sparse column is
      multivalent. Currently "mean", "sqrtn" and "sum" are supported, with "sum"
      the default. "sqrtn" often achieves good accuracy, in particular with
      bag-of-words columns. It combines each sparse columns independently.
        * "sum": do not normalize features in the column
        * "mean": do l1 normalization on features in the column
        * "sqrtn": do l2 normalization on features in the column
    weight_collections: A list of collection names to which the Variable will be
      added. Note that, variables will also be added to collections
      `tf.GraphKeys.GLOBAL_VARIABLES` and `ops.GraphKeys.MODEL_VARIABLES`.
    trainable: If `True` also add the variable to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).

  Returns:
    A `Tensor` which represents predictions/logits of a linear model. Its shape
    is (batch_size, units) and its dtype is `float32`.

  Raises:
    ValueError: if an item in `feature_columns` is neither a `_DenseColumn`
      nor `_CategoricalColumn`.
  """
  _check_feature_columns(feature_columns)
  for column in feature_columns:
    if not isinstance(column, (_DenseColumn, _CategoricalColumn)):
      raise ValueError('Items of feature_columns must be either a _DenseColumn '
                       'or _CategoricalColumn. Given: {}'.format(column))
  weight_collections = list(weight_collections or [])
  weight_collections += [
      ops.GraphKeys.GLOBAL_VARIABLES, ops.GraphKeys.MODEL_VARIABLES
  ]
  with variable_scope.variable_scope(
      None, default_name='make_linear_model', values=features.values()):
    weigthed_sums = []
    builder = _LazyBuilder(features)
    for column in sorted(feature_columns, key=lambda x: x.name):
      with variable_scope.variable_scope(None, default_name=column.name):
        if isinstance(column, _CategoricalColumn):
          weigthed_sums.append(_create_categorical_column_weighted_sum(
              column, builder, units, sparse_combiner, weight_collections,
              trainable))
        else:
          weigthed_sums.append(_create_dense_column_weighted_sum(
              column, builder, units, weight_collections, trainable))
    predictions_no_bias = math_ops.add_n(
        weigthed_sums, name='weighted_sum_no_bias')
    bias = variable_scope.get_variable(
        'bias_weight',
        shape=[units],
        initializer=init_ops.zeros_initializer(),
        trainable=trainable,
        collections=weight_collections)
    predictions = nn_ops.bias_add(
        predictions_no_bias, bias, name='weighted_sum')

    return predictions


def numeric_column(key,
                   shape=(1,),
                   default_value=None,
                   dtype=dtypes.float32,
                   normalizer_fn=None):
  """Represents real valued or numerical features.

  An example:
  ```python
  price = numeric_column('price')
  all_feature_columns = [price, ...]
  dense_tensor = make_input_layer(features, all_feature_columns)

  # or
  bucketized_price = bucketized_column(price, boundaries=[...])
  all_feature_columns = [bucketized_price, ...]
  linear_prediction = make_linear_model(features, all_feature_columns)

  ```

  Args:
    key: A string providing key to look up corresponding `Tensor`.
    shape: An iterable of integers specifies the shape of the `Tensor`. An
      integer can be given which means a single dimension `Tensor` with given
      width. The `Tensor` representing the column will have the shape of
      [batch_size] + `shape`.
    default_value: A single value compatible with `dtype` or an iterable of
      values compatible with `dtype` which the column takes on during
      `tf.Example` parsing if data is missing. A default value of `None` will
      cause `tf.parse_example` to fail if an example does not contain this
      column. If a single value is provided, the same value will be applied as
      the default value for every item. If an iterable of values is provided,
      the shape of the `default_value` should be equal to the given `shape`.
    dtype: defines the type of values. Default value is `tf.float32`. Must be a
      non-quantized, real integer or floating point type.
    normalizer_fn: If not `None`, a function that can be used to normalize the
      value of the tensor after `default_value` is applied for parsing.
      Normalizer function takes the input `Tensor` as its argument, and returns
      the output `Tensor`. (e.g. lambda x: (x - 3.0) / 4.2). Please note that
      even though most common use case of this function is normalization, it can
      be used for any kind of Tensorflow transformations.

  Returns:
    A _NumericColumn.

  Raises:
    TypeError: if any dimension in shape is not an int
    ValueError: if any dimension in shape is not a positive integer
    TypeError: if `default_value` is an iterable but not compatible with `shape`
    TypeError: if `default_value` is not compatible with `dtype`.
    ValueError: if `dtype` is not convertible to `tf.float32`.
  """
  shape = _check_shape(shape, key)
  if not (dtype.is_integer or dtype.is_floating):
    raise ValueError('dtype must be convertible to float. '
                     'dtype: {}, key: {}'.format(dtype, key))
  default_value = _check_default_value(shape, default_value, dtype, key)

  if normalizer_fn is not None and not callable(normalizer_fn):
    raise TypeError(
        'normalizer_fn must be a callable. Given: {}'.format(normalizer_fn))

  return _NumericColumn(
      key,
      shape=shape,
      default_value=default_value,
      dtype=dtype,
      normalizer_fn=normalizer_fn)


def bucketized_column(source_column, boundaries):
  """Represents discretized dense input.

  Buckets include the left boundary, and exclude the right boundary. Namely,
  `boundaries=[0., 1., 2.]` generates buckets `(-inf, 0.)`, `[0., 1.)`,
  `[1., 2.)`, and `[2., +inf)`.

  An example:
  ```python
  price = numeric_column('price')
  bucketized_price = bucketized_column(price, boundaries=[...])
  all_feature_columns = [bucketized_price, ...]
  linear_prediction = make_linear_model(features, all_feature_columns)

  # or
  all_feature_columns = [bucketized_price, ...]
  dense_tensor = make_input_layer(features, all_feature_columns)
  ```

  Args:
    source_column: A one-dimensional dense column which is generated with
      `numeric_column`.
    boundaries: A sorted list or tuple of floats specifying the boundaries.

  Returns:
    A `_BucketizedColumn`.

  Raises:
    ValueError: If `source_column` is not a numeric column, or if it is not
      one-dimensional.
    ValueError: If `boundaries` is not a sorted list or tuple.
  """
  if not isinstance(source_column, _NumericColumn):
    raise ValueError(
        'source_column must be a column generated with numeric_column(). '
        'Given: {}'.format(source_column))
  if len(source_column.shape) > 1:
    raise ValueError(
        'source_column must be one-dimensional column. '
        'Given: {}'.format(source_column))
  if (not boundaries or
      not (isinstance(boundaries, list) or isinstance(boundaries, tuple))):
    raise ValueError('boundaries must be a sorted list.')
  for i in range(len(boundaries) - 1):
    if boundaries[i] >= boundaries[i + 1]:
      raise ValueError('boundaries must be a sorted list.')
  return _BucketizedColumn(source_column, tuple(boundaries))


def categorical_column_with_hash_bucket(key,
                                        hash_bucket_size,
                                        dtype=dtypes.string):
  """Represents sparse feature where ids are set by hashing.

  Use this when your sparse features are in string or integer format where you
  want to distribute your inputs into a finite number of buckets by hashing.
  output_id = Hash(input_feature_string) % bucket_size

  An example:
  ```python
  keywords = categorical_column_with_hash_bucket("keywords", 10K)
  all_feature_columns = [keywords, ...]
  linear_prediction = make_linear_model(features, all_feature_columns)

  # or
  keywords_embedded = embedding_column(keywords, 16)
  all_feature_columns = [keywords_embedded, ...]
  dense_tensor = make_input_layer(features, all_feature_columns)
  ```

  Args:
    key: A string providing key to look up corresponding `Tensor`.
    hash_bucket_size: An int > 1. The number of buckets.
    dtype: The type of features. Only string and integer types are supported.

  Returns:
    A `_CategoricalColumnHashed`.

  Raises:
    ValueError: `hash_bucket_size` is not greater than 1.
    ValueError: `dtype` is neither string nor integer.
  """
  if hash_bucket_size is None:
    raise ValueError('hash_bucket_size must be set. ' 'key: {}'.format(key))

  if hash_bucket_size < 1:
    raise ValueError('hash_bucket_size must be at least 1. '
                     'hash_bucket_size: {}, key: {}'.format(
                         hash_bucket_size, key))

  if dtype != dtypes.string and not dtype.is_integer:
    raise ValueError('dtype must be string or integer. '
                     'dtype: {}, column_name: {}'.format(dtype, key))

  return _CategoricalColumnHashed(key, hash_bucket_size, dtype)


class _FeatureColumn(object):
  """Represents a feature column abstraction.

  WARNING: Do not subclass this layer unless you know what you are doing:
  the API is subject to future changes.

  To distinguish the concept of a feature family and a specific binary feature
  within a family, we refer to a feature family like "country" as a feature
  column. Following is an example feature in a `tf.Example` format:
    {key: "country",  value: [ "US" ]}
  In this example the value of feature is "US" and "country" refers to the
  column of the feature.

  This class is an abstract class. User should not create instances of this.
  """
  __metaclass__ = abc.ABCMeta

  @abc.abstractproperty
  def name(self):
    """Returns string. used for variable_scope and naming."""
    pass

  @abc.abstractmethod
  def _transform_feature(self, inputs):
    """Returns transformed `Tensor`, uses `inputs` to access input tensors.

    It uses `inputs` to get either raw feature or transformation of other
    FeatureColumns.

    Example input access:
    Let's say a Feature column depends on raw feature ('raw') and another
    `_FeatureColumn` (input_fc). To access corresponding Tensors, inputs will
    be used as follows:

    ```python
    raw_tensor = inputs.get('raw')
    fc_tensor = inputs.get(input_fc)
    ```

    Args:
      inputs: A `_LazyBuilder` object to access inputs.

    Returns:
      Transformed feature `Tensor`.
    """
    pass

  @abc.abstractproperty
  def _parse_example_config(self):
    """Returns a `tf.Example` parsing spec as dict.

    It is used for get_parsing_spec for `tf.parse_example`. Returned spec is a
    dict from keys ('string') to `VarLenFeature`, `FixedLenFeature`, and other
    supported objects. Please check documentation of ${tf.parse_example} for all
    supported spec objects.

    Let's say a Feature column depends on raw feature ('raw') and another
    `_FeatureColumn` (input_fc). One possible implementation of
    _parse_example_config is as follows:

    ```python
    spec = {'raw': tf.FixedLenFeature(...)}
    spec.update(input_fc._parse_example_config)
    return spec
    ```
    """
    pass


class _DenseColumn(_FeatureColumn):
  """Represents a column which can be represented as `Tensor`.

  WARNING: Do not subclass this layer unless you know what you are doing:
  the API is subject to future changes.

  Some examples of this type are: numeric_column, embedding_column,
  indicator_column.
  """

  __metaclass__ = abc.ABCMeta

  @abc.abstractproperty
  def _variable_shape(self):
    """Returns shape of variable which is compatible with _get_dense_tensor."""
    pass

  @abc.abstractmethod
  def _get_dense_tensor(self, inputs, weight_collections=None, trainable=None):
    """Returns a `Tensor`.

    The output of this function will be used by model-buildier-functions. For
    example the pseudo code of `make_input_layer` will be like that:
    ```python
    def make_input_layer(features, feature_columns, ...):
      outputs = [fc._get_dense_tensor(...) for fc in feature_columns]
      return tf.concat(outputs)
    ```

    Args:
      inputs: A `_LazyBuilder` object to access inputs.
      weight_collections: List of graph collections to which Variables (if any
        will be created) are added.
      trainable: If `True` also add variables to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see ${tf.Variable}).
    """
    pass


def _create_dense_column_weighted_sum(
    column, builder, units, weight_collections, trainable):
  """Create a weighted sum of a dense column for make_linear_model."""
  tensor = column._get_dense_tensor(  # pylint: disable=protected-access
      builder,
      weight_collections=weight_collections,
      trainable=trainable)
  num_elements = tensor_shape.TensorShape(column._variable_shape).num_elements()  # pylint: disable=protected-access
  batch_size = array_ops.shape(tensor)[0]
  tensor = array_ops.reshape(tensor, shape=(batch_size, num_elements))
  weight = variable_scope.get_variable(
      name='weight',
      shape=[num_elements, units],
      initializer=init_ops.zeros_initializer(),
      trainable=trainable,
      collections=weight_collections)
  return math_ops.matmul(tensor, weight, name='weighted_sum')


class _CategoricalColumn(_FeatureColumn):
  """Represents a categorical feautre.

  WARNING: Do not subclass this layer unless you know what you are doing:
  the API is subject to future changes.

  A categorical feature typically handled with a ${tf.SparseTensor} of IDs.
  """
  __metaclass__ = abc.ABCMeta

  IdWeightPair = collections.namedtuple(  # pylint: disable=invalid-name
      'IdWeightPair', ['id_tensor', 'weight_tensor'])

  @abc.abstractproperty
  def _num_buckets(self):
    """Returns number of buckets in this sparse feature."""
    pass

  @abc.abstractmethod
  def _get_sparse_tensors(self,
                          inputs,
                          weight_collections=None,
                          trainable=None):
    """Returns an IdWeightPair.

    `IdWeightPair` is a pair of `SparseTensor`s which represents ids and
    weights.

    `IdWeightPair.id_tensor` is typically a `batch_size` x `num_buckets`
    `SparseTensor` of `int64`. `IdWeightPair.weight_tensor` is either a
    `SparseTensor` of `float` or `None` to indicate all weights should be
    taken to be 1. If specified, `weight_tensor` must have exactly the same
    shape and indices as `sp_ids`. Expected `SparseTensor` is same as parsing
    output of a `VarLenFeature` which is a ragged matrix.

    Args:
      inputs: A `LazyBuilder` as a cache to get input tensors required to
        create `IdWeightPair`.
      weight_collections: List of graph collections to which variables (if any
        will be created) are added.
      trainable: If `True` also add variables to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see ${tf.get_variable}).
    """
    pass


def _create_categorical_column_weighted_sum(
    column, builder, units, sparse_combiner, weight_collections, trainable):
  """Create a weighted sum of a categorical column for make_linear_model."""
  sparse_tensors = column._get_sparse_tensors(  # pylint: disable=protected-access
      builder,
      weight_collections=weight_collections,
      trainable=trainable)
  weight = variable_scope.get_variable(
      name='weight',
      shape=[column._num_buckets, units],  # pylint: disable=protected-access
      initializer=init_ops.zeros_initializer(),
      trainable=trainable,
      collections=weight_collections)
  return _safe_embedding_lookup_sparse(
      weight,
      sparse_tensors.id_tensor,
      sparse_weights=sparse_tensors.weight_tensor,
      combiner=sparse_combiner,
      name='weighted_sum')


class _LazyBuilder(object):
  """Handles caching of transformations while building the model.

  `FeatureColumn` specifies how to digest an input column to the network. Some
  feature columns require data transformations. This class caches those
  transformations.

  Some features may be used in more than one place. For example, one can use a
  bucketized feature by itself and a cross with it. In that case we
  should create only one bucketization op instead of creating ops for each
  feature column separately. To handle re-use of transformed columns,
  `_LazyBuilder` caches all previously transformed columns.

  Example:
  We're trying to use the following `FeatureColumns`:

  ```python
    bucketized_age = fc.bucketized_column(fc.numeric_column("age"), ...)
    keywords = fc.categorical_column_with_hash_buckets("keywords", ...)
    age_X_keywords = fc.crossed_column([bucketized_age, keywords])
    ... = make_linear_model(features,
                            [bucketized_age, keywords, age_X_keywords]
  ```

  If we transform each column independently, then we'll get duplication of
  bucketization (one for cross, one for bucketization itself).
  The `_LazyBuilder` eliminates this duplication.
  """

  def __init__(self, features):
    """Creates a `_LazyBuilder`.

    Args:
      features: A mapping from feature column to tensors. A `string` key
        signifies a base feature (not-transformed). A `FeatureColumn` key
        means that this `Tensor` is the output of an existing `FeatureColumn`
        which can be reused.
    """
    self._columns_to_tensors = features.copy()

  def get(self, key):
    """Returns a `Tensor` for the given key.

    A `str` key is used to access a base feature (not-transformed). When a
    `_FeatureColumn` is passed, the transformed feature is returned if it
    already exists, otherwise the given `_FeatureColumn` is asked to provide its
    transformed output, which is then cached.

    Args:
      key: a `str` or a `_FeatureColumn`.

    Returns:
      The transformed `Tensor` corresponding to the `key`.

    Raises:
      ValueError: if key is not found or a transformed `Tensor` cannot be
        computed.
    """
    if key in self._columns_to_tensors:
      # Feature_column is already transformed or it's a raw feature.
      return self._columns_to_tensors[key]

    if not isinstance(key, (str, _FeatureColumn)):
      raise TypeError('"key" must be either a "str" or "_FeatureColumn". '
                      'Provided: {}'.format(key))

    if not isinstance(key, _FeatureColumn):
      raise ValueError('Feature {} is not in features dictionary.'.format(key))

    column = key
    logging.debug('Transforming feature_column %s.', column)
    transformed = column._transform_feature(self)  # pylint: disable=protected-access
    if transformed is None:
      raise ValueError('Column {} is not supported.'.format(column.name))
    self._columns_to_tensors[column] = transformed
    return self._columns_to_tensors[column]


def _check_feature_columns(feature_columns):
  if isinstance(feature_columns, dict):
    raise ValueError('Expected feature_columns to be iterable, found dict.')
  for column in feature_columns:
    if not isinstance(column, _FeatureColumn):
      raise ValueError('Items of feature_columns must be a _FeatureColumn.')
  name_to_column = dict()
  for column in feature_columns:
    if column.name in name_to_column:
      raise ValueError('Duplicate feature column name found for columns: {} '
                       'and {}. This usually means that these columns refer to '
                       'same base feature. Either one must be discarded or a '
                       'duplicated but renamed item must be inserted in '
                       'features dict.'.format(column,
                                               name_to_column[column.name]))
    name_to_column[column.name] = column


class _NumericColumn(_DenseColumn,
                     collections.namedtuple('_NumericColumn', [
                         'key', 'shape', 'default_value', 'dtype',
                         'normalizer_fn'
                     ])):
  """see `numeric_column`."""

  @property
  def name(self):
    return self.key

  @property
  def _parse_example_config(self):
    return {
        self.key:
            parsing_ops.FixedLenFeature(self.shape, self.dtype,
                                        self.default_value)
    }

  def _transform_feature(self, inputs):
    input_tensor = inputs.get(self.key)
    if isinstance(input_tensor, sparse_tensor_lib.SparseTensor):
      raise ValueError(
          'The corresponding Tensor of numerical column must be a Tensor. '
          'SparseTensor is not supported. key: {}'.format(self.key))
    if self.normalizer_fn is not None:
      input_tensor = self.normalizer_fn(input_tensor)
    return math_ops.to_float(input_tensor)

  @property
  def _variable_shape(self):
    return self.shape

  def _get_dense_tensor(self, inputs, weight_collections=None, trainable=None):
    del weight_collections
    del trainable
    return inputs.get(self)


class _BucketizedColumn(_DenseColumn, _CategoricalColumn,
                        collections.namedtuple('_BucketizedColumn', [
                            'source_column', 'boundaries'])):
  """See `bucketized_column`."""

  @property
  def name(self):
    return '{}_bucketized'.format(self.source_column.name)

  @property
  def _parse_example_config(self):
    return self.source_column._parse_example_config  # pylint: disable=protected-access

  def _transform_feature(self, inputs):
    source_tensor = inputs.get(self.source_column)
    return math_ops._bucketize(  # pylint: disable=protected-access
        source_tensor,
        boundaries=self.boundaries)

  @property
  def _variable_shape(self):
    return tuple(self.source_column.shape) + (len(self.boundaries) + 1,)

  def _get_dense_tensor(self, inputs, weight_collections=None, trainable=None):
    del weight_collections
    del trainable
    input_tensor = inputs.get(self)
    return array_ops.one_hot(
        indices=math_ops.to_int64(input_tensor),
        depth=len(self.boundaries) + 1,
        on_value=1.,
        off_value=0.)

  @property
  def _num_buckets(self):
    # By construction, source_column is always one-dimensional.
    return (len(self.boundaries) + 1) * self.source_column.shape[0]

  def _get_sparse_tensors(self, inputs, weight_collections=None,
                          trainable=None):
    input_tensor = inputs.get(self)
    batch_size = array_ops.shape(input_tensor)[0]
    # By construction, source_column is always one-dimensional.
    source_dimension = self.source_column.shape[0]

    i1 = array_ops.reshape(
        array_ops.tile(
            array_ops.expand_dims(math_ops.range(0, batch_size), 1),
            [1, source_dimension]),
        (-1,))
    i2 = array_ops.tile(math_ops.range(0, source_dimension), [batch_size])
    # Flatten the bucket indices and unique them across dimensions
    # E.g. 2nd dimension indices will range from k to 2*k-1 with k buckets
    bucket_indices = (
        array_ops.reshape(input_tensor, (-1,)) +
        (len(self.boundaries) + 1) * i2)

    indices = math_ops.to_int64(array_ops.transpose(array_ops.stack((i1, i2))))
    dense_shape = math_ops.to_int64(array_ops.stack(
        [batch_size, source_dimension]))
    sparse_tensor = sparse_tensor_lib.SparseTensor(
        indices=indices,
        values=bucket_indices,
        dense_shape=dense_shape)
    return _CategoricalColumn.IdWeightPair(sparse_tensor, None)


def _create_tuple(shape, value):
  """Returns a tuple with given shape and filled with value."""
  if shape:
    return tuple([_create_tuple(shape[1:], value) for _ in range(shape[0])])
  return value


def _as_tuple(value):
  if not nest.is_sequence(value):
    return value
  return tuple([_as_tuple(v) for v in value])


def _check_shape(shape, key):
  """Returns shape if it's valid, raises error otherwise."""
  assert shape is not None
  if not nest.is_sequence(shape):
    shape = [shape]
  shape = tuple(shape)
  for dimension in shape:
    if not isinstance(dimension, int):
      raise TypeError('shape dimensions must be integer. '
                      'shape: {}, key: {}'.format(shape, key))
    if dimension < 1:
      raise ValueError('shape dimensions must be greater than 0. '
                       'shape: {}, key: {}'.format(shape, key))
  return shape


def _is_shape_and_default_value_compatible(default_value, shape):
  """Verifies compatibility of shape and default_value."""
  # Invalid condition:
  #  * if default_value is not a scalar and shape is empty
  #  * or if default_value is an iterable and shape is not empty
  if nest.is_sequence(default_value) != bool(shape):
    return False
  if not shape:
    return True
  if len(default_value) != shape[0]:
    return False
  for i in range(shape[0]):
    if not _is_shape_and_default_value_compatible(default_value[i], shape[1:]):
      return False
  return True


def _check_default_value(shape, default_value, dtype, key):
  """Returns default value as tuple if it's valid, otherwise raises errors.

  This function verifies that `default_value` is compatible with both `shape`
  and `dtype`. If it is not compatible, it raises an error. If it is compatible,
  it casts default_value to a tuple and returns it. `key` is used only
  for error message.

  Args:
    shape: An iterable of integers specifies the shape of the `Tensor`.
    default_value: If a single value is provided, the same value will be applied
      as the default value for every item. If an iterable of values is
      provided, the shape of the `default_value` should be equal to the given
      `shape`.
    dtype: defines the type of values. Default value is `tf.float32`. Must be a
      non-quantized, real integer or floating point type.
    key: A string providing key to look up corresponding `Tensor`.

  Returns:
    A tuple which will be used as default value.

  Raises:
    TypeError: if `default_value` is an iterable but not compatible with `shape`
    TypeError: if `default_value` is not compatible with `dtype`.
    ValueError: if `dtype` is not convertible to `tf.float32`.
  """
  if default_value is None:
    return None

  if isinstance(default_value, int):
    return _create_tuple(shape, default_value)

  if isinstance(default_value, float) and dtype.is_floating:
    return _create_tuple(shape, default_value)

  if callable(getattr(default_value, 'tolist', None)):  # Handles numpy arrays
    default_value = default_value.tolist()

  if nest.is_sequence(default_value):
    if not _is_shape_and_default_value_compatible(default_value, shape):
      raise ValueError(
          'The shape of default_value must be equal to given shape. '
          'default_value: {}, shape: {}, key: {}'.format(
              default_value, shape, key))
    # Check if the values in the list are all integers or are convertible to
    # floats.
    is_list_all_int = all(
        isinstance(v, int) for v in nest.flatten(default_value))
    is_list_has_float = any(
        isinstance(v, float) for v in nest.flatten(default_value))
    if is_list_all_int:
      return _as_tuple(default_value)
    if is_list_has_float and dtype.is_floating:
      return _as_tuple(default_value)
  raise TypeError('default_value must be compatible with dtype. '
                  'default_value: {}, dtype: {}, key: {}'.format(
                      default_value, dtype, key))


class _CategoricalColumnHashed(
    _CategoricalColumn,
    collections.namedtuple('_CategoricalColumnHashed',
                           ['key', 'hash_bucket_size', 'dtype'])):
  """see `categorical_column_with_hash_bucket`."""

  @property
  def name(self):
    return self.key

  @property
  def _parse_example_config(self):
    return {self.key: parsing_ops.VarLenFeature(self.dtype)}

  def _transform_feature(self, inputs):
    input_tensor = inputs.get(self.key)
    if not isinstance(input_tensor, sparse_tensor_lib.SparseTensor):
      raise ValueError('SparseColumn input must be a SparseTensor.')

    if (input_tensor.dtype != dtypes.string and
        not input_tensor.dtype.is_integer):
      raise ValueError('input tensors dtype must be string or integer. '
                       'dtype: {}, column_name: {}'.format(
                           input_tensor.dtype, self.key))

    if self.dtype.is_integer != input_tensor.dtype.is_integer:
      raise ValueError(
          'Column dtype and SparseTensors dtype must be compatible. '
          'key: {}, column dtype: {}, tensor dtype: {}'.format(
              self.key, self.dtype, input_tensor.dtype))

    if self.dtype == dtypes.string:
      sparse_values = input_tensor.values
    else:
      sparse_values = string_ops.as_string(input_tensor.values)

    sparse_id_values = string_ops.string_to_hash_bucket_fast(
        sparse_values, self.hash_bucket_size, name='lookup')
    return sparse_tensor_lib.SparseTensor(
        input_tensor.indices, sparse_id_values, input_tensor.dense_shape)

  @property
  def _num_buckets(self):
    """Returns number of buckets in this sparse feature."""
    return self.hash_bucket_size

  def _get_sparse_tensors(self, inputs, weight_collections=None,
                          trainable=None):
    return _CategoricalColumn.IdWeightPair(inputs.get(self), None)


# TODO(zakaria): Move this to embedding_ops and make it public.
def _safe_embedding_lookup_sparse(embedding_weights,
                                  sparse_ids,
                                  sparse_weights=None,
                                  combiner=None,
                                  default_id=None,
                                  name=None,
                                  partition_strategy='div',
                                  max_norm=None):
  """Lookup embedding results, accounting for invalid IDs and empty features.

  The partitioned embedding in `embedding_weights` must all be the same shape
  except for the first dimension. The first dimension is allowed to vary as the
  vocabulary size is not necessarily a multiple of `P`.  `embedding_weights`
  may be a `PartitionedVariable` as returned by using `tf.get_variable()` with a
  partitioner.

  Invalid IDs (< 0) are pruned from input IDs and weights, as well as any IDs
  with non-positive weight. For an entry with no features, the embedding vector
  for `default_id` is returned, or the 0-vector if `default_id` is not supplied.

  The ids and weights may be multi-dimensional. Embeddings are always aggregated
  along the last dimension.

  Args:
    embedding_weights:  A list of `P` float tensors or values representing
        partitioned embedding tensors.  Alternatively, a `PartitionedVariable`,
        created by partitioning along dimension 0.  The total unpartitioned
        shape should be `[e_0, e_1, ..., e_m]`, where `e_0` represents the
        vocab size and `e_1, ..., e_m` are the embedding dimensions.
    sparse_ids: `SparseTensor` of shape `[d_0, d_1, ..., d_n]` containing the
        ids. `d_0` is typically batch size.
    sparse_weights: `SparseTensor` of same shape as `sparse_ids`, containing
        float weights corresponding to `sparse_ids`, or `None` if all weights
        are be assumed to be 1.0.
    combiner: A string specifying how to combine embedding results for each
        entry. Currently "mean", "sqrtn" and "sum" are supported, with "mean"
        the default.
    default_id: The id to use for an entry with no features.
    name: A name for this operation (optional).
    partition_strategy: A string specifying the partitioning strategy.
        Currently `"div"` and `"mod"` are supported. Default is `"div"`.
    max_norm: If not None, all embeddings are l2-normalized to max_norm before
        combining.


  Returns:
    Dense tensor of shape `[d_0, d_1, ..., d_{n-1}, e_1, ..., e_m]`.

  Raises:
    ValueError: if `embedding_weights` is empty.
  """
  if combiner is None:
    logging.warn('The default value of combiner will change from \"mean\" '
                 'to \"sqrtn\" after 2016/11/01.')
    combiner = 'mean'
  if embedding_weights is None:
    raise ValueError('Missing embedding_weights %s.' % embedding_weights)
  if isinstance(embedding_weights, variables.PartitionedVariable):
    embedding_weights = list(embedding_weights)  # get underlying Variables.
  if not isinstance(embedding_weights, list):
    embedding_weights = [embedding_weights]
  if len(embedding_weights) < 1:
    raise ValueError('Missing embedding_weights %s.' % embedding_weights)

  dtype = sparse_weights.dtype if sparse_weights is not None else None
  if isinstance(embedding_weights, variables.PartitionedVariable):
    embedding_weights = list(embedding_weights)
  embedding_weights = [
      ops.convert_to_tensor(w, dtype=dtype) for w in embedding_weights
  ]

  with ops.name_scope(name, 'embedding_lookup',
                      embedding_weights + [sparse_ids,
                                           sparse_weights]) as scope:
    # Reshape higher-rank sparse ids and weights to linear segment ids.
    original_shape = sparse_ids.dense_shape
    original_rank_dim = sparse_ids.dense_shape.get_shape()[0]
    original_rank = (
        array_ops.size(original_shape)
        if original_rank_dim.value is None
        else original_rank_dim.value)
    sparse_ids = sparse_ops.sparse_reshape(sparse_ids, [
        math_ops.reduce_prod(
            array_ops.slice(original_shape, [0], [original_rank - 1])),
        array_ops.gather(original_shape, original_rank - 1)])
    if sparse_weights is not None:
      sparse_weights = sparse_tensor_lib.SparseTensor(
          sparse_ids.indices,
          sparse_weights.values, sparse_ids.dense_shape)

    # Prune invalid ids and weights.
    sparse_ids, sparse_weights = _prune_invalid_ids(sparse_ids, sparse_weights)

    # Fill in dummy values for empty features, if necessary.
    sparse_ids, is_row_empty = sparse_ops.sparse_fill_empty_rows(sparse_ids,
                                                                 default_id or
                                                                 0)
    if sparse_weights is not None:
      sparse_weights, _ = sparse_ops.sparse_fill_empty_rows(sparse_weights, 1.0)

    result = embedding_ops.embedding_lookup_sparse(
        embedding_weights,
        sparse_ids,
        sparse_weights,
        combiner=combiner,
        partition_strategy=partition_strategy,
        name=None if default_id is None else scope,
        max_norm=max_norm)

    if default_id is None:
      # Broadcast is_row_empty to the same shape as embedding_lookup_result,
      # for use in Select.
      is_row_empty = array_ops.tile(
          array_ops.reshape(is_row_empty, [-1, 1]),
          array_ops.stack([1, array_ops.shape(result)[1]]))

      result = array_ops.where(is_row_empty,
                               array_ops.zeros_like(result),
                               result,
                               name=scope)

    # Reshape back from linear ids back into higher-dimensional dense result.
    final_result = array_ops.reshape(
        result,
        array_ops.concat([
            array_ops.slice(
                math_ops.cast(original_shape, dtypes.int32), [0],
                [original_rank - 1]),
            array_ops.slice(array_ops.shape(result), [1], [-1])
        ], 0))
    final_result.set_shape(tensor_shape.unknown_shape(
        (original_rank_dim - 1).value).concatenate(result.get_shape()[1:]))
    return final_result


def _prune_invalid_ids(sparse_ids, sparse_weights):
  """Prune invalid IDs (< 0) from the input ids and weights."""
  is_id_valid = math_ops.greater_equal(sparse_ids.values, 0)
  if sparse_weights is not None:
    is_id_valid = math_ops.logical_and(
        is_id_valid, math_ops.greater(sparse_weights.values, 0))
  sparse_ids = sparse_ops.sparse_retain(sparse_ids, is_id_valid)
  if sparse_weights is not None:
    sparse_weights = sparse_ops.sparse_retain(sparse_weights, is_id_valid)
  return sparse_ids, sparse_weights
