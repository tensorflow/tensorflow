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
"""Tests for linear_optimizer.sdca_estimator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.layers.python.layers import feature_column as feature_column_lib
from tensorflow.contrib.linear_optimizer.python.sdca_estimator import SDCALogisticClassifier
from tensorflow.contrib.linear_optimizer.python.sdca_estimator import SDCARegressor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.platform import test


class SDCALogisticClassifierTest(test.TestCase):

  def testRealValuedFeatures(self):
    """Tests SDCALogisticClassifier works with real valued features."""

    def input_fn():
      return {
          'example_id': constant_op.constant(['1', '2']),
          'maintenance_cost': constant_op.constant([[500.0], [200.0]]),
          'sq_footage': constant_op.constant([[800.0], [600.0]]),
          'weights': constant_op.constant([[1.0], [1.0]])
      }, constant_op.constant([[0], [1]])

    maintenance_cost = feature_column_lib.real_valued_column('maintenance_cost')
    sq_footage = feature_column_lib.real_valued_column('sq_footage')
    classifier = SDCALogisticClassifier(
        example_id_column='example_id',
        feature_columns=[maintenance_cost, sq_footage],
        weight_column_name='weights')
    classifier.fit(input_fn=input_fn, steps=100)
    loss = classifier.evaluate(input_fn=input_fn, steps=1)['loss']
    self.assertLess(loss, 0.05)

  def testRealValuedFeatureWithHigherDimension(self):
    """Tests SDCALogisticClassifier with high-dimension real valued features."""

    # input_fn is identical to the one in testRealValuedFeatures where 2
    # 1-dimensional dense features are replaced by a 2-dimensional feature.
    def input_fn():
      return {
          'example_id':
              constant_op.constant(['1', '2']),
          'dense_feature':
              constant_op.constant([[500.0, 800.0], [200.0, 600.0]])
      }, constant_op.constant([[0], [1]])

    dense_feature = feature_column_lib.real_valued_column(
        'dense_feature', dimension=2)
    classifier = SDCALogisticClassifier(
        example_id_column='example_id', feature_columns=[dense_feature])
    classifier.fit(input_fn=input_fn, steps=100)
    loss = classifier.evaluate(input_fn=input_fn, steps=1)['loss']
    self.assertLess(loss, 0.05)

  def testBucketizedFeatures(self):
    """Tests SDCALogisticClassifier with bucketized features."""

    def input_fn():
      return {
          'example_id': constant_op.constant(['1', '2', '3']),
          'price': constant_op.constant([[600.0], [1000.0], [400.0]]),
          'sq_footage': constant_op.constant([[1000.0], [600.0], [700.0]]),
          'weights': constant_op.constant([[1.0], [1.0], [1.0]])
      }, constant_op.constant([[1], [0], [1]])

    price_bucket = feature_column_lib.bucketized_column(
        feature_column_lib.real_valued_column('price'),
        boundaries=[500.0, 700.0])
    sq_footage_bucket = feature_column_lib.bucketized_column(
        feature_column_lib.real_valued_column('sq_footage'), boundaries=[650.0])
    classifier = SDCALogisticClassifier(
        example_id_column='example_id',
        feature_columns=[price_bucket, sq_footage_bucket],
        weight_column_name='weights',
        l2_regularization=1.0)
    classifier.fit(input_fn=input_fn, steps=50)
    metrics = classifier.evaluate(input_fn=input_fn, steps=1)
    self.assertGreater(metrics['accuracy'], 0.9)

  def testSparseFeatures(self):
    """Tests SDCALogisticClassifier with sparse features."""

    def input_fn():
      return {
          'example_id':
              constant_op.constant(['1', '2', '3']),
          'price':
              constant_op.constant([[0.4], [0.6], [0.3]]),
          'country':
              sparse_tensor.SparseTensor(
                  values=['IT', 'US', 'GB'],
                  indices=[[0, 0], [1, 3], [2, 1]],
                  dense_shape=[3, 5]),
          'weights':
              constant_op.constant([[1.0], [1.0], [1.0]])
      }, constant_op.constant([[1], [0], [1]])

    price = feature_column_lib.real_valued_column('price')
    country = feature_column_lib.sparse_column_with_hash_bucket(
        'country', hash_bucket_size=5)
    classifier = SDCALogisticClassifier(
        example_id_column='example_id',
        feature_columns=[price, country],
        weight_column_name='weights')
    classifier.fit(input_fn=input_fn, steps=50)
    metrics = classifier.evaluate(input_fn=input_fn, steps=1)
    self.assertGreater(metrics['accuracy'], 0.9)

  def testWeightedSparseFeatures(self):
    """Tests SDCALogisticClassifier with weighted sparse features."""

    def input_fn():
      return {
          'example_id':
              constant_op.constant(['1', '2', '3']),
          'price':
              sparse_tensor.SparseTensor(
                  values=[2., 3., 1.],
                  indices=[[0, 0], [1, 0], [2, 0]],
                  dense_shape=[3, 5]),
          'country':
              sparse_tensor.SparseTensor(
                  values=['IT', 'US', 'GB'],
                  indices=[[0, 0], [1, 0], [2, 0]],
                  dense_shape=[3, 5])
      }, constant_op.constant([[1], [0], [1]])

    country = feature_column_lib.sparse_column_with_hash_bucket(
        'country', hash_bucket_size=5)
    country_weighted_by_price = feature_column_lib.weighted_sparse_column(
        country, 'price')
    classifier = SDCALogisticClassifier(
        example_id_column='example_id',
        feature_columns=[country_weighted_by_price])
    classifier.fit(input_fn=input_fn, steps=50)
    metrics = classifier.evaluate(input_fn=input_fn, steps=1)
    self.assertGreater(metrics['accuracy'], 0.9)

  def testCrossedFeatures(self):
    """Tests SDCALogisticClassifier with crossed features."""

    def input_fn():
      return {
          'example_id':
              constant_op.constant(['1', '2', '3']),
          'language':
              sparse_tensor.SparseTensor(
                  values=['english', 'italian', 'spanish'],
                  indices=[[0, 0], [1, 0], [2, 0]],
                  dense_shape=[3, 1]),
          'country':
              sparse_tensor.SparseTensor(
                  values=['US', 'IT', 'MX'],
                  indices=[[0, 0], [1, 0], [2, 0]],
                  dense_shape=[3, 1])
      }, constant_op.constant([[0], [0], [1]])

    language = feature_column_lib.sparse_column_with_hash_bucket(
        'language', hash_bucket_size=5)
    country = feature_column_lib.sparse_column_with_hash_bucket(
        'country', hash_bucket_size=5)
    country_language = feature_column_lib.crossed_column(
        [language, country], hash_bucket_size=10)
    classifier = SDCALogisticClassifier(
        example_id_column='example_id', feature_columns=[country_language])
    classifier.fit(input_fn=input_fn, steps=10)
    metrics = classifier.evaluate(input_fn=input_fn, steps=1)
    self.assertGreater(metrics['accuracy'], 0.9)

  def testMixedFeatures(self):
    """Tests SDCALogisticClassifier with a mix of features."""

    def input_fn():
      return {
          'example_id':
              constant_op.constant(['1', '2', '3']),
          'price':
              constant_op.constant([[0.6], [0.8], [0.3]]),
          'sq_footage':
              constant_op.constant([[900.0], [700.0], [600.0]]),
          'country':
              sparse_tensor.SparseTensor(
                  values=['IT', 'US', 'GB'],
                  indices=[[0, 0], [1, 3], [2, 1]],
                  dense_shape=[3, 5]),
          'weights':
              constant_op.constant([[3.0], [1.0], [1.0]])
      }, constant_op.constant([[1], [0], [1]])

    price = feature_column_lib.real_valued_column('price')
    sq_footage_bucket = feature_column_lib.bucketized_column(
        feature_column_lib.real_valued_column('sq_footage'),
        boundaries=[650.0, 800.0])
    country = feature_column_lib.sparse_column_with_hash_bucket(
        'country', hash_bucket_size=5)
    sq_footage_country = feature_column_lib.crossed_column(
        [sq_footage_bucket, country], hash_bucket_size=10)
    classifier = SDCALogisticClassifier(
        example_id_column='example_id',
        feature_columns=[price, sq_footage_bucket, country, sq_footage_country],
        weight_column_name='weights')
    classifier.fit(input_fn=input_fn, steps=50)
    metrics = classifier.evaluate(input_fn=input_fn, steps=1)
    self.assertGreater(metrics['accuracy'], 0.9)


class SDCARegressorTest(test.TestCase):

  def testRealValuedLinearFeatures(self):
    """Tests SDCARegressor works with real valued features."""
    x = [[1.2, 2.0, -1.5], [-2.0, 3.0, -0.5], [1.0, -0.5, 4.0]]
    weights = [[3.0], [-1.2], [0.5]]
    y = np.dot(x, weights)

    def input_fn():
      return {
          'example_id': constant_op.constant(['1', '2', '3']),
          'x': constant_op.constant(x),
          'weights': constant_op.constant([[10.0], [10.0], [10.0]])
      }, constant_op.constant(y)

    x_column = feature_column_lib.real_valued_column('x', dimension=3)
    regressor = SDCARegressor(
        example_id_column='example_id',
        feature_columns=[x_column],
        weight_column_name='weights')
    regressor.fit(input_fn=input_fn, steps=20)
    loss = regressor.evaluate(input_fn=input_fn, steps=1)['loss']
    self.assertLess(loss, 0.01)
    self.assertIn('linear/x/weight', regressor.get_variable_names())
    regressor_weights = regressor.get_variable_value('linear/x/weight')
    self.assertAllClose(
        [w[0] for w in weights], regressor_weights.flatten(), rtol=0.1)

  def testMixedFeaturesArbitraryWeights(self):
    """Tests SDCARegressor works with a mix of features."""

    def input_fn():
      return {
          'example_id':
              constant_op.constant(['1', '2', '3']),
          'price':
              constant_op.constant([[0.6], [0.8], [0.3]]),
          'sq_footage':
              constant_op.constant([[900.0], [700.0], [600.0]]),
          'country':
              sparse_tensor.SparseTensor(
                  values=['IT', 'US', 'GB'],
                  indices=[[0, 0], [1, 3], [2, 1]],
                  dense_shape=[3, 5]),
          'weights':
              constant_op.constant([[3.0], [5.0], [7.0]])
      }, constant_op.constant([[1.55], [-1.25], [-3.0]])

    price = feature_column_lib.real_valued_column('price')
    sq_footage_bucket = feature_column_lib.bucketized_column(
        feature_column_lib.real_valued_column('sq_footage'),
        boundaries=[650.0, 800.0])
    country = feature_column_lib.sparse_column_with_hash_bucket(
        'country', hash_bucket_size=5)
    sq_footage_country = feature_column_lib.crossed_column(
        [sq_footage_bucket, country], hash_bucket_size=10)
    regressor = SDCARegressor(
        example_id_column='example_id',
        feature_columns=[price, sq_footage_bucket, country, sq_footage_country],
        l2_regularization=1.0,
        weight_column_name='weights')
    regressor.fit(input_fn=input_fn, steps=20)
    loss = regressor.evaluate(input_fn=input_fn, steps=1)['loss']
    self.assertLess(loss, 0.05)

  def testSdcaOptimizerSparseFeaturesWithL1Reg(self):
    """Tests SDCARegressor works with sparse features and L1 regularization."""

    def input_fn():
      return {
          'example_id':
              constant_op.constant(['1', '2', '3']),
          'price':
              constant_op.constant([[0.4], [0.6], [0.3]]),
          'country':
              sparse_tensor.SparseTensor(
                  values=['IT', 'US', 'GB'],
                  indices=[[0, 0], [1, 3], [2, 1]],
                  dense_shape=[3, 5]),
          'weights':
              constant_op.constant([[10.0], [10.0], [10.0]])
      }, constant_op.constant([[1.4], [-0.8], [2.6]])

    price = feature_column_lib.real_valued_column('price')
    country = feature_column_lib.sparse_column_with_hash_bucket(
        'country', hash_bucket_size=5)
    # Regressor with no L1 regularization.
    regressor = SDCARegressor(
        example_id_column='example_id',
        feature_columns=[price, country],
        weight_column_name='weights')
    regressor.fit(input_fn=input_fn, steps=20)
    no_l1_reg_loss = regressor.evaluate(input_fn=input_fn, steps=1)['loss']
    variable_names = regressor.get_variable_names()
    self.assertIn('linear/price/weight', variable_names)
    self.assertIn('linear/country/weights', variable_names)
    no_l1_reg_weights = {
        'linear/price/weight':
            regressor.get_variable_value('linear/price/weight'),
        'linear/country/weights':
            regressor.get_variable_value('linear/country/weights'),
    }

    # Regressor with L1 regularization.
    regressor = SDCARegressor(
        example_id_column='example_id',
        feature_columns=[price, country],
        l1_regularization=1.0,
        weight_column_name='weights')
    regressor.fit(input_fn=input_fn, steps=20)
    l1_reg_loss = regressor.evaluate(input_fn=input_fn, steps=1)['loss']
    l1_reg_weights = {
        'linear/price/weight':
            regressor.get_variable_value('linear/price/weight'),
        'linear/country/weights':
            regressor.get_variable_value('linear/country/weights'),
    }

    # Unregularized loss is lower when there is no L1 regularization.
    self.assertLess(no_l1_reg_loss, l1_reg_loss)
    self.assertLess(no_l1_reg_loss, 0.05)

    # But weights returned by the regressor with L1 regularization have smaller
    # L1 norm.
    l1_reg_weights_norm, no_l1_reg_weights_norm = 0.0, 0.0
    for var_name in sorted(l1_reg_weights):
      l1_reg_weights_norm += sum(
          np.absolute(l1_reg_weights[var_name].flatten()))
      no_l1_reg_weights_norm += sum(
          np.absolute(no_l1_reg_weights[var_name].flatten()))
      print('Var name: %s, value: %s' % (var_name,
                                         no_l1_reg_weights[var_name].flatten()))
    self.assertLess(l1_reg_weights_norm, no_l1_reg_weights_norm)

  def testBiasOnly(self):
    """Tests SDCARegressor has a valid bias weight."""

    def input_fn():
      """Testing the bias weight when it's the only feature present.

      All of the instances in this input only have the bias feature, and a
      1/4 of the labels are positive. This means that the expected weight for
      the bias should be close to the average prediction, i.e 0.25.
      Returns:
        Training data for the test.
      """
      num_examples = 40
      return {
          'example_id':
              constant_op.constant([str(x + 1) for x in range(num_examples)]),
          # place_holder is an empty column which is always 0 (absent), because
          # LinearClassifier requires at least one column.
          'place_holder':
              constant_op.constant([[0.0]] * num_examples),
      }, constant_op.constant([[1 if i % 4 is 0 else 0]
                               for i in range(num_examples)])

    place_holder = feature_column_lib.real_valued_column('place_holder')
    regressor = SDCARegressor(
        example_id_column='example_id', feature_columns=[place_holder])
    regressor.fit(input_fn=input_fn, steps=100)
    self.assertNear(
        regressor.get_variable_value('linear/bias_weight')[0], 0.25, err=0.1)

  def testBiasAndOtherColumns(self):
    """SDCARegressor has valid bias weight when other columns are present."""

    def input_fn():
      """Testing the bias weight when there are other features present.

      1/2 of the instances in this input have feature 'a', the rest have
      feature 'b', and we expect the bias to be added to each instance as well.
      0.4 of all instances that have feature 'a' are positive, and 0.2 of all
      instances that have feature 'b' are positive. The labels in the dataset
      are ordered to appear shuffled since SDCA expects shuffled data, and
      converges faster with this pseudo-random ordering.
      If the bias was centered we would expect the weights to be:
      bias: 0.3
      a: 0.1
      b: -0.1
      Until b/29339026 is resolved, the bias gets regularized with the same
      global value for the other columns, and so the expected weights get
      shifted and are:
      bias: 0.2
      a: 0.2
      b: 0.0
      Returns:
        The test dataset.
      """
      num_examples = 200
      half = int(num_examples / 2)
      return {
          'example_id':
              constant_op.constant([str(x + 1) for x in range(num_examples)]),
          'a':
              constant_op.constant([[1]] * int(half) + [[0]] * int(half)),
          'b':
              constant_op.constant([[0]] * int(half) + [[1]] * int(half)),
      }, constant_op.constant(
          [[x]
           for x in [1, 0, 0, 1, 1, 0, 0, 0, 1, 0] * int(half / 10) +
           [0, 1, 0, 0, 0, 0, 0, 0, 1, 0] * int(half / 10)])

    regressor = SDCARegressor(
        example_id_column='example_id',
        feature_columns=[
            feature_column_lib.real_valued_column('a'),
            feature_column_lib.real_valued_column('b')
        ])

    regressor.fit(input_fn=input_fn, steps=200)

    variable_names = regressor.get_variable_names()
    self.assertIn('linear/bias_weight', variable_names)
    self.assertIn('linear/a/weight', variable_names)
    self.assertIn('linear/b/weight', variable_names)
    # TODO(b/29339026): Change the expected results to expect a centered bias.
    self.assertNear(
        regressor.get_variable_value('linear/bias_weight')[0], 0.2, err=0.05)
    self.assertNear(
        regressor.get_variable_value('linear/a/weight')[0], 0.2, err=0.05)
    self.assertNear(
        regressor.get_variable_value('linear/b/weight')[0], 0.0, err=0.05)

  def testBiasAndOtherColumnsFabricatedCentered(self):
    """SDCARegressor has valid bias weight when instances are centered."""

    def input_fn():
      """Testing the bias weight when there are other features present.

      1/2 of the instances in this input have feature 'a', the rest have
      feature 'b', and we expect the bias to be added to each instance as well.
      0.1 of all instances that have feature 'a' have a label of 1, and 0.1 of
      all instances that have feature 'b' have a label of -1.
      We can expect the weights to be:
      bias: 0.0
      a: 0.1
      b: -0.1
      Returns:
        The test dataset.
      """
      num_examples = 200
      half = int(num_examples / 2)
      return {
          'example_id':
              constant_op.constant([str(x + 1) for x in range(num_examples)]),
          'a':
              constant_op.constant([[1]] * int(half) + [[0]] * int(half)),
          'b':
              constant_op.constant([[0]] * int(half) + [[1]] * int(half)),
      }, constant_op.constant([[1 if x % 10 == 0 else 0] for x in range(half)] +
                              [[-1 if x % 10 == 0 else 0] for x in range(half)])

    regressor = SDCARegressor(
        example_id_column='example_id',
        feature_columns=[
            feature_column_lib.real_valued_column('a'),
            feature_column_lib.real_valued_column('b')
        ])

    regressor.fit(input_fn=input_fn, steps=100)

    variable_names = regressor.get_variable_names()
    self.assertIn('linear/bias_weight', variable_names)
    self.assertIn('linear/a/weight', variable_names)
    self.assertIn('linear/b/weight', variable_names)
    self.assertNear(
        regressor.get_variable_value('linear/bias_weight')[0], 0.0, err=0.05)
    self.assertNear(
        regressor.get_variable_value('linear/a/weight')[0], 0.1, err=0.05)
    self.assertNear(
        regressor.get_variable_value('linear/b/weight')[0], -0.1, err=0.05)


if __name__ == '__main__':
  test.main()
