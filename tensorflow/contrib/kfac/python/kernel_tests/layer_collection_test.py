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
"""Tests for tf.contrib.kfac.layer_collection."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.kfac.python.ops import fisher_factors
from tensorflow.contrib.kfac.python.ops import layer_collection
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import test


class LayerCollectionTest(test.TestCase):

  def testLayerCollectionInit(self):
    lc = layer_collection.LayerCollection()
    self.assertEqual(0, len(lc.get_blocks()))
    self.assertEqual(0, len(lc.get_factors()))
    self.assertFalse(lc.losses)

  def testRegisterBlocks(self):
    with ops.Graph().as_default():
      random_seed.set_random_seed(200)
      lc = layer_collection.LayerCollection()
      lc.register_fully_connected(
          array_ops.constant(1), array_ops.constant(2), array_ops.constant(3))
      lc.register_conv2d(
          array_ops.constant(4), [1, 1, 1, 1], 'SAME',
          array_ops.ones((1, 1, 1, 1)), array_ops.constant(3))
      lc.register_generic(
          array_ops.constant(5), 16, approx=layer_collection.APPROX_FULL_NAME)
      lc.register_generic(
          array_ops.constant(6),
          16,
          approx=layer_collection.APPROX_DIAGONAL_NAME)

      self.assertEqual(4, len(lc.get_blocks()))

  def testRegisterBlocksMultipleRegistrations(self):
    with ops.Graph().as_default():
      random_seed.set_random_seed(200)
      lc = layer_collection.LayerCollection()
      key = array_ops.constant(1)
      lc.register_fully_connected(key,
                                  array_ops.constant(2), array_ops.constant(3))
      with self.assertRaises(ValueError):
        lc.register_generic(key, 16)

  def testRegisterSingleParamNotRegistered(self):
    x = variable_scope.get_variable('x', initializer=array_ops.constant(1,))
    lc = layer_collection.LayerCollection()
    lc.fisher_blocks = {
        variable_scope.get_variable('y', initializer=array_ops.constant(1,)):
            '1'
    }
    lc.register_block(x, 'foo')

  def testShouldRegisterSingleParamRegistered(self):
    x = variable_scope.get_variable('x', initializer=array_ops.constant(1,))
    lc = layer_collection.LayerCollection()
    lc.fisher_blocks = {x: '1'}
    with self.assertRaises(ValueError):
      lc.register_block(x, 'foo')

  def testRegisterSingleParamRegisteredInTuple(self):
    x = variable_scope.get_variable('x', initializer=array_ops.constant(1,))
    y = variable_scope.get_variable('y', initializer=array_ops.constant(1,))
    lc = layer_collection.LayerCollection()
    lc.fisher_blocks = {(x, y): '1'}
    lc.register_block(x, 'foo')
    self.assertEqual(set(['1']), set(lc.get_blocks()))

  def testRegisterTupleParamNotRegistered(self):
    x = variable_scope.get_variable('x', initializer=array_ops.constant(1,))
    y = variable_scope.get_variable('y', initializer=array_ops.constant(1,))
    lc = layer_collection.LayerCollection()
    lc.fisher_blocks = {
        variable_scope.get_variable('z', initializer=array_ops.constant(1,)):
            '1'
    }

    lc.register_block((x, y), 'foo')
    self.assertEqual(set(['1', 'foo']), set(lc.get_blocks()))

  def testRegisterTupleParamRegistered(self):
    x = variable_scope.get_variable('x', initializer=array_ops.constant(1,))
    y = variable_scope.get_variable('y', initializer=array_ops.constant(1,))
    lc = layer_collection.LayerCollection()
    lc.fisher_blocks = {(x, y): '1'}

    with self.assertRaises(ValueError):
      lc.register_block((x, y), 'foo')

  def testRegisterTupleParamRegisteredInSuperset(self):
    x = variable_scope.get_variable('x', initializer=array_ops.constant(1,))
    y = variable_scope.get_variable('y', initializer=array_ops.constant(1,))
    z = variable_scope.get_variable('z', initializer=array_ops.constant(1,))
    lc = layer_collection.LayerCollection()
    lc.fisher_blocks = {(x, y, z): '1'}

    lc.register_block((x, y), 'foo')
    self.assertEqual(set(['1']), set(lc.get_blocks()))

  def testRegisterTupleParamSomeRegistered(self):
    x = variable_scope.get_variable('x', initializer=array_ops.constant(1,))
    y = variable_scope.get_variable('y', initializer=array_ops.constant(1,))
    z = variable_scope.get_variable('z', initializer=array_ops.constant(1,))
    lc = layer_collection.LayerCollection()
    lc.fisher_blocks = {x: '1', z: '2'}

    lc.register_block((x, y), 'foo')
    self.assertEqual(set(['2', 'foo']), set(lc.get_blocks()))

  def testRegisterTupleVarSomeRegisteredInOtherTuples(self):
    x = variable_scope.get_variable('x', initializer=array_ops.constant(1,))
    y = variable_scope.get_variable('y', initializer=array_ops.constant(1,))
    z = variable_scope.get_variable('z', initializer=array_ops.constant(1,))
    w = variable_scope.get_variable('w', initializer=array_ops.constant(1,))
    lc = layer_collection.LayerCollection()
    lc.fisher_blocks = {(x, z): '1', (z, w): '2'}

    with self.assertRaises(ValueError):
      lc.register_block((x, y), 'foo')

  def testRegisterCategoricalPredictiveDistribution(self):
    with ops.Graph().as_default(), self.test_session() as sess:
      random_seed.set_random_seed(200)
      logits = linalg_ops.eye(2)

      lc = layer_collection.LayerCollection()
      lc.register_categorical_predictive_distribution(logits, seed=200)
      single_loss = sess.run(lc.total_sampled_loss())

      lc2 = layer_collection.LayerCollection()
      lc2.register_categorical_predictive_distribution(logits, seed=200)
      lc2.register_categorical_predictive_distribution(logits, seed=200)
      double_loss = sess.run(lc2.total_sampled_loss())
      self.assertAlmostEqual(2 * single_loss, double_loss)

  def testRegisterCategoricalPredictiveDistributionBatchSize1(self):
    with ops.Graph().as_default():
      random_seed.set_random_seed(200)
      logits = random_ops.random_normal((1, 2))
      lc = layer_collection.LayerCollection()

      lc.register_categorical_predictive_distribution(logits, seed=200)

  def testRegisterCategoricalPredictiveDistributionSpecifiedTargets(self):
    with ops.Graph().as_default(), self.test_session() as sess:
      random_seed.set_random_seed(200)
      logits = array_ops.constant([[1., 2.], [3., 4.]], dtype=dtypes.float32)
      lc = layer_collection.LayerCollection()
      targets = array_ops.constant([0, 1], dtype=dtypes.int32)

      lc.register_categorical_predictive_distribution(logits, targets=targets)
      single_loss = sess.run(lc.total_loss())
      self.assertAlmostEqual(1.6265233, single_loss)

  def testRegisterNormalPredictiveDistribution(self):
    with ops.Graph().as_default(), self.test_session() as sess:
      random_seed.set_random_seed(200)
      predictions = array_ops.constant(
          [[1., 2.], [3., 4]], dtype=dtypes.float32)

      lc = layer_collection.LayerCollection()
      lc.register_normal_predictive_distribution(predictions, 1., seed=200)
      single_loss = sess.run(lc.total_sampled_loss())

      lc2 = layer_collection.LayerCollection()
      lc2.register_normal_predictive_distribution(predictions, 1., seed=200)
      lc2.register_normal_predictive_distribution(predictions, 1., seed=200)
      double_loss = sess.run(lc2.total_sampled_loss())

      self.assertAlmostEqual(2 * single_loss, double_loss)

  def testRegisterNormalPredictiveDistributionSpecifiedTargets(self):
    with ops.Graph().as_default(), self.test_session() as sess:
      random_seed.set_random_seed(200)
      predictions = array_ops.constant(
          [[1., 2.], [3., 4.]], dtype=dtypes.float32)
      lc = layer_collection.LayerCollection()
      targets = array_ops.constant([[3., 1.], [4., 2.]], dtype=dtypes.float32)

      lc.register_normal_predictive_distribution(
          predictions, 2.**2, targets=targets)
      single_loss = sess.run(lc.total_loss())
      self.assertAlmostEqual(7.6983433, single_loss)

  def testMakeOrGetFactor(self):
    with ops.Graph().as_default():
      random_seed.set_random_seed(200)
      lc = layer_collection.LayerCollection()
      key = array_ops.constant(1)
      lc.make_or_get_factor(fisher_factors.FullFactor, ((key,), 16))
      lc.make_or_get_factor(fisher_factors.FullFactor, ((key,), 16))
      lc.make_or_get_factor(fisher_factors.FullFactor,
                            ((array_ops.constant(2),), 16))

      self.assertEqual(2, len(lc.get_factors()))
      variables = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
      self.assertTrue(
          all([var.name.startswith('LayerCollection') for var in variables]))

  def testMakeOrGetFactorCustomScope(self):
    with ops.Graph().as_default():
      random_seed.set_random_seed(200)
      scope = 'Foo'
      lc = layer_collection.LayerCollection(name=scope)
      key = array_ops.constant(1)
      lc.make_or_get_factor(fisher_factors.FullFactor, ((key,), 16))
      lc.make_or_get_factor(fisher_factors.FullFactor, ((key,), 16))
      lc.make_or_get_factor(fisher_factors.FullFactor,
                            ((array_ops.constant(2),), 16))

      self.assertEqual(2, len(lc.get_factors()))
      variables = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
      self.assertTrue(all([var.name.startswith(scope) for var in variables]))

  def testGetUseCountMap(self):
    lc = layer_collection.LayerCollection()
    lc.fisher_blocks = {'a': 1, ('a', 'c'): 2, ('b', 'c'): 2}
    use_count_map = lc.get_use_count_map()
    self.assertDictEqual({'a': 2, 'b': 1, 'c': 2}, use_count_map)


if __name__ == '__main__':
  test.main()
