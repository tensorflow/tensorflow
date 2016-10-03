# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""Estimator regression tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.learn_io import data_feeder


def _get_input_fn(x, y, batch_size=None):
  df = data_feeder.setup_train_data_feeder(
      x, y, n_classes=None, batch_size=batch_size)
  return df.input_builder, df.get_feed_dict_fn()


# We use a null optimizer since we can't get deterministic results out of
# supervisor's multiple threads.
class _NullOptimizer(tf.train.Optimizer):

  def __init__(self):
    super(_NullOptimizer, self).__init__(use_locking=False, name='Null')

  def _apply_dense(self, grad, var):
    return tf.no_op()

  def _apply_sparse(self, grad, var):
    return tf.no_op()

  def _prepare(self):
    pass


_NULL_OPTIMIZER = _NullOptimizer()


class StabilityTest(tf.test.TestCase):
  """Tests that estiamtors are reproducible."""

  def testRandomStability(self):
    my_seed = 42
    minval = -0.3333
    maxval = 0.3333
    with tf.Graph().as_default() as g:
      with self.test_session(graph=g) as session:
        g.seed = my_seed
        x = tf.random_uniform([10, 10], minval=minval, maxval=maxval)
        val1 = session.run(x)
    with tf.Graph().as_default() as g:
      with self.test_session(graph=g) as session:
        g.seed = my_seed
        x = tf.random_uniform([10, 10], minval=minval, maxval=maxval)
        val2 = session.run(x)
    self.assertAllClose(val1, val2)

  def testLinearRegression(self):
    my_seed = 42
    config = tf.contrib.learn.RunConfig(tf_random_seed=my_seed)
    boston = tf.contrib.learn.datasets.load_boston()
    columns = [tf.contrib.layers.real_valued_column('', dimension=13)]

    # We train with

    with tf.Graph().as_default() as g1:
      random.seed(my_seed)
      g1.seed = my_seed
      tf.contrib.framework.create_global_step()
      regressor1 = tf.contrib.learn.LinearRegressor(optimizer=_NULL_OPTIMIZER,
                                                    feature_columns=columns,
                                                    config=config)
      regressor1.fit(x=boston.data, y=boston.target, steps=1)

    with tf.Graph().as_default() as g2:
      random.seed(my_seed)
      g2.seed = my_seed
      tf.contrib.framework.create_global_step()
      regressor2 = tf.contrib.learn.LinearRegressor(optimizer=_NULL_OPTIMIZER,
                                                    feature_columns=columns,
                                                    config=config)
      regressor2.fit(x=boston.data, y=boston.target, steps=1)

    self.assertAllClose(regressor1.weights_, regressor2.weights_)
    self.assertAllClose(regressor1.bias_, regressor2.bias_)
    self.assertAllClose(
        list(regressor1.predict(boston.data, as_iterable=True)),
        list(regressor2.predict(boston.data, as_iterable=True)), atol=1e-05)

  def testDNNRegression(self):
    my_seed = 42
    config = tf.contrib.learn.RunConfig(tf_random_seed=my_seed)
    boston = tf.contrib.learn.datasets.load_boston()
    columns = [tf.contrib.layers.real_valued_column('', dimension=13)]

    with tf.Graph().as_default() as g1:
      random.seed(my_seed)
      g1.seed = my_seed
      tf.contrib.framework.create_global_step()
      regressor1 = tf.contrib.learn.DNNRegressor(
          hidden_units=[10], feature_columns=columns,
          optimizer=_NULL_OPTIMIZER, config=config)
      regressor1.fit(x=boston.data, y=boston.target, steps=1)

    with tf.Graph().as_default() as g2:
      random.seed(my_seed)
      g2.seed = my_seed
      tf.contrib.framework.create_global_step()
      regressor2 = tf.contrib.learn.DNNRegressor(
          hidden_units=[10], feature_columns=columns,
          optimizer=_NULL_OPTIMIZER, config=config)
      regressor2.fit(x=boston.data, y=boston.target, steps=1)

    for w1, w2 in zip(regressor1.weights_, regressor2.weights_):
      self.assertAllClose(w1, w2)
    for b1, b2 in zip(regressor2.bias_, regressor2.bias_):
      self.assertAllClose(b1, b2)
    self.assertAllClose(
        list(regressor1.predict(boston.data, as_iterable=True)),
        list(regressor2.predict(boston.data, as_iterable=True)), atol=1e-05)


if __name__ == '__main__':
  tf.test.main()
