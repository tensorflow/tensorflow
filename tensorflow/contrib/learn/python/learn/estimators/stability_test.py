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
import sys

# TODO: #6568 Remove this hack that makes dlopen() not crash.
if hasattr(sys, 'getdlopenflags') and hasattr(sys, 'setdlopenflags'):
  import ctypes
  sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)

from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import feature_column
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.contrib.learn.python.learn.estimators import dnn
from tensorflow.contrib.learn.python.learn.estimators import linear
from tensorflow.contrib.learn.python.learn.estimators import run_config
from tensorflow.contrib.learn.python.learn.learn_io import data_feeder
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test
from tensorflow.python.training import optimizer as optimizer_lib


def _get_input_fn(x, y, batch_size=None):
  df = data_feeder.setup_train_data_feeder(
      x, y, n_classes=None, batch_size=batch_size)
  return df.input_builder, df.get_feed_dict_fn()


# We use a null optimizer since we can't get deterministic results out of
# supervisor's multiple threads.
class _NullOptimizer(optimizer_lib.Optimizer):

  def __init__(self):
    super(_NullOptimizer, self).__init__(use_locking=False, name='Null')

  def _apply_dense(self, grad, var):
    return control_flow_ops.no_op()

  def _apply_sparse(self, grad, var):
    return control_flow_ops.no_op()

  def _prepare(self):
    pass


_NULL_OPTIMIZER = _NullOptimizer()


class StabilityTest(test.TestCase):
  """Tests that estiamtors are reproducible."""

  def testRandomStability(self):
    my_seed = 42
    minval = -0.3333
    maxval = 0.3333
    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as session:
        g.seed = my_seed
        x = random_ops.random_uniform([10, 10], minval=minval, maxval=maxval)
        val1 = session.run(x)
    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as session:
        g.seed = my_seed
        x = random_ops.random_uniform([10, 10], minval=minval, maxval=maxval)
        val2 = session.run(x)
    self.assertAllClose(val1, val2)

  def testLinearRegression(self):
    my_seed = 42
    config = run_config.RunConfig(tf_random_seed=my_seed)
    boston = base.load_boston()
    columns = [feature_column.real_valued_column('', dimension=13)]

    # We train with

    with ops.Graph().as_default() as g1:
      random.seed(my_seed)
      g1.seed = my_seed
      variables.create_global_step()
      regressor1 = linear.LinearRegressor(
          optimizer=_NULL_OPTIMIZER, feature_columns=columns, config=config)
      regressor1.fit(x=boston.data, y=boston.target, steps=1)

    with ops.Graph().as_default() as g2:
      random.seed(my_seed)
      g2.seed = my_seed
      variables.create_global_step()
      regressor2 = linear.LinearRegressor(
          optimizer=_NULL_OPTIMIZER, feature_columns=columns, config=config)
      regressor2.fit(x=boston.data, y=boston.target, steps=1)

    self.assertAllClose(regressor1.weights_, regressor2.weights_)
    self.assertAllClose(regressor1.bias_, regressor2.bias_)
    self.assertAllClose(
        list(regressor1.predict_scores(
            boston.data, as_iterable=True)),
        list(regressor2.predict_scores(
            boston.data, as_iterable=True)),
        atol=1e-05)

  def testDNNRegression(self):
    my_seed = 42
    config = run_config.RunConfig(tf_random_seed=my_seed)
    boston = base.load_boston()
    columns = [feature_column.real_valued_column('', dimension=13)]

    with ops.Graph().as_default() as g1:
      random.seed(my_seed)
      g1.seed = my_seed
      variables.create_global_step()
      regressor1 = dnn.DNNRegressor(
          hidden_units=[10],
          feature_columns=columns,
          optimizer=_NULL_OPTIMIZER,
          config=config)
      regressor1.fit(x=boston.data, y=boston.target, steps=1)

    with ops.Graph().as_default() as g2:
      random.seed(my_seed)
      g2.seed = my_seed
      variables.create_global_step()
      regressor2 = dnn.DNNRegressor(
          hidden_units=[10],
          feature_columns=columns,
          optimizer=_NULL_OPTIMIZER,
          config=config)
      regressor2.fit(x=boston.data, y=boston.target, steps=1)

    weights1 = ([regressor1.get_variable_value('dnn/hiddenlayer_0/weights')] +
                [regressor1.get_variable_value('dnn/logits/weights')])
    weights2 = ([regressor2.get_variable_value('dnn/hiddenlayer_0/weights')] +
                [regressor2.get_variable_value('dnn/logits/weights')])
    for w1, w2 in zip(weights1, weights2):
      self.assertAllClose(w1, w2)

    biases1 = ([regressor1.get_variable_value('dnn/hiddenlayer_0/biases')] +
               [regressor1.get_variable_value('dnn/logits/biases')])
    biases2 = ([regressor2.get_variable_value('dnn/hiddenlayer_0/biases')] +
               [regressor2.get_variable_value('dnn/logits/biases')])
    for b1, b2 in zip(biases1, biases2):
      self.assertAllClose(b1, b2)
    self.assertAllClose(
        list(regressor1.predict_scores(
            boston.data, as_iterable=True)),
        list(regressor2.predict_scores(
            boston.data, as_iterable=True)),
        atol=1e-05)


if __name__ == '__main__':
  test.main()
