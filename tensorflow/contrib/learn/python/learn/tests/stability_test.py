# pylint: disable=g-bad-file-header
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

"""Non-linear estimator tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import random

import tensorflow as tf


class StabilityTest(tf.test.TestCase):
  """Tests that estiamtors are reproducible."""

  def testRandomStability(self):
    my_seed, minval, maxval = 42, -0.3333, 0.3333
    with tf.Graph().as_default() as g:
      with self.test_session(graph=g) as session:
        tf.set_random_seed(my_seed)
        x = tf.random_uniform([10, 10], minval=minval, maxval=maxval)
        val1 = session.run(x)
    with tf.Graph().as_default() as g:
      with self.test_session(graph=g) as session:
        tf.set_random_seed(my_seed)
        x = tf.random_uniform([10, 10], minval=minval, maxval=maxval)
        val2 = session.run(x)
    self.assertAllClose(val1, val2)

  def testLinearRegression(self):
    # TODO(ipolosukhin): This doesn't pass at all, but should...
    pass
#     random.seed(42)
#     boston = tf.contrib.learn.datasets.load_boston()
#     regressor = tf.contrib.learn.LinearRegressor()
#     regressor.fit(x=boston.data, y=boston.target, steps=1)
#     regressor2 = tf.contrib.learn.LinearRegressor()
#     regressor2.fit(x=boston.data, y=boston.target, steps=1)
#     self.assertAllClose(regressor.weights_, regressor2.weights_)
#     self.assertAllClose(regressor.bias_, regressor2.bias_)
#     self.assertAllClose(regressor.predict(boston.data),
#                         regressor2.predict(boston.data), atol=1e-05)

  def testDNNRegression(self):
    # TODO(ipolosukhin): This doesn't pass at all, but should...
    # Either bugs or just general instability.
    pass
#     random.seed(42)
#     boston = tf.contrib.learn.datasets.load_boston()
#     regressor = tf.contrib.learn.DNNRegressor(
#         hidden_units=[10],
#         optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.001))
#     graph_dump = tf.contrib.learn.monitors.GraphDump()
#     regressor.fit(x=boston.data, y=boston.target, steps=1,
#                   monitors=[graph_dump], batch_size=1)
#     regressor2 = tf.contrib.learn.DNNRegressor(
#         hidden_units=[10],
#         optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.001))
#     graph_dump2 = tf.contrib.learn.monitors.GraphDump()
#     regressor2.fit(x=boston.data, y=boston.target, steps=1,
#                    monitors=[graph_dump2], batch_size=1)
#     _, non_match = graph_dump.compare(graph_dump2, 0, atol=1e-02)
#     self.assertEmpty(non_match.keys())
#     for weight1, weight2 in zip(regressor.weights_, regressor2.weights_):
#       self.assertAllClose(weight1, weight2)
#     for bias1, bias2 in zip(regressor.bias_, regressor2.bias_):
#       self.assertAllClose(bias1, bias2)
#     self.assertAllClose(regressor.predict(boston.data),
#                         regressor2.predict(boston.data), atol=1e-05)


if __name__ == '__main__':
  tf.test.main()
