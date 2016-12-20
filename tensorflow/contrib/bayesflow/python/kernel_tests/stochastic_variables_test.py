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
"""Tests for stochastic graphs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

sv = tf.contrib.bayesflow.stochastic_variables
st = tf.contrib.bayesflow.stochastic_tensor
vi = tf.contrib.bayesflow.variational_inference
dist = tf.contrib.distributions


class StochasticVariablesTest(tf.test.TestCase):

  def testStochasticVariables(self):
    shape = (10, 20)
    with tf.variable_scope(
        "stochastic_variables",
        custom_getter=sv.make_stochastic_variable_getter(
            dist_cls=dist.NormalWithSoftplusSigma)):
      v = tf.get_variable("sv", shape)

    self.assertTrue(isinstance(v, st.StochasticTensor))
    self.assertTrue(isinstance(v.distribution, dist.NormalWithSoftplusSigma))

    self.assertEqual(
        {"stochastic_variables/sv_mu", "stochastic_variables/sv_sigma"},
        set([v.op.name for v in tf.global_variables()]))
    self.assertEqual(set(tf.trainable_variables()), set(tf.global_variables()))

    v = tf.convert_to_tensor(v)
    self.assertEqual(list(shape), v.get_shape().as_list())
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertEqual(shape, sess.run(v).shape)

  def testStochasticVariablesWithConstantInitializer(self):
    shape = (10, 20)
    with tf.variable_scope(
        "stochastic_variables",
        custom_getter=sv.make_stochastic_variable_getter(
            dist_cls=dist.NormalWithSoftplusSigma,
            dist_kwargs={"validate_args": True},
            param_initializers={
                "mu": np.ones(shape) * 4.,
                "sigma": np.ones(shape) * 2.
            })):
      v = tf.get_variable("sv")

    for var in tf.global_variables():
      if "mu" in var.name:
        mu_var = var
      if "sigma" in var.name:
        sigma_var = var

    v = tf.convert_to_tensor(v)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertAllEqual(np.ones(shape) * 4., sess.run(mu_var))
      self.assertAllEqual(np.ones(shape) * 2., sess.run(sigma_var))
      self.assertEqual(shape, sess.run(v).shape)

  def testStochasticVariablesWithCallableInitializer(self):
    shape = (10, 20)

    def sigma_init(shape, dtype, partition_info):
      _ = partition_info
      return tf.ones(shape, dtype=dtype) * 2.

    with tf.variable_scope(
        "stochastic_variables",
        custom_getter=sv.make_stochastic_variable_getter(
            dist_cls=dist.NormalWithSoftplusSigma,
            dist_kwargs={"validate_args": True},
            param_initializers={
                "mu": np.ones(
                    shape, dtype=np.float32) * 4.,
                "sigma": sigma_init
            })):
      v = tf.get_variable("sv", shape)

    for var in tf.global_variables():
      if "mu" in var.name:
        mu_var = var
      if "sigma" in var.name:
        sigma_var = var

    v = tf.convert_to_tensor(v)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertAllEqual(np.ones(shape) * 4., sess.run(mu_var))
      self.assertAllEqual(np.ones(shape) * 2., sess.run(sigma_var))
      self.assertEqual(shape, sess.run(v).shape)

  def testStochasticVariablesWithPrior(self):
    shape = (10, 20)
    prior = dist.Normal(0., 1.)
    with tf.variable_scope(
        "stochastic_variables",
        custom_getter=sv.make_stochastic_variable_getter(
            dist_cls=dist.NormalWithSoftplusSigma, prior=prior)):
      w = tf.get_variable("weights", shape)

    x = tf.random_uniform((8, 10))
    y = tf.matmul(x, w)

    prior_map = vi._find_variational_and_priors(y, None)
    self.assertEqual(prior_map[w], prior)
    elbo = vi.elbo(y, keep_batch_dim=False)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(elbo)

  def testStochasticVariablesWithCallablePriorInitializer(self):

    def prior_init(shape, dtype):
      return dist.Normal(tf.zeros(shape, dtype), tf.ones(shape, dtype))

    with tf.variable_scope(
        "stochastic_variables",
        custom_getter=sv.make_stochastic_variable_getter(
            dist_cls=dist.NormalWithSoftplusSigma, prior=prior_init)):
      w = tf.get_variable("weights", (10, 20))

    x = tf.random_uniform((8, 10))
    y = tf.matmul(x, w)

    prior_map = vi._find_variational_and_priors(y, None)
    self.assertTrue(isinstance(prior_map[w], dist.Normal))
    elbo = vi.elbo(y, keep_batch_dim=False)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(elbo)


if __name__ == "__main__":
  tf.test.main()
