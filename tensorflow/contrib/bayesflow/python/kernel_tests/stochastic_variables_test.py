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
from tensorflow.contrib import distributions
from tensorflow.contrib.bayesflow.python.ops import stochastic_tensor
from tensorflow.contrib.bayesflow.python.ops import stochastic_variables
from tensorflow.contrib.bayesflow.python.ops import variational_inference
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

sv = stochastic_variables
st = stochastic_tensor
vi = variational_inference
dist = distributions


class StochasticVariablesTest(test.TestCase):

  def testStochasticVariables(self):
    shape = (10, 20)
    with variable_scope.variable_scope(
        "stochastic_variables",
        custom_getter=sv.make_stochastic_variable_getter(
            dist_cls=dist.NormalWithSoftplusScale)):
      v = variable_scope.get_variable("sv", shape)

    self.assertTrue(isinstance(v, st.StochasticTensor))
    self.assertTrue(isinstance(v.distribution, dist.NormalWithSoftplusScale))

    self.assertEqual(
        {"stochastic_variables/sv_loc", "stochastic_variables/sv_scale"},
        set([v.op.name for v in variables.global_variables()]))
    self.assertEqual(
        set(variables.trainable_variables()), set(variables.global_variables()))

    v = ops.convert_to_tensor(v)
    self.assertEqual(list(shape), v.get_shape().as_list())
    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())
      self.assertEqual(shape, sess.run(v).shape)

  def testStochasticVariablesWithConstantInitializer(self):
    shape = (10, 20)
    with variable_scope.variable_scope(
        "stochastic_variables",
        custom_getter=sv.make_stochastic_variable_getter(
            dist_cls=dist.NormalWithSoftplusScale,
            dist_kwargs={"validate_args": True},
            param_initializers={
                "loc": np.ones(shape) * 4.,
                "scale": np.ones(shape) * 2.
            })):
      v = variable_scope.get_variable("sv")

    for var in variables.global_variables():
      if "loc" in var.name:
        mu_var = var
      if "scale" in var.name:
        sigma_var = var

    v = ops.convert_to_tensor(v)
    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())
      self.assertAllEqual(np.ones(shape) * 4., sess.run(mu_var))
      self.assertAllEqual(np.ones(shape) * 2., sess.run(sigma_var))
      self.assertEqual(shape, sess.run(v).shape)

  def testStochasticVariablesWithCallableInitializer(self):
    shape = (10, 20)

    def sigma_init(shape, dtype, partition_info):
      _ = partition_info
      return array_ops.ones(shape, dtype=dtype) * 2.

    with variable_scope.variable_scope(
        "stochastic_variables",
        custom_getter=sv.make_stochastic_variable_getter(
            dist_cls=dist.NormalWithSoftplusScale,
            dist_kwargs={"validate_args": True},
            param_initializers={
                "loc": np.ones(
                    shape, dtype=np.float32) * 4.,
                "scale": sigma_init
            })):
      v = variable_scope.get_variable("sv", shape)

    for var in variables.global_variables():
      if "loc" in var.name:
        mu_var = var
      if "scale" in var.name:
        sigma_var = var

    v = ops.convert_to_tensor(v)
    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())
      self.assertAllEqual(np.ones(shape) * 4., sess.run(mu_var))
      self.assertAllEqual(np.ones(shape) * 2., sess.run(sigma_var))
      self.assertEqual(shape, sess.run(v).shape)

  def testStochasticVariablesWithPrior(self):
    shape = (10, 20)
    prior = dist.Normal(0., 1.)
    with variable_scope.variable_scope(
        "stochastic_variables",
        custom_getter=sv.make_stochastic_variable_getter(
            dist_cls=dist.NormalWithSoftplusScale, prior=prior)):
      w = variable_scope.get_variable("weights", shape)

    x = random_ops.random_uniform((8, 10))
    y = math_ops.matmul(x, w)

    prior_map = vi._find_variational_and_priors(y, None)
    self.assertEqual(prior_map[w], prior)
    elbo = vi.elbo(y, keep_batch_dim=False)

    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())
      sess.run(elbo)

  def testStochasticVariablesWithCallablePriorInitializer(self):

    def prior_init(shape, dtype):
      return dist.Normal(
          array_ops.zeros(shape, dtype), array_ops.ones(shape, dtype))

    with variable_scope.variable_scope(
        "stochastic_variables",
        custom_getter=sv.make_stochastic_variable_getter(
            dist_cls=dist.NormalWithSoftplusScale, prior=prior_init)):
      w = variable_scope.get_variable("weights", (10, 20))

    x = random_ops.random_uniform((8, 10))
    y = math_ops.matmul(x, w)

    prior_map = vi._find_variational_and_priors(y, None)
    self.assertTrue(isinstance(prior_map[w], dist.Normal))
    elbo = vi.elbo(y, keep_batch_dim=False)

    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())
      sess.run(elbo)


if __name__ == "__main__":
  test.main()
