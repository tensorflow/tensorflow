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
"""Tests for ConditionalTransformedDistribution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib import distributions
from tensorflow.contrib.distributions.python.kernel_tests import transformed_distribution_test
from tensorflow.contrib.distributions.python.ops import conditional_bijector
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

ds = distributions


class _ChooseLocation(conditional_bijector.ConditionalBijector):
  """A Bijector which chooses between one of two location parameters."""

  def __init__(self, loc, name="ChooseLocation"):
    self._graph_parents = []
    self._name = name
    with self._name_scope("init", values=[loc]):
      self._loc = ops.convert_to_tensor(loc, name="loc")
      super(_ChooseLocation, self).__init__(
          graph_parents=[self._loc],
          is_constant_jacobian=True,
          validate_args=False,
          name=name)

  def _forward(self, x, z):
    return x + self._gather_loc(z)

  def _inverse(self, x, z):
    return x - self._gather_loc(z)

  def _inverse_log_det_jacobian(self, x, z=None):
    return 0.

  def _gather_loc(self, z):
    z = ops.convert_to_tensor(z)
    z = math_ops.cast((1 + z) / 2, dtypes.int32)
    return array_ops.gather(self._loc, z)


class ConditionalTransformedDistributionTest(
    transformed_distribution_test.TransformedDistributionTest):

  def _cls(self):
    return ds.ConditionalTransformedDistribution

  def testConditioning(self):
    with self.test_session():
      conditional_normal = ds.ConditionalTransformedDistribution(
          distribution=ds.Normal(loc=0., scale=1.),
          bijector=_ChooseLocation(loc=[-100., 100.]))
      z = [-1, +1, -1, -1, +1]
      self.assertAllClose(
          np.sign(conditional_normal.sample(
              5, bijector_kwargs={"z": z}).eval()), z)


class ConditionalScalarToMultiTest(
    transformed_distribution_test.ScalarToMultiTest):

  def _cls(self):
    return ds.ConditionalTransformedDistribution


if __name__ == "__main__":
  test.main()
