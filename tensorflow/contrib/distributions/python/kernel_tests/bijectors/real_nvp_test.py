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
"""Tests for MaskedAutoregressiveFlow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensorflow.contrib.distributions.python.ops import test_util
from tensorflow.contrib.distributions.python.ops.bijectors.invert import Invert
from tensorflow.contrib.distributions.python.ops.bijectors.real_nvp import real_nvp_default_template
from tensorflow.contrib.distributions.python.ops.bijectors.real_nvp import RealNVP
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.distributions import normal as normal_lib
from tensorflow.python.ops.distributions import transformed_distribution as transformed_distribution_lib
from tensorflow.python.platform import test


class RealNVPTest(test_util.VectorDistributionTestHelpers, test.TestCase):

  @property
  def _real_nvp_kwargs(self):
    return {
        "shift_and_log_scale_fn": real_nvp_default_template(
            hidden_layers=[3], shift_only=False),
        "is_constant_jacobian": False,
    }

  def testBijector(self):
    x_ = np.arange(3 * 4 * 2).astype(np.float32).reshape(3, 4 * 2)
    with self.test_session() as sess:
      nvp = RealNVP(
          num_masked=4,
          validate_args=True,
          **self._real_nvp_kwargs)
      x = constant_op.constant(x_)
      forward_x = nvp.forward(x)
      # Use identity to invalidate cache.
      inverse_y = nvp.inverse(array_ops.identity(forward_x))
      forward_inverse_y = nvp.forward(inverse_y)
      fldj = nvp.forward_log_det_jacobian(x, event_ndims=1)
      # Use identity to invalidate cache.
      ildj = nvp.inverse_log_det_jacobian(
          array_ops.identity(forward_x), event_ndims=1)
      variables.global_variables_initializer().run()
      [
          forward_x_,
          inverse_y_,
          forward_inverse_y_,
          ildj_,
          fldj_,
      ] = sess.run([
          forward_x,
          inverse_y,
          forward_inverse_y,
          ildj,
          fldj,
      ])
      self.assertEqual("real_nvp", nvp.name)
      self.assertAllClose(forward_x_, forward_inverse_y_, rtol=1e-1, atol=0.)
      self.assertAllClose(x_, inverse_y_, rtol=1e-1, atol=0.)
      self.assertAllClose(ildj_, -fldj_, rtol=1e-6, atol=0.)

  def testMutuallyConsistent(self):
    dims = 4
    with self.test_session() as sess:
      nvp = RealNVP(
          num_masked=3,
          validate_args=True,
          **self._real_nvp_kwargs)
      dist = transformed_distribution_lib.TransformedDistribution(
          distribution=normal_lib.Normal(loc=0., scale=1.),
          bijector=nvp,
          event_shape=[dims],
          validate_args=True)
      self.run_test_sample_consistent_log_prob(
          sess_run_fn=sess.run,
          dist=dist,
          num_samples=int(1e5),
          radius=1.,
          center=0.,
          rtol=0.02)

  def testInvertMutuallyConsistent(self):
    dims = 4
    with self.test_session() as sess:
      nvp = Invert(RealNVP(
          num_masked=3,
          validate_args=True,
          **self._real_nvp_kwargs))
      dist = transformed_distribution_lib.TransformedDistribution(
          distribution=normal_lib.Normal(loc=0., scale=1.),
          bijector=nvp,
          event_shape=[dims],
          validate_args=True)
      self.run_test_sample_consistent_log_prob(
          sess_run_fn=sess.run,
          dist=dist,
          num_samples=int(1e5),
          radius=1.,
          center=0.,
          rtol=0.02)


class NICETest(RealNVPTest):

  @property
  def _real_nvp_kwargs(self):
    return {
        "shift_and_log_scale_fn": real_nvp_default_template(
            hidden_layers=[2], shift_only=True),
        "is_constant_jacobian": True,
    }


class RealNVPConstantShiftScaleTest(RealNVPTest):

  @property
  def _real_nvp_kwargs(self):

    def constant_shift_log_scale_fn(x0, output_units):
      del x0, output_units
      shift = constant_op.constant([0.1])
      log_scale = constant_op.constant([0.5])
      return shift, log_scale

    return {
        "shift_and_log_scale_fn": constant_shift_log_scale_fn,
        "is_constant_jacobian": True,
    }

if __name__ == "__main__":
  test.main()
