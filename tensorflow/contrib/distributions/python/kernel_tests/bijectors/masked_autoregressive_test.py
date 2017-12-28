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
from tensorflow.contrib.distributions.python.ops.bijectors.masked_autoregressive import _gen_mask
from tensorflow.contrib.distributions.python.ops.bijectors.masked_autoregressive import masked_autoregressive_default_template
from tensorflow.contrib.distributions.python.ops.bijectors.masked_autoregressive import MaskedAutoregressiveFlow
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.distributions import normal as normal_lib
from tensorflow.python.ops.distributions import transformed_distribution as transformed_distribution_lib
from tensorflow.python.platform import test


class GenMaskTest(test.TestCase):

  def test346Exclusive(self):
    expected_mask = np.array(
        [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [1, 0, 0, 0],
         [1, 0, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 0, 0]])
    mask = _gen_mask(num_blocks=3, n_in=4, n_out=6, mask_type="exclusive")
    self.assertAllEqual(expected_mask, mask)

  def test346Inclusive(self):
    expected_mask = np.array(
        [[1, 0, 0, 0],
         [1, 0, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 0],
         [1, 1, 1, 0]])
    mask = _gen_mask(num_blocks=3, n_in=4, n_out=6, mask_type="inclusive")
    self.assertAllEqual(expected_mask, mask)


class MaskedAutoregressiveFlowTest(test_util.VectorDistributionTestHelpers,
                                   test.TestCase):

  @property
  def _autoregressive_flow_kwargs(self):
    return {
        "shift_and_log_scale_fn": masked_autoregressive_default_template(
            hidden_layers=[2], shift_only=False),
        "is_constant_jacobian": False,
    }

  def testBijector(self):
    x_ = np.arange(3 * 4 * 2).astype(np.float32).reshape(3, 4, 2)
    with self.test_session() as sess:
      ma = MaskedAutoregressiveFlow(
          validate_args=True,
          **self._autoregressive_flow_kwargs)
      x = constant_op.constant(x_)
      forward_x = ma.forward(x)
      # Use identity to invalidate cache.
      inverse_y = ma.inverse(array_ops.identity(forward_x))
      fldj = ma.forward_log_det_jacobian(x)
      # Use identity to invalidate cache.
      ildj = ma.inverse_log_det_jacobian(array_ops.identity(forward_x))
      variables.global_variables_initializer().run()
      [
          forward_x_,
          inverse_y_,
          ildj_,
          fldj_,
      ] = sess.run([
          forward_x,
          inverse_y,
          ildj,
          fldj,
      ])
      self.assertEqual("masked_autoregressive_flow", ma.name)
      self.assertAllClose(forward_x_, forward_x_, rtol=1e-6, atol=0.)
      self.assertAllClose(x_, inverse_y_, rtol=1e-5, atol=0.)
      self.assertAllClose(ildj_, -fldj_, rtol=1e-6, atol=0.)

  def testMutuallyConsistent(self):
    dims = 4
    with self.test_session() as sess:
      ma = MaskedAutoregressiveFlow(
          validate_args=True,
          **self._autoregressive_flow_kwargs)
      dist = transformed_distribution_lib.TransformedDistribution(
          distribution=normal_lib.Normal(loc=0., scale=1.),
          bijector=ma,
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
      ma = Invert(MaskedAutoregressiveFlow(
          validate_args=True,
          **self._autoregressive_flow_kwargs))
      dist = transformed_distribution_lib.TransformedDistribution(
          distribution=normal_lib.Normal(loc=0., scale=1.),
          bijector=ma,
          event_shape=[dims],
          validate_args=True)
      self.run_test_sample_consistent_log_prob(
          sess_run_fn=sess.run,
          dist=dist,
          num_samples=int(1e5),
          radius=1.,
          center=0.,
          rtol=0.02)


class MaskedAutoregressiveFlowShiftOnlyTest(MaskedAutoregressiveFlowTest):

  @property
  def _autoregressive_flow_kwargs(self):
    return {
        "shift_and_log_scale_fn": masked_autoregressive_default_template(
            hidden_layers=[2], shift_only=True),
        "is_constant_jacobian": True,
    }


if __name__ == "__main__":
  test.main()
