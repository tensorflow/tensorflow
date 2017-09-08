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
"""Tests for estimator.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six

from tensorflow.contrib.distributions.python.ops import estimator as estimator_lib
from tensorflow.contrib.learn.python.learn.estimators import constants
from tensorflow.contrib.learn.python.learn.estimators import head as head_lib
from tensorflow.contrib.learn.python.learn.estimators import model_fn
from tensorflow.contrib.learn.python.learn.estimators.head_test import _assert_metrics
from tensorflow.contrib.learn.python.learn.estimators.head_test import _assert_no_variables
from tensorflow.contrib.learn.python.learn.estimators.head_test import _assert_summary_tags
from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.distributions import normal as normal_lib
from tensorflow.python.platform import test


class EstimatorHeadDistributionRegressionTest(test.TestCase):

  def _assert_output_alternatives(self, model_fn_ops):
    self.assertEquals({
        None: constants.ProblemType.LINEAR_REGRESSION
    }, {
        k: v[0] for k, v in six.iteritems(model_fn_ops.output_alternatives)
    })

  def testNormalLocScaleLogits(self):
    # We will bias logits[..., 1] so that: logits[..., 1]=0 implies scale=1.
    scale_bias = np.log(np.expm1(1.))

    def softplus(x):
      return np.log1p(np.exp(x))

    def actual_loss(logits, labels):
      mu = actual_mean(logits)
      sigma = actual_stddev(logits)
      labels = np.squeeze(labels, -1)
      z = (labels - mu) / sigma
      loss = 0.5 * (z**2. + np.log(2. * np.pi)) + np.log(sigma)
      return loss.mean()

    def actual_mean(logits):
      return logits[..., 0]

    def actual_stddev(logits):
      return softplus(logits[..., 1] + scale_bias)

    def make_distribution_fn(logits):
      return normal_lib.Normal(
          loc=logits[..., 0],
          scale=nn_ops.softplus(logits[..., 1] + scale_bias))

    head = estimator_lib.estimator_head_distribution_regression(
        make_distribution_fn,
        logits_dimension=2)
    labels = np.float32([[-1.],
                         [0.],
                         [1.]])
    logits = np.float32([[0., -1],
                         [1, 0.5],
                         [-1, 1]])
    with ops.Graph().as_default(), session.Session():
      # Convert to tensor so we can index into head.distributions.
      tflogits = ops.convert_to_tensor(logits, name="logits")
      model_fn_ops = head.create_model_fn_ops(
          {},
          labels=labels,
          mode=model_fn.ModeKeys.TRAIN,
          train_op_fn=head_lib.no_op_train_fn,
          logits=tflogits)
      self._assert_output_alternatives(model_fn_ops)
      _assert_summary_tags(self, ["loss"])
      _assert_no_variables(self)
      loss = actual_loss(logits, labels)
      _assert_metrics(self, loss, {"loss": loss}, model_fn_ops)

      # Now we verify the underlying distribution was correctly constructed.
      expected_mean = logits[..., 0]
      self.assertAllClose(
          expected_mean,
          head.distribution(tflogits).mean().eval(),
          rtol=1e-6, atol=0.)

      expected_stddev = softplus(logits[..., 1] + scale_bias)
      self.assertAllClose(
          expected_stddev,
          head.distribution(tflogits).stddev().eval(),
          rtol=1e-6, atol=0.)
      # Should have created only one distribution.
      self.assertEqual(1, len(head.distributions))


if __name__ == "__main__":
  test.main()
