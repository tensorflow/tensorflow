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
"""Tests for Relaxed One-Hot Categorical distribution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.special import gamma

from tensorflow.contrib.distributions.python.ops import relaxed_onehot_categorical
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test


def make_relaxed_categorical(batch_shape, num_classes, dtype=dtypes.float32):
  logits = random_ops.random_uniform(
      list(batch_shape) + [num_classes], -10, 10, dtype=dtype) - 50.
  temperatures = random_ops.random_uniform(
      list(batch_shape), 0.1, 10, dtype=dtypes.float32)
  return relaxed_onehot_categorical.RelaxedOneHotCategorical(
      temperatures, logits, dtype=dtype)


class ExpRelaxedOneHotCategoricalTest(test.TestCase):

  def testP(self):
    temperature = 1.0
    logits = [2.0, 3.0, -4.0]
    dist = relaxed_onehot_categorical.ExpRelaxedOneHotCategorical(temperature,
                                                                  logits)
    expected_p = np.exp(logits)/np.sum(np.exp(logits))
    with self.cached_session():
      self.assertAllClose(expected_p, dist.probs.eval())
      self.assertAllEqual([3], dist.probs.get_shape())

  def testPdf(self):
    temperature = .4
    logits = [.3, .1, .4]
    k = len(logits)
    p = np.exp(logits)/np.sum(np.exp(logits))
    dist = relaxed_onehot_categorical.ExpRelaxedOneHotCategorical(temperature,
                                                                  logits)
    with self.cached_session():
      x = dist.sample().eval()
      # analytical ExpConcrete density presented in Maddison et al. 2016
      prod_term = p*np.exp(-temperature * x)
      expected_pdf = (gamma(k) * np.power(temperature, k-1) *
                      np.prod(prod_term/np.sum(prod_term)))
      pdf = dist.prob(x).eval()
      self.assertAllClose(expected_pdf, pdf)


class RelaxedOneHotCategoricalTest(test.TestCase):

  def testLogits(self):
    temperature = 1.0
    logits = [2.0, 3.0, -4.0]
    dist = relaxed_onehot_categorical.RelaxedOneHotCategorical(temperature,
                                                               logits)
    with self.cached_session():
      # check p for ExpRelaxed base distribution
      self.assertAllClose(logits, dist._distribution.logits.eval())
      self.assertAllEqual([3], dist._distribution.logits.get_shape())

  def testSample(self):
    temperature = 1.4
    with self.cached_session():
      # single logit
      logits = [.3, .1, .4]
      dist = relaxed_onehot_categorical.RelaxedOneHotCategorical(temperature,
                                                                 logits)
      self.assertAllEqual([3], dist.sample().eval().shape)
      self.assertAllEqual([5, 3], dist.sample(5).eval().shape)
      # multiple distributions
      logits = [[2.0, 3.0, -4.0], [.3, .1, .4]]
      dist = relaxed_onehot_categorical.RelaxedOneHotCategorical(temperature,
                                                                 logits)
      self.assertAllEqual([2, 3], dist.sample().eval().shape)
      self.assertAllEqual([5, 2, 3], dist.sample(5).eval().shape)
      # multiple distributions
      logits = np.random.uniform(size=(4, 1, 3)).astype(np.float32)
      dist = relaxed_onehot_categorical.RelaxedOneHotCategorical(temperature,
                                                                 logits)
      self.assertAllEqual([4, 1, 3], dist.sample().eval().shape)
      self.assertAllEqual([5, 4, 1, 3], dist.sample(5).eval().shape)

  def testPdf(self):
    def analytical_pdf(x, temperature, logits):
      # analytical density of RelaxedOneHotCategorical
      temperature = np.reshape(temperature, (-1, 1))
      if len(x.shape) == 1:
        x = np.expand_dims(x, 0)
      k = logits.shape[1]
      p = np.exp(logits)/np.sum(np.exp(logits), axis=1, keepdims=True)
      term1 = gamma(k)*np.power(temperature, k-1)
      term2 = np.sum(p/(np.power(x, temperature)), axis=1, keepdims=True)
      term3 = np.prod(p/(np.power(x, temperature+1)), axis=1, keepdims=True)
      expected_pdf = term1*np.power(term2, -k)*term3
      return expected_pdf

    with self.cached_session():
      temperature = .4
      logits = np.array([[.3, .1, .4]]).astype(np.float32)
      dist = relaxed_onehot_categorical.RelaxedOneHotCategorical(temperature,
                                                                 logits)
      x = dist.sample().eval()
      pdf = dist.prob(x).eval()
      expected_pdf = analytical_pdf(x, temperature, logits)
      self.assertAllClose(expected_pdf.flatten(), pdf, rtol=1e-4)

      # variable batch size
      logits = np.array([[.3, .1, .4], [.6, -.1, 2.]]).astype(np.float32)
      temperatures = np.array([0.4, 2.3]).astype(np.float32)
      dist = relaxed_onehot_categorical.RelaxedOneHotCategorical(temperatures,
                                                                 logits)
      x = dist.sample().eval()
      pdf = dist.prob(x).eval()
      expected_pdf = analytical_pdf(x, temperatures, logits)
      self.assertAllClose(expected_pdf.flatten(), pdf, rtol=1e-4)

  def testShapes(self):
    with self.cached_session():
      for batch_shape in ([], [1], [2, 3, 4]):
        dist = make_relaxed_categorical(batch_shape, 10)
        self.assertAllEqual(batch_shape, dist.batch_shape.as_list())
        self.assertAllEqual(batch_shape, dist.batch_shape_tensor().eval())
        self.assertAllEqual([10], dist.event_shape_tensor().eval())
        self.assertAllEqual([10], dist.event_shape_tensor().eval())

      for batch_shape in ([], [1], [2, 3, 4]):
        dist = make_relaxed_categorical(
            batch_shape, constant_op.constant(10, dtype=dtypes.int32))
        self.assertAllEqual(len(batch_shape), dist.batch_shape.ndims)
        self.assertAllEqual(batch_shape, dist.batch_shape_tensor().eval())
        self.assertAllEqual([10], dist.event_shape_tensor().eval())
        self.assertAllEqual([10], dist.event_shape_tensor().eval())

  def testUnknownShape(self):
    with self.cached_session():
      logits_pl = array_ops.placeholder(dtypes.float32)
      temperature = 1.0
      dist = relaxed_onehot_categorical.ExpRelaxedOneHotCategorical(temperature,
                                                                    logits_pl)
      with self.cached_session():
        feed_dict = {logits_pl: [.3, .1, .4]}
        self.assertAllEqual([3], dist.sample().eval(feed_dict=feed_dict).shape)
        self.assertAllEqual([5, 3],
                            dist.sample(5).eval(feed_dict=feed_dict).shape)

  def testDTypes(self):
    # check that sampling and log_prob work for a range of dtypes
    with self.cached_session():
      for dtype in (dtypes.float16, dtypes.float32, dtypes.float64):
        logits = random_ops.random_uniform(shape=[3, 3], dtype=dtype)
        dist = relaxed_onehot_categorical.RelaxedOneHotCategorical(
            temperature=0.5, logits=logits)
        dist.log_prob(dist.sample())

if __name__ == "__main__":
  test.main()
