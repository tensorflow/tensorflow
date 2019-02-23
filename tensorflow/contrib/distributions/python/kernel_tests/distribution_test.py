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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib import distributions
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test

tfd = distributions


class DistributionTest(test.TestCase):

  def testParamShapesAndFromParams(self):
    classes = [
        tfd.Normal,
        tfd.Bernoulli,
        tfd.Beta,
        tfd.Chi2,
        tfd.Exponential,
        tfd.Gamma,
        tfd.InverseGamma,
        tfd.Laplace,
        tfd.StudentT,
        tfd.Uniform,
    ]

    sample_shapes = [(), (10,), (10, 20, 30)]
    with self.cached_session():
      for cls in classes:
        for sample_shape in sample_shapes:
          param_shapes = cls.param_shapes(sample_shape)
          params = dict([(name, random_ops.random_normal(shape))
                         for name, shape in param_shapes.items()])
          dist = cls(**params)
          self.assertAllEqual(sample_shape,
                              array_ops.shape(dist.sample()).eval())
          dist_copy = dist.copy()
          self.assertAllEqual(sample_shape,
                              array_ops.shape(dist_copy.sample()).eval())
          self.assertEqual(dist.parameters, dist_copy.parameters)

  def testCopyExtraArgs(self):
    with self.cached_session():
      # Note: we cannot easily test all distributions since each requires
      # different initialization arguments. We therefore spot test a few.
      normal = tfd.Normal(loc=1., scale=2., validate_args=True)
      self.assertEqual(normal.parameters, normal.copy().parameters)
      wishart = tfd.WishartFull(df=2, scale=[[1., 2], [2, 5]],
                                validate_args=True)
      self.assertEqual(wishart.parameters, wishart.copy().parameters)

  def testCopyOverride(self):
    with self.cached_session():
      normal = tfd.Normal(loc=1., scale=2., validate_args=True)
      unused_normal_copy = normal.copy(validate_args=False)
      base_params = normal.parameters.copy()
      copy_params = normal.copy(validate_args=False).parameters.copy()
      self.assertNotEqual(
          base_params.pop("validate_args"), copy_params.pop("validate_args"))
      self.assertEqual(base_params, copy_params)

  def testIsScalar(self):
    with self.cached_session():
      mu = 1.
      sigma = 2.

      normal = tfd.Normal(mu, sigma, validate_args=True)
      self.assertTrue(tensor_util.constant_value(normal.is_scalar_event()))
      self.assertTrue(tensor_util.constant_value(normal.is_scalar_batch()))

      normal = tfd.Normal([mu], [sigma], validate_args=True)
      self.assertTrue(tensor_util.constant_value(normal.is_scalar_event()))
      self.assertFalse(tensor_util.constant_value(normal.is_scalar_batch()))

      mvn = tfd.MultivariateNormalDiag([mu], [sigma], validate_args=True)
      self.assertFalse(tensor_util.constant_value(mvn.is_scalar_event()))
      self.assertTrue(tensor_util.constant_value(mvn.is_scalar_batch()))

      mvn = tfd.MultivariateNormalDiag([[mu]], [[sigma]], validate_args=True)
      self.assertFalse(tensor_util.constant_value(mvn.is_scalar_event()))
      self.assertFalse(tensor_util.constant_value(mvn.is_scalar_batch()))

      # We now test every codepath within the underlying is_scalar_helper
      # function.

      # Test case 1, 2.
      x = array_ops.placeholder(dtype=dtypes.int32, shape=[])
      # None would fire an exception were it actually executed.
      self.assertTrue(normal._is_scalar_helper(x.get_shape(), lambda: None))
      self.assertTrue(
          normal._is_scalar_helper(tensor_shape.TensorShape(None),
                                   lambda: array_ops.shape(x)))

      x = array_ops.placeholder(dtype=dtypes.int32, shape=[1])
      # None would fire an exception were it actually executed.
      self.assertFalse(normal._is_scalar_helper(x.get_shape(), lambda: None))
      self.assertFalse(
          normal._is_scalar_helper(tensor_shape.TensorShape(None),
                                   lambda: array_ops.shape(x)))

      # Test case 3.
      x = array_ops.placeholder(dtype=dtypes.int32)
      is_scalar = normal._is_scalar_helper(x.get_shape(),
                                           lambda: array_ops.shape(x))
      self.assertTrue(is_scalar.eval(feed_dict={x: 1}))
      self.assertFalse(is_scalar.eval(feed_dict={x: [1]}))

  def _GetFakeDistribution(self):
    class FakeDistribution(tfd.Distribution):
      """Fake Distribution for testing _set_sample_static_shape."""

      def __init__(self, batch_shape=None, event_shape=None):
        self._static_batch_shape = tensor_shape.TensorShape(batch_shape)
        self._static_event_shape = tensor_shape.TensorShape(event_shape)
        super(FakeDistribution, self).__init__(
            dtype=dtypes.float32,
            reparameterization_type=distributions.NOT_REPARAMETERIZED,
            validate_args=True,
            allow_nan_stats=True,
            name="DummyDistribution")

      def _batch_shape(self):
        return self._static_batch_shape

      def _event_shape(self):
        return self._static_event_shape

    return FakeDistribution

  def testSampleShapeHints(self):
    fake_distribution = self._GetFakeDistribution()

    with self.cached_session():
      # Make a new session since we're playing with static shapes. [And below.]
      x = array_ops.placeholder(dtype=dtypes.float32)
      dist = fake_distribution(batch_shape=[2, 3], event_shape=[5])
      sample_shape = ops.convert_to_tensor([6, 7], dtype=dtypes.int32)
      y = dist._set_sample_static_shape(x, sample_shape)
      # We use as_list since TensorShape comparison does not work correctly for
      # unknown values, ie, Dimension(None).
      self.assertAllEqual([6, 7, 2, 3, 5], y.get_shape().as_list())

    with self.cached_session():
      x = array_ops.placeholder(dtype=dtypes.float32)
      dist = fake_distribution(batch_shape=[None, 3], event_shape=[5])
      sample_shape = ops.convert_to_tensor([6, 7], dtype=dtypes.int32)
      y = dist._set_sample_static_shape(x, sample_shape)
      self.assertAllEqual([6, 7, None, 3, 5], y.get_shape().as_list())

    with self.cached_session():
      x = array_ops.placeholder(dtype=dtypes.float32)
      dist = fake_distribution(batch_shape=[None, 3], event_shape=[None])
      sample_shape = ops.convert_to_tensor([6, 7], dtype=dtypes.int32)
      y = dist._set_sample_static_shape(x, sample_shape)
      self.assertAllEqual([6, 7, None, 3, None], y.get_shape().as_list())

    with self.cached_session():
      x = array_ops.placeholder(dtype=dtypes.float32)
      dist = fake_distribution(batch_shape=None, event_shape=None)
      sample_shape = ops.convert_to_tensor([6, 7], dtype=dtypes.int32)
      y = dist._set_sample_static_shape(x, sample_shape)
      self.assertTrue(y.get_shape().ndims is None)

    with self.cached_session():
      x = array_ops.placeholder(dtype=dtypes.float32)
      dist = fake_distribution(batch_shape=[None, 3], event_shape=None)
      sample_shape = ops.convert_to_tensor([6, 7], dtype=dtypes.int32)
      y = dist._set_sample_static_shape(x, sample_shape)
      self.assertTrue(y.get_shape().ndims is None)

  def testNameScopeWorksCorrectly(self):
    x = tfd.Normal(loc=0., scale=1., name="x")
    x_duplicate = tfd.Normal(loc=0., scale=1., name="x")
    with ops.name_scope("y") as name:
      y = tfd.Bernoulli(logits=0., name=name)
    x_sample = x.sample(name="custom_sample")
    x_sample_duplicate = x.sample(name="custom_sample")
    x_log_prob = x.log_prob(0., name="custom_log_prob")
    x_duplicate_sample = x_duplicate.sample(name="custom_sample")

    self.assertEqual(x.name, "x/")
    self.assertEqual(x_duplicate.name, "x_1/")
    self.assertEqual(y.name, "y/")
    self.assertTrue(x_sample.name.startswith("x/custom_sample"))
    self.assertTrue(x_sample_duplicate.name.startswith("x/custom_sample_1"))
    self.assertTrue(x_log_prob.name.startswith("x/custom_log_prob"))
    self.assertTrue(x_duplicate_sample.name.startswith(
        "x_1/custom_sample"))

  def testStrWorksCorrectlyScalar(self):
    normal = tfd.Normal(loc=np.float16(0), scale=np.float16(1))
    self.assertEqual(
        ("tfp.distributions.Normal("
         "\"Normal/\", "
         "batch_shape=(), "
         "event_shape=(), "
         "dtype=float16)"),  # Got the dtype right.
        str(normal))

    chi2 = tfd.Chi2(df=np.float32([1., 2.]), name="silly")
    self.assertEqual(
        ("tfp.distributions.Chi2("
         "\"silly/\", "  # What a silly name that is!
         "batch_shape=(2,), "
         "event_shape=(), "
         "dtype=float32)"),
        str(chi2))

    exp = tfd.Exponential(rate=array_ops.placeholder(dtype=dtypes.float32))
    self.assertEqual(
        ("tfp.distributions.Exponential(\"Exponential/\", "
         # No batch shape.
         "event_shape=(), "
         "dtype=float32)"),
        str(exp))

  def testStrWorksCorrectlyMultivariate(self):
    mvn_static = tfd.MultivariateNormalDiag(
        loc=np.zeros([2, 2]), name="MVN")
    self.assertEqual(
        ("tfp.distributions.MultivariateNormalDiag("
         "\"MVN/\", "
         "batch_shape=(2,), "
         "event_shape=(2,), "
         "dtype=float64)"),
        str(mvn_static))

    mvn_dynamic = tfd.MultivariateNormalDiag(
        loc=array_ops.placeholder(shape=[None, 3], dtype=dtypes.float32),
        name="MVN2")
    if mvn_dynamic.batch_shape._v2_behavior:
      self.assertEqual(
          ("tfp.distributions.MultivariateNormalDiag("
           "\"MVN2/\", "
           "batch_shape=(None,), "  # Partially known.
           "event_shape=(3,), "
           "dtype=float32)"),
          str(mvn_dynamic))
    else:
      self.assertEqual(
          ("tfp.distributions.MultivariateNormalDiag("
           "\"MVN2/\", "
           "batch_shape=(?,), "  # Partially known.
           "event_shape=(3,), "
           "dtype=float32)"),
          str(mvn_dynamic))

  def testReprWorksCorrectlyScalar(self):
    normal = tfd.Normal(loc=np.float16(0), scale=np.float16(1))
    self.assertEqual(
        ("<tfp.distributions.Normal"
         " 'Normal/'"
         " batch_shape=()"
         " event_shape=()"
         " dtype=float16>"),  # Got the dtype right.
        repr(normal))

    chi2 = tfd.Chi2(df=np.float32([1., 2.]), name="silly")
    self.assertEqual(
        ("<tfp.distributions.Chi2"
         " 'silly/'"  # What a silly name that is!
         " batch_shape=(2,)"
         " event_shape=()"
         " dtype=float32>"),
        repr(chi2))

    exp = tfd.Exponential(rate=array_ops.placeholder(dtype=dtypes.float32))
    self.assertEqual(
        ("<tfp.distributions.Exponential"
         " 'Exponential/'"
         " batch_shape=<unknown>"
         " event_shape=()"
         " dtype=float32>"),
        repr(exp))

  def testReprWorksCorrectlyMultivariate(self):
    mvn_static = tfd.MultivariateNormalDiag(
        loc=np.zeros([2, 2]), name="MVN")
    self.assertEqual(
        ("<tfp.distributions.MultivariateNormalDiag"
         " 'MVN/'"
         " batch_shape=(2,)"
         " event_shape=(2,)"
         " dtype=float64>"),
        repr(mvn_static))

    mvn_dynamic = tfd.MultivariateNormalDiag(
        loc=array_ops.placeholder(shape=[None, 3], dtype=dtypes.float32),
        name="MVN2")
    if mvn_dynamic.batch_shape._v2_behavior:
      self.assertEqual(
          ("<tfp.distributions.MultivariateNormalDiag"
           " 'MVN2/'"
           " batch_shape=(None,)"  # Partially known.
           " event_shape=(3,)"
           " dtype=float32>"),
          repr(mvn_dynamic))
    else:
      self.assertEqual(
          ("<tfp.distributions.MultivariateNormalDiag"
           " 'MVN2/'"
           " batch_shape=(?,)"  # Partially known.
           " event_shape=(3,)"
           " dtype=float32>"),
          repr(mvn_dynamic))


if __name__ == "__main__":
  test.main()
