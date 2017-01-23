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

from tensorflow.contrib import distributions
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test

ds = distributions


class DistributionTest(test.TestCase):

  def testParamShapesAndFromParams(self):
    classes = [
        ds.Normal,
        ds.Bernoulli,
        ds.Beta,
        ds.Chi2,
        ds.Exponential,
        ds.Gamma,
        ds.InverseGamma,
        ds.Laplace,
        ds.StudentT,
        ds.Uniform,
    ]

    sample_shapes = [(), (10,), (10, 20, 30)]
    with self.test_session():
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
    with self.test_session():
      # Note: we cannot easily test all distributions since each requires
      # different initialization arguments. We therefore spot test a few.
      normal = ds.Normal(mu=1., sigma=2., validate_args=True)
      self.assertEqual(normal.parameters, normal.copy().parameters)
      wishart = ds.WishartFull(df=2, scale=[[1., 2], [2, 5]],
                               validate_args=True)
      self.assertEqual(wishart.parameters, wishart.copy().parameters)

  def testCopyOverride(self):
    with self.test_session():
      normal = ds.Normal(mu=1., sigma=2., validate_args=True)
      normal_copy = normal.copy(validate_args=False)
      base_params = normal.parameters.copy()
      copy_params = normal.copy(validate_args=False).parameters.copy()
      self.assertNotEqual(
          base_params.pop("validate_args"), copy_params.pop("validate_args"))
      self.assertEqual(base_params, copy_params)

  def testIsScalar(self):
    with self.test_session():
      mu = 1.
      sigma = 2.

      normal = ds.Normal(mu, sigma, validate_args=True)
      self.assertTrue(tensor_util.constant_value(normal.is_scalar_event()))
      self.assertTrue(tensor_util.constant_value(normal.is_scalar_batch()))

      normal = ds.Normal([mu], [sigma], validate_args=True)
      self.assertTrue(tensor_util.constant_value(normal.is_scalar_event()))
      self.assertFalse(tensor_util.constant_value(normal.is_scalar_batch()))

      mvn = ds.MultivariateNormalDiag([mu], [sigma], validate_args=True)
      self.assertFalse(tensor_util.constant_value(mvn.is_scalar_event()))
      self.assertTrue(tensor_util.constant_value(mvn.is_scalar_batch()))

      mvn = ds.MultivariateNormalDiag([[mu]], [[sigma]], validate_args=True)
      self.assertFalse(tensor_util.constant_value(mvn.is_scalar_event()))
      self.assertFalse(tensor_util.constant_value(mvn.is_scalar_batch()))

      # We now test every codepath within the underlying is_scalar_helper
      # function.

      # Test case 1, 2.
      x = array_ops.placeholder(dtype=dtypes.int32, shape=[])
      # None would fire an exception were it actually executed.
      self.assertTrue(normal._is_scalar_helper(x.get_shape, lambda: None))
      self.assertTrue(
          normal._is_scalar_helper(lambda: tensor_shape.TensorShape(None),
                                   lambda: array_ops.shape(x)))

      x = array_ops.placeholder(dtype=dtypes.int32, shape=[1])
      # None would fire an exception were it actually executed.
      self.assertFalse(normal._is_scalar_helper(x.get_shape, lambda: None))
      self.assertFalse(
          normal._is_scalar_helper(lambda: tensor_shape.TensorShape(None),
                                   lambda: array_ops.shape(x)))

      # Test case 3.
      x = array_ops.placeholder(dtype=dtypes.int32)
      is_scalar = normal._is_scalar_helper(x.get_shape,
                                           lambda: array_ops.shape(x))
      self.assertTrue(is_scalar.eval(feed_dict={x: 1}))
      self.assertFalse(is_scalar.eval(feed_dict={x: [1]}))

  def testSampleShapeHints(self):
    class _FakeDistribution(ds.Distribution):
      """Fake Distribution for testing _set_sample_static_shape."""

      def __init__(self, batch_shape=None, event_shape=None):
        self._static_batch_shape = tensor_shape.TensorShape(batch_shape)
        self._static_event_shape = tensor_shape.TensorShape(event_shape)
        super(_FakeDistribution, self).__init__(
            dtype=dtypes.float32,
            is_continuous=False,
            is_reparameterized=False,
            validate_args=True,
            allow_nan_stats=True,
            name="DummyDistribution")

      def _get_batch_shape(self):
        return self._static_batch_shape

      def _get_event_shape(self):
        return self._static_event_shape

    with self.test_session():
      # Make a new session since we're playing with static shapes. [And below.]
      x = array_ops.placeholder(dtype=dtypes.float32)
      dist = _FakeDistribution(batch_shape=[2, 3], event_shape=[5])
      sample_shape = ops.convert_to_tensor([6, 7], dtype=dtypes.int32)
      y = dist._set_sample_static_shape(x, sample_shape)
      # We use as_list since TensorShape comparison does not work correctly for
      # unknown values, ie, Dimension(None).
      self.assertAllEqual([6, 7, 2, 3, 5], y.get_shape().as_list())

    with self.test_session():
      x = array_ops.placeholder(dtype=dtypes.float32)
      dist = _FakeDistribution(batch_shape=[None, 3], event_shape=[5])
      sample_shape = ops.convert_to_tensor([6, 7], dtype=dtypes.int32)
      y = dist._set_sample_static_shape(x, sample_shape)
      self.assertAllEqual([6, 7, None, 3, 5], y.get_shape().as_list())

    with self.test_session():
      x = array_ops.placeholder(dtype=dtypes.float32)
      dist = _FakeDistribution(batch_shape=[None, 3], event_shape=[None])
      sample_shape = ops.convert_to_tensor([6, 7], dtype=dtypes.int32)
      y = dist._set_sample_static_shape(x, sample_shape)
      self.assertAllEqual([6, 7, None, 3, None], y.get_shape().as_list())

    with self.test_session():
      x = array_ops.placeholder(dtype=dtypes.float32)
      dist = _FakeDistribution(batch_shape=None, event_shape=None)
      sample_shape = ops.convert_to_tensor([6, 7], dtype=dtypes.int32)
      y = dist._set_sample_static_shape(x, sample_shape)
      self.assertTrue(y.get_shape().ndims is None)

    with self.test_session():
      x = array_ops.placeholder(dtype=dtypes.float32)
      dist = _FakeDistribution(batch_shape=[None, 3], event_shape=None)
      sample_shape = ops.convert_to_tensor([6, 7], dtype=dtypes.int32)
      y = dist._set_sample_static_shape(x, sample_shape)
      self.assertTrue(y.get_shape().ndims is None)


if __name__ == "__main__":
  test.main()
