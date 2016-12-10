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

import tensorflow as tf

dists = tf.contrib.distributions


class DistributionTest(tf.test.TestCase):

  def testParamShapesAndFromParams(self):
    classes = [
        dists.Normal,
        dists.Bernoulli,
        dists.Beta,
        dists.Chi2,
        dists.Exponential,
        dists.Gamma,
        dists.InverseGamma,
        dists.Laplace,
        dists.StudentT,
        dists.Uniform]

    sample_shapes = [(), (10,), (10, 20, 30)]
    with self.test_session():
      for cls in classes:
        for sample_shape in sample_shapes:
          param_shapes = cls.param_shapes(sample_shape)
          params = dict([(name, tf.random_normal(shape))
                         for name, shape in param_shapes.items()])
          dist = cls(**params)
          self.assertAllEqual(sample_shape, tf.shape(dist.sample()).eval())
          dist_copy = dist.copy()
          self.assertAllEqual(sample_shape,
                              tf.shape(dist_copy.sample()).eval())
          self.assertEqual(dist.parameters, dist_copy.parameters)

  def testCopyExtraArgs(self):
    with self.test_session():
      # Note: we cannot easily test all distributions since each requires
      # different initialization arguments. We therefore spot test a few.
      normal = dists.Normal(mu=1., sigma=2., validate_args=True)
      self.assertEqual(normal.parameters, normal.copy().parameters)
      wishart = dists.WishartFull(df=2, scale=[[1., 2], [2, 5]],
                                  validate_args=True)
      self.assertEqual(wishart.parameters, wishart.copy().parameters)

  def testCopyOverride(self):
    with self.test_session():
      normal = dists.Normal(mu=1., sigma=2., validate_args=True)
      normal_copy = normal.copy(validate_args=False)
      base_params = normal.parameters.copy()
      copy_params = normal.copy(validate_args=False).parameters.copy()
      self.assertNotEqual(base_params.pop("validate_args"),
                          copy_params.pop("validate_args"))
      self.assertEqual(base_params, copy_params)


if __name__ == '__main__':
  tf.test.main()
