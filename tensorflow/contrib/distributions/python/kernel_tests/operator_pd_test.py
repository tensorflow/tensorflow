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
import tensorflow as tf

distributions = tf.contrib.distributions


class BaseCovarianceTest(tf.test.TestCase):

  def test_all_shapes_methods_defined_by_the_one_abstractproperty_shape(self):

    class OperatorShape(distributions.OperatorPDBase):
      """Operator implements the ABC method .shape."""

      def __init__(self, shape):
        self._shape = shape

      @property
      def verify_pd(self):
        return True

      def get_shape(self):
        return tf.TensorShape(self._shape)

      def shape(self, name='shape'):
        return tf.shape(np.random.rand(*self._shape))

      @property
      def name(self):
        return 'OperatorShape'

      def dtype(self):
        raise tf.int32

      def inv_quadratic_form(
          self, x, name='inv_quadratic_form'):
        return x

      def log_det(self, name='log_det'):
        raise NotImplementedError()

      @property
      def inputs(self):
        return []

      def sqrt_matmul(self, x, name='sqrt_matmul'):
        return x

    shape = (1, 2, 3, 3)
    with self.test_session():
      operator = OperatorShape(shape)

      self.assertAllEqual(shape, operator.shape().eval())
      self.assertAllEqual(4, operator.rank().eval())
      self.assertAllEqual((1, 2), operator.batch_shape().eval())
      self.assertAllEqual((1, 2, 3), operator.vector_shape().eval())
      self.assertAllEqual(3, operator.vector_space_dimension().eval())

      self.assertEqual(shape, operator.get_shape())
      self.assertEqual((1, 2), operator.get_batch_shape())
      self.assertEqual((1, 2, 3), operator.get_vector_shape())


if __name__ == '__main__':
  tf.test.main()
