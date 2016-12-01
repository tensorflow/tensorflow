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

linalg = tf.contrib.linalg


class LinearOperatorShape(linalg.LinearOperator):
  """LinearOperator that implements the methods ._shape and _shape_dynamic."""

  def __init__(self,
               shape,
               is_non_singular=None,
               is_self_adjoint=None,
               is_positive_definite=None):
    self._stored_shape = shape
    super(LinearOperatorShape, self).__init__(
        dtype=tf.float32,
        graph_parents=None,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,)

  def _shape(self):
    return tf.TensorShape(self._stored_shape)

  def _shape_dynamic(self):
    return tf.constant(self._stored_shape, dtype=tf.int32)


class LinearOperatorTest(tf.test.TestCase):

  def test_all_shape_properties_defined_by_the_one_property_shape(self):

    shape = (1, 2, 3, 4)
    operator = LinearOperatorShape(shape)

    self.assertAllEqual(shape, operator.shape)
    self.assertAllEqual(4, operator.tensor_rank)
    self.assertAllEqual((1, 2), operator.batch_shape)
    self.assertAllEqual(4, operator.domain_dimension)
    self.assertAllEqual(3, operator.range_dimension)

  def test_all_shape_methods_defined_by_the_one_method_shape(self):
    with self.test_session():
      shape = (1, 2, 3, 4)
      operator = LinearOperatorShape(shape)

      self.assertAllEqual(shape, operator.shape_dynamic().eval())
      self.assertAllEqual(4, operator.tensor_rank_dynamic().eval())
      self.assertAllEqual((1, 2), operator.batch_shape_dynamic().eval())
      self.assertAllEqual(4, operator.domain_dimension_dynamic().eval())
      self.assertAllEqual(3, operator.range_dimension_dynamic().eval())

  def test_is_x_properties(self):
    operator = LinearOperatorShape(
        shape=(2, 2),
        is_non_singular=False,
        is_self_adjoint=True,
        is_positive_definite=False)
    self.assertFalse(operator.is_non_singular)
    self.assertTrue(operator.is_self_adjoint)
    self.assertFalse(operator.is_positive_definite)


if __name__ == '__main__':
  tf.test.main()
