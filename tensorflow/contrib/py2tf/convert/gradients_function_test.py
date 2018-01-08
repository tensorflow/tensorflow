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
"""Tests for gradients_function module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.eager.python import tfe
from tensorflow.contrib.py2tf.convert import gradients_function
from tensorflow.contrib.py2tf.pyct import compiler
from tensorflow.contrib.py2tf.pyct import parser
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import gradients_impl
from tensorflow.python.platform import test


class GradientsFunctionTest(test.TestCase):

  def test_transform(self):

    def loss(x, w):
      return x * w

    def test_fn(x, w):
      l, (dw,) = tfe.value_and_gradients_function(loss, [1])(x, w)  # pylint:disable=undefined-variable
      return l, dw

    node = parser.parse_object(test_fn)
    node = gradients_function.transform(node)
    result = compiler.ast_to_object(node)
    setattr(result, 'tf', gradients_impl)
    setattr(result, 'loss', loss)

    with self.test_session() as sess:
      self.assertEqual(
          (12, 3),
          sess.run(
              result.test_fn(constant_op.constant(3), constant_op.constant(4))))


if __name__ == '__main__':
  test.main()
