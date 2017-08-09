# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for modrelu.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.nn.python.ops import mod_relu
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

class ModreluTest(test.TestCase):

  def testModRelu(self):
    with self.test_session() as sess:
      input_real = constant_op.constant(
          [
              [2.5, 0.5, 0.3, 0.4],
              [0.7, 0, -0.3, 0.2]
          ]
      )
      input_imag = constant_op.constant(
          [
              [0.2, 0.0, 0.5, 0.7],
              [0.5, -2.5, 0.0, 0.8]
          ]
      )
      input_complex = math_ops.complex(input_real, input_imag)
      biases = constant_op.constant([0.0, 1.0, 0.0, -1.0])
      computed = sess.run(mod_relu.modrelu(input_complex, biases))

      # Compute the expected target value.
      target_real = constant_op.constant(
          [
              [2.5, 1.5, 0.3, 0.0],
              [0.7, 0, -0.3, 0.0]
          ]
      )
      target_imag = constant_op.constant(
          [
              [0.2, 0.0, 0.5, 0.0],
              [0.5, -3.5, 0.0, 0.0]
          ]
      )
      target = math_ops.complex(target_real, target_imag)
      expected = sess.run(target)

      self.assertAllClose(expected, computed)


if __name__ == '__main__':
  test.main()
