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
import modrelu
# from tensorflow.contrib.nn.python.ops import modrelu
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

class ModreluTest(test.TestCase):
  def test_tensor(self):
    with self.test_session() as sess:
      inp_re = constant_op.constant(
          [
              [2.5, 0.5, 0.3, 0.4],
              [0.7, 0, -0.3, 0.2]
          ]
      )
      inp_im = constant_op.constant(
          [
              [0.2, 0.0, 0.5, 0.7],
              [0.5, -2.5, 0.0, 0.8]
          ]
      )
      inp = math_ops.complex(inp_re, inp_im)
      biases = constant_op.constant([0.0, 1.0, 0.0, -1.0])

      target_re = constant_op.constant(
          [
              [2.5, 1.5, 0.3, 0.0],
              [0.7, 0, -0.3, 0.0]
          ]
      )
      target_im = constant_op.constant(
          [
              [0.2, 0.0, 0.5, 0.0],
              [0.5, -3.5, 0.0, 0.0]
          ]
      )
      target = math_ops.complex(target_re, target_im)

      out = sess.run(modrelu.modrelu(inp, biases))
      tar = sess.run(target)

      self.assertAllClose(out, tar)

if __name__ == '__main__':
  test.main()
