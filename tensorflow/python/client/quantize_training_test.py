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
"""Tests for the SWIG-wrapped quantize training rewriting."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class PywrapQuantizeTrainingTest(test.TestCase):

  # Mainly to verify the python interface is working.
  # More tests for this function can be found in the related c++ tests.
  def testQuantizeTraining(self):
    with session.Session() as sess:
      a = constant_op.constant(6.0, shape=[1, 1])
      b = constant_op.constant(7.0, shape=[1, 1])
      c = math_ops.matmul(a, b, name='matmul')

      self.assertEquals(len(sess.graph_def.node), 3)

      result = pywrap_tensorflow.do_quantize_training_on_graphdef(
          sess.graph_def, 8)

      # We just want to guarantee that some rewrite happened.
      self.assertGreater(len(result.node), 3)

      self.assertEquals(c.eval(), 42)


if __name__ == '__main__':
  test.main()
