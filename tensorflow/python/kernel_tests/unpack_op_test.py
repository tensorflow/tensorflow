# Copyright 2015 Google Inc. All Rights Reserved.
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

"""Functional tests for Unpack Op."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf


class UnpackOpTest(tf.test.TestCase):

  def testSimple(self):
    np.random.seed(7)
    for use_gpu in False, True:
      with self.test_session(use_gpu=use_gpu):
        for shape in (2,), (3,), (2, 3), (3, 2), (4, 3, 2):
          data = np.random.randn(*shape)
          # Convert data to a single tensorflow tensor
          x = tf.constant(data)
          # Unpack into a list of tensors
          cs = tf.unpack(x, num=shape[0])
          self.assertEqual(type(cs), list)
          self.assertEqual(len(cs), shape[0])
          cs = [c.eval() for c in cs]
          self.assertAllEqual(cs, data)

  def testGradients(self):
    for use_gpu in False, True:
      for shape in (2,), (3,), (2, 3), (3, 2), (4, 3, 2):
        data = np.random.randn(*shape)
        shapes = [shape[1:]] * shape[0]
        for i in xrange(shape[0]):
          with self.test_session(use_gpu=use_gpu):
            x = tf.constant(data)
            cs = tf.unpack(x, num=shape[0])
            err = tf.test.compute_gradient_error(x, shape, cs[i], shapes[i])
            self.assertLess(err, 1e-6)

  def testInferNum(self):
    with self.test_session():
      for shape in (2,), (3,), (2, 3), (3, 2), (4, 3, 2):
        x = tf.placeholder(np.float32, shape=shape)
        cs = tf.unpack(x)
        self.assertEqual(type(cs), list)
        self.assertEqual(len(cs), shape[0])

  def testCannotInferNum(self):
    x = tf.placeholder(np.float32)
    with self.assertRaisesRegexp(
        ValueError, r'Cannot infer num from shape <unknown>'):
      tf.unpack(x)


if __name__ == '__main__':
  tf.test.main()
