# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for 1D LSTM."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

# TODO: #6568 Remove this hack that makes dlopen() not crash.
if hasattr(sys, "getdlopenflags") and hasattr(sys, "setdlopenflags"):
  import ctypes
  sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)

import numpy as np

from tensorflow.contrib.ndlstm.python import lstm1d as lstm1d_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

lstm1d = lstm1d_lib


def _rand(*size):
  return np.random.uniform(size=size).astype("f")


class Lstm1DTest(test.TestCase):

  def testSequenceToSequenceDims(self):
    with self.test_session():
      inputs = constant_op.constant(_rand(17, 1, 5))
      outputs = lstm1d.ndlstm_base(inputs, 8)
      variables.global_variables_initializer().run()
      names = [v.name for v in variables.trainable_variables()]
      self.assertEqual(len(names), 2)
      result = outputs.eval()
      self.assertEqual(tuple(result.shape), (17, 1, 8))

  def testSequenceToSequenceGradient(self):
    with self.test_session():
      size = (17, 1, 15)
      output_size = (17, 1, 8)
      inputs = constant_op.constant(_rand(*size))
      outputs = lstm1d.ndlstm_base(inputs, 8, dynamic=False)
      variables.global_variables_initializer().run()
      gradients = gradients_impl.gradients(outputs, inputs)
      if 1:  # pylint: disable=using-constant-test
        gradients = gradients_impl.gradients(outputs, inputs)[0].eval()
        self.assertEqual(gradients.shape, size)
      else:
        # TODO(tmb) tf.test.compute_gradient error is currently broken
        # with dynamic_rnn. Enable this test case eventually.
        err = gradient_checker.compute_gradient_error(
            inputs, size, outputs, output_size, delta=1e-4)
        self.assert_(not np.isnan(err))
        self.assert_(err < 0.1)

  def testSequenceToSequenceGradientReverse(self):
    with self.test_session():
      size = (17, 1, 15)
      output_size = (17, 1, 8)
      inputs = constant_op.constant(_rand(*size))
      outputs = lstm1d.ndlstm_base(inputs, 8, reverse=1, dynamic=False)
      variables.global_variables_initializer().run()
      if 1:  # pylint: disable=using-constant-test
        gradients = gradients_impl.gradients(outputs, inputs)[0].eval()
        self.assertEqual(gradients.shape, size)
      else:
        # TODO(tmb) tf.test.compute_gradient error is currently broken
        # with dynamic_rnn. Enable this test case eventually.
        err = gradient_checker.compute_gradient_error(
            inputs, size, outputs, output_size, delta=1e-4)
        self.assert_(not np.isnan(err))
        self.assert_(err < 0.1)

  def testSequenceToFinalDims(self):
    with self.test_session():
      inputs = constant_op.constant(_rand(17, 6, 5))
      outputs = lstm1d.sequence_to_final(inputs, 8)
      variables.global_variables_initializer().run()
      names = [v.name for v in variables.trainable_variables()]
      self.assertEqual(len(names), 2)
      result = outputs.eval()
      self.assertEqual(tuple(result.shape), (6, 8))

  def testSequenceSoftmaxDims(self):
    with self.test_session():
      inputs = constant_op.constant(_rand(17, 1, 5))
      outputs = lstm1d.sequence_softmax(inputs, 8)
      variables.global_variables_initializer().run()
      result = outputs.eval()
      self.assertEqual(tuple(result.shape), (17, 1, 8))


if __name__ == "__main__":
  test.main()
