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
"""Miscellaneous tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

# TODO: #6568 Remove this hack that makes dlopen() not crash.
if hasattr(sys, "getdlopenflags") and hasattr(sys, "setdlopenflags"):
  import ctypes
  sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)

import numpy as np

from tensorflow.contrib.ndlstm.python import misc as misc_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

misc = misc_lib


def _rand(*size):
  return np.random.uniform(size=size).astype("f")


class LstmMiscTest(test_util.TensorFlowTestCase):

  def testPixelsAsVectorDims(self):
    with self.test_session():
      inputs = constant_op.constant(_rand(2, 7, 11, 5))
      outputs = misc.pixels_as_vector(inputs)
      variables.global_variables_initializer().run()
      result = outputs.eval()
      self.assertEqual(tuple(result.shape), (2, 7 * 11 * 5))

  def testPoolAsVectorDims(self):
    with self.test_session():
      inputs = constant_op.constant(_rand(2, 7, 11, 5))
      outputs = misc.pool_as_vector(inputs)
      variables.global_variables_initializer().run()
      result = outputs.eval()
      self.assertEqual(tuple(result.shape), (2, 5))

  def testOneHotPlanes(self):
    with self.test_session():
      inputs = constant_op.constant([0, 1, 3])
      outputs = misc.one_hot_planes(inputs, 4)
      variables.global_variables_initializer().run()
      result = outputs.eval()
      self.assertEqual(tuple(result.shape), (3, 1, 1, 4))
      target = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
      self.assertAllClose(result.reshape(-1), target.reshape(-1))

  def testOneHotMask(self):
    with self.test_session():
      data = np.array([[0, 1, 2], [2, 0, 1]]).reshape(2, 3, 1)
      inputs = constant_op.constant(data)
      outputs = misc.one_hot_mask(inputs, 3)
      variables.global_variables_initializer().run()
      result = outputs.eval()
      self.assertEqual(tuple(result.shape), (2, 3, 3))
      target = np.array([[[1, 0, 0], [0, 1, 0]], [[0, 1, 0], [0, 0, 1]],
                         [[0, 0, 1], [1, 0, 0]]]).transpose(1, 2, 0)
      self.assertAllClose(result.reshape(-1), target.reshape(-1))


if __name__ == "__main__":
  test.main()
