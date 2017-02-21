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
"""Tests for tf.contrib.tensor_forest.ops.scatter_add_ndim_op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

# TODO: #6568 Remove this hack that makes dlopen() not crash.
if hasattr(sys, 'getdlopenflags') and hasattr(sys, 'setdlopenflags'):
  import ctypes
  sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)

from tensorflow.contrib.tensor_forest.python.ops import tensor_forest_ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest


class ScatterAddNdimTest(test_util.TensorFlowTestCase):

  def test1dim(self):
    input_data = variables.Variable(
        [1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.])
    indices = [[1], [10]]
    updates = [100., 200.]

    with self.test_session():
      variables.global_variables_initializer().run()
      tensor_forest_ops.scatter_add_ndim(input_data, indices, updates).run()
      self.assertAllEqual(
          [1., 102., 3., 4., 5., 6., 7., 8., 9., 10., 211., 12.],
          input_data.eval())

  def test3dim(self):
    input_data = variables.Variable([[[1., 2., 3.], [4., 5., 6.]],
                                     [[7., 8., 9.], [10., 11., 12.]]])
    indices = [[0, 0, 1], [1, 1, 2]]
    updates = [100., 200.]

    with self.test_session():
      variables.global_variables_initializer().run()
      tensor_forest_ops.scatter_add_ndim(input_data, indices, updates).run()
      self.assertAllEqual([[[1., 102., 3.], [4., 5., 6.]],
                           [[7., 8., 9.], [10., 11., 212.]]], input_data.eval())

  def testNoUpdates(self):
    init_val = [[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]]]
    input_data = variables.Variable(init_val)
    indices = []
    updates = []

    with self.test_session():
      variables.global_variables_initializer().run()
      tensor_forest_ops.scatter_add_ndim(input_data, indices, updates).run()
      self.assertAllEqual(init_val, input_data.eval())

  def testBadInput(self):
    init_val = [[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]]]
    input_data = variables.Variable(init_val)
    indices = [[0, 0, 1], [1, 1, 2]]
    updates = [100.]
    with self.test_session():
      variables.global_variables_initializer().run()
      with self.assertRaisesOpError(
          'Number of updates should be same as number of indices.'):
        tensor_forest_ops.scatter_add_ndim(input_data, indices, updates).run()
        self.assertAllEqual(init_val, input_data.eval())

  def testIncompleteIndices(self):
    input_data = variables.Variable([[[1., 2., 3.], [4., 5., 6.]],
                                     [[7., 8., 9.], [10., 11., 12.]]])
    indices = [[0, 0], [1, 1]]
    updates = [[100., 200., 300.], [400., 500., 600.]]

    with self.test_session():
      variables.global_variables_initializer().run()
      tensor_forest_ops.scatter_add_ndim(input_data, indices, updates).run()
      self.assertAllEqual([[[101., 202., 303.], [4., 5., 6.]],
                           [[7., 8., 9.], [410., 511., 612.]]],
                          input_data.eval())


if __name__ == '__main__':
  googletest.main()
