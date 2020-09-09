# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for functional operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import functional_ops
from tensorflow.python.platform import test


class FunctionalOpsTest(test.TestCase):

  @test_util.deprecated_graph_mode_only
  def testIfWithDefun(self):

    @function.Defun(dtypes.float32)
    def Then(x):
      return x + 1

    @function.Defun(dtypes.float32)
    def Else(x):
      return x - 1

    with self.cached_session():
      inputs = [10.]
      result = self.evaluate(functional_ops.If(False, inputs, Then, Else))
      self.assertEqual([9.0], result)

  def testIfWithFunction(self):

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec((), dtypes.float32)])
    def Then(x):
      return x + 1

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec((), dtypes.float32)])
    def Else(x):
      return x - 1

    with self.cached_session():
      inputs = [10.]
      result = self.evaluate(
          functional_ops.If(False, inputs, Then.get_concrete_function(),
                            Else.get_concrete_function()))
      self.assertEqual([9.0], result)


if __name__ == '__main__':
  test.main()
