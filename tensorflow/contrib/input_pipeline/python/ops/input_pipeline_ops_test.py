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
"""Tests for input_pipeline_ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.input_pipeline.python.ops import input_pipeline_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class InputPipelineOpsTest(test.TestCase):

  def testObtainNext(self):
    with self.test_session():
      var = state_ops.variable_op([], dtypes.int64)
      state_ops.assign(var, -1).op.run()
      c = constant_op.constant(["a", "b"])
      sample1 = input_pipeline_ops.obtain_next(c, var)
      self.assertEqual(b"a", sample1.eval())
      self.assertEqual(0, var.eval())
      sample2 = input_pipeline_ops.obtain_next(c, var)
      self.assertEqual(b"b", sample2.eval())
      self.assertEqual(1, var.eval())
      sample3 = input_pipeline_ops.obtain_next(c, var)
      self.assertEqual(b"a", sample3.eval())
      self.assertEqual(0, var.eval())

  def testSeekNext(self):
    string_list = ["a", "b", "c"]
    with self.test_session() as session:
      elem = input_pipeline_ops.seek_next(string_list)
      session.run([variables.global_variables_initializer()])
      self.assertEqual(b"a", session.run(elem))
      self.assertEqual(b"b", session.run(elem))
      self.assertEqual(b"c", session.run(elem))
      # Make sure we loop.
      self.assertEqual(b"a", session.run(elem))

  # Helper method that runs the op len(expected_list) number of times, asserts
  # that the results are elements of the expected_list and then throws an
  # OutOfRangeError.
  def _assert_output(self, expected_list, session, op):
    for element in expected_list:
      self.assertEqual(element, session.run(op))
    with self.assertRaises(errors.OutOfRangeError):
      session.run(op)

  def testSeekNextLimitEpochs(self):
    string_list = ["a", "b", "c"]
    with self.test_session() as session:
      elem = input_pipeline_ops.seek_next(string_list, num_epochs=1)
      session.run([
          variables.local_variables_initializer(),
          variables.global_variables_initializer()
      ])
      self._assert_output([b"a", b"b", b"c"], session, elem)

  def testSeekNextLimitEpochsThree(self):
    string_list = ["a", "b", "c"]
    with self.test_session() as session:
      elem = input_pipeline_ops.seek_next(string_list, num_epochs=3)
      session.run([
          variables.local_variables_initializer(),
          variables.global_variables_initializer()
      ])
      # Expect to see [a, b, c] three times.
      self._assert_output([b"a", b"b", b"c"] * 3, session, elem)


if __name__ == "__main__":
  test.main()
