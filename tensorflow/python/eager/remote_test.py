# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for remote execution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import def_function
from tensorflow.python.eager import remote
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import variables


class SingleWorkerTest(test.TestCase):

  def setUp(self):
    super(SingleWorkerTest, self).setUp()

    workers, _ = test_util.create_local_cluster(1, 0)
    remote.connect_to_remote_host(workers[0].target)

  def testMultiDeviceFunctionBasic(self):

    @def_function.function
    def basic(i):
      with ops.device('/job:localhost/replica:0/task:0/cpu:0'):
        a = constant_op.constant([2]) + i
      with ops.device('/job:worker/replica:0/task:0/cpu:0'):
        b = constant_op.constant([1])

      return a + b

    self.assertAllEqual(basic(constant_op.constant([2])).numpy(), [5])
    self.assertAllEqual(basic(constant_op.constant([1])).numpy(), [4])

  def testMultiDeviceFunctionVariable(self):
    with ops.device('/job:worker/replica:0/task:0/cpu:0'):
      variable_b = variables.Variable(1)

    @def_function.function
    def with_variable(i):
      return i + variable_b

    self.assertAllEqual(with_variable(constant_op.constant([2])).numpy(), [3])

  def testMultiDeviceFunctionRemoteOutput(self):
    with ops.device('/job:worker/replica:0/task:0/cpu:0'):
      variable_b = variables.Variable(1)

    @def_function.function
    def remote_output(i):
      return variable_b, i + variable_b

    with self.assertRaises(errors.UnimplementedError) as cm:
      remote_output(constant_op.constant([1]))

    self.assertIn(
        'Currently, outputting tensors on remote devices is not supported.',
        cm.exception.message)

  def testMultiDeviceFunctionAmbiguousDevice(self):

    @def_function.function
    def ambiguous_device(i):
      with ops.device('cpu:0'):
        return i + constant_op.constant([2])

    with self.assertRaises(errors.InvalidArgumentError) as cm:
      with ops.device('/job:worker/replica:0/task:0/cpu:0'):
        self.assertAllEqual(
            ambiguous_device(constant_op.constant([2])).numpy(), [3])

    self.assertIn('the output node must match exactly one device',
                  cm.exception.message)


class MultiWorkersTest(test.TestCase):

  def setUp(self):
    super(MultiWorkersTest, self).setUp()

    workers, _ = test_util.create_local_cluster(2, 0)
    remote.connect_to_remote_host([workers[0].target, workers[1].target])

  def testMultiDeviceFunctionOnRemoteDevice(self):
    with ops.device('/job:worker/replica:0/task:1'):
      variable_b = variables.Variable(1.0)

    @def_function.function
    def remote_function(i):
      with ops.device('/job:worker/replica:0/task:0'):
        a = i + variable_b
      c = a + 1.0
      return c

    with ops.device('/job:worker/replica:0/task:0'):
      self.assertAllEqual(remote_function(constant_op.constant([1.0])), [3.0])


if __name__ == '__main__':
  test.main()
