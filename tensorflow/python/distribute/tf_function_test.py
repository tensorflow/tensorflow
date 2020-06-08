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
"""Tests for tf.function + distribution strategies."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.compat import v2_compat
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.distribute import values
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables


class TFFunctionTest(test.TestCase, parameterized.TestCase):

  def setup(self):
    # Clear the state for every test.
    def_function.run_functions_eagerly(False)

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.all_strategies,
          mode=["eager"],
          run_functions_eagerly=[True, False]
      ))
  def testDefaultDeviceInsideFunctionWithScope(
      self, distribution, run_functions_eagerly):

    def_function.run_functions_eagerly(run_functions_eagerly)
    expected_device = (device_util.canonicalize("cpu:0")
                       if run_functions_eagerly else "")
    with distribution.scope():
      with ops.device_v2("cpu:0"):
        @def_function.function
        def add():
          one = array_ops.ones([])
          self.assertEqual(expected_device, one.device)
          return one + 1

        add()

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.all_strategies,
          mode=["eager"],
          run_functions_eagerly=[True, False]
      ))
  def testDefaultDeviceInsideNestedFunctionWithScope(
      self, distribution, run_functions_eagerly):

    def_function.run_functions_eagerly(run_functions_eagerly)
    expected_device = (device_util.canonicalize("cpu:0")
                       if run_functions_eagerly else "")
    with distribution.scope():
      @def_function.function
      def foo():
        with ops.device("cpu:0"):

          @def_function.function
          def bar():
            one = array_ops.ones([])
            self.assertEqual(expected_device, one.device)
            return one + 1

          bar()

      foo()

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.all_strategies,
          mode=["eager"],
          run_functions_eagerly=[True, False]
      ))
  def testReadVariableInsideFunction(self, distribution, run_functions_eagerly):

    # Get devices on which variables will be placed. Default strategy does not
    # define this, so assume cpu:0 in that case.
    try:
      devices = distribution.extended.parameter_devices
    except RuntimeError:
      devices = ["cpu:0"]

    with distribution.scope():
      v = variables.Variable(0.)
      if isinstance(v, values.DistributedVariable):
        for i in range(len(devices)):
          # NOTE: Assigning manually to component variables so we can test
          # different values on different devices. Using .assign on the
          # mirrored variable itself will lead to a synchronization which
          # will prohibit testing different values.
          replica_variable = v._values[i]
          replica_variable.assign(math_ops.cast(i, dtypes.float32))

    @def_function.function
    def read():
      return v.read_value()

    for i, d in enumerate(devices):
      with ops.device(d):
        # Verify that the value from each device is read, when in that device
        # scope.
        self.assertEqual(math_ops.cast(i, dtypes.float32), read())


if __name__ == "__main__":
  v2_compat.enable_v2_behavior()
  test.main()
