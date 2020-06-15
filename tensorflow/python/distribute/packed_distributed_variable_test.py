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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import packed_distributed_variable
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import test


class PackedDistributedVariableTest(test.TestCase):

  def setUp(self):
    super(PackedDistributedVariableTest, self).setUp()
    cpus = config.list_physical_devices('CPU')
    # Set 2 virtual CPUs
    config.set_logical_device_configuration(cpus[0], [
        context.LogicalDeviceConfiguration(),
        context.LogicalDeviceConfiguration(),
    ])

  def testPackedVariable(self):
    with ops.device('/cpu:0'):
      v0 = resource_variable_ops.ResourceVariable(1.0, name='var0')
    with ops.device('/cpu:1'):
      v1 = resource_variable_ops.ResourceVariable(2.0, name='var1')

    packed_var = packed_distributed_variable.PackedDistributedVariable([v0, v1])
    self.assertTrue(packed_var.handle.is_packed)
    self.assertTrue(packed_var.is_initialized)

    with ops.device('/cpu:0'):
      self.assertAllEqual(packed_var.get_var_on_current_device(), v0)
      val0 = packed_var.assign(2.0).assign_add(1.0)
      self.assertAllEqual(val0, 3.0)

    with ops.device('/cpu:1'):
      self.assertAllEqual(packed_var.get_var_on_current_device(), v1)
      val0 = packed_var.assign(2.0).assign_add(1.0)
      self.assertAllEqual(val0, 3.0)

    @def_function.function
    def update_var():
      with ops.device('/cpu:0'):
        packed_var.assign_add(3.0).assign_sub(1.0)
        read0 = packed_var.value()
      with ops.device('/cpu:1'):
        packed_var.assign_sub(4.0).assign_sub(2.0)
        read1 = packed_var.value()

      return read0, read1

    self.assertAllEqual(update_var(), (5.0, -3.0))

  def testPackedVarAndDevice(self):
    device0 = device_util.canonicalize('/cpu:0')
    device1 = device_util.canonicalize('/cpu:1')

    with ops.device(device0):
      v0 = resource_variable_ops.ResourceVariable(1.0)
    with ops.device(device1):
      v1 = resource_variable_ops.ResourceVariable(2.0)

    packed_var = packed_distributed_variable.PackedDistributedVariable([v0, v1])

    packed_var0 = packed_distributed_variable.PackedVarAndDevice(
        packed_var, device0)
    self.assertTrue(packed_var0.handle.is_packed)
    self.assertAllEqual(math_ops.mul(packed_var0, 2.0), 2.0)

    packed_var1 = packed_distributed_variable.PackedVarAndDevice(
        packed_var, device1)
    self.assertAllEqual(packed_var1.assign(3.0), 3.0)

    @def_function.function
    def func():
      var0 = packed_distributed_variable.PackedVarAndDevice(packed_var, device0)
      var0.assign_add(3.0)
      var1 = packed_distributed_variable.PackedVarAndDevice(packed_var, device1)
      return var0.value(), math_ops.add(var1, 2.0)

    self.assertAllEqual(func(), (4.0, 5.0))


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
