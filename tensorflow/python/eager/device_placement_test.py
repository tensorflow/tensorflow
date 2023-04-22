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
"""Tests for device placement."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import remote
from tensorflow.python.eager import test
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops


class SoftDevicePlacementTest(test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(SoftDevicePlacementTest, self).setUp()
    context._reset_context()
    context.ensure_initialized()
    config.set_soft_device_placement(enabled=True)
    context.context().log_device_placement = True

  @test_util.run_gpu_only
  def testDefaultPlacement(self):
    a = constant_op.constant(1)
    b = constant_op.constant(2)
    c = a + b
    with ops.device('CPU'):
      d = a + b
    self.assertIn('GPU', c.device)
    self.assertIn('CPU', d.device)

  @test_util.run_gpu_only
  def testUnsupportedDevice(self):
    a = constant_op.constant(1)
    b = constant_op.constant(2)
    s = constant_op.constant(list('hello world'))
    with ops.device('GPU:0'):
      c = a + b
      t = s[a]
    self.assertIn('GPU:0', c.device)
    self.assertIn('CPU', t.device)

  @test_util.run_gpu_only
  def testUnknownDevice(self):
    a = constant_op.constant(1)
    b = constant_op.constant(2)
    with ops.device('GPU:42'):
      c = a + b
    self.assertIn('GPU:0', c.device)

  def testNoGpu(self):
    if test_util.is_gpu_available():
      # CPU only test.
      return
    a = constant_op.constant(1)
    b = constant_op.constant(2)
    c = a + b
    with ops.device('GPU'):
      d = a + b
    self.assertIn('CPU', c.device)
    self.assertIn('CPU', d.device)

  @test_util.run_gpu_only
  def testSoftPlacedGPU(self):
    a = constant_op.constant(1)
    b = constant_op.constant(2)
    with ops.device('GPU:110'):
      c = a + b
    self.assertIn('GPU:0', c.device)

  @test_util.run_gpu_only
  def testNestedDeviceScope(self):
    a = constant_op.constant(1)
    b = constant_op.constant(2)
    with ops.device('CPU:0'):
      with ops.device('GPU:42'):
        c = a + b
    # We don't support nested device placement right now.
    self.assertIn('GPU:0', c.device)

  @parameterized.named_parameters(('float', 1.0, None),
                                  ('int32', [1], dtypes.int32),
                                  ('string', ['a'], None))
  def testSoftPlacedCPUConstant(self, value, dtype):
    if test_util.is_gpu_available():
      self.skipTest('CPU only test')
    with ops.device('GPU:0'):
      a = constant_op.constant(value, dtype=dtype)
    self.assertIn('CPU:0', a.device)
    self.assertIn('CPU:0', a.backing_device)

  def testPlacedToDeviceInFunction(self):

    @def_function.function
    def f():
      a = random_ops.random_uniform([32, 32])
      return math_ops.matmul(a, a)

    gpus = config.list_physical_devices('GPU')
    if not gpus:
      self.assertIn('CPU:0', f().device)
    else:
      self.assertIn('GPU:0', f().device)

  @test_util.disable_tfrt('b/173726713: Support properly inserting device at '
                          'tf_to_corert lowering.')
  def testUnknownDeviceInFunction(self):

    @def_function.function
    def f():
      with ops.device('GPU:42'):
        # With placer, the unknown GPU:42 will be replaced with GPU:0.
        a = constant_op.constant(1) + constant_op.constant(2)
      return a + constant_op.constant(2)

    gpus = config.list_physical_devices('GPU')
    if not gpus:
      self.assertIn('CPU:0', f().device)
    else:
      self.assertIn('GPU:0', f().device)


class HardDevicePlacementTest(test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(HardDevicePlacementTest, self).setUp()
    context._reset_context()
    config.set_soft_device_placement(enabled=False)
    context.context().log_device_placement = True
    cpus = context.context().list_physical_devices('CPU')
    # Set 2 virtual CPUs
    context.context().set_logical_device_configuration(cpus[0], [
        context.LogicalDeviceConfiguration(),
        context.LogicalDeviceConfiguration()
    ])
    self.assertEqual(config.get_soft_device_placement(), False)
    self.assertEqual(context.context().soft_device_placement, False)

  @test_util.run_gpu_only
  def testIdentityCanCopy(self):
    config.set_device_policy('explicit')
    with ops.device('CPU:0'):
      x = constant_op.constant(1.0)
      self.assertIn('CPU:0', x.device)
      self.assertIn('CPU:0', x.backing_device)
    with ops.device('GPU:0'):
      y = array_ops.identity(x)
      self.assertIn('GPU:0', y.device)
      self.assertIn('GPU:0', y.backing_device)

  @test_util.run_gpu_only
  def testSimpleConstantsExplicitGPU(self):
    config.set_device_policy('explicit')
    with ops.device('GPU:0'):
      self.assertAllClose(1., array_ops.ones([]))
      self.assertAllClose(0., array_ops.zeros([]))
      self.assertAllClose([1.], array_ops.fill([1], 1.))

  def testSimpleConstantsExplicitCPU(self):
    config.set_device_policy('explicit')
    with ops.device('CPU:1'):
      self.assertAllClose(1., array_ops.ones([]))
      self.assertAllClose(0., array_ops.zeros([]))
      self.assertAllClose([1.], array_ops.fill([1], 1.))
      self.assertAllClose(2., constant_op.constant(1.) * 2.)

  @parameterized.named_parameters(
      ('float_cpu0', 'CPU:0', 1.0, None),
      ('int32_cpu0', 'CPU:0', [1], dtypes.int32),
      ('string_cpu0', 'CPU:0', ['a'], None),
      ('float_cpu1', 'CPU:1', 1.0, None),
      ('int32_cpu1', 'CPU:1', [1], dtypes.int32),
      ('string_cpu1', 'CPU:1', ['a'], None),
  )
  def testHardPlacedCPUConstant(self, device, value, dtype):
    with ops.device(device):
      a = constant_op.constant(value, dtype=dtype)
      self.assertIn(device, a.device)

  @parameterized.named_parameters(
      ('float', 'GPU:0', 1.0, None),
      ('int32', 'GPU:0', [1], dtypes.int32),
      ('string', 'GPU:0', ['a'], None),
  )
  def testHardPlacedGPUConstant(self, device, value, dtype):
    if not test_util.is_gpu_available():
      self.skipTest('Test requires a GPU')
    with ops.device(device):
      a = constant_op.constant(value, dtype=dtype)
      self.assertIn(device, a.device)
      if a.dtype == dtypes.float32:
        self.assertIn(device, a.backing_device)


class ClusterPlacementTest(test.TestCase):

  def setUp(self):
    super(ClusterPlacementTest, self).setUp()
    context._reset_context()
    config.set_soft_device_placement(enabled=True)
    context.context().log_device_placement = True
    workers, _ = test_util.create_local_cluster(2, 0)
    remote.connect_to_remote_host([workers[0].target, workers[1].target])

  @test_util.disable_tfrt('remote host not supported yet.')
  def testNotFullySpecifiedTask(self):
    a = constant_op.constant(1)
    b = constant_op.constant(2)
    with ops.device('/job:worker'):
      c = a + b
    self.assertIn('/job:worker/replica:0/task:0', c.device)

  @test_util.disable_tfrt('remote host not supported yet.')
  def testRemoteUnknownDevice(self):
    a = constant_op.constant(1)
    b = constant_op.constant(2)
    # Right now we don't support soft device place on remote worker.
    with self.assertRaises(errors.InvalidArgumentError) as cm:
      with ops.device('/job:worker/replica:0/task:0/device:GPU:42'):
        c = a + b
        del c
      self.assertIn('unknown device', cm.exception.message)

  @test_util.disable_tfrt('remote host not supported yet.')
  def testUnknownDeviceInFunctionReturnUnknowDevice(self):

    @def_function.function
    def f():
      with ops.device('GPU:42'):
        return constant_op.constant(1) + constant_op.constant(2)

    gpus = config.list_physical_devices('GPU')
    if not gpus:
      self.assertIn('CPU:0', f().device)
    else:
      self.assertIn('GPU:0', f().device)

  @test_util.disable_tfrt('remote host not supported yet.')
  def testUnknownDeviceInFunction(self):

    @def_function.function
    def f():
      with ops.device('GPU:42'):
        a = constant_op.constant(1) + constant_op.constant(2)
      return a + constant_op.constant(2)

    gpus = config.list_physical_devices('GPU')
    if not gpus:
      self.assertIn('CPU:0', f().device)
    else:
      self.assertIn('GPU:0', f().device)


if __name__ == '__main__':
  test.main()
