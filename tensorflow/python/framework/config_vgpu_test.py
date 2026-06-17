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
"""Tests that the system configuration methods work properly."""

from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import ops
from tensorflow.python.platform import test


def reset_eager(fn):

  def wrapper(*args, **kwargs):
    try:
      return fn(*args, **kwargs)
    finally:
      # Reset the context.
      context._reset_jit_compiler_flags()
      context._reset_context()
      ops.enable_eager_execution_internal()
      assert context._context is not None

  return wrapper


class DeviceTest(test.TestCase):

  # Due to a limitation in TensorFlow virtual GPU initialization, tests in
  # this file must run with a new process per test. This is achieved by
  # setting the number of shards to a large number.
  already_run = False

  def setUp(self):
    super().setUp()
    if self.already_run:
      raise RuntimeError(
          'Each test in this test suite must run in a separate process. '
          'Increase number of shards used to run this test.')
    self.already_run = True

  @reset_eager
  def testVGpu1P2V(self):
    gpus = config.list_physical_devices('GPU')
    if len(gpus) != 1:
      self.skipTest('Need 1 GPUs')

    config.set_logical_device_configuration(gpus[0], [
        context.LogicalDeviceConfiguration(
            memory_limit=100, experimental_device_ordinal=0),
        context.LogicalDeviceConfiguration(
            memory_limit=100, experimental_device_ordinal=1)
    ])

    context.ensure_initialized()

    vcpus = config.list_logical_devices('GPU')
    self.assertEqual(len(vcpus), 2)

  @reset_eager
  def testVGpu2P2V1Default(self):
    gpus = config.list_physical_devices('GPU')
    if len(gpus) != 2:
      self.skipTest('Need 2 GPUs')

    config.set_logical_device_configuration(gpus[0], [
        context.LogicalDeviceConfiguration(memory_limit=100),
        context.LogicalDeviceConfiguration(memory_limit=100)
    ])

    context.ensure_initialized()

    vcpus = config.list_logical_devices('GPU')
    self.assertEqual(len(vcpus), 3)

  @reset_eager
  def testGpu2P2V2V(self):
    gpus = config.list_physical_devices('GPU')
    if len(gpus) != 2:
      self.skipTest('Need 2 GPUs')

    config.set_logical_device_configuration(gpus[0], [
        context.LogicalDeviceConfiguration(
            memory_limit=100, experimental_device_ordinal=0),
        context.LogicalDeviceConfiguration(
            memory_limit=100, experimental_device_ordinal=1)
    ])

    config.set_logical_device_configuration(gpus[1], [
        context.LogicalDeviceConfiguration(
            memory_limit=100, experimental_device_ordinal=0),
        context.LogicalDeviceConfiguration(
            memory_limit=100, experimental_device_ordinal=1)
    ])

    context.ensure_initialized()

    vcpus = config.list_logical_devices('GPU')
    self.assertEqual(len(vcpus), 4)

  @reset_eager
  def testGpu2P2D2D(self):
    gpus = config.list_physical_devices('GPU')
    if len(gpus) != 2:
      self.skipTest('Need 2 GPUs')

    config.set_logical_device_configuration(gpus[0], [
        context.LogicalDeviceConfiguration(memory_limit=100),
        context.LogicalDeviceConfiguration(memory_limit=100)
    ])

    config.set_logical_device_configuration(gpus[1], [
        context.LogicalDeviceConfiguration(memory_limit=100),
        context.LogicalDeviceConfiguration(memory_limit=100)
    ])

    context.ensure_initialized()

    vcpus = config.list_logical_devices('GPU')
    self.assertEqual(len(vcpus), 4)


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
