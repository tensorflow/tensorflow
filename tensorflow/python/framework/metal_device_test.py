# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
"""End-to-end tests for the Metal PluggableDevice backend.

These only run on an Apple silicon machine with a TensorFlow built using
--config=metal. Everywhere else they skip, so the file is harmless in the
normal CI matrix.
"""

import platform

import numpy as np

from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


def _metal_is_available():
  """True when a Metal device from this backend is present."""
  if platform.system() != "Darwin" or platform.machine() != "arm64":
    return False
  # The backend registers under the GPU device type, and on macOS no other
  # backend does, so any GPU device here is a Metal one.
  return bool(config.list_physical_devices("GPU"))


class MetalDeviceTest(test.TestCase):

  def setUp(self):
    super().setUp()
    if not _metal_is_available():
      self.skipTest("No Metal device; TensorFlow was probably not built with "
                    "--config=metal.")

  def testDeviceIsListed(self):
    devices = config.list_physical_devices("GPU")
    self.assertNotEmpty(devices)
    self.assertEqual(devices[0].device_type, "GPU")

  def testDeviceDetailsNameTheMetalPlatform(self):
    device = config.list_physical_devices("GPU")[0]
    details = config.get_device_details(device)
    # hardware_name comes from MTLDevice.name, so it should say something
    # about an Apple GPU rather than being blank.
    self.assertIn("device_name", details)
    self.assertNotEmpty(details["device_name"])

  def testTensorPlacement(self):
    with ops.device("/GPU:0"):
      value = constant_op.constant([1.0, 2.0, 3.0], dtype=dtypes.float32)
    self.assertIn("GPU:0", value.device)

  def testRoundTripPreservesValues(self):
    """Exercises the host-to-device and device-to-host copy paths."""
    expected = np.arange(1024, dtype=np.float32).reshape(32, 32)
    with ops.device("/GPU:0"):
      on_device = constant_op.constant(expected)
    self.assertAllEqual(expected, on_device.numpy())

  def testAdd(self):
    lhs = np.random.rand(64, 64).astype(np.float32)
    rhs = np.random.rand(64, 64).astype(np.float32)
    with ops.device("/GPU:0"):
      result = math_ops.add(constant_op.constant(lhs),
                            constant_op.constant(rhs))
    self.assertIn("GPU:0", result.device)
    self.assertAllClose(lhs + rhs, result.numpy())

  def testAddScalarBroadcast(self):
    values = np.random.rand(16, 16).astype(np.float32)
    with ops.device("/GPU:0"):
      result = math_ops.add(constant_op.constant(values),
                            constant_op.constant(2.0, dtype=dtypes.float32))
    self.assertAllClose(values + 2.0, result.numpy())

  def testMul(self):
    lhs = np.random.rand(8, 8).astype(np.float32)
    rhs = np.random.rand(8, 8).astype(np.float32)
    with ops.device("/GPU:0"):
      result = math_ops.multiply(constant_op.constant(lhs),
                                 constant_op.constant(rhs))
    self.assertAllClose(lhs * rhs, result.numpy())

  def testMatMul(self):
    lhs = np.random.rand(64, 96).astype(np.float32)
    rhs = np.random.rand(96, 32).astype(np.float32)
    with ops.device("/GPU:0"):
      result = math_ops.matmul(constant_op.constant(lhs),
                               constant_op.constant(rhs))
    self.assertIn("GPU:0", result.device)
    self.assertAllClose(np.matmul(lhs, rhs), result.numpy(), rtol=1e-5,
                        atol=1e-5)

  def testMatMulTransposed(self):
    lhs = np.random.rand(96, 64).astype(np.float32)
    rhs = np.random.rand(32, 96).astype(np.float32)
    with ops.device("/GPU:0"):
      result = math_ops.matmul(constant_op.constant(lhs),
                               constant_op.constant(rhs),
                               transpose_a=True, transpose_b=True)
    self.assertAllClose(np.matmul(lhs.T, rhs.T), result.numpy(), rtol=1e-5,
                        atol=1e-5)

  def testMatMulFloat16(self):
    lhs = np.random.rand(32, 32).astype(np.float16)
    rhs = np.random.rand(32, 32).astype(np.float16)
    with ops.device("/GPU:0"):
      result = math_ops.matmul(constant_op.constant(lhs),
                               constant_op.constant(rhs))
    self.assertAllClose(np.matmul(lhs.astype(np.float32),
                                  rhs.astype(np.float32)),
                        result.numpy().astype(np.float32),
                        rtol=1e-2, atol=1e-2)

  def testIdentity(self):
    values = np.random.rand(16, 16).astype(np.float32)
    with ops.device("/GPU:0"):
      result = array_ops.identity(constant_op.constant(values))
    self.assertIn("GPU:0", result.device)
    self.assertAllEqual(values, result.numpy())

  def testChainedOpsStayOnDevice(self):
    """A short chain, to check that stream ordering holds across kernels."""
    lhs = np.random.rand(48, 48).astype(np.float32)
    rhs = np.random.rand(48, 48).astype(np.float32)
    with ops.device("/GPU:0"):
      product = math_ops.matmul(constant_op.constant(lhs),
                                constant_op.constant(rhs))
      shifted = math_ops.add(product, constant_op.constant(1.0))
      doubled = math_ops.multiply(shifted, constant_op.constant(2.0))
    self.assertAllClose((np.matmul(lhs, rhs) + 1.0) * 2.0, doubled.numpy(),
                        rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
  test.main()
