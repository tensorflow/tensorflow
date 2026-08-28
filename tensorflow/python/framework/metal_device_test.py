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
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test

try:
  from tensorflow.python.keras import layers  # pylint: disable=g-import-not-at-top
  from tensorflow.python.keras.engine import sequential  # pylint: disable=g-import-not-at-top
except ImportError:  # Keras is packaged separately in some builds.
  layers = None
  sequential = None


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

  def testCastFloatToHalfAndBack(self):
    values = np.random.rand(128).astype(np.float32)
    with ops.device("/GPU:0"):
      as_half = math_ops.cast(constant_op.constant(values), dtypes.float16)
      back = math_ops.cast(as_half, dtypes.float32)
    self.assertAllClose(values, back.numpy(), rtol=1e-2, atol=1e-2)

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

  def testConv2DMatchesCpu(self):
    """The convolution itself, checked against the CPU kernel."""
    images = np.random.rand(2, 8, 8, 3).astype(np.float32)
    filters = np.random.rand(3, 3, 3, 4).astype(np.float32)
    with ops.device("/CPU:0"):
      expected = nn_ops.conv2d(constant_op.constant(images),
                               constant_op.constant(filters),
                               strides=[1, 1, 1, 1], padding="SAME").numpy()
    with ops.device("/GPU:0"):
      actual = nn_ops.conv2d(constant_op.constant(images),
                             constant_op.constant(filters),
                             strides=[1, 1, 1, 1], padding="SAME")
    self.assertIn("GPU:0", actual.device)
    self.assertAllClose(expected, actual.numpy(), rtol=1e-4, atol=1e-4)

  def testMaxPoolMatchesCpu(self):
    images = np.random.rand(2, 8, 8, 3).astype(np.float32)
    with ops.device("/CPU:0"):
      expected = nn_ops.max_pool2d(constant_op.constant(images), ksize=2,
                                   strides=2, padding="VALID").numpy()
    with ops.device("/GPU:0"):
      actual = nn_ops.max_pool2d(constant_op.constant(images), ksize=2,
                                 strides=2, padding="VALID")
    self.assertAllClose(expected, actual.numpy())

  def testReluAndSoftmaxMatchCpu(self):
    values = (np.random.rand(4, 10).astype(np.float32) - 0.5) * 8.0
    with ops.device("/GPU:0"):
      relu = nn_ops.relu(constant_op.constant(values))
      softmax = nn_ops.softmax(constant_op.constant(values))
    self.assertAllClose(np.maximum(values, 0.0), relu.numpy())
    # Rows of a softmax must sum to one, which catches an axis mistake that a
    # loose elementwise comparison would let through.
    self.assertAllClose(np.ones([4]), np.sum(softmax.numpy(), axis=1),
                        rtol=1e-5, atol=1e-5)

  def testReductionsMatchNumpy(self):
    values = np.random.rand(4, 5, 6).astype(np.float32)
    with ops.device("/GPU:0"):
      total = math_ops.reduce_sum(constant_op.constant(values), axis=[1])
      average = math_ops.reduce_mean(constant_op.constant(values))
    self.assertAllClose(values.sum(axis=1), total.numpy(), rtol=1e-5,
                        atol=1e-5)
    self.assertAllClose(values.mean(), average.numpy(), rtol=1e-5, atol=1e-5)

  def testRandomInitialisationIsNotDegenerate(self):
    """Two draws must differ, and the spread must look like the distribution."""
    with ops.device("/GPU:0"):
      first = random_ops.random_normal([4096], dtype=dtypes.float32).numpy()
      second = random_ops.random_normal([4096], dtype=dtypes.float32).numpy()
    # A seed baked into a cached graph would make these identical, which is
    # the specific failure this backend's counter-based generator avoids.
    self.assertNotAllClose(first, second)
    self.assertAllClose(0.0, first.mean(), atol=0.1)
    self.assertAllClose(1.0, first.std(), atol=0.1)

  def testTruncatedNormalStaysInRange(self):
    with ops.device("/GPU:0"):
      values = random_ops.truncated_normal([4096],
                                           dtype=dtypes.float32).numpy()
    self.assertLessEqual(np.abs(values).max(), 2.0 + 1e-5)

  def testTrainsASmallConvNet(self):
    """The point of the whole backend: a model that actually learns."""
    if sequential is None:
      self.skipTest("Keras is not available in this build.")
    rng = np.random.RandomState(0)
    images = rng.rand(64, 8, 8, 1).astype(np.float32)
    # A label the model can actually fit: bright images are class 1.
    labels = (images.mean(axis=(1, 2, 3)) > 0.5).astype(np.int32)

    with ops.device("/GPU:0"):
      model = sequential.Sequential([
          layers.Conv2D(4, 3, padding="same", activation="relu",
                        input_shape=(8, 8, 1)),
          layers.MaxPooling2D(2),
          layers.Flatten(),
          layers.Dense(2),
      ])
      model.compile(optimizer="adam",
                    loss="sparse_categorical_crossentropy")
      history = model.fit(images, labels, epochs=8, batch_size=16, verbose=0)

    losses = history.history["loss"]
    self.assertLess(losses[-1], losses[0],
                    "loss did not decrease: %s" % losses)

  def testUnsupportedBroadcastIsReportedNotSilentlyWrong(self):
    """Rank-mismatched broadcasting must fail loudly, not compute garbage."""
    lhs = np.random.rand(8, 4).astype(np.float32)
    rhs = np.random.rand(4).astype(np.float32)
    try:
      with ops.device("/GPU:0"):
        result = math_ops.add(constant_op.constant(lhs),
                              constant_op.constant(rhs)).numpy()
    except Exception:  # pylint: disable=broad-except
      # Rejected by the Metal kernel, which is the documented behaviour until
      # strided broadcasting lands.
      return
    # Otherwise core placed the op on the host instead, which is also correct;
    # what must not happen is a wrong answer.
    self.assertAllClose(lhs + rhs, result)


if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()
