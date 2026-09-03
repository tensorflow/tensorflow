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
"""Tests for DLPack functions.

Coverage in this file is split into two groups:

  * ``testRoundTrip`` is parameterized over every dtype/shape combination in
    ``dlpack_dtypes`` x ``testcase_shapes`` and checks that a tensor survives
    an export-then-import round trip unchanged, on whichever device it
    started on. If you add support for a new dtype in the C++ dlpack
    converter, add it to ``dlpack_dtypes`` and this test will exercise it
    automatically -- no new test method needed.
  * The remaining ``test*`` methods each cover one specific behavioral
    contract (single-use capsules, error messages, context-reset safety,
    etc.) rather than dtype/shape coverage; add a new method here for a new
    *behavior*, not a new *type*.
"""

from typing import Any, Dict, List, Sequence, Tuple

from absl.testing import parameterized
import numpy as np

from tensorflow.python.dlpack import dlpack
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


# Dtypes accepted by the dlpack converter, grouped the same way the DLPack
# spec groups them (dlpack.h's DLDataTypeCode). Kept as separate lists so a
# contributor adding e.g. a new int width only has to touch `int_dtypes`.
int_dtypes = [
    np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32,
    np.uint64
]
float_dtypes = [np.float16, np.float32, np.float64]
complex_dtypes = [np.complex64, np.complex128]
dlpack_dtypes = (
    int_dtypes + float_dtypes + [dtypes.bfloat16] + complex_dtypes + [np.bool_]
)

# Deliberately includes a scalar (()), a couple of zero-sized dims ((2, 0),
# (0, 7)) to catch off-by-one bugs in empty-buffer handling, and a
# multi-dimensional shape.
testcase_shapes: List[Tuple[int, ...]] = [(), (1,), (2, 3), (2, 0), (0, 7),
                                          (4, 1, 2)]


def FormatShapeAndDtype(shape: Sequence[int], dtype: Any) -> str:
  """Builds a readable parameterized-test suffix, e.g. '_float32[2,3]'."""
  return "_{}[{}]".format(str(dtype), ",".join(map(str, shape)))


def GetNamedTestParameters() -> List[Dict[str, Any]]:
  """Cross product of dlpack_dtypes x testcase_shapes for @parameterized."""
  return [{
      "testcase_name": FormatShapeAndDtype(shape, dtype),
      "dtype": dtype,
      "shape": shape,
  } for dtype in dlpack_dtypes for shape in testcase_shapes]


def _make_test_tensor(shape: Sequence[int], dtype: Any) -> "ops.Tensor":
  """Creates a small deterministic tensor of the given shape/dtype.

  Values are drawn as plain Python ints from `np.random.randint` and then
  cast to `dtype` by `constant_op.constant`, so this works uniformly for
  int/float/complex/bfloat16 dtypes. Bool tensors need a 0/1 range instead
  of the 0-9 range used for everything else.
  """
  high = 2 if dtype == np.bool_ else 10
  np_array = np.random.randint(0, high, shape)
  return constant_op.constant(np_array, dtype=dtype)


class DLPackTest(parameterized.TestCase, test.TestCase):
  """Tests for `tf.experimental.dlpack.{to,from}_dlpack`."""

  @parameterized.named_parameters(GetNamedTestParameters())
  def testRoundTrip(self, dtype: Any, shape: Sequence[int]) -> None:
    """A tensor exported to dlpack and re-imported should be unchanged."""
    np.random.seed(42)
    # array_ops.identity forces a copy onto whatever device is default
    # (e.g. GPU, if one is available), so this test also covers non-CPU
    # tensors when run on a GPU-enabled build.
    source_tensor = _make_test_tensor(shape, dtype)
    np_array = source_tensor.numpy()
    tf_tensor = array_ops.identity(source_tensor)
    tf_tensor_device = tf_tensor.device
    tf_tensor_dtype = tf_tensor.dtype

    dlcapsule = dlpack.to_dlpack(tf_tensor)
    del tf_tensor  # The capsule should keep the underlying buffer alive.
    tf_tensor2 = dlpack.from_dlpack(dlcapsule)

    self.assertAllClose(np_array, tf_tensor2)
    if tf_tensor_dtype == dtypes.int32:
      # int32 tensors are always placed on CPU today (see int32 host-memory
      # pinning in the eager runtime), regardless of the source device.
      self.assertEqual(tf_tensor2.device,
                       "/job:localhost/replica:0/task:0/device:CPU:0")
    else:
      self.assertEqual(tf_tensor_device, tf_tensor2.device)

  def testRoundTripWithoutToDlpack(self) -> None:
    """`np.from_dlpack` should work directly on a CPU eager tensor.

    This exercises the numpy-side entry point (which calls our
    `__dlpack__` protocol method under the hood) rather than the explicit
    `dlpack.to_dlpack`/`from_dlpack` pair covered by `testRoundTrip`.
    """
    np_array = np.random.randint(0, 10, [42])
    self.assertAllEqual(
        np.from_dlpack(constant_op.constant(np_array).cpu()), np_array
    )

  def testTensorsCanBeConsumedOnceOnly(self) -> None:
    """A dlpack capsule must raise, not silently misbehave, on reuse."""
    np.random.seed(42)
    np_array = np.random.randint(0, 10, (2, 3, 4))
    tf_tensor = constant_op.constant(np_array, dtype=np.float32)
    dlcapsule = dlpack.to_dlpack(tf_tensor)
    del tf_tensor  # should still work
    _ = dlpack.from_dlpack(dlcapsule)  # First consumption: fine.

    def ConsumeDLPackTensor() -> None:
      dlpack.from_dlpack(dlcapsule)  # Second consumption: should raise.

    self.assertRaisesRegex(Exception,
                           ".*a DLPack tensor may be consumed at most once.*",
                           ConsumeDLPackTensor)

  def testDLPackFromWithoutContextInitialization(self) -> None:
    """`from_dlpack` must (re-)initialize the eager context itself."""
    tf_tensor = constant_op.constant(1)
    dlcapsule = dlpack.to_dlpack(tf_tensor)
    # Resetting the context doesn't cause an error.
    context._reset_context()  # pylint: disable=protected-access
    _ = dlpack.from_dlpack(dlcapsule)

  def testUnsupportedTypeToDLPack(self) -> None:
    """Quantized dtypes have no DLPack equivalent and should error clearly."""

    def UnsupportedQint16() -> None:
      tf_tensor = constant_op.constant([[1, 4], [5, 2]], dtype=dtypes.qint16)
      _ = dlpack.to_dlpack(tf_tensor)

    self.assertRaisesRegex(Exception, ".* is not supported by dlpack",
                           UnsupportedQint16)

  def testMustPassTensorArgumentToDLPack(self) -> None:
    """Passing a non-Tensor should fail fast with a clear message."""
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "The argument to `to_dlpack` must be a TF tensor, not Python object"):
      dlpack.to_dlpack([1])


if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()
