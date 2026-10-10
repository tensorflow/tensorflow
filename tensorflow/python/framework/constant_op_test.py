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
"""Tests for tensorflow.python.framework.constant_op."""

from absl.testing import parameterized
import numpy as np

from google.protobuf import text_format

from tensorflow.core.framework import graph_pb2
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops.parallel_for import control_flow_ops
from tensorflow.python.platform import test


class _ArrayProtocol:
  """Minimal stand-in for a container exposing `__array__` (e.g. a Series)."""

  def __init__(self, array):
    self._array = array

  def __array__(self, dtype=None, copy=None):  # pylint: disable=unused-argument
    return np.asarray(self._array, dtype=dtype)


class _ArrayInterfaceProtocol:
  """Minimal stand-in for a container exposing `__array_interface__`."""

  def __init__(self, array):
    self._array = np.asarray(array)

  @property
  def __array_interface__(self):
    return self._array.__array_interface__


class ConstantOpTest(test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      dtypes.bfloat16,
      dtypes.complex128,
      dtypes.complex64,
      dtypes.double,
      dtypes.float16,
      dtypes.float32,
      dtypes.float64,
      dtypes.half,
      dtypes.int16,
      dtypes.int32,
      dtypes.int64,
      dtypes.int8,
      dtypes.qint16,
      dtypes.qint32,
      dtypes.qint8,
      dtypes.quint16,
      dtypes.quint8,
      dtypes.uint16,
      dtypes.uint32,
      dtypes.uint64,
      dtypes.uint8,
  )
  def test_convert_string_to_number(self, dtype):
    with self.assertRaises(TypeError):
      constant_op.constant("hello", dtype)

  def _make_graph_def(self, text):
    ret = graph_pb2.GraphDef()
    text_format.Parse(text, ret)
    return ret

  def test_eager_const_xla(self):

    @def_function.function(jit_compile=True)
    def f_using_eagerconst(x):
      graph_def = self._make_graph_def("""
         node { name: 'x' op: 'Const'
           attr { key: 'dtype' value { type: DT_FLOAT } }
           attr { key: 'value' value { tensor {
             dtype: DT_FLOAT tensor_shape {} float_val: NaN } } } }
         node { name: 'const' op: '_EagerConst' input: 'x:0'
                attr { key: 'T' value { type: DT_FLOAT } }}""")
      x_id = importer.import_graph_def(
          graph_def,
          input_map={"x:0": x},
          return_elements=["const"],
          name="import")[0].outputs[0]
      return x_id

    self.assertAllClose(3.14, f_using_eagerconst(constant_op.constant(3.14)))

  def test_np_array_memory_not_shared(self):
    # An arbitrarily large loop number to test memory sharing
    for _ in range(10000):
      x = np.arange(10)
      xt = constant_op.constant(x)
      x[3] = 42
      # Changing the input array after `xt` is created should not affect `xt`
      self.assertEqual(xt.numpy()[3], 3)

  @parameterized.named_parameters(
      *[{
          "testcase_name": f"_{int_d.name}_{np.dtype(flt_d).name}",
          "int_dtype": int_d,
          "float_dtype": flt_d,
      } for int_d in (dtypes.int8, dtypes.int32, dtypes.int64, dtypes.uint8)
        for flt_d in (np.float16, np.float32, np.float64,
                      dtypes.bfloat16.as_numpy_dtype,
                      dtypes.float8_e4m3fn.as_numpy_dtype,
                      dtypes.float8_e5m2.as_numpy_dtype)]
  )
  def test_non_finite_to_integer_dtype_raises(self, int_dtype, float_dtype):
    # A NumPy array holding NaN or Inf must not be silently mapped to the
    # smallest representable integer. That disagrees with the Python-list
    # conversion path, which rejects such values with a TypeError. The check
    # must hold regardless of the floating point precision of the input.
    non_finite = (
        # NumPy scalars and 0-D arrays (ndim == 0, size == 1) must be rejected
        # as well.
        float_dtype(np.nan),
        np.array(np.nan, dtype=float_dtype),
        np.array([np.nan], dtype=float_dtype),
        np.array([np.inf], dtype=float_dtype),
        np.array([-np.inf], dtype=float_dtype),
        np.array([1.0, np.nan], dtype=float_dtype),
        # NaN must also be detected when it is not an extreme element.
        np.array([[1.0, np.nan], [3.0, 4.0]], dtype=float_dtype),
        # Array-like containers never pass through the Python wrappers, so the
        # C++ conversion routine has to reject them on its own. `pandas.Series`
        # is the motivating example; the two stubs stand in for the `__array__`
        # and `__array_interface__` protocols without the extra dependency.
        _ArrayProtocol(np.array([np.nan], dtype=float_dtype)),
        _ArrayProtocol(np.array([1.0, np.inf], dtype=float_dtype)),
    )
    expected_regex = r"NaN and Inf cannot be represented as an integer"
    for value in non_finite:
      with self.assertRaisesRegex(TypeError, expected_regex):
        constant_op.constant(value, dtype=int_dtype)
      with self.assertRaisesRegex(TypeError, expected_regex):
        ops.convert_to_tensor(value, dtype=int_dtype)

    # `__array_interface__` containers (e.g. a PIL image) are opaque to the
    # Python wrappers as well. The protocol only round-trips plain NumPy
    # dtypes, so this is checked with float32 rather than the 16/8-bit types.
    for value in (_ArrayInterfaceProtocol(np.array([np.nan], dtype=np.float32)),
                  _ArrayInterfaceProtocol(
                      np.array([1.0, np.inf], dtype=np.float32))):
      with self.assertRaisesRegex(TypeError, expected_regex):
        constant_op.constant(value, dtype=int_dtype)
      with self.assertRaisesRegex(TypeError, expected_regex):
        ops.convert_to_tensor(value, dtype=int_dtype)

    # Graph construction must reject them as well -- both the plain arrays
    # (which reach `make_tensor_proto`) and the array-like containers.
    with ops.Graph().as_default():
      with self.assertRaisesRegex(TypeError, expected_regex):
        constant_op.constant(np.array([np.nan], dtype=float_dtype),
                             dtype=int_dtype)
      with self.assertRaisesRegex(TypeError, expected_regex):
        constant_op.constant(_ArrayProtocol(np.array([np.nan],
                                                     dtype=float_dtype)),
                             dtype=int_dtype)

    # Complex inputs with a non-finite real or imaginary plane are rejected
    # too: complex kinds were previously skipped by the guard. Scalars, 0-D
    # arrays, 1-D arrays and 2-D arrays must all pass through the guard, so each
    # container shape is exercised here. `cdt.as_numpy_dtype` is a callable
    # scalar type, which `np.dtype(cdt.as_numpy_dtype)` is not.
    for cdt in (dtypes.complex64, dtypes.complex128):
      cfloat_dtype = cdt.as_numpy_dtype
      for value in (cfloat_dtype(complex(np.nan, 1.0)),
                    np.array(complex(1.0, np.inf), dtype=cfloat_dtype),
                    np.array([complex(np.nan, 1.0)], dtype=cfloat_dtype),
                    np.array([complex(1.0, np.inf)], dtype=cfloat_dtype),
                    np.array([[complex(1.0, 0.0), complex(1.0, np.nan)],
                              [complex(3.0, 0.0), complex(4.0, 0.0)]],
                             dtype=cfloat_dtype)):
        with self.assertRaisesRegex(TypeError, expected_regex):
          constant_op.constant(value, dtype=int_dtype)
        with self.assertRaisesRegex(TypeError, expected_regex):
          ops.convert_to_tensor(value, dtype=int_dtype)

    # Empty arrays and NumPy scalars take their respective shortcuts. 1.25 is
    # exactly representable in every dtype under test -- including the float8
    # variants, which have only two or three mantissa bits -- so the truncating
    # result is unambiguous.
    self.assertAllEqual([], constant_op.constant(
        np.array([], dtype=float_dtype), dtype=int_dtype))
    self.assertAllEqual(1, constant_op.constant(
        np.array(1.25, dtype=float_dtype), dtype=int_dtype))

    # Finite floats keep the previous (truncating) behaviour.
    self.assertAllEqual([1], constant_op.constant(
        np.array([1.25], dtype=float_dtype), dtype=int_dtype))

  def test_eager_const_grad_error(self):

    @def_function.function
    def f_using_eagerconst():
      x = constant_op.constant(1.)
      graph_def = self._make_graph_def("""
         node { name: 'x' op: 'Placeholder'
                attr { key: 'dtype' value { type: DT_FLOAT } }}
         node { name: 'const' op: '_EagerConst' input: 'x:0'
                attr { key: 'T' value { type: DT_FLOAT } }}""")
      x_id = importer.import_graph_def(
          graph_def,
          input_map={"x:0": x},
          return_elements=["const"],
          name="import")[0].outputs[0]
      gradients_impl.gradients(x_id, x)
      return x_id

    with self.assertRaisesRegex(AssertionError, "Please file a bug"):
      f_using_eagerconst()

  def test_eager_const_pfor(self):

    @def_function.function
    def f_using_eagerconst():

      def vec_fn(x):
        graph_def = self._make_graph_def("""
           node { name: 'x' op: 'Const'
             attr { key: 'dtype' value { type: DT_FLOAT } }
             attr { key: 'value' value { tensor {
               dtype: DT_FLOAT tensor_shape {} float_val: 3.14 } } } }
           node { name: 'const' op: '_EagerConst' input: 'x:0'
                  attr { key: 'T' value { type: DT_FLOAT } }}""")
        return importer.import_graph_def(
            graph_def,
            input_map={"x:0": x},
            return_elements=["const"],
            name="import")[0].outputs[0]

      return control_flow_ops.vectorized_map(
          vec_fn, constant_op.constant([1., 2.]), fallback_to_while_loop=False)

    self.assertAllClose([1., 2.], f_using_eagerconst())

  def test_eager_tensor_numpy_array_protocol_scalar(self):
    t = constant_op.constant(42.0)
    arr = t.__array__()
    self.assertIsInstance(arr, np.ndarray)
    self.assertEqual(arr.shape, ())
    self.assertEqual(arr.item(), 42.0)


if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()
