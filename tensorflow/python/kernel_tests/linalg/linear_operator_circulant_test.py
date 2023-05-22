# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
import contextlib

from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import config
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.ops.linalg import linalg
from tensorflow.python.ops.linalg import linear_operator_circulant
from tensorflow.python.ops.linalg import linear_operator_test_util
from tensorflow.python.ops.signal import fft_ops
from tensorflow.python.platform import test

rng = np.random.RandomState(0)
_to_complex = linear_operator_circulant._to_complex

exponential_power_convolution_kernel = (
    linear_operator_circulant.exponential_power_convolution_kernel)


def _operator_from_kernel(kernel, d, **kwargs):
  spectrum = linear_operator_circulant._FFT_OP[d](
      math_ops.cast(kernel, dtypes.complex64))
  if d == 1:
    return linear_operator_circulant.LinearOperatorCirculant(spectrum, **kwargs)
  elif d == 2:
    return linear_operator_circulant.LinearOperatorCirculant2D(
        spectrum, **kwargs)
  elif d == 3:
    return linear_operator_circulant.LinearOperatorCirculant3D(
        spectrum, **kwargs)


def _spectrum_for_symmetric_circulant(
    spectrum_shape,
    d,
    ensure_self_adjoint_and_pd,
    dtype,
):
  """Spectrum for d-dimensional real/symmetric circulant."""
  grid_shape = spectrum_shape[-d:]

  if grid_shape == (0,) * d:
    kernel = array_ops.reshape(math_ops.cast([], dtype), grid_shape)
  else:
    kernel = exponential_power_convolution_kernel(
        grid_shape=grid_shape,
        # power=2 with this scale and no inflation will have some negative
        # spectra. It will still be real/symmetric.
        length_scale=math_ops.cast([0.2] * d, dtype.real_dtype),
        power=1 if ensure_self_adjoint_and_pd else 2,
        zero_inflation=0.2 if ensure_self_adjoint_and_pd else None,
    )
  spectrum = linear_operator_circulant._FFT_OP[d](_to_complex(kernel))
  spectrum = math_ops.cast(spectrum, dtype)
  return array_ops.broadcast_to(spectrum, spectrum_shape)


@test_util.run_all_in_graph_and_eager_modes
class ExponentialPowerConvolutionKernelTest(parameterized.TestCase,
                                            test.TestCase):

  def assert_diag_is_ones(self, matrix, rtol):
    self.assertAllClose(
        np.ones_like(np.diag(matrix)), np.diag(matrix), rtol=rtol)

  def assert_real_symmetric(self, matrix, tol):
    self.assertAllClose(np.zeros_like(matrix.imag), matrix.imag, atol=tol)
    self.assertAllClose(matrix.real, matrix.real.T, rtol=tol)

  @parameterized.named_parameters(
      dict(testcase_name="1Deven_power1", grid_shape=[10], power=1.),
      dict(testcase_name="2Deven_power1", grid_shape=[4, 6], power=1.),
      dict(testcase_name="3Deven_power1", grid_shape=[4, 6, 8], power=1.),
      dict(testcase_name="3Devenodd_power1", grid_shape=[4, 5, 7], power=1.),
      dict(testcase_name="1Dodd_power2", grid_shape=[9], power=2.),
      dict(testcase_name="2Deven_power2", grid_shape=[8, 4], power=2.),
      dict(testcase_name="3Devenodd_power2", grid_shape=[4, 5, 3], power=2.),
  )
  def test_makes_symmetric_and_real_circulant_with_ones_diag(
      self, grid_shape, power):
    d = len(grid_shape)
    length_scale = [0.2] * d
    kernel = exponential_power_convolution_kernel(
        grid_shape=grid_shape,
        length_scale=length_scale,
        power=power)
    operator = _operator_from_kernel(kernel, d)

    matrix = self.evaluate(operator.to_dense())

    tol = np.finfo(matrix.dtype).eps * np.prod(grid_shape)
    self.assert_real_symmetric(matrix, tol)
    self.assert_diag_is_ones(matrix, rtol=tol)

  @parameterized.named_parameters(
      dict(testcase_name="1D", grid_shape=[10]),
      dict(testcase_name="2D", grid_shape=[5, 5]),
      dict(testcase_name="3D", grid_shape=[5, 4, 3]),
  )
  def test_zero_inflation(self, grid_shape):
    d = len(grid_shape)
    length_scale = [0.2] * d

    kernel_no_inflation = exponential_power_convolution_kernel(
        grid_shape=grid_shape,
        length_scale=length_scale,
        zero_inflation=None,
    )
    matrix_no_inflation = self.evaluate(
        _operator_from_kernel(kernel_no_inflation, d).to_dense())

    kernel_inflation_one_half = exponential_power_convolution_kernel(
        grid_shape=grid_shape,
        length_scale=length_scale,
        zero_inflation=0.5,
    )
    matrix_inflation_one_half = self.evaluate(
        _operator_from_kernel(kernel_inflation_one_half, d).to_dense())

    kernel_inflation_one = exponential_power_convolution_kernel(
        grid_shape=grid_shape,
        length_scale=length_scale,
        zero_inflation=1.0,
    )
    matrix_inflation_one = self.evaluate(
        _operator_from_kernel(kernel_inflation_one, d).to_dense())

    tol = np.finfo(matrix_no_inflation.dtype).eps * np.prod(grid_shape)

    # In all cases, matrix should be real and symmetric.
    self.assert_real_symmetric(matrix_no_inflation, tol)
    self.assert_real_symmetric(matrix_inflation_one, tol)
    self.assert_real_symmetric(matrix_inflation_one_half, tol)

    # In all cases, the diagonal should be all ones.
    self.assert_diag_is_ones(matrix_no_inflation, rtol=tol)
    self.assert_diag_is_ones(matrix_inflation_one_half, rtol=tol)
    self.assert_diag_is_ones(matrix_inflation_one, rtol=tol)

    def _matrix_with_zerod_diag(matrix):
      return matrix - np.diag(np.diag(matrix))

    # Inflation = 0.5 means the off-diagonal is deflated by factor (1 - .5) = .5
    self.assertAllClose(
        _matrix_with_zerod_diag(matrix_no_inflation) * 0.5,
        _matrix_with_zerod_diag(matrix_inflation_one_half), rtol=tol)

    # Inflation = 1.0 means the off-diagonal is deflated by factor (1 - 1) = 0
    self.assertAllClose(
        np.zeros_like(matrix_inflation_one),
        _matrix_with_zerod_diag(matrix_inflation_one), rtol=tol)

  @parameterized.named_parameters(
      dict(testcase_name="1D", grid_shape=[10]),
      dict(testcase_name="2D", grid_shape=[5, 5]),
      dict(testcase_name="3D", grid_shape=[5, 4, 3]),
  )
  def test_tiny_scale_corresponds_to_identity_matrix(self, grid_shape):
    d = len(grid_shape)

    kernel = exponential_power_convolution_kernel(
        grid_shape=grid_shape, length_scale=[0.001] * d, power=2)
    matrix = self.evaluate(_operator_from_kernel(kernel, d).to_dense())

    tol = np.finfo(matrix.dtype).eps * np.prod(grid_shape)
    self.assertAllClose(matrix, np.eye(np.prod(grid_shape)), atol=tol)
    self.assert_real_symmetric(matrix, tol)

  @parameterized.named_parameters(
      dict(testcase_name="1D", grid_shape=[10]),
      dict(testcase_name="2D", grid_shape=[5, 5]),
      dict(testcase_name="3D", grid_shape=[5, 4, 3]),
  )
  def test_huge_scale_corresponds_to_ones_matrix(self, grid_shape):
    d = len(grid_shape)

    kernel = exponential_power_convolution_kernel(
        grid_shape=grid_shape, length_scale=[100.] * d, power=2)
    matrix = self.evaluate(_operator_from_kernel(kernel, d).to_dense())

    tol = np.finfo(matrix.dtype).eps * np.prod(grid_shape) * 50
    self.assert_real_symmetric(matrix, tol)
    self.assertAllClose(np.ones_like(matrix), matrix, rtol=tol)


@test_util.run_all_in_graph_and_eager_modes
class LinearOperatorCirculantBaseTest(object):
  """Common class for circulant tests."""

  _atol = {
      dtypes.float16: 1e-3,
      dtypes.float32: 1e-6,
      dtypes.float64: 1e-7,
      dtypes.complex64: 1e-6,
      dtypes.complex128: 1e-7
  }
  _rtol = {
      dtypes.float16: 1e-3,
      dtypes.float32: 1e-6,
      dtypes.float64: 1e-7,
      dtypes.complex64: 1e-6,
      dtypes.complex128: 1e-7
  }

  @contextlib.contextmanager
  def _constrain_devices_and_set_default(self, sess, use_gpu, force_gpu):
    """We overwrite the FFT operation mapping for testing."""
    with test.TestCase._constrain_devices_and_set_default(
        self, sess, use_gpu, force_gpu) as sess:
      yield sess

  def _shape_to_spectrum_shape(self, shape):
    # If spectrum.shape = batch_shape + [N],
    # this creates an operator of shape batch_shape + [N, N]
    return shape[:-1]

  def _spectrum_to_circulant_1d(self, spectrum, shape, dtype):
    """Creates a circulant matrix from a spectrum.

    Intentionally done in an explicit yet inefficient way.  This provides a
    cross check to the main code that uses fancy reshapes.

    Args:
      spectrum: Float or complex `Tensor`.
      shape:  Python list.  Desired shape of returned matrix.
      dtype:  Type to cast the returned matrix to.

    Returns:
      Circulant (batch) matrix of desired `dtype`.
    """
    spectrum = _to_complex(spectrum)
    spectrum_shape = self._shape_to_spectrum_shape(shape)
    domain_dimension = spectrum_shape[-1]
    if not domain_dimension:
      return array_ops.zeros(shape, dtype)

    # Explicitly compute the action of spectrum on basis vectors.
    matrix_rows = []
    for m in range(domain_dimension):
      x = np.zeros([domain_dimension])
      # x is a basis vector.
      x[m] = 1.0
      fft_x = fft_ops.fft(math_ops.cast(x, spectrum.dtype))
      h_convolve_x = fft_ops.ifft(spectrum * fft_x)
      matrix_rows.append(h_convolve_x)
    matrix = array_ops_stack.stack(matrix_rows, axis=-1)
    return math_ops.cast(matrix, dtype)


class LinearOperatorCirculantTestSelfAdjointOperator(
    LinearOperatorCirculantBaseTest,
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
  """Test of LinearOperatorCirculant when operator is self-adjoint.

  Real spectrum <==> Self adjoint operator.
  Note that when the spectrum is real, the operator may still be complex.
  """

  @staticmethod
  def dtypes_to_test():
    # This operator will always be complex because, although the spectrum is
    # real, the matrix will not be real.
    return [dtypes.complex64, dtypes.complex128]

  @staticmethod
  def optional_tests():
    """List of optional test names to run."""
    return [
        "operator_matmul_with_same_type",
        "operator_solve_with_same_type",
    ]

  def operator_and_matrix(self,
                          shape_info,
                          dtype,
                          use_placeholder,
                          ensure_self_adjoint_and_pd=False):
    shape = shape_info.shape
    # For this test class, we are creating real spectrums.
    # We also want the spectrum to have eigenvalues bounded away from zero.
    #
    # spectrum is bounded away from zero.
    spectrum = linear_operator_test_util.random_sign_uniform(
        shape=self._shape_to_spectrum_shape(shape), minval=1., maxval=2.)
    if ensure_self_adjoint_and_pd:
      spectrum = math_ops.abs(spectrum)
    # If dtype is complex, cast spectrum to complex.  The imaginary part will be
    # zero, so the operator will still be self-adjoint.
    spectrum = math_ops.cast(spectrum, dtype)

    lin_op_spectrum = spectrum

    if use_placeholder:
      lin_op_spectrum = array_ops.placeholder_with_default(spectrum, shape=None)

    operator = linalg.LinearOperatorCirculant(
        lin_op_spectrum,
        is_self_adjoint=True,
        is_positive_definite=True if ensure_self_adjoint_and_pd else None,
        input_output_dtype=dtype)

    mat = self._spectrum_to_circulant_1d(spectrum, shape, dtype=dtype)

    return operator, mat

  @test_util.disable_xla("No registered Const")
  def test_simple_hermitian_spectrum_gives_operator_with_zero_imag_part(self):
    with self.cached_session():
      spectrum = math_ops.cast([1. + 0j, 1j, -1j], dtypes.complex64)
      operator = linalg.LinearOperatorCirculant(
          spectrum, input_output_dtype=dtypes.complex64)
      matrix = operator.to_dense()
      imag_matrix = math_ops.imag(matrix)
      eps = np.finfo(np.float32).eps
      np.testing.assert_allclose(
          0, self.evaluate(imag_matrix), rtol=0, atol=eps * 3)

  def test_tape_safe(self):
    spectrum = variables_module.Variable(
        math_ops.cast([1. + 0j, 1. + 0j], dtypes.complex64))
    operator = linalg.LinearOperatorCirculant(spectrum, is_self_adjoint=True)
    self.check_tape_safe(operator)

  def test_convert_variables_to_tensors(self):
    spectrum = variables_module.Variable(
        math_ops.cast([1. + 0j, 1. + 0j], dtypes.complex64))
    operator = linalg.LinearOperatorCirculant(spectrum, is_self_adjoint=True)
    with self.cached_session() as sess:
      sess.run([spectrum.initializer])
      self.check_convert_variables_to_tensors(operator)


class LinearOperatorCirculantTestHermitianSpectrum(
    LinearOperatorCirculantBaseTest,
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
  """Test of LinearOperatorCirculant when the spectrum is Hermitian.

  Hermitian spectrum <==> Real valued operator.  We test both real and complex
  dtypes here though.  So in some cases the matrix will be complex but with
  zero imaginary part.
  """

  def tearDown(self):
    config.enable_tensor_float_32_execution(self.tf32_keep_)

  def setUp(self):
    self.tf32_keep_ = config.tensor_float_32_execution_enabled()
    config.enable_tensor_float_32_execution(False)

  @staticmethod
  def optional_tests():
    """List of optional test names to run."""
    return [
        "operator_matmul_with_same_type",
        "operator_solve_with_same_type",
    ]

  def operator_and_matrix(self,
                          shape_info,
                          dtype,
                          use_placeholder,
                          ensure_self_adjoint_and_pd=False):
    shape = shape_info.shape
    spectrum = _spectrum_for_symmetric_circulant(
        spectrum_shape=self._shape_to_spectrum_shape(shape),
        d=1,
        ensure_self_adjoint_and_pd=ensure_self_adjoint_and_pd,
        dtype=dtype)

    lin_op_spectrum = spectrum

    if use_placeholder:
      lin_op_spectrum = array_ops.placeholder_with_default(spectrum, shape=None)

    operator = linalg.LinearOperatorCirculant(
        lin_op_spectrum,
        input_output_dtype=dtype,
        is_positive_definite=True if ensure_self_adjoint_and_pd else None,
        is_self_adjoint=True if ensure_self_adjoint_and_pd else None,
    )

    mat = self._spectrum_to_circulant_1d(spectrum, shape, dtype=dtype)

    return operator, mat

  @test_util.disable_xla("No registered Const")
  def test_simple_hermitian_spectrum_gives_operator_with_zero_imag_part(self):
    with self.cached_session():
      spectrum = math_ops.cast([1. + 0j, 1j, -1j], dtypes.complex64)
      operator = linalg.LinearOperatorCirculant(
          spectrum, input_output_dtype=dtypes.complex64)
      matrix = operator.to_dense()
      imag_matrix = math_ops.imag(matrix)
      eps = np.finfo(np.float32).eps
      np.testing.assert_allclose(
          0, self.evaluate(imag_matrix), rtol=0, atol=eps * 3)

  def test_tape_safe(self):
    spectrum = variables_module.Variable(
        math_ops.cast([1. + 0j, 1. + 1j], dtypes.complex64))
    operator = linalg.LinearOperatorCirculant(spectrum, is_self_adjoint=False)
    self.check_tape_safe(operator)


class LinearOperatorCirculantTestNonHermitianSpectrum(
    LinearOperatorCirculantBaseTest,
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
  """Test of LinearOperatorCirculant when the spectrum is not Hermitian.

  Non-Hermitian spectrum <==> Complex valued operator.
  We test only complex dtypes here.
  """

  @staticmethod
  def dtypes_to_test():
    return [dtypes.complex64, dtypes.complex128]

  # Skip Cholesky since we are explicitly testing non-hermitian
  # spectra.
  @staticmethod
  def skip_these_tests():
    return ["cholesky", "eigvalsh"]

  @staticmethod
  def optional_tests():
    """List of optional test names to run."""
    return [
        "operator_matmul_with_same_type",
        "operator_solve_with_same_type",
    ]

  def operator_and_matrix(self,
                          shape_info,
                          dtype,
                          use_placeholder,
                          ensure_self_adjoint_and_pd=False):
    del ensure_self_adjoint_and_pd
    shape = shape_info.shape
    # Will be well conditioned enough to get accurate solves.
    spectrum = linear_operator_test_util.random_sign_uniform(
        shape=self._shape_to_spectrum_shape(shape),
        dtype=dtype,
        minval=1.,
        maxval=2.)

    lin_op_spectrum = spectrum

    if use_placeholder:
      lin_op_spectrum = array_ops.placeholder_with_default(spectrum, shape=None)

    operator = linalg.LinearOperatorCirculant(
        lin_op_spectrum, input_output_dtype=dtype)

    self.assertEqual(
        operator.parameters,
        {
            "input_output_dtype": dtype,
            "is_non_singular": None,
            "is_positive_definite": None,
            "is_self_adjoint": None,
            "is_square": True,
            "name": "LinearOperatorCirculant",
            "spectrum": lin_op_spectrum,
        })

    mat = self._spectrum_to_circulant_1d(spectrum, shape, dtype=dtype)

    return operator, mat

  @test_util.disable_xla("No registered Const")
  def test_simple_hermitian_spectrum_gives_operator_with_zero_imag_part(self):
    with self.cached_session():
      spectrum = math_ops.cast([1. + 0j, 1j, -1j], dtypes.complex64)
      operator = linalg.LinearOperatorCirculant(
          spectrum, input_output_dtype=dtypes.complex64)
      matrix = operator.to_dense()
      imag_matrix = math_ops.imag(matrix)
      eps = np.finfo(np.float32).eps
      np.testing.assert_allclose(
          0, self.evaluate(imag_matrix), rtol=0, atol=eps * 3)

  def test_simple_positive_real_spectrum_gives_self_adjoint_pos_def_oper(self):
    with self.cached_session() as sess:
      spectrum = math_ops.cast([6., 4, 2], dtypes.complex64)
      operator = linalg.LinearOperatorCirculant(
          spectrum, input_output_dtype=dtypes.complex64)
      matrix, matrix_h = sess.run(
          [operator.to_dense(),
           linalg.adjoint(operator.to_dense())])
      self.assertAllClose(matrix, matrix_h)
      self.evaluate(operator.assert_positive_definite())  # Should not fail
      self.evaluate(operator.assert_self_adjoint())  # Should not fail

  def test_defining_operator_using_real_convolution_kernel(self):
    with self.cached_session():
      convolution_kernel = [1., 2., 1.]
      spectrum = fft_ops.fft(
          math_ops.cast(convolution_kernel, dtypes.complex64))

      # spectrum is shape [3] ==> operator is shape [3, 3]
      # spectrum is Hermitian ==> operator is real.
      operator = linalg.LinearOperatorCirculant(spectrum)

      # Allow for complex output so we can make sure it has zero imag part.
      self.assertEqual(operator.dtype, dtypes.complex64)

      matrix = self.evaluate(operator.to_dense())
      np.testing.assert_allclose(0, np.imag(matrix), atol=1e-6)

  @test_util.run_v1_only("currently failing on v2")
  def test_hermitian_spectrum_gives_operator_with_zero_imag_part(self):
    with self.cached_session():
      # Make spectrum the FFT of a real convolution kernel h.  This ensures that
      # spectrum is Hermitian.
      h = linear_operator_test_util.random_normal(shape=(3, 4))
      spectrum = fft_ops.fft(math_ops.cast(h, dtypes.complex64))
      operator = linalg.LinearOperatorCirculant(
          spectrum, input_output_dtype=dtypes.complex64)
      matrix = operator.to_dense()
      imag_matrix = math_ops.imag(matrix)
      eps = np.finfo(np.float32).eps
      np.testing.assert_allclose(
          0, self.evaluate(imag_matrix), rtol=0, atol=eps * 3 * 4)

  def test_convolution_kernel_same_as_first_row_of_to_dense(self):
    spectrum = [[3., 2., 1.], [2., 1.5, 1.]]
    with self.cached_session():
      operator = linalg.LinearOperatorCirculant(spectrum)
      h = operator.convolution_kernel()
      c = operator.to_dense()

      self.assertAllEqual((2, 3), h.shape)
      self.assertAllEqual((2, 3, 3), c.shape)
      self.assertAllClose(self.evaluate(h), self.evaluate(c)[:, :, 0])

  def test_assert_non_singular_fails_for_singular_operator(self):
    spectrum = math_ops.cast([0 + 0j, 4 + 0j, 2j + 2], dtypes.complex64)
    operator = linalg.LinearOperatorCirculant(spectrum)
    with self.cached_session():
      with self.assertRaisesOpError("Singular operator"):
        self.evaluate(operator.assert_non_singular())

  def test_assert_non_singular_does_not_fail_for_non_singular_operator(self):
    spectrum = math_ops.cast([-3j, 4 + 0j, 2j + 2], dtypes.complex64)
    operator = linalg.LinearOperatorCirculant(spectrum)
    with self.cached_session():
      self.evaluate(operator.assert_non_singular())  # Should not fail

  def test_assert_positive_definite_fails_for_non_positive_definite(self):
    spectrum = math_ops.cast([6. + 0j, 4 + 0j, 2j], dtypes.complex64)
    operator = linalg.LinearOperatorCirculant(spectrum)
    with self.cached_session():
      with self.assertRaisesOpError("Not positive definite"):
        self.evaluate(operator.assert_positive_definite())

  def test_assert_positive_definite_does_not_fail_when_pos_def(self):
    spectrum = math_ops.cast([6. + 0j, 4 + 0j, 2j + 2], dtypes.complex64)
    operator = linalg.LinearOperatorCirculant(spectrum)
    with self.cached_session():
      self.evaluate(operator.assert_positive_definite())  # Should not fail

  def test_real_spectrum_and_not_self_adjoint_hint_raises(self):
    spectrum = [1., 2.]
    with self.assertRaisesRegex(ValueError, "real.*always.*self-adjoint"):
      linalg.LinearOperatorCirculant(spectrum, is_self_adjoint=False)

  def test_real_spectrum_auto_sets_is_self_adjoint_to_true(self):
    spectrum = [1., 2.]
    operator = linalg.LinearOperatorCirculant(spectrum)
    self.assertTrue(operator.is_self_adjoint)


@test_util.run_all_in_graph_and_eager_modes
class LinearOperatorCirculant2DBaseTest(object):
  """Common class for 2D circulant tests."""

  _atol = {
      dtypes.float16: 1e-3,
      dtypes.float32: 1e-6,
      dtypes.float64: 1e-7,
      dtypes.complex64: 1e-6,
      dtypes.complex128: 1e-7
  }
  _rtol = {
      dtypes.float16: 1e-3,
      dtypes.float32: 1e-6,
      dtypes.float64: 1e-7,
      dtypes.complex64: 1e-6,
      dtypes.complex128: 1e-7
  }

  @contextlib.contextmanager
  def _constrain_devices_and_set_default(self, sess, use_gpu, force_gpu):
    """We overwrite the FFT operation mapping for testing."""
    with test.TestCase._constrain_devices_and_set_default(
        self, sess, use_gpu, force_gpu) as sess:
      yield sess

  @staticmethod
  def operator_shapes_infos():
    shape_info = linear_operator_test_util.OperatorShapesInfo
    # non-batch operators (n, n) and batch operators.
    return [
        shape_info((0, 0)),
        shape_info((1, 1)),
        shape_info((1, 6, 6)),
        shape_info((3, 4, 4)),
        shape_info((2, 1, 3, 3))
    ]

  @staticmethod
  def optional_tests():
    """List of optional test names to run."""
    return [
        "operator_matmul_with_same_type",
        "operator_solve_with_same_type",
    ]

  def _shape_to_spectrum_shape(self, shape):
    """Get a spectrum shape that will make an operator of desired shape."""
    # This 2D block circulant operator takes a spectrum of shape
    # batch_shape + [N0, N1],
    # and creates and operator of shape
    # batch_shape + [N0*N1, N0*N1]
    if shape == (0, 0):
      return (0, 0)
    elif shape == (1, 1):
      return (1, 1)
    elif shape == (1, 6, 6):
      return (1, 2, 3)
    elif shape == (3, 4, 4):
      return (3, 2, 2)
    elif shape == (2, 1, 3, 3):
      return (2, 1, 3, 1)
    else:
      raise ValueError("Unhandled shape: %s" % shape)

  def _spectrum_to_circulant_2d(self, spectrum, shape, dtype):
    """Creates a block circulant matrix from a spectrum.

    Intentionally done in an explicit yet inefficient way.  This provides a
    cross check to the main code that uses fancy reshapes.

    Args:
      spectrum: Float or complex `Tensor`.
      shape:  Python list.  Desired shape of returned matrix.
      dtype:  Type to cast the returned matrix to.

    Returns:
      Block circulant (batch) matrix of desired `dtype`.
    """
    spectrum = _to_complex(spectrum)
    spectrum_shape = self._shape_to_spectrum_shape(shape)
    domain_dimension = spectrum_shape[-1]
    if not domain_dimension:
      return array_ops.zeros(shape, dtype)

    block_shape = spectrum_shape[-2:]

    # Explicitly compute the action of spectrum on basis vectors.
    matrix_rows = []
    for n0 in range(block_shape[0]):
      for n1 in range(block_shape[1]):
        x = np.zeros(block_shape)
        # x is a basis vector.
        x[n0, n1] = 1.0
        fft_x = fft_ops.fft2d(math_ops.cast(x, spectrum.dtype))
        h_convolve_x = fft_ops.ifft2d(spectrum * fft_x)
        # We want the flat version of the action of the operator on a basis
        # vector, not the block version.
        h_convolve_x = array_ops.reshape(h_convolve_x, shape[:-1])
        matrix_rows.append(h_convolve_x)
    matrix = array_ops_stack.stack(matrix_rows, axis=-1)
    return math_ops.cast(matrix, dtype)


class LinearOperatorCirculant2DTestHermitianSpectrum(
    LinearOperatorCirculant2DBaseTest,
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
  """Test of LinearOperatorCirculant2D when the spectrum is Hermitian.

  Hermitian spectrum <==> Real valued operator.  We test both real and complex
  dtypes here though.  So in some cases the matrix will be complex but with
  zero imaginary part.
  """

  def tearDown(self):
    config.enable_tensor_float_32_execution(self.tf32_keep_)

  def setUp(self):
    self.tf32_keep_ = config.tensor_float_32_execution_enabled()
    config.enable_tensor_float_32_execution(False)

  def operator_and_matrix(self,
                          shape_info,
                          dtype,
                          use_placeholder,
                          ensure_self_adjoint_and_pd=False):
    shape = shape_info.shape
    spectrum = _spectrum_for_symmetric_circulant(
        spectrum_shape=self._shape_to_spectrum_shape(shape),
        d=2,
        ensure_self_adjoint_and_pd=ensure_self_adjoint_and_pd,
        dtype=dtype)

    lin_op_spectrum = spectrum

    if use_placeholder:
      lin_op_spectrum = array_ops.placeholder_with_default(spectrum, shape=None)

    operator = linalg.LinearOperatorCirculant2D(
        lin_op_spectrum,
        is_positive_definite=True if ensure_self_adjoint_and_pd else None,
        is_self_adjoint=True if ensure_self_adjoint_and_pd else None,
        input_output_dtype=dtype)

    self.assertEqual(
        operator.parameters,
        {
            "input_output_dtype": dtype,
            "is_non_singular": None,
            "is_positive_definite": (
                True if ensure_self_adjoint_and_pd else None),
            "is_self_adjoint": (
                True if ensure_self_adjoint_and_pd else None),
            "is_square": True,
            "name": "LinearOperatorCirculant2D",
            "spectrum": lin_op_spectrum,
        })

    mat = self._spectrum_to_circulant_2d(spectrum, shape, dtype=dtype)

    return operator, mat


class LinearOperatorCirculant2DTestNonHermitianSpectrum(
    LinearOperatorCirculant2DBaseTest,
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
  """Test of LinearOperatorCirculant when the spectrum is not Hermitian.

  Non-Hermitian spectrum <==> Complex valued operator.
  We test only complex dtypes here.
  """

  @staticmethod
  def dtypes_to_test():
    return [dtypes.complex64, dtypes.complex128]

  @staticmethod
  def skip_these_tests():
    return ["cholesky", "eigvalsh"]

  def operator_and_matrix(self,
                          shape_info,
                          dtype,
                          use_placeholder,
                          ensure_self_adjoint_and_pd=False):
    del ensure_self_adjoint_and_pd
    shape = shape_info.shape
    # Will be well conditioned enough to get accurate solves.
    spectrum = linear_operator_test_util.random_sign_uniform(
        shape=self._shape_to_spectrum_shape(shape),
        dtype=dtype,
        minval=1.,
        maxval=2.)

    lin_op_spectrum = spectrum

    if use_placeholder:
      lin_op_spectrum = array_ops.placeholder_with_default(spectrum, shape=None)

    operator = linalg.LinearOperatorCirculant2D(
        lin_op_spectrum, input_output_dtype=dtype)

    self.assertEqual(
        operator.parameters,
        {
            "input_output_dtype": dtype,
            "is_non_singular": None,
            "is_positive_definite": None,
            "is_self_adjoint": None,
            "is_square": True,
            "name": "LinearOperatorCirculant2D",
            "spectrum": lin_op_spectrum,
        }
    )

    mat = self._spectrum_to_circulant_2d(spectrum, shape, dtype=dtype)

    return operator, mat

  def test_real_hermitian_spectrum_gives_real_symmetric_operator(self):
    with self.cached_session():  # Necessary for fft_kernel_label_map
      # This is a real and hermitian spectrum.
      spectrum = [[1., 2., 2.], [3., 4., 4.], [3., 4., 4.]]
      operator = linalg.LinearOperatorCirculant(spectrum)

      matrix_tensor = operator.to_dense()
      self.assertEqual(matrix_tensor.dtype, dtypes.complex64)
      matrix_t = array_ops.matrix_transpose(matrix_tensor)
      imag_matrix = math_ops.imag(matrix_tensor)
      matrix, matrix_transpose, imag_matrix = self.evaluate(
          [matrix_tensor, matrix_t, imag_matrix])

      np.testing.assert_allclose(0, imag_matrix, atol=1e-6)
      self.assertAllClose(matrix, matrix_transpose, atol=1e-6)

  def test_real_spectrum_gives_self_adjoint_operator(self):
    with self.cached_session():
      # This is a real and hermitian spectrum.
      spectrum = linear_operator_test_util.random_normal(
          shape=(3, 3), dtype=dtypes.float32)
      operator = linalg.LinearOperatorCirculant2D(spectrum)

      matrix_tensor = operator.to_dense()
      self.assertEqual(matrix_tensor.dtype, dtypes.complex64)
      matrix_h = linalg.adjoint(matrix_tensor)
      matrix, matrix_h = self.evaluate([matrix_tensor, matrix_h])
      self.assertAllClose(matrix, matrix_h, atol=1e-5)

  def test_assert_non_singular_fails_for_singular_operator(self):
    spectrum = math_ops.cast([[0 + 0j, 4 + 0j], [2j + 2, 3. + 0j]],
                             dtypes.complex64)
    operator = linalg.LinearOperatorCirculant2D(spectrum)
    with self.cached_session():
      with self.assertRaisesOpError("Singular operator"):
        self.evaluate(operator.assert_non_singular())

  def test_assert_non_singular_does_not_fail_for_non_singular_operator(self):
    spectrum = math_ops.cast([[-3j, 4 + 0j], [2j + 2, 3. + 0j]],
                             dtypes.complex64)
    operator = linalg.LinearOperatorCirculant2D(spectrum)
    with self.cached_session():
      self.evaluate(operator.assert_non_singular())  # Should not fail

  def test_assert_positive_definite_fails_for_non_positive_definite(self):
    spectrum = math_ops.cast([[6. + 0j, 4 + 0j], [2j, 3. + 0j]],
                             dtypes.complex64)
    operator = linalg.LinearOperatorCirculant2D(spectrum)
    with self.cached_session():
      with self.assertRaisesOpError("Not positive definite"):
        self.evaluate(operator.assert_positive_definite())

  def test_assert_positive_definite_does_not_fail_when_pos_def(self):
    spectrum = math_ops.cast([[6. + 0j, 4 + 0j], [2j + 2, 3. + 0j]],
                             dtypes.complex64)
    operator = linalg.LinearOperatorCirculant2D(spectrum)
    with self.cached_session():
      self.evaluate(operator.assert_positive_definite())  # Should not fail

  def test_real_spectrum_and_not_self_adjoint_hint_raises(self):
    spectrum = [[1., 2.], [3., 4]]
    with self.assertRaisesRegex(ValueError, "real.*always.*self-adjoint"):
      linalg.LinearOperatorCirculant2D(spectrum, is_self_adjoint=False)

  def test_real_spectrum_auto_sets_is_self_adjoint_to_true(self):
    spectrum = [[1., 2.], [3., 4]]
    operator = linalg.LinearOperatorCirculant2D(spectrum)
    self.assertTrue(operator.is_self_adjoint)

  def test_invalid_rank_raises(self):
    spectrum = array_ops.constant(np.float32(rng.rand(2)))
    with self.assertRaisesRegex(ValueError, "must have at least 2 dimensions"):
      linalg.LinearOperatorCirculant2D(spectrum)

  def test_tape_safe(self):
    spectrum = variables_module.Variable(
        math_ops.cast([[1. + 0j, 1. + 0j], [1. + 1j, 2. + 2j]],
                      dtypes.complex64))
    operator = linalg.LinearOperatorCirculant2D(spectrum)
    self.check_tape_safe(operator)


@test_util.run_all_in_graph_and_eager_modes
class LinearOperatorCirculant3DTest(test.TestCase):
  """Simple test of the 3D case.  See also the 1D and 2D tests."""

  _atol = {
      dtypes.float16: 1e-3,
      dtypes.float32: 1e-6,
      dtypes.float64: 1e-7,
      dtypes.complex64: 1e-6,
      dtypes.complex128: 1e-7
  }
  _rtol = {
      dtypes.float16: 1e-3,
      dtypes.float32: 1e-6,
      dtypes.float64: 1e-7,
      dtypes.complex64: 1e-6,
      dtypes.complex128: 1e-7
  }

  @contextlib.contextmanager
  def _constrain_devices_and_set_default(self, sess, use_gpu, force_gpu):
    """We overwrite the FFT operation mapping for testing."""
    with test.TestCase._constrain_devices_and_set_default(
        self, sess, use_gpu, force_gpu) as sess:
      yield sess

  def test_real_spectrum_gives_self_adjoint_operator(self):
    with self.cached_session():
      # This is a real and hermitian spectrum.
      spectrum = linear_operator_test_util.random_normal(
          shape=(2, 2, 3, 5), dtype=dtypes.float32)
      operator = linalg.LinearOperatorCirculant3D(spectrum)
      self.assertAllEqual((2, 2 * 3 * 5, 2 * 3 * 5), operator.shape)

      self.assertEqual(
          operator.parameters,
          {
              "input_output_dtype": dtypes.complex64,
              "is_non_singular": None,
              "is_positive_definite": None,
              "is_self_adjoint": None,
              "is_square": True,
              "name": "LinearOperatorCirculant3D",
              "spectrum": spectrum,
          })

      matrix_tensor = operator.to_dense()
      self.assertEqual(matrix_tensor.dtype, dtypes.complex64)
      matrix_h = linalg.adjoint(matrix_tensor)

      matrix, matrix_h = self.evaluate([matrix_tensor, matrix_h])
      self.assertAllEqual((2, 2 * 3 * 5, 2 * 3 * 5), matrix.shape)
      self.assertAllClose(matrix, matrix_h)

  def test_defining_operator_using_real_convolution_kernel(self):
    with self.cached_session():
      convolution_kernel = linear_operator_test_util.random_normal(
          shape=(2, 2, 3, 5), dtype=dtypes.float32)
      # Convolution kernel is real ==> spectrum is Hermitian.
      spectrum = fft_ops.fft3d(
          math_ops.cast(convolution_kernel, dtypes.complex64))

      # spectrum is Hermitian ==> operator is real.
      operator = linalg.LinearOperatorCirculant3D(spectrum)
      self.assertAllEqual((2, 2 * 3 * 5, 2 * 3 * 5), operator.shape)

      # Allow for complex output so we can make sure it has zero imag part.
      self.assertEqual(operator.dtype, dtypes.complex64)
      matrix = self.evaluate(operator.to_dense())
      self.assertAllEqual((2, 2 * 3 * 5, 2 * 3 * 5), matrix.shape)
      np.testing.assert_allclose(0, np.imag(matrix), atol=1e-5)

  def test_defining_spd_operator_by_taking_real_part(self):
    with self.cached_session():  # Necessary for fft_kernel_label_map
      # S is real and positive.
      s = linear_operator_test_util.random_uniform(
          shape=(10, 2, 3, 4), dtype=dtypes.float32, minval=1., maxval=2.)

      # Let S = S1 + S2, the Hermitian and anti-hermitian parts.
      # S1 = 0.5 * (S + S^H), S2 = 0.5 * (S - S^H),
      # where ^H is the Hermitian transpose of the function:
      #    f(n0, n1, n2)^H := ComplexConjugate[f(N0-n0, N1-n1, N2-n2)].
      # We want to isolate S1, since
      #   S1 is Hermitian by construction
      #   S1 is real since S is
      #   S1 is positive since it is the sum of two positive kernels

      # IDFT[S] = IDFT[S1] + IDFT[S2]
      #         =      H1  +      H2
      # where H1 is real since it is Hermitian,
      # and H2 is imaginary since it is anti-Hermitian.
      ifft_s = fft_ops.ifft3d(math_ops.cast(s, dtypes.complex64))

      # Throw away H2, keep H1.
      real_ifft_s = math_ops.real(ifft_s)

      # This is the perfect spectrum!
      # spectrum = DFT[H1]
      #          = S1,
      fft_real_ifft_s = fft_ops.fft3d(
          math_ops.cast(real_ifft_s, dtypes.complex64))

      # S1 is Hermitian ==> operator is real.
      # S1 is real ==> operator is self-adjoint.
      # S1 is positive ==> operator is positive-definite.
      operator = linalg.LinearOperatorCirculant3D(fft_real_ifft_s)

      # Allow for complex output so we can check operator has zero imag part.
      self.assertEqual(operator.dtype, dtypes.complex64)
      matrix, matrix_t = self.evaluate([
          operator.to_dense(),
          array_ops.matrix_transpose(operator.to_dense())
      ])
      self.evaluate(operator.assert_positive_definite())  # Should not fail.
      np.testing.assert_allclose(0, np.imag(matrix), atol=1e-6)
      self.assertAllClose(matrix, matrix_t)

      # Just to test the theory, get S2 as well.
      # This should create an imaginary operator.
      # S2 is anti-Hermitian ==> operator is imaginary.
      # S2 is real ==> operator is self-adjoint.
      imag_ifft_s = math_ops.imag(ifft_s)
      fft_imag_ifft_s = fft_ops.fft3d(
          1j * math_ops.cast(imag_ifft_s, dtypes.complex64))
      operator_imag = linalg.LinearOperatorCirculant3D(fft_imag_ifft_s)

      matrix, matrix_h = self.evaluate([
          operator_imag.to_dense(),
          array_ops.matrix_transpose(math_ops.conj(operator_imag.to_dense()))
      ])
      self.assertAllClose(matrix, matrix_h)
      np.testing.assert_allclose(0, np.real(matrix), atol=1e-7)


if __name__ == "__main__":
  linear_operator_test_util.add_tests(
      LinearOperatorCirculantTestSelfAdjointOperator)
  linear_operator_test_util.add_tests(
      LinearOperatorCirculantTestHermitianSpectrum)
  linear_operator_test_util.add_tests(
      LinearOperatorCirculantTestNonHermitianSpectrum)
  linear_operator_test_util.add_tests(
      LinearOperatorCirculant2DTestHermitianSpectrum)
  linear_operator_test_util.add_tests(
      LinearOperatorCirculant2DTestNonHermitianSpectrum)
  test.main()
