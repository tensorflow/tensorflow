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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import spectral_ops_test_util
from tensorflow.python.ops.linalg import linalg
from tensorflow.python.ops.linalg import linear_operator_circulant
from tensorflow.python.ops.linalg import linear_operator_test_util
from tensorflow.python.platform import test

rng = np.random.RandomState(0)
_to_complex = linear_operator_circulant._to_complex


class LinearOperatorCirculantBaseTest(object):
  """Common class for circulant tests."""

  @contextlib.contextmanager
  def test_session(self, *args, **kwargs):
    with test.TestCase.test_session(self, *args, **kwargs) as sess:
      with spectral_ops_test_util.fft_kernel_label_map():
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
      fft_x = math_ops.fft(x)
      h_convolve_x = math_ops.ifft(spectrum * fft_x)
      matrix_rows.append(h_convolve_x)
    matrix = array_ops.stack(matrix_rows, axis=-1)
    return math_ops.cast(matrix, dtype)


class LinearOperatorCirculantTestSelfAdjointOperator(
    LinearOperatorCirculantBaseTest,
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
  """Test of LinearOperatorCirculant when operator is self-adjoint.

  Real spectrum <==> Self adjoint operator.
  Note that when the spectrum is real, the operator may still be complex.
  """

  @property
  def _dtypes_to_test(self):
    # This operator will always be complex because, although the specturm is
    # real, the matrix will not be real.
    return [dtypes.complex64]

  def _operator_and_mat_and_feed_dict(self, build_info, dtype, use_placeholder):
    shape = build_info.shape
    # For this test class, we are creating real spectrums.
    # We also want the spectrum to have eigenvalues bounded away from zero.
    #
    # spectrum is bounded away from zero.
    spectrum = linear_operator_test_util.random_sign_uniform(
        shape=self._shape_to_spectrum_shape(shape), minval=1., maxval=2.)
    # If dtype is complex, cast spectrum to complex.  The imaginary part will be
    # zero, so the operator will still be self-adjoint.
    spectrum = math_ops.cast(spectrum, dtype)

    if use_placeholder:
      spectrum_ph = array_ops.placeholder(dtypes.complex64)
      # Evaluate here because (i) you cannot feed a tensor, and (ii)
      # it is random and we want the same value used for both mat and feed_dict.
      spectrum = spectrum.eval()
      operator = linalg.LinearOperatorCirculant(
          spectrum_ph, is_self_adjoint=True, input_output_dtype=dtype)
      feed_dict = {spectrum_ph: spectrum}
    else:
      operator = linalg.LinearOperatorCirculant(
          spectrum, is_self_adjoint=True, input_output_dtype=dtype)
      feed_dict = None

    mat = self._spectrum_to_circulant_1d(spectrum, shape, dtype=dtype)

    return operator, mat, feed_dict

  def test_simple_hermitian_spectrum_gives_operator_with_zero_imag_part(self):
    with self.test_session():
      spectrum = math_ops.cast([1., 1j, -1j], dtypes.complex64)
      operator = linalg.LinearOperatorCirculant(
          spectrum, input_output_dtype=dtypes.complex64)
      matrix = operator.to_dense()
      imag_matrix = math_ops.imag(matrix)
      eps = np.finfo(np.float32).eps
      np.testing.assert_allclose(0, imag_matrix.eval(), rtol=0, atol=eps * 3)


class LinearOperatorCirculantTestHermitianSpectrum(
    LinearOperatorCirculantBaseTest,
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
  """Test of LinearOperatorCirculant when the spectrum is Hermitian.

  Hermitian spectrum <==> Real valued operator.  We test both real and complex
  dtypes here though.  So in some cases the matrix will be complex but with
  zero imaginary part.
  """

  @property
  def _dtypes_to_test(self):
    return [dtypes.float32, dtypes.complex64]

  def _operator_and_mat_and_feed_dict(self, build_info, dtype, use_placeholder):
    shape = build_info.shape
    # For this test class, we are creating Hermitian spectrums.
    # We also want the spectrum to have eigenvalues bounded away from zero.
    #
    # pre_spectrum is bounded away from zero.
    pre_spectrum = linear_operator_test_util.random_uniform(
        shape=self._shape_to_spectrum_shape(shape), minval=1., maxval=2.)
    pre_spectrum_c = _to_complex(pre_spectrum)

    # Real{IFFT[pre_spectrum]}
    #  = IFFT[EvenPartOf[pre_spectrum]]
    # is the IFFT of something that is also bounded away from zero.
    # Therefore, FFT[pre_h] would be a well-conditioned spectrum.
    pre_h = math_ops.ifft(pre_spectrum_c)

    # A spectrum is Hermitian iff it is the DFT of a real convolution kernel.
    # So we will make spectrum = FFT[h], for real valued h.
    h = math_ops.real(pre_h)
    h_c = _to_complex(h)

    spectrum = math_ops.fft(h_c)

    if use_placeholder:
      spectrum_ph = array_ops.placeholder(dtypes.complex64)
      # Evaluate here because (i) you cannot feed a tensor, and (ii)
      # it is random and we want the same value used for both mat and feed_dict.
      spectrum = spectrum.eval()
      operator = linalg.LinearOperatorCirculant(
          spectrum_ph, input_output_dtype=dtype)
      feed_dict = {spectrum_ph: spectrum}
    else:
      operator = linalg.LinearOperatorCirculant(
          spectrum, input_output_dtype=dtype)
      feed_dict = None

    mat = self._spectrum_to_circulant_1d(spectrum, shape, dtype=dtype)

    return operator, mat, feed_dict

  def test_simple_hermitian_spectrum_gives_operator_with_zero_imag_part(self):
    with self.test_session():
      spectrum = math_ops.cast([1., 1j, -1j], dtypes.complex64)
      operator = linalg.LinearOperatorCirculant(
          spectrum, input_output_dtype=dtypes.complex64)
      matrix = operator.to_dense()
      imag_matrix = math_ops.imag(matrix)
      eps = np.finfo(np.float32).eps
      np.testing.assert_allclose(0, imag_matrix.eval(), rtol=0, atol=eps * 3)


class LinearOperatorCirculantTestNonHermitianSpectrum(
    LinearOperatorCirculantBaseTest,
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
  """Test of LinearOperatorCirculant when the spectrum is not Hermitian.

  Non-Hermitian spectrum <==> Complex valued operator.
  We test only complex dtypes here.
  """

  @property
  def _dtypes_to_test(self):
    return [dtypes.complex64]

  def _operator_and_mat_and_feed_dict(self, build_info, dtype, use_placeholder):
    shape = build_info.shape
    # Will be well conditioned enough to get accurate solves.
    spectrum = linear_operator_test_util.random_sign_uniform(
        shape=self._shape_to_spectrum_shape(shape),
        dtype=dtypes.complex64,
        minval=1.,
        maxval=2.)

    if use_placeholder:
      spectrum_ph = array_ops.placeholder(dtypes.complex64)
      # Evaluate here because (i) you cannot feed a tensor, and (ii)
      # it is random and we want the same value used for both mat and feed_dict.
      spectrum = spectrum.eval()
      operator = linalg.LinearOperatorCirculant(
          spectrum_ph, input_output_dtype=dtype)
      feed_dict = {spectrum_ph: spectrum}
    else:
      operator = linalg.LinearOperatorCirculant(
          spectrum, input_output_dtype=dtype)
      feed_dict = None

    mat = self._spectrum_to_circulant_1d(spectrum, shape, dtype=dtype)

    return operator, mat, feed_dict

  def test_simple_hermitian_spectrum_gives_operator_with_zero_imag_part(self):
    with self.test_session():
      spectrum = math_ops.cast([1., 1j, -1j], dtypes.complex64)
      operator = linalg.LinearOperatorCirculant(
          spectrum, input_output_dtype=dtypes.complex64)
      matrix = operator.to_dense()
      imag_matrix = math_ops.imag(matrix)
      eps = np.finfo(np.float32).eps
      np.testing.assert_allclose(0, imag_matrix.eval(), rtol=0, atol=eps * 3)

  def test_simple_positive_real_spectrum_gives_self_adjoint_pos_def_oper(self):
    with self.test_session() as sess:
      spectrum = math_ops.cast([6., 4, 2], dtypes.complex64)
      operator = linalg.LinearOperatorCirculant(
          spectrum, input_output_dtype=dtypes.complex64)
      matrix, matrix_h = sess.run(
          [operator.to_dense(),
           linalg.adjoint(operator.to_dense())])
      self.assertAllClose(matrix, matrix_h)
      operator.assert_positive_definite().run()  # Should not fail
      operator.assert_self_adjoint().run()  # Should not fail

  def test_defining_operator_using_real_convolution_kernel(self):
    with self.test_session():
      convolution_kernel = [1., 2., 1.]
      spectrum = math_ops.fft(
          math_ops.cast(convolution_kernel, dtypes.complex64))

      # spectrum is shape [3] ==> operator is shape [3, 3]
      # spectrum is Hermitian ==> operator is real.
      operator = linalg.LinearOperatorCirculant(spectrum)

      # Allow for complex output so we can make sure it has zero imag part.
      self.assertEqual(operator.dtype, dtypes.complex64)

      matrix = operator.to_dense().eval()
      np.testing.assert_allclose(0, np.imag(matrix), atol=1e-6)

  def test_hermitian_spectrum_gives_operator_with_zero_imag_part(self):
    with self.test_session():
      # Make spectrum the FFT of a real convolution kernel h.  This ensures that
      # spectrum is Hermitian.
      h = linear_operator_test_util.random_normal(shape=(3, 4))
      spectrum = math_ops.fft(math_ops.cast(h, dtypes.complex64))
      operator = linalg.LinearOperatorCirculant(
          spectrum, input_output_dtype=dtypes.complex64)
      matrix = operator.to_dense()
      imag_matrix = math_ops.imag(matrix)
      eps = np.finfo(np.float32).eps
      np.testing.assert_allclose(
          0, imag_matrix.eval(), rtol=0, atol=eps * 3 * 4)

  def test_convolution_kernel_same_as_first_row_of_to_dense(self):
    spectrum = [[3., 2., 1.], [2., 1.5, 1.]]
    with self.test_session():
      operator = linalg.LinearOperatorCirculant(spectrum)
      h = operator.convolution_kernel()
      c = operator.to_dense()

      self.assertAllEqual((2, 3), h.get_shape())
      self.assertAllEqual((2, 3, 3), c.get_shape())
      self.assertAllClose(h.eval(), c.eval()[:, :, 0])

  def test_assert_non_singular_fails_for_singular_operator(self):
    spectrum = math_ops.cast([0, 4, 2j + 2], dtypes.complex64)
    operator = linalg.LinearOperatorCirculant(spectrum)
    with self.test_session():
      with self.assertRaisesOpError("Singular operator"):
        operator.assert_non_singular().run()

  def test_assert_non_singular_does_not_fail_for_non_singular_operator(self):
    spectrum = math_ops.cast([-3j, 4, 2j + 2], dtypes.complex64)
    operator = linalg.LinearOperatorCirculant(spectrum)
    with self.test_session():
      operator.assert_non_singular().run()  # Should not fail

  def test_assert_positive_definite_fails_for_non_positive_definite(self):
    spectrum = math_ops.cast([6., 4, 2j], dtypes.complex64)
    operator = linalg.LinearOperatorCirculant(spectrum)
    with self.test_session():
      with self.assertRaisesOpError("Not positive definite"):
        operator.assert_positive_definite().run()

  def test_assert_positive_definite_does_not_fail_when_pos_def(self):
    spectrum = math_ops.cast([6., 4, 2j + 2], dtypes.complex64)
    operator = linalg.LinearOperatorCirculant(spectrum)
    with self.test_session():
      operator.assert_positive_definite().run()  # Should not fail

  def test_real_spectrum_and_not_self_adjoint_hint_raises(self):
    spectrum = [1., 2.]
    with self.assertRaisesRegexp(ValueError, "real.*always.*self-adjoint"):
      linalg.LinearOperatorCirculant(spectrum, is_self_adjoint=False)

  def test_real_spectrum_auto_sets_is_self_adjoint_to_true(self):
    spectrum = [1., 2.]
    operator = linalg.LinearOperatorCirculant(spectrum)
    self.assertTrue(operator.is_self_adjoint)


class LinearOperatorCirculant2DBaseTest(object):
  """Common class for 2D circulant tests."""

  @contextlib.contextmanager
  def test_session(self, *args, **kwargs):
    with test.TestCase.test_session(self, *args, **kwargs) as sess:
      with spectral_ops_test_util.fft_kernel_label_map():
        yield sess

  @property
  def _operator_build_infos(self):
    build_info = linear_operator_test_util.OperatorBuildInfo
    # non-batch operators (n, n) and batch operators.
    return [
        build_info((0, 0)),
        build_info((1, 1)),
        build_info((1, 6, 6)),
        build_info((3, 4, 4)),
        build_info((2, 1, 3, 3))
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
        fft_x = math_ops.fft2d(x)
        h_convolve_x = math_ops.ifft2d(spectrum * fft_x)
        # We want the flat version of the action of the operator on a basis
        # vector, not the block version.
        h_convolve_x = array_ops.reshape(h_convolve_x, shape[:-1])
        matrix_rows.append(h_convolve_x)
    matrix = array_ops.stack(matrix_rows, axis=-1)
    return math_ops.cast(matrix, dtype)


class LinearOperatorCirculant2DTestHermitianSpectrum(
    LinearOperatorCirculant2DBaseTest,
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
  """Test of LinearOperatorCirculant2D when the spectrum is Hermitian.

  Hermitian spectrum <==> Real valued operator.  We test both real and complex
  dtypes here though.  So in some cases the matrix will be complex but with
  zero imaginary part.
  """

  @property
  def _dtypes_to_test(self):
    return [dtypes.float32, dtypes.complex64]

  def _operator_and_mat_and_feed_dict(self, build_info, dtype, use_placeholder):
    shape = build_info.shape
    # For this test class, we are creating Hermitian spectrums.
    # We also want the spectrum to have eigenvalues bounded away from zero.
    #
    # pre_spectrum is bounded away from zero.
    pre_spectrum = linear_operator_test_util.random_uniform(
        shape=self._shape_to_spectrum_shape(shape), minval=1., maxval=2.)
    pre_spectrum_c = _to_complex(pre_spectrum)

    # Real{IFFT[pre_spectrum]}
    #  = IFFT[EvenPartOf[pre_spectrum]]
    # is the IFFT of something that is also bounded away from zero.
    # Therefore, FFT[pre_h] would be a well-conditioned spectrum.
    pre_h = math_ops.ifft2d(pre_spectrum_c)

    # A spectrum is Hermitian iff it is the DFT of a real convolution kernel.
    # So we will make spectrum = FFT[h], for real valued h.
    h = math_ops.real(pre_h)
    h_c = _to_complex(h)

    spectrum = math_ops.fft2d(h_c)

    if use_placeholder:
      spectrum_ph = array_ops.placeholder(dtypes.complex64)
      # Evaluate here because (i) you cannot feed a tensor, and (ii)
      # it is random and we want the same value used for both mat and feed_dict.
      spectrum = spectrum.eval()
      operator = linalg.LinearOperatorCirculant2D(
          spectrum_ph, input_output_dtype=dtype)
      feed_dict = {spectrum_ph: spectrum}
    else:
      operator = linalg.LinearOperatorCirculant2D(
          spectrum, input_output_dtype=dtype)
      feed_dict = None

    mat = self._spectrum_to_circulant_2d(spectrum, shape, dtype=dtype)

    return operator, mat, feed_dict


class LinearOperatorCirculant2DTestNonHermitianSpectrum(
    LinearOperatorCirculant2DBaseTest,
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
  """Test of LinearOperatorCirculant when the spectrum is not Hermitian.

  Non-Hermitian spectrum <==> Complex valued operator.
  We test only complex dtypes here.
  """

  @property
  def _dtypes_to_test(self):
    return [dtypes.complex64]

  def _operator_and_mat_and_feed_dict(self, build_info, dtype, use_placeholder):
    shape = build_info.shape
    # Will be well conditioned enough to get accurate solves.
    spectrum = linear_operator_test_util.random_sign_uniform(
        shape=self._shape_to_spectrum_shape(shape),
        dtype=dtype,
        minval=1.,
        maxval=2.)

    if use_placeholder:
      spectrum_ph = array_ops.placeholder(dtypes.complex64)
      # Evaluate here because (i) you cannot feed a tensor, and (ii)
      # it is random and we want the same value used for both mat and feed_dict.
      spectrum = spectrum.eval()
      operator = linalg.LinearOperatorCirculant2D(
          spectrum_ph, input_output_dtype=dtype)
      feed_dict = {spectrum_ph: spectrum}
    else:
      operator = linalg.LinearOperatorCirculant2D(
          spectrum, input_output_dtype=dtype)
      feed_dict = None

    mat = self._spectrum_to_circulant_2d(spectrum, shape, dtype=dtype)

    return operator, mat, feed_dict

  def test_real_hermitian_spectrum_gives_real_symmetric_operator(self):
    with self.test_session() as sess:
      # This is a real and hermitian spectrum.
      spectrum = [[1., 2., 2.], [3., 4., 4.], [3., 4., 4.]]
      operator = linalg.LinearOperatorCirculant(spectrum)

      matrix_tensor = operator.to_dense()
      self.assertEqual(matrix_tensor.dtype,
                       linear_operator_circulant._DTYPE_COMPLEX)
      matrix_t = array_ops.matrix_transpose(matrix_tensor)
      imag_matrix = math_ops.imag(matrix_tensor)
      matrix, matrix_transpose, imag_matrix = sess.run(
          [matrix_tensor, matrix_t, imag_matrix])

      np.testing.assert_allclose(0, imag_matrix, atol=1e-6)
      self.assertAllClose(matrix, matrix_transpose, atol=0)

  def test_real_spectrum_gives_self_adjoint_operator(self):
    with self.test_session() as sess:
      # This is a real and hermitian spectrum.
      spectrum = linear_operator_test_util.random_normal(
          shape=(3, 3), dtype=dtypes.float32)
      operator = linalg.LinearOperatorCirculant2D(spectrum)

      matrix_tensor = operator.to_dense()
      self.assertEqual(matrix_tensor.dtype,
                       linear_operator_circulant._DTYPE_COMPLEX)
      matrix_h = linalg.adjoint(matrix_tensor)
      matrix, matrix_h = sess.run([matrix_tensor, matrix_h])
      self.assertAllClose(matrix, matrix_h, atol=0)

  def test_assert_non_singular_fails_for_singular_operator(self):
    spectrum = math_ops.cast([[0, 4], [2j + 2, 3.]], dtypes.complex64)
    operator = linalg.LinearOperatorCirculant2D(spectrum)
    with self.test_session():
      with self.assertRaisesOpError("Singular operator"):
        operator.assert_non_singular().run()

  def test_assert_non_singular_does_not_fail_for_non_singular_operator(self):
    spectrum = math_ops.cast([[-3j, 4], [2j + 2, 3.]], dtypes.complex64)
    operator = linalg.LinearOperatorCirculant2D(spectrum)
    with self.test_session():
      operator.assert_non_singular().run()  # Should not fail

  def test_assert_positive_definite_fails_for_non_positive_definite(self):
    spectrum = math_ops.cast([[6., 4], [2j, 3.]], dtypes.complex64)
    operator = linalg.LinearOperatorCirculant2D(spectrum)
    with self.test_session():
      with self.assertRaisesOpError("Not positive definite"):
        operator.assert_positive_definite().run()

  def test_assert_positive_definite_does_not_fail_when_pos_def(self):
    spectrum = math_ops.cast([[6., 4], [2j + 2, 3.]], dtypes.complex64)
    operator = linalg.LinearOperatorCirculant2D(spectrum)
    with self.test_session():
      operator.assert_positive_definite().run()  # Should not fail

  def test_real_spectrum_and_not_self_adjoint_hint_raises(self):
    spectrum = [[1., 2.], [3., 4]]
    with self.assertRaisesRegexp(ValueError, "real.*always.*self-adjoint"):
      linalg.LinearOperatorCirculant2D(spectrum, is_self_adjoint=False)

  def test_real_spectrum_auto_sets_is_self_adjoint_to_true(self):
    spectrum = [[1., 2.], [3., 4]]
    operator = linalg.LinearOperatorCirculant2D(spectrum)
    self.assertTrue(operator.is_self_adjoint)

  def test_invalid_dtype_raises(self):
    spectrum = array_ops.constant(rng.rand(2, 2, 2))
    with self.assertRaisesRegexp(TypeError, "must have dtype"):
      linalg.LinearOperatorCirculant2D(spectrum)

  def test_invalid_rank_raises(self):
    spectrum = array_ops.constant(np.float32(rng.rand(2)))
    with self.assertRaisesRegexp(ValueError, "must have at least 2 dimensions"):
      linalg.LinearOperatorCirculant2D(spectrum)


class LinearOperatorCirculant3DTest(test.TestCase):
  """Simple test of the 3D case.  See also the 1D and 2D tests."""

  @contextlib.contextmanager
  def test_session(self, *args, **kwargs):
    with test.TestCase.test_session(self, *args, **kwargs) as sess:
      with spectral_ops_test_util.fft_kernel_label_map():
        yield sess

  def test_real_spectrum_gives_self_adjoint_operator(self):
    with self.test_session() as sess:
      # This is a real and hermitian spectrum.
      spectrum = linear_operator_test_util.random_normal(
          shape=(2, 2, 3, 5), dtype=dtypes.float32)
      operator = linalg.LinearOperatorCirculant3D(spectrum)
      self.assertAllEqual((2, 2 * 3 * 5, 2 * 3 * 5), operator.shape)

      matrix_tensor = operator.to_dense()
      self.assertEqual(matrix_tensor.dtype,
                       linear_operator_circulant._DTYPE_COMPLEX)
      matrix_h = linalg.adjoint(matrix_tensor)

      matrix, matrix_h = sess.run([matrix_tensor, matrix_h])
      self.assertAllEqual((2, 2 * 3 * 5, 2 * 3 * 5), matrix.shape)
      self.assertAllClose(matrix, matrix_h)

  def test_defining_operator_using_real_convolution_kernel(self):
    with self.test_session():
      convolution_kernel = linear_operator_test_util.random_normal(
          shape=(2, 2, 3, 5), dtype=dtypes.float32)
      # Convolution kernel is real ==> spectrum is Hermitian.
      spectrum = math_ops.fft3d(
          math_ops.cast(convolution_kernel, dtypes.complex64))

      # spectrum is Hermitian ==> operator is real.
      operator = linalg.LinearOperatorCirculant3D(spectrum)
      self.assertAllEqual((2, 2 * 3 * 5, 2 * 3 * 5), operator.shape)

      # Allow for complex output so we can make sure it has zero imag part.
      self.assertEqual(operator.dtype, dtypes.complex64)
      matrix = operator.to_dense().eval()
      self.assertAllEqual((2, 2 * 3 * 5, 2 * 3 * 5), matrix.shape)
      np.testing.assert_allclose(0, np.imag(matrix), atol=1e-6)

  def test_defining_spd_operator_by_taking_real_part(self):
    with self.test_session() as sess:
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
      ifft_s = math_ops.ifft3d(math_ops.cast(s, dtypes.complex64))

      # Throw away H2, keep H1.
      real_ifft_s = math_ops.real(ifft_s)

      # This is the perfect spectrum!
      # spectrum = DFT[H1]
      #          = S1,
      fft_real_ifft_s = math_ops.fft3d(
          math_ops.cast(real_ifft_s, dtypes.complex64))

      # S1 is Hermitian ==> operator is real.
      # S1 is real ==> operator is self-adjoint.
      # S1 is positive ==> operator is positive-definite.
      operator = linalg.LinearOperatorCirculant3D(fft_real_ifft_s)

      # Allow for complex output so we can check operator has zero imag part.
      self.assertEqual(operator.dtype, dtypes.complex64)
      matrix, matrix_t = sess.run([
          operator.to_dense(),
          array_ops.matrix_transpose(operator.to_dense())
      ])
      operator.assert_positive_definite().run()  # Should not fail.
      np.testing.assert_allclose(0, np.imag(matrix), atol=1e-6)
      self.assertAllClose(matrix, matrix_t)

      # Just to test the theory, get S2 as well.
      # This should create an imaginary operator.
      # S2 is anti-Hermitian ==> operator is imaginary.
      # S2 is real ==> operator is self-adjoint.
      imag_ifft_s = math_ops.imag(ifft_s)
      fft_imag_ifft_s = math_ops.fft3d(
          1j * math_ops.cast(imag_ifft_s, dtypes.complex64))
      operator_imag = linalg.LinearOperatorCirculant3D(fft_imag_ifft_s)

      matrix, matrix_h = sess.run([
          operator_imag.to_dense(),
          array_ops.matrix_transpose(math_ops.conj(operator_imag.to_dense()))
      ])
      self.assertAllClose(matrix, matrix_h)
      np.testing.assert_allclose(0, np.real(matrix), atol=1e-7)


if __name__ == "__main__":
  test.main()
