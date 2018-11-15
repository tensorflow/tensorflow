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
"""`LinearOperator` coming from a [[nested] block] circulant matrix."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import util as distribution_util
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.ops.signal import fft_ops
from tensorflow.python.util.tf_export import tf_export

__all__ = [
    "LinearOperatorCirculant",
    "LinearOperatorCirculant2D",
    "LinearOperatorCirculant3D",
]

# Different FFT Ops will be used for different block depths.
_FFT_OP = {1: fft_ops.fft, 2: fft_ops.fft2d, 3: fft_ops.fft3d}
_IFFT_OP = {1: fft_ops.ifft, 2: fft_ops.ifft2d, 3: fft_ops.ifft3d}

# This is the only dtype allowed with fft ops.
# TODO(langmore) Add other types once available.
_DTYPE_COMPLEX = dtypes.complex64


# TODO(langmore) Add transformations that create common spectrums, e.g.
#   starting with the convolution kernel
#   start with half a spectrum, and create a Hermitian one.
#   common filters.
# TODO(langmore) Support rectangular Toeplitz matrices.
class _BaseLinearOperatorCirculant(linear_operator.LinearOperator):
  """Base class for circulant operators.  Not user facing.

  `LinearOperator` acting like a [batch] [[nested] block] circulant matrix.
  """

  def __init__(self,
               spectrum,
               block_depth,
               input_output_dtype=_DTYPE_COMPLEX,
               is_non_singular=None,
               is_self_adjoint=None,
               is_positive_definite=None,
               is_square=True,
               name="LinearOperatorCirculant"):
    r"""Initialize an `_BaseLinearOperatorCirculant`.

    Args:
      spectrum:  Shape `[B1,...,Bb, N]` `Tensor`.  Allowed dtypes are
        `float32`, `complex64`.  Type can be different than `input_output_dtype`
      block_depth:  Python integer, either 1, 2, or 3.  Will be 1 for circulant,
        2 for block circulant, and 3 for nested block circulant.
      input_output_dtype: `dtype` for input/output.  Must be either
        `float32` or `complex64`.
      is_non_singular:  Expect that this operator is non-singular.
      is_self_adjoint:  Expect that this operator is equal to its hermitian
        transpose.  If `spectrum` is real, this will always be true.
      is_positive_definite:  Expect that this operator is positive definite,
        meaning the quadratic form `x^H A x` has positive real part for all
        nonzero `x`.  Note that we do not require the operator to be
        self-adjoint to be positive-definite.  See:
        https://en.wikipedia.org/wiki/Positive-definite_matrix\
            #Extension_for_non_symmetric_matrices
      is_square:  Expect that this operator acts like square [batch] matrices.
      name:  A name to prepend to all ops created by this class.

    Raises:
      ValueError:  If `block_depth` is not an allowed value.
      TypeError:  If `spectrum` is not an allowed type.
    """

    allowed_block_depths = [1, 2, 3]

    self._name = name

    if block_depth not in allowed_block_depths:
      raise ValueError("Expected block_depth to be in %s.  Found: %s." %
                       (allowed_block_depths, block_depth))
    self._block_depth = block_depth

    with ops.name_scope(name, values=[spectrum]):
      self._spectrum = self._check_spectrum_and_return_tensor(spectrum)

      # Check and auto-set hints.
      if not self.spectrum.dtype.is_complex:
        if is_self_adjoint is False:
          raise ValueError(
              "A real spectrum always corresponds to a self-adjoint operator.")
        is_self_adjoint = True

      if is_square is False:
        raise ValueError(
            "A [[nested] block] circulant operator is always square.")
      is_square = True

      # If spectrum.shape = [s0, s1, s2], and block_depth = 2,
      # block_shape = [s1, s2]
      s_shape = array_ops.shape(self.spectrum)
      self._block_shape_tensor = s_shape[-self.block_depth:]

      # Add common variants of spectrum to the graph.
      self._spectrum_complex = _to_complex(self.spectrum)
      self._abs_spectrum = math_ops.abs(self.spectrum)
      self._conj_spectrum = math_ops.conj(self._spectrum_complex)

      super(_BaseLinearOperatorCirculant, self).__init__(
          dtype=dtypes.as_dtype(input_output_dtype),
          graph_parents=[self.spectrum],
          is_non_singular=is_non_singular,
          is_self_adjoint=is_self_adjoint,
          is_positive_definite=is_positive_definite,
          is_square=is_square,
          name=name)

  def _check_spectrum_and_return_tensor(self, spectrum):
    """Static check of spectrum.  Then return `Tensor` version."""
    spectrum = ops.convert_to_tensor(spectrum, name="spectrum")

    allowed_dtypes = [dtypes.float32, dtypes.complex64]
    if spectrum.dtype not in allowed_dtypes:
      raise TypeError("Argument spectrum must have dtype in %s.  Found: %s" %
                      (allowed_dtypes, spectrum.dtype))
    if spectrum.get_shape().ndims is not None:
      if spectrum.get_shape().ndims < self.block_depth:
        raise ValueError(
            "Argument spectrum must have at least %d dimensions.  Found: %s" %
            (self.block_depth, spectrum))
    return spectrum

  @property
  def block_depth(self):
    """Depth of recursively defined circulant blocks defining this `Operator`.

    With `A` the dense representation of this `Operator`,

    `block_depth = 1` means `A` is symmetric circulant.  For example,

    ```
    A = |w z y x|
        |x w z y|
        |y x w z|
        |z y x w|
    ```

    `block_depth = 2` means `A` is block symmetric circulant with symemtric
    circulant blocks.  For example, with `W`, `X`, `Y`, `Z` symmetric circulant,

    ```
    A = |W Z Y X|
        |X W Z Y|
        |Y X W Z|
        |Z Y X W|
    ```

    `block_depth = 3` means `A` is block symmetric circulant with block
    symmetric circulant blocks.

    Returns:
      Python `integer`.
    """
    return self._block_depth

  def block_shape_tensor(self):
    """Shape of the block dimensions of `self.spectrum`."""
    return self._block_shape_tensor

  @property
  def block_shape(self):
    return self.spectrum.get_shape()[-self.block_depth:]

  @property
  def spectrum(self):
    return self._spectrum

  def _vectorize_then_blockify(self, matrix):
    """Shape batch matrix to batch vector, then blockify trailing dimensions."""
    # Suppose
    #   matrix.shape = [m0, m1, m2, m3],
    # and matrix is a matrix because the final two dimensions are matrix dims.
    #   self.block_depth = 2,
    #   self.block_shape = [b0, b1]  (note b0 * b1 = m2).
    # We will reshape matrix to
    #   [m3, m0, m1, b0, b1].

    # Vectorize: Reshape to batch vector.
    #   [m0, m1, m2, m3] --> [m3, m0, m1, m2]
    # This is called "vectorize" because we have taken the final two matrix dims
    # and turned this into a size m3 batch of vectors.
    vec = distribution_util.rotate_transpose(matrix, shift=1)

    # Blockify: Blockfy trailing dimensions.
    #   [m3, m0, m1, m2] --> [m3, m0, m1, b0, b1]
    if (vec.get_shape().is_fully_defined() and
        self.block_shape.is_fully_defined()):
      # vec_leading_shape = [m3, m0, m1],
      # the parts of vec that will not be blockified.
      vec_leading_shape = vec.get_shape()[:-1]
      final_shape = vec_leading_shape.concatenate(self.block_shape)
    else:
      vec_leading_shape = array_ops.shape(vec)[:-1]
      final_shape = array_ops.concat(
          (vec_leading_shape, self.block_shape_tensor()), 0)
    return array_ops.reshape(vec, final_shape)

  def _unblockify_then_matricize(self, vec):
    """Flatten the block dimensions then reshape to a batch matrix."""
    # Suppose
    #   vec.shape = [v0, v1, v2, v3],
    #   self.block_depth = 2.
    # Then
    #   leading shape = [v0, v1]
    #   block shape = [v2, v3].
    # We will reshape vec to
    #   [v1, v2*v3, v0].

    # Un-blockify: Flatten block dimensions.  Reshape
    #   [v0, v1, v2, v3] --> [v0, v1, v2*v3].
    if vec.get_shape().is_fully_defined():
      # vec_shape = [v0, v1, v2, v3]
      vec_shape = vec.get_shape().as_list()
      # vec_leading_shape = [v0, v1]
      vec_leading_shape = vec_shape[:-self.block_depth]
      # vec_block_shape = [v2, v3]
      vec_block_shape = vec_shape[-self.block_depth:]
      # flat_shape = [v0, v1, v2*v3]
      flat_shape = vec_leading_shape + [np.prod(vec_block_shape)]
    else:
      vec_shape = array_ops.shape(vec)
      vec_leading_shape = vec_shape[:-self.block_depth]
      vec_block_shape = vec_shape[-self.block_depth:]
      flat_shape = array_ops.concat(
          (vec_leading_shape, [math_ops.reduce_prod(vec_block_shape)]), 0)
    vec_flat = array_ops.reshape(vec, flat_shape)

    # Matricize:  Reshape to batch matrix.
    #   [v0, v1, v2*v3] --> [v1, v2*v3, v0],
    # representing a shape [v1] batch of [v2*v3, v0] matrices.
    matrix = distribution_util.rotate_transpose(vec_flat, shift=-1)
    return matrix

  def _fft(self, x):
    """FFT along the last self.block_depth dimensions of x.

    Args:
      x: `Tensor` with floating or complex `dtype`.
        Should be in the form returned by self._vectorize_then_blockify.

    Returns:
      `Tensor` with `dtype` `complex64`.
    """
    x_complex = _to_complex(x)
    return _FFT_OP[self.block_depth](x_complex)

  def _ifft(self, x):
    """IFFT along the last self.block_depth dimensions of x.

    Args:
      x: `Tensor` with floating or complex dtype.  Should be in the form
        returned by self._vectorize_then_blockify.

    Returns:
      `Tensor` with `dtype` `complex64`.
    """
    x_complex = _to_complex(x)
    return _IFFT_OP[self.block_depth](x_complex)

  def convolution_kernel(self, name="convolution_kernel"):
    """Convolution kernel corresponding to `self.spectrum`.

    The `D` dimensional DFT of this kernel is the frequency domain spectrum of
    this operator.

    Args:
      name:  A name to give this `Op`.

    Returns:
      `Tensor` with `dtype` `self.dtype`.
    """
    with self._name_scope(name):
      h = self._ifft(self._spectrum_complex)
      return math_ops.cast(h, self.dtype)

  def _shape(self):
    s_shape = self._spectrum.get_shape()
    # Suppose spectrum.shape = [a, b, c, d]
    # block_depth = 2
    # Then:
    #   batch_shape = [a, b]
    #   N = c*d
    # and we want to return
    #   [a, b, c*d, c*d]
    batch_shape = s_shape[:-self.block_depth]
    # trailing_dims = [c, d]
    trailing_dims = s_shape[-self.block_depth:]
    if trailing_dims.is_fully_defined():
      n = np.prod(trailing_dims.as_list())
    else:
      n = None
    n_x_n = tensor_shape.TensorShape([n, n])
    return batch_shape.concatenate(n_x_n)

  def _shape_tensor(self):
    # See self.shape for explanation of steps
    s_shape = array_ops.shape(self._spectrum)
    batch_shape = s_shape[:-self.block_depth]
    trailing_dims = s_shape[-self.block_depth:]
    n = math_ops.reduce_prod(trailing_dims)
    n_x_n = [n, n]
    return array_ops.concat((batch_shape, n_x_n), 0)

  def assert_hermitian_spectrum(self, name="assert_hermitian_spectrum"):
    """Returns an `Op` that asserts this operator has Hermitian spectrum.

    This operator corresponds to a real-valued matrix if and only if its
    spectrum is Hermitian.

    Args:
      name:  A name to give this `Op`.

    Returns:
      An `Op` that asserts this operator has Hermitian spectrum.
    """
    eps = np.finfo(self.dtype.real_dtype.as_numpy_dtype).eps
    with self._name_scope(name):
      # Assume linear accumulation of error.
      max_err = eps * self.domain_dimension_tensor()
      imag_convolution_kernel = math_ops.imag(self.convolution_kernel())
      return check_ops.assert_less(
          math_ops.abs(imag_convolution_kernel),
          max_err,
          message="Spectrum was not Hermitian")

  def _assert_non_singular(self):
    return linear_operator_util.assert_no_entries_with_modulus_zero(
        self.spectrum,
        message="Singular operator:  Spectrum contained zero values.")

  def _assert_positive_definite(self):
    # This operator has the action  Ax = F^H D F x,
    # where D is the diagonal matrix with self.spectrum on the diag.  Therefore,
    # <x, Ax> = <Fx, DFx>,
    # Since F is bijective, the condition for positive definite is the same as
    # for a diagonal matrix, i.e. real part of spectrum is positive.
    message = (
        "Not positive definite:  Real part of spectrum was not all positive.")
    return check_ops.assert_positive(
        math_ops.real(self.spectrum), message=message)

  def _assert_self_adjoint(self):
    # Recall correspondence between symmetry and real transforms.  See docstring
    return linear_operator_util.assert_zero_imag_part(
        self.spectrum,
        message=(
            "Not self-adjoint:  The spectrum contained non-zero imaginary part."
        ))

  def _broadcast_batch_dims(self, x, spectrum):
    """Broadcast batch dims of batch matrix `x` and spectrum."""
    # spectrum.shape = batch_shape + block_shape
    # First make spectrum a batch matrix with
    #   spectrum.shape = batch_shape + [prod(block_shape), 1]
    spec_mat = array_ops.reshape(
        spectrum, array_ops.concat(
            (self.batch_shape_tensor(), [-1, 1]), axis=0))
    # Second, broadcast, possibly requiring an addition of array of zeros.
    x, spec_mat = linear_operator_util.broadcast_matrix_batch_dims((x,
                                                                    spec_mat))
    # Third, put the block shape back into spectrum.
    batch_shape = array_ops.shape(x)[:-2]
    spectrum = array_ops.reshape(
        spec_mat,
        array_ops.concat((batch_shape, self.block_shape_tensor()), axis=0))

    return x, spectrum

  def _matmul(self, x, adjoint=False, adjoint_arg=False):
    x = linalg.adjoint(x) if adjoint_arg else x
    # With F the matrix of a DFT, and F^{-1}, F^H the inverse and Hermitian
    # transpose, one can show that F^{-1} = F^{H} is the IDFT matrix.  Therefore
    # matmul(x) = F^{-1} diag(spectrum) F x,
    #           = F^{H} diag(spectrum) F x,
    # so that
    # matmul(x, adjoint=True) = F^{H} diag(conj(spectrum)) F x.
    spectrum = self._conj_spectrum if adjoint else self._spectrum_complex

    x, spectrum = self._broadcast_batch_dims(x, spectrum)

    x_vb = self._vectorize_then_blockify(x)
    fft_x_vb = self._fft(x_vb)
    block_vector_result = self._ifft(spectrum * fft_x_vb)
    y = self._unblockify_then_matricize(block_vector_result)

    return math_ops.cast(y, self.dtype)

  def _determinant(self):
    reduction_indices = [-(i + 1) for i in range(self.block_depth)]
    det = math_ops.reduce_prod(
        self.spectrum, reduction_indices=reduction_indices)
    return math_ops.cast(det, self.dtype)

  def _log_abs_determinant(self):
    reduction_indices = [-(i + 1) for i in range(self.block_depth)]
    lad = math_ops.reduce_sum(
        math_ops.log(self._abs_spectrum), reduction_indices=reduction_indices)
    return math_ops.cast(lad, self.dtype)

  def _solve(self, rhs, adjoint=False, adjoint_arg=False):
    rhs = linalg.adjoint(rhs) if adjoint_arg else rhs
    spectrum = self._conj_spectrum if adjoint else self._spectrum_complex

    rhs, spectrum = self._broadcast_batch_dims(rhs, spectrum)

    rhs_vb = self._vectorize_then_blockify(rhs)
    fft_rhs_vb = self._fft(rhs_vb)
    solution_vb = self._ifft(fft_rhs_vb / spectrum)
    x = self._unblockify_then_matricize(solution_vb)
    return math_ops.cast(x, self.dtype)

  def _diag_part(self):
    # Get ones in shape of diag, which is [B1,...,Bb, N]
    # Also get the size of the diag, "N".
    if self.shape.is_fully_defined():
      diag_shape = self.shape[:-1]
      diag_size = self.domain_dimension.value
    else:
      diag_shape = self.shape_tensor()[:-1]
      diag_size = self.domain_dimension_tensor()
    ones_diag = array_ops.ones(diag_shape, dtype=self.dtype)

    # As proved in comments in self._trace, the value on the diag is constant,
    # repeated N times.  This value is the trace divided by N.

    # The handling of self.shape = (0, 0) is tricky, and is the reason we choose
    # to compute trace and use that to compute diag_part, rather than computing
    # the value on the diagonal ("diag_value") directly.  Both result in a 0/0,
    # but in different places, and the current method gives the right result in
    # the end.

    # Here, if self.shape = (0, 0), then self.trace() = 0., and then
    # diag_value = 0. / 0. = NaN.
    diag_value = self.trace() / math_ops.cast(diag_size, self.dtype)

    # If self.shape = (0, 0), then ones_diag = [] (empty tensor), and then
    # the following line is NaN * [] = [], as needed.
    return diag_value[..., array_ops.newaxis] * ones_diag

  def _trace(self):
    # The diagonal of the [[nested] block] circulant operator is the mean of
    # the spectrum.
    # Proof:  For the [0,...,0] element, this follows from the IDFT formula.
    # Then the result follows since all diagonal elements are the same.

    # Therefore, the trace is the sum of the spectrum.

    # Get shape of diag along with the axis over which to reduce the spectrum.
    # We will reduce the spectrum over all block indices.
    if self.spectrum.get_shape().is_fully_defined():
      spec_rank = self.spectrum.get_shape().ndims
      axis = np.arange(spec_rank - self.block_depth, spec_rank, dtype=np.int32)
    else:
      spec_rank = array_ops.rank(self.spectrum)
      axis = math_ops.range(spec_rank - self.block_depth, spec_rank)

    # Real diag part "re_d".
    # Suppose spectrum.shape = [B1,...,Bb, N1, N2]
    # self.shape = [B1,...,Bb, N, N], with N1 * N2 = N.
    # re_d_value.shape = [B1,...,Bb]
    re_d_value = math_ops.reduce_sum(math_ops.real(self.spectrum), axis=axis)

    if not self.dtype.is_complex:
      return math_ops.cast(re_d_value, self.dtype)

    # Imaginary part, "im_d".
    if self.is_self_adjoint:
      im_d_value = 0.
    else:
      im_d_value = math_ops.reduce_sum(math_ops.imag(self.spectrum), axis=axis)

    return math_ops.cast(math_ops.complex(re_d_value, im_d_value), self.dtype)


@tf_export("linalg.LinearOperatorCirculant")
class LinearOperatorCirculant(_BaseLinearOperatorCirculant):
  """`LinearOperator` acting like a circulant matrix.

  This operator acts like a circulant matrix `A` with
  shape `[B1,...,Bb, N, N]` for some `b >= 0`.  The first `b` indices index a
  batch member.  For every batch index `(i1,...,ib)`, `A[i1,...,ib, : :]` is
  an `N x N` matrix.  This matrix `A` is not materialized, but for
  purposes of broadcasting this shape will be relevant.

  #### Description in terms of circulant matrices

  Circulant means the entries of `A` are generated by a single vector, the
  convolution kernel `h`: `A_{mn} := h_{m-n mod N}`.  With `h = [w, x, y, z]`,

  ```
  A = |w z y x|
      |x w z y|
      |y x w z|
      |z y x w|
  ```

  This means that the result of matrix multiplication `v = Au` has `Lth` column
  given circular convolution between `h` with the `Lth` column of `u`.

  See http://ee.stanford.edu/~gray/toeplitz.pdf

  #### Description in terms of the frequency spectrum

  There is an equivalent description in terms of the [batch] spectrum `H` and
  Fourier transforms.  Here we consider `A.shape = [N, N]` and ignore batch
  dimensions.  Define the discrete Fourier transform (DFT) and its inverse by

  ```
  DFT[ h[n] ] = H[k] := sum_{n = 0}^{N - 1} h_n e^{-i 2pi k n / N}
  IDFT[ H[k] ] = h[n] = N^{-1} sum_{k = 0}^{N - 1} H_k e^{i 2pi k n / N}
  ```

  From these definitions, we see that

  ```
  H[0] = sum_{n = 0}^{N - 1} h_n
  H[1] = "the first positive frequency"
  H[N - 1] = "the first negative frequency"
  ```

  Loosely speaking, with `*` element-wise multiplication, matrix multiplication
  is equal to the action of a Fourier multiplier: `A u = IDFT[ H * DFT[u] ]`.
  Precisely speaking, given `[N, R]` matrix `u`, let `DFT[u]` be the `[N, R]`
  matrix with `rth` column equal to the DFT of the `rth` column of `u`.
  Define the `IDFT` similarly.
  Matrix multiplication may be expressed columnwise:

  ```(A u)_r = IDFT[ H * (DFT[u])_r ]```

  #### Operator properties deduced from the spectrum.

  Letting `U` be the `kth` Euclidean basis vector, and `U = IDFT[u]`.
  The above formulas show that`A U = H_k * U`.  We conclude that the elements
  of `H` are the eigenvalues of this operator.   Therefore

  * This operator is positive definite if and only if `Real{H} > 0`.

  A general property of Fourier transforms is the correspondence between
  Hermitian functions and real valued transforms.

  Suppose `H.shape = [B1,...,Bb, N]`.  We say that `H` is a Hermitian spectrum
  if, with `%` meaning modulus division,

  ```H[..., n % N] = ComplexConjugate[ H[..., (-n) % N] ]```

  * This operator corresponds to a real matrix if and only if `H` is Hermitian.
  * This operator is self-adjoint if and only if `H` is real.

  See e.g. "Discrete-Time Signal Processing", Oppenheim and Schafer.

  #### Example of a self-adjoint positive definite operator

  ```python
  # spectrum is real ==> operator is self-adjoint
  # spectrum is positive ==> operator is positive definite
  spectrum = [6., 4, 2]

  operator = LinearOperatorCirculant(spectrum)

  # IFFT[spectrum]
  operator.convolution_kernel()
  ==> [4 + 0j, 1 + 0.58j, 1 - 0.58j]

  operator.to_dense()
  ==> [[4 + 0.0j, 1 - 0.6j, 1 + 0.6j],
       [1 + 0.6j, 4 + 0.0j, 1 - 0.6j],
       [1 - 0.6j, 1 + 0.6j, 4 + 0.0j]]
  ```

  #### Example of defining in terms of a real convolution kernel

  ```python
  # convolution_kernel is real ==> spectrum is Hermitian.
  convolution_kernel = [1., 2., 1.]]
  spectrum = tf.fft(tf.cast(convolution_kernel, tf.complex64))

  # spectrum is Hermitian ==> operator is real.
  # spectrum is shape [3] ==> operator is shape [3, 3]
  # We force the input/output type to be real, which allows this to operate
  # like a real matrix.
  operator = LinearOperatorCirculant(spectrum, input_output_dtype=tf.float32)

  operator.to_dense()
  ==> [[ 1, 1, 2],
       [ 2, 1, 1],
       [ 1, 2, 1]]
  ```

  #### Example of Hermitian spectrum

  ```python
  # spectrum is shape [3] ==> operator is shape [3, 3]
  # spectrum is Hermitian ==> operator is real.
  spectrum = [1, 1j, -1j]

  operator = LinearOperatorCirculant(spectrum)

  operator.to_dense()
  ==> [[ 0.33 + 0j,  0.91 + 0j, -0.24 + 0j],
       [-0.24 + 0j,  0.33 + 0j,  0.91 + 0j],
       [ 0.91 + 0j, -0.24 + 0j,  0.33 + 0j]
  ```

  #### Example of forcing real `dtype` when spectrum is Hermitian

  ```python
  # spectrum is shape [4] ==> operator is shape [4, 4]
  # spectrum is real ==> operator is self-adjoint
  # spectrum is Hermitian ==> operator is real
  # spectrum has positive real part ==> operator is positive-definite.
  spectrum = [6., 4, 2, 4]

  # Force the input dtype to be float32.
  # Cast the output to float32.  This is fine because the operator will be
  # real due to Hermitian spectrum.
  operator = LinearOperatorCirculant(spectrum, input_output_dtype=tf.float32)

  operator.shape
  ==> [4, 4]

  operator.to_dense()
  ==> [[4, 1, 0, 1],
       [1, 4, 1, 0],
       [0, 1, 4, 1],
       [1, 0, 1, 4]]

  # convolution_kernel = tf.ifft(spectrum)
  operator.convolution_kernel()
  ==> [4, 1, 0, 1]
  ```

  #### Performance

  Suppose `operator` is a `LinearOperatorCirculant` of shape `[N, N]`,
  and `x.shape = [N, R]`.  Then

  * `operator.matmul(x)` is `O(R*N*Log[N])`
  * `operator.solve(x)` is `O(R*N*Log[N])`
  * `operator.determinant()` involves a size `N` `reduce_prod`.

  If instead `operator` and `x` have shape `[B1,...,Bb, N, N]` and
  `[B1,...,Bb, N, R]`, every operation increases in complexity by `B1*...*Bb`.

  #### Matrix property hints

  This `LinearOperator` is initialized with boolean flags of the form `is_X`,
  for `X = non_singular, self_adjoint, positive_definite, square`.
  These have the following meaning:

  * If `is_X == True`, callers should expect the operator to have the
    property `X`.  This is a promise that should be fulfilled, but is *not* a
    runtime assert.  For example, finite floating point precision may result
    in these promises being violated.
  * If `is_X == False`, callers should expect the operator to not have `X`.
  * If `is_X == None` (the default), callers should have no expectation either
    way.
  """

  def __init__(self,
               spectrum,
               input_output_dtype=_DTYPE_COMPLEX,
               is_non_singular=None,
               is_self_adjoint=None,
               is_positive_definite=None,
               is_square=True,
               name="LinearOperatorCirculant"):
    r"""Initialize an `LinearOperatorCirculant`.

    This `LinearOperator` is initialized to have shape `[B1,...,Bb, N, N]`
    by providing `spectrum`, a `[B1,...,Bb, N]` `Tensor`.

    If `input_output_dtype = DTYPE`:

    * Arguments to methods such as `matmul` or `solve` must be `DTYPE`.
    * Values returned by all methods, such as `matmul` or `determinant` will be
      cast to `DTYPE`.

    Note that if the spectrum is not Hermitian, then this operator corresponds
    to a complex matrix with non-zero imaginary part.  In this case, setting
    `input_output_dtype` to a real type will forcibly cast the output to be
    real, resulting in incorrect results!

    If on the other hand the spectrum is Hermitian, then this operator
    corresponds to a real-valued matrix, and setting `input_output_dtype` to
    a real type is fine.

    Args:
      spectrum:  Shape `[B1,...,Bb, N]` `Tensor`.  Allowed dtypes are
        `float32`, `complex64`.  Type can be different than `input_output_dtype`
      input_output_dtype: `dtype` for input/output.  Must be either
        `float32` or `complex64`.
      is_non_singular:  Expect that this operator is non-singular.
      is_self_adjoint:  Expect that this operator is equal to its hermitian
        transpose.  If `spectrum` is real, this will always be true.
      is_positive_definite:  Expect that this operator is positive definite,
        meaning the quadratic form `x^H A x` has positive real part for all
        nonzero `x`.  Note that we do not require the operator to be
        self-adjoint to be positive-definite.  See:
        https://en.wikipedia.org/wiki/Positive-definite_matrix\
            #Extension_for_non_symmetric_matrices
      is_square:  Expect that this operator acts like square [batch] matrices.
      name:  A name to prepend to all ops created by this class.
    """
    super(LinearOperatorCirculant, self).__init__(
        spectrum,
        block_depth=1,
        input_output_dtype=input_output_dtype,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        name=name)


@tf_export("linalg.LinearOperatorCirculant2D")
class LinearOperatorCirculant2D(_BaseLinearOperatorCirculant):
  """`LinearOperator` acting like a block circulant matrix.

  This operator acts like a block circulant matrix `A` with
  shape `[B1,...,Bb, N, N]` for some `b >= 0`.  The first `b` indices index a
  batch member.  For every batch index `(i1,...,ib)`, `A[i1,...,ib, : :]` is
  an `N x N` matrix.  This matrix `A` is not materialized, but for
  purposes of broadcasting this shape will be relevant.

  #### Description in terms of block circulant matrices

  If `A` is block circulant, with block sizes `N0, N1` (`N0 * N1 = N`):
  `A` has a block circulant structure, composed of `N0 x N0` blocks, with each
  block an `N1 x N1` circulant matrix.

  For example, with `W`, `X`, `Y`, `Z` each circulant,

  ```
  A = |W Z Y X|
      |X W Z Y|
      |Y X W Z|
      |Z Y X W|
  ```

  Note that `A` itself will not in general be circulant.

  #### Description in terms of the frequency spectrum

  There is an equivalent description in terms of the [batch] spectrum `H` and
  Fourier transforms.  Here we consider `A.shape = [N, N]` and ignore batch
  dimensions.

  If `H.shape = [N0, N1]`, (`N0 * N1 = N`):
  Loosely speaking, matrix multiplication is equal to the action of a
  Fourier multiplier:  `A u = IDFT2[ H DFT2[u] ]`.
  Precisely speaking, given `[N, R]` matrix `u`, let `DFT2[u]` be the
  `[N0, N1, R]` `Tensor` defined by re-shaping `u` to `[N0, N1, R]` and taking
  a two dimensional DFT across the first two dimensions.  Let `IDFT2` be the
  inverse of `DFT2`.  Matrix multiplication may be expressed columnwise:

  ```(A u)_r = IDFT2[ H * (DFT2[u])_r ]```

  #### Operator properties deduced from the spectrum.

  * This operator is positive definite if and only if `Real{H} > 0`.

  A general property of Fourier transforms is the correspondence between
  Hermitian functions and real valued transforms.

  Suppose `H.shape = [B1,...,Bb, N0, N1]`, we say that `H` is a Hermitian
  spectrum if, with `%` indicating modulus division,

  ```
  H[..., n0 % N0, n1 % N1] = ComplexConjugate[ H[..., (-n0) % N0, (-n1) % N1 ].
  ```

  * This operator corresponds to a real matrix if and only if `H` is Hermitian.
  * This operator is self-adjoint if and only if `H` is real.

  See e.g. "Discrete-Time Signal Processing", Oppenheim and Schafer.

  ### Example of a self-adjoint positive definite operator

  ```python
  # spectrum is real ==> operator is self-adjoint
  # spectrum is positive ==> operator is positive definite
  spectrum = [[1., 2., 3.],
              [4., 5., 6.],
              [7., 8., 9.]]

  operator = LinearOperatorCirculant2D(spectrum)

  # IFFT[spectrum]
  operator.convolution_kernel()
  ==> [[5.0+0.0j, -0.5-.3j, -0.5+.3j],
       [-1.5-.9j,        0,        0],
       [-1.5+.9j,        0,        0]]

  operator.to_dense()
  ==> Complex self adjoint 9 x 9 matrix.
  ```

  #### Example of defining in terms of a real convolution kernel,

  ```python
  # convolution_kernel is real ==> spectrum is Hermitian.
  convolution_kernel = [[1., 2., 1.], [5., -1., 1.]]
  spectrum = tf.fft2d(tf.cast(convolution_kernel, tf.complex64))

  # spectrum is shape [2, 3] ==> operator is shape [6, 6]
  # spectrum is Hermitian ==> operator is real.
  operator = LinearOperatorCirculant2D(spectrum, input_output_dtype=tf.float32)
  ```

  #### Performance

  Suppose `operator` is a `LinearOperatorCirculant` of shape `[N, N]`,
  and `x.shape = [N, R]`.  Then

  * `operator.matmul(x)` is `O(R*N*Log[N])`
  * `operator.solve(x)` is `O(R*N*Log[N])`
  * `operator.determinant()` involves a size `N` `reduce_prod`.

  If instead `operator` and `x` have shape `[B1,...,Bb, N, N]` and
  `[B1,...,Bb, N, R]`, every operation increases in complexity by `B1*...*Bb`.

  #### Matrix property hints

  This `LinearOperator` is initialized with boolean flags of the form `is_X`,
  for `X = non_singular, self_adjoint, positive_definite, square`.
  These have the following meaning
  * If `is_X == True`, callers should expect the operator to have the
    property `X`.  This is a promise that should be fulfilled, but is *not* a
    runtime assert.  For example, finite floating point precision may result
    in these promises being violated.
  * If `is_X == False`, callers should expect the operator to not have `X`.
  * If `is_X == None` (the default), callers should have no expectation either
    way.
  """

  def __init__(self,
               spectrum,
               input_output_dtype=_DTYPE_COMPLEX,
               is_non_singular=None,
               is_self_adjoint=None,
               is_positive_definite=None,
               is_square=True,
               name="LinearOperatorCirculant2D"):
    r"""Initialize an `LinearOperatorCirculant2D`.

    This `LinearOperator` is initialized to have shape `[B1,...,Bb, N, N]`
    by providing `spectrum`, a `[B1,...,Bb, N0, N1]` `Tensor` with `N0*N1 = N`.

    If `input_output_dtype = DTYPE`:

    * Arguments to methods such as `matmul` or `solve` must be `DTYPE`.
    * Values returned by all methods, such as `matmul` or `determinant` will be
      cast to `DTYPE`.

    Note that if the spectrum is not Hermitian, then this operator corresponds
    to a complex matrix with non-zero imaginary part.  In this case, setting
    `input_output_dtype` to a real type will forcibly cast the output to be
    real, resulting in incorrect results!

    If on the other hand the spectrum is Hermitian, then this operator
    corresponds to a real-valued matrix, and setting `input_output_dtype` to
    a real type is fine.

    Args:
      spectrum:  Shape `[B1,...,Bb, N]` `Tensor`.  Allowed dtypes are
        `float32`, `complex64`.  Type can be different than `input_output_dtype`
      input_output_dtype: `dtype` for input/output.  Must be either
        `float32` or `complex64`.
      is_non_singular:  Expect that this operator is non-singular.
      is_self_adjoint:  Expect that this operator is equal to its hermitian
        transpose.  If `spectrum` is real, this will always be true.
      is_positive_definite:  Expect that this operator is positive definite,
        meaning the quadratic form `x^H A x` has positive real part for all
        nonzero `x`.  Note that we do not require the operator to be
        self-adjoint to be positive-definite.  See:
        https://en.wikipedia.org/wiki/Positive-definite_matrix\
            #Extension_for_non_symmetric_matrices
      is_square:  Expect that this operator acts like square [batch] matrices.
      name:  A name to prepend to all ops created by this class.
    """
    super(LinearOperatorCirculant2D, self).__init__(
        spectrum,
        block_depth=2,
        input_output_dtype=input_output_dtype,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        name=name)


@tf_export("linalg.LinearOperatorCirculant3D")
class LinearOperatorCirculant3D(_BaseLinearOperatorCirculant):
  """`LinearOperator` acting like a nested block circulant matrix.

  This operator acts like a block circulant matrix `A` with
  shape `[B1,...,Bb, N, N]` for some `b >= 0`.  The first `b` indices index a
  batch member.  For every batch index `(i1,...,ib)`, `A[i1,...,ib, : :]` is
  an `N x N` matrix.  This matrix `A` is not materialized, but for
  purposes of broadcasting this shape will be relevant.

  #### Description in terms of block circulant matrices

  If `A` is nested block circulant, with block sizes `N0, N1, N2`
  (`N0 * N1 * N2 = N`):
  `A` has a block structure, composed of `N0 x N0` blocks, with each
  block an `N1 x N1` block circulant matrix.

  For example, with `W`, `X`, `Y`, `Z` each block circulant,

  ```
  A = |W Z Y X|
      |X W Z Y|
      |Y X W Z|
      |Z Y X W|
  ```

  Note that `A` itself will not in general be circulant.

  #### Description in terms of the frequency spectrum

  There is an equivalent description in terms of the [batch] spectrum `H` and
  Fourier transforms.  Here we consider `A.shape = [N, N]` and ignore batch
  dimensions.

  If `H.shape = [N0, N1, N2]`, (`N0 * N1 * N2 = N`):
  Loosely speaking, matrix multiplication is equal to the action of a
  Fourier multiplier:  `A u = IDFT3[ H DFT3[u] ]`.
  Precisely speaking, given `[N, R]` matrix `u`, let `DFT3[u]` be the
  `[N0, N1, N2, R]` `Tensor` defined by re-shaping `u` to `[N0, N1, N2, R]` and
  taking a three dimensional DFT across the first three dimensions.  Let `IDFT3`
  be the inverse of `DFT3`.  Matrix multiplication may be expressed columnwise:

  ```(A u)_r = IDFT3[ H * (DFT3[u])_r ]```

  #### Operator properties deduced from the spectrum.

  * This operator is positive definite if and only if `Real{H} > 0`.

  A general property of Fourier transforms is the correspondence between
  Hermitian functions and real valued transforms.

  Suppose `H.shape = [B1,...,Bb, N0, N1, N2]`, we say that `H` is a Hermitian
  spectrum if, with `%` meaning modulus division,

  ```
  H[..., n0 % N0, n1 % N1, n2 % N2]
    = ComplexConjugate[ H[..., (-n0) % N0, (-n1) % N1, (-n2) % N2] ].
  ```

  * This operator corresponds to a real matrix if and only if `H` is Hermitian.
  * This operator is self-adjoint if and only if `H` is real.

  See e.g. "Discrete-Time Signal Processing", Oppenheim and Schafer.

  ### Examples

  See `LinearOperatorCirculant` and `LinearOperatorCirculant2D` for examples.

  #### Performance

  Suppose `operator` is a `LinearOperatorCirculant` of shape `[N, N]`,
  and `x.shape = [N, R]`.  Then

  * `operator.matmul(x)` is `O(R*N*Log[N])`
  * `operator.solve(x)` is `O(R*N*Log[N])`
  * `operator.determinant()` involves a size `N` `reduce_prod`.

  If instead `operator` and `x` have shape `[B1,...,Bb, N, N]` and
  `[B1,...,Bb, N, R]`, every operation increases in complexity by `B1*...*Bb`.

  #### Matrix property hints

  This `LinearOperator` is initialized with boolean flags of the form `is_X`,
  for `X = non_singular, self_adjoint, positive_definite, square`.
  These have the following meaning
  * If `is_X == True`, callers should expect the operator to have the
    property `X`.  This is a promise that should be fulfilled, but is *not* a
    runtime assert.  For example, finite floating point precision may result
    in these promises being violated.
  * If `is_X == False`, callers should expect the operator to not have `X`.
  * If `is_X == None` (the default), callers should have no expectation either
    way.
  """

  def __init__(self,
               spectrum,
               input_output_dtype=_DTYPE_COMPLEX,
               is_non_singular=None,
               is_self_adjoint=None,
               is_positive_definite=None,
               is_square=True,
               name="LinearOperatorCirculant3D"):
    """Initialize an `LinearOperatorCirculant`.

    This `LinearOperator` is initialized to have shape `[B1,...,Bb, N, N]`
    by providing `spectrum`, a `[B1,...,Bb, N0, N1, N2]` `Tensor`
    with `N0*N1*N2 = N`.

    If `input_output_dtype = DTYPE`:

    * Arguments to methods such as `matmul` or `solve` must be `DTYPE`.
    * Values returned by all methods, such as `matmul` or `determinant` will be
      cast to `DTYPE`.

    Note that if the spectrum is not Hermitian, then this operator corresponds
    to a complex matrix with non-zero imaginary part.  In this case, setting
    `input_output_dtype` to a real type will forcibly cast the output to be
    real, resulting in incorrect results!

    If on the other hand the spectrum is Hermitian, then this operator
    corresponds to a real-valued matrix, and setting `input_output_dtype` to
    a real type is fine.

    Args:
      spectrum:  Shape `[B1,...,Bb, N]` `Tensor`.  Allowed dtypes are
        `float32`, `complex64`.  Type can be different than `input_output_dtype`
      input_output_dtype: `dtype` for input/output.  Must be either
        `float32` or `complex64`.
      is_non_singular:  Expect that this operator is non-singular.
      is_self_adjoint:  Expect that this operator is equal to its hermitian
        transpose.  If `spectrum` is real, this will always be true.
      is_positive_definite:  Expect that this operator is positive definite,
        meaning the real part of all eigenvalues is positive.  We do not require
        the operator to be self-adjoint to be positive-definite.  See:
        https://en.wikipedia.org/wiki/Positive-definite_matrix
            #Extension_for_non_symmetric_matrices
      is_square:  Expect that this operator acts like square [batch] matrices.
      name:  A name to prepend to all ops created by this class.
    """
    super(LinearOperatorCirculant3D, self).__init__(
        spectrum,
        block_depth=3,
        input_output_dtype=input_output_dtype,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        name=name)


def _to_complex(x):
  return math_ops.cast(x, _DTYPE_COMPLEX)
