/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/determinant_op.h"

#include <complex>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/cuda_solvers.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;
namespace {
__device__ int PermutationOrder(int n, const int* pivots) {
  // Compute the order of the permutation from the number of transpositions
  // encoded in the pivot array, see:
  // http://icl.cs.utk.edu/lapack-forum/viewtopic.php?f=2&t=340
  int order = 0;
  for (int i = 0; i < n - 1; ++i) {
    // Notice: Internally, the cuBlas code uses Fortran convention (1-based)
    // indexing so we expect pivots[i] == i + 1 for rows that were not moved.
    order += pivots[i] != (i + 1);
  }
  return order;
}

#if defined(__CUDACC__)
// Hack around missing support for complex in NVCC.
template <typename T>
__device__ inline std::complex<T> complex_multiply(const std::complex<T>& a,
                                                   const std::complex<T>& b) {
  const T a_real = Eigen::numext::real(a);
  const T a_imag = Eigen::numext::imag(a);
  const T b_real = Eigen::numext::real(b);
  const T b_imag = Eigen::numext::imag(b);
  return std::complex<T>(a_real * b_real - a_imag * b_imag,
                         a_real * b_imag + a_imag * b_real);
}
__device__ inline complex64 operator*(const complex64& a, const complex64& b) {
  return complex_multiply<float>(a, b);
}
__device__ inline complex64 operator*(const complex64& a, const float& b) {
  return complex64(Eigen::numext::real(a) * b, Eigen::numext::imag(a) * b);
}
__device__ inline complex64 operator/(const complex64& a, const float& b) {
  const float inv_b = 1.0f / b;
  return a * inv_b;
}
__device__ inline complex128 operator*(const complex128& a,
                                       const complex128& b) {
  return complex_multiply<double>(a, b);
}
__device__ inline complex128 operator*(const complex128& a, const double& b) {
  return complex128(Eigen::numext::real(a) * b, Eigen::numext::imag(a) * b);
}
__device__ inline complex128 operator/(const complex128& a, const double& b) {
  const double inv_b = 1.0 / b;
  return a * inv_b;
}
#endif
}  // namespace

// This kernel computes either determinant or log_abs_determinant, depending
// on the value of the template parameter. If compute_log_abs_det is false,
// the sign argument is ignored.
template <typename Scalar, bool compute_log_abs_det = true>
__global__ void DeterminantFromPivotedLUKernel(int nthreads, int n,
                                               const Scalar* lu_factor,
                                               const int* all_pivots,
                                               Scalar* sign,
                                               Scalar* log_abs_det) {
  typedef typename Eigen::NumTraits<Scalar>::Real RealScalar;
  const int matrix_size = n * n;
  const int stride = n + 1;
  // We only parallelize over batches here. Performance is not critical,
  // since this cheap O(n) kernel always follows an O(n^3) LU factorization.
  // The main purpose is to avoid having to copy the LU decomposition to
  // host memory.
  GPU_1D_KERNEL_LOOP(o_idx, nthreads) {
    // Initialize sign to (-1)^order.
    const int order = PermutationOrder(n, all_pivots + o_idx * n);
    Scalar prod_sign = order % 2 ? Scalar(-1) : Scalar(1);
    RealScalar sum_log_abs_det = RealScalar(0);
    int i_idx = matrix_size * o_idx;
    for (int i = 0; i < n; ++i, i_idx += stride) {
      const RealScalar abs_i = Eigen::numext::abs(lu_factor[i_idx]);
      sum_log_abs_det += Eigen::numext::log(abs_i);
      prod_sign = prod_sign * (lu_factor[i_idx] / abs_i);
    }
    if (!Eigen::numext::isfinite(sum_log_abs_det)) {
      prod_sign = Scalar(0);
      sum_log_abs_det = sum_log_abs_det > 0 ? -Eigen::numext::log(RealScalar(0))
                                            : Eigen::numext::log(RealScalar(0));
    }
    if (compute_log_abs_det) {
      sign[o_idx] = prod_sign;
      log_abs_det[o_idx] = Scalar(sum_log_abs_det);
    } else {
      log_abs_det[o_idx] = prod_sign * Eigen::numext::exp(sum_log_abs_det);
    }
  }
}

template <typename Scalar>
struct DeterminantFromPivotedLUFunctor<GPUDevice, Scalar> {
  void operator()(const GPUDevice& device,
                  typename TTypes<Scalar, 3>::ConstTensor lu_factor,
                  const int* pivots, typename TTypes<Scalar, 1>::Tensor output,
                  int* info) {
    const int64 num_matrices = output.size();
    const int64 n = lu_factor.dimension(2);
    GpuLaunchConfig config = GetGpuLaunchConfig(num_matrices, device);
    DeterminantFromPivotedLUKernel<Scalar, /*compute_log_abs_det=*/false>
        <<<config.block_count, config.thread_per_block, 0, device.stream()>>>(
            config.virtual_thread_count, n, lu_factor.data(), pivots, nullptr,
            output.data());
  }
};

template struct DeterminantFromPivotedLUFunctor<GPUDevice, float>;
template struct DeterminantFromPivotedLUFunctor<GPUDevice, double>;
template struct DeterminantFromPivotedLUFunctor<GPUDevice, complex64>;
template struct DeterminantFromPivotedLUFunctor<GPUDevice, complex128>;

template <typename Scalar>
struct LogDeterminantFromPivotedLUFunctor<GPUDevice, Scalar> {
  void operator()(const GPUDevice& device,
                  typename TTypes<Scalar, 3>::ConstTensor lu_factor,
                  const int* pivots, typename TTypes<Scalar, 1>::Tensor sign,
                  typename TTypes<Scalar, 1>::Tensor log_abs_det) {
    const int64 num_matrices = sign.size();
    const int64 n = lu_factor.dimension(2);
    GpuLaunchConfig config = GetGpuLaunchConfig(num_matrices, device);
    DeterminantFromPivotedLUKernel<Scalar, /*compute_log_abs_det=*/true>
        <<<config.block_count, config.thread_per_block, 0, device.stream()>>>(
            config.virtual_thread_count, n, lu_factor.data(), pivots,
            sign.data(), log_abs_det.data());
  }
};

template struct LogDeterminantFromPivotedLUFunctor<GPUDevice, float>;
template struct LogDeterminantFromPivotedLUFunctor<GPUDevice, double>;
template struct LogDeterminantFromPivotedLUFunctor<GPUDevice, complex64>;
template struct LogDeterminantFromPivotedLUFunctor<GPUDevice, complex128>;

}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
