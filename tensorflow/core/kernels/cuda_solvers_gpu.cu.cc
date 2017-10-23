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

#include "tensorflow/core/kernels/cuda_solvers.h"

#include <complex>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

namespace {

// Hacks around missing support for complex arithmetic in nvcc.
template <typename Scalar>
__device__ inline Scalar Multiply(Scalar x, Scalar y) {
  return x * y;
}

template <>
__device__ inline cuComplex Multiply(cuComplex x, cuComplex y) {
  return cuCmulf(x, y);
}

template <>
__device__ inline cuDoubleComplex Multiply(cuDoubleComplex x,
                                           cuDoubleComplex y) {
  return cuCmul(x, y);
}

template <typename Scalar>
__device__ inline Scalar Negate(Scalar x) {
  return -x;
}

template <>
__device__ inline cuComplex Negate(cuComplex x) {
  return make_cuComplex(-cuCrealf(x), -cuCimagf(x));
}

template <>
__device__ inline cuDoubleComplex Negate(cuDoubleComplex x) {
  return make_cuDoubleComplex(-cuCreal(x), -cuCimag(x));
}

template <typename Scalar>
__device__ inline bool IsFinite(Scalar x) {
  return Eigen::numext::isfinite(x);
}

template <>
__device__ inline bool IsFinite(cuComplex x) {
  return Eigen::numext::isfinite(cuCrealf(x)) &&
         Eigen::numext::isfinite(cuCimagf(x));
}

template <>
__device__ inline bool IsFinite(cuDoubleComplex x) {
  return Eigen::numext::isfinite(cuCreal(x)) &&
         Eigen::numext::isfinite(cuCimag(x));
}

template <typename Scalar>
struct Const {
  template <typename RealScalar>
  __device__ static inline Scalar make_const(const RealScalar x) {
    return Scalar(x);
  }
};

template <>
struct Const<cuComplex> {
  template <typename RealScalar>
  __device__ static inline cuComplex make_const(const RealScalar x) {
    return make_cuComplex(x, 0.0f);
  }
};

template <>
struct Const<cuDoubleComplex> {
  template <typename RealScalar>
  __device__ static inline cuDoubleComplex make_const(const RealScalar x) {
    return make_cuDoubleComplex(x, 0.0f);
  }
};

}  // namespace

template <typename Scalar>
__global__ void DeterminantFromPivotedLUKernel(int nthreads, int n,
                                               const Scalar* lu_factor,
                                               const int* all_pivots,
                                               Scalar* dst, int* info) {
  const int matrix_size = n * n;
  const int stride = n + 1;
  // We only parallelize over batches here. Performance is not critical,
  // since this cheap O(n) kernel always follows an O(n^3) LU factorization.
  // The main purpose is to avoid having to copy the LU decomposition to
  // host memory.
  CUDA_1D_KERNEL_LOOP(o_idx, nthreads) {
    // Compute the order of the permutation from the number of transpositions
    // encoded in the pivot array, see:
    // http://icl.cs.utk.edu/lapack-forum/viewtopic.php?f=2&t=340
    const int* pivots = all_pivots + o_idx * n;
    int order = 0;
    for (int i = 0; i < n - 1; ++i) {
      // Notice: Internally, the cuBlas code uses Fortran convention (1-based)
      // indexing so we expect pivots[i] == i + 1 for rows that were not moved.
      order += pivots[i] != (i + 1);
    }

    // Compute the product of the diagonal elements of U from the partially
    // pivoted LU factorization.
    // TODO(rmlarsen): This naive implementation (matching that in Eigen used
    // for the CPU kernel) is pathetically unstable. Should we implement
    // log-determinant instead (a different set of ops altogether) or something
    // like the method used in the old LINPACK code:
    // http://www.netlib.org/linpack/dgedi.f ?
    int i_idx = matrix_size * o_idx;
    Scalar prod = lu_factor[i_idx];
    for (int i = 1; i < n; ++i) {
      i_idx += stride;
      prod = Multiply(prod, lu_factor[i_idx]);
    }
    // Finally set the determinant to (-1)^order * prod(diag(U)).
    dst[o_idx] = order % 2 ? Negate(prod) : prod;

    // We write a magic value into the info array if the result was infinite.
    if (!IsFinite(prod)) {
      info[o_idx] = kint32min;
    }
  }
}

template <typename Scalar>
struct DeterminantFromPivotedLUFunctor<GPUDevice, Scalar> {
  void operator()(const GPUDevice& device,
                  typename TTypes<Scalar, 3>::ConstTensor lu_factor,
                  const int* pivots, typename TTypes<Scalar, 1>::Tensor output,
                  int* info) {
    using CudaType = typename CUDAComplexT<Scalar>::type;
    const int64 num_matrices = output.size();
    const int64 n = lu_factor.dimension(2);
    const CudaType* lu_factor_ptr =
        reinterpret_cast<const CudaType*>(lu_factor.data());
    CudaType* output_ptr = reinterpret_cast<CudaType*>(output.data());
    CudaLaunchConfig config = GetCudaLaunchConfig(num_matrices, device);
    DeterminantFromPivotedLUKernel<<<
        config.block_count, config.thread_per_block, 0, device.stream()>>>(
        config.virtual_thread_count, n, lu_factor_ptr, pivots, output_ptr,
        info);
  }
};

template struct DeterminantFromPivotedLUFunctor<GPUDevice, float>;
template struct DeterminantFromPivotedLUFunctor<GPUDevice, double>;
template struct DeterminantFromPivotedLUFunctor<GPUDevice, complex64>;
template struct DeterminantFromPivotedLUFunctor<GPUDevice, complex128>;

template <typename Scalar>
__global__ void EyeKernel(Cuda3DLaunchConfig config, int batch_size, int m,
                          int n, Scalar* matrix_batch_ptr) {
  const int matrix_size = m * n;
  const Scalar one = Const<Scalar>::make_const(1.0);
  CUDA_AXIS_KERNEL_LOOP(batch, config.virtual_thread_count, x) {
    if (batch >= batch_size) {
      break;
    }
    CUDA_AXIS_KERNEL_LOOP(row, config.virtual_thread_count, y) {
      if (row >= m) {
        break;
      }
      const int row_start = batch * matrix_size + row * n;
      CUDA_AXIS_KERNEL_LOOP(col, config.virtual_thread_count, z) {
        if (col >= n) {
          break;
        }
        matrix_batch_ptr[row_start + col] = row == col ? one : Scalar();
      }
    }
  }
}

template <typename Scalar>
struct EyeFunctor<GPUDevice, Scalar> {
  void operator()(const GPUDevice& device,
                  typename TTypes<Scalar, 3>::Tensor matrix_batch) {
    using CudaType = typename CUDAComplexT<Scalar>::type;
    const int batch_size = matrix_batch.dimension(0);
    const int m = matrix_batch.dimension(1);
    const int n = matrix_batch.dimension(2);
    CudaType* matrix_batch_ptr =
        reinterpret_cast<CudaType*>(matrix_batch.data());
    Cuda3DLaunchConfig config = GetCuda3DLaunchConfig(batch_size, m, n, device,
                                                      EyeKernel<Scalar>, 0, 0);
    EyeKernel<<<config.block_count, config.thread_per_block, 0,
                device.stream()>>>(config, batch_size, m, n, matrix_batch_ptr);
  }
};

template struct EyeFunctor<GPUDevice, float>;
template struct EyeFunctor<GPUDevice, double>;
template struct EyeFunctor<GPUDevice, complex64>;
template struct EyeFunctor<GPUDevice, complex128>;

}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
