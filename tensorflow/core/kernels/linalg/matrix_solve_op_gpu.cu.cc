/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include <complex>

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/linalg/matrix_solve_op.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

// GPU kernel that checks the diagonal of the LU factorization for exact zeros.
// If a zero diagonal is found, the corresponding info entry is set to the
// 1-based index of the first zero pivot (matching LAPACK convention for
// singular matrix detection).
template <typename Scalar>
__global__ void CheckLUDiagonalForZerosKernel(
    int nthreads, int n, const Scalar* __restrict__ lu_factor,
    int* __restrict__ info) {
  typedef typename Eigen::NumTraits<Scalar>::Real RealScalar;
  const int matrix_size = n * n;
  const int stride = n + 1;
  GPU_1D_KERNEL_LOOP(batch, nthreads) {
    // Only update info if it was not already set to a positive value
    // (i.e., getrfBatched didn't already detect the singularity).
    if (info[batch] == 0) {
      int base = matrix_size * batch;
      for (int i = 0; i < n; ++i) {
        if (Eigen::numext::abs(lu_factor[base + i * stride]) ==
            RealScalar(0)) {
          // Set info to 1-based index of the first zero pivot.
          info[batch] = i + 1;
          break;
        }
      }
    }
  }
}

template <typename Scalar>
struct CheckLUDiagonalForZerosFunctor<GPUDevice, Scalar> {
  void operator()(const GPUDevice& device,
                  typename TTypes<Scalar, 3>::ConstTensor lu_factor,
                  int* info) {
    const int64_t num_matrices = lu_factor.dimension(0);
    const int64_t n = lu_factor.dimension(2);
    GpuLaunchConfig config = GetGpuLaunchConfig(num_matrices, device);
    TF_CHECK_OK(GpuLaunchKernel(CheckLUDiagonalForZerosKernel<Scalar>,
                                config.block_count, config.thread_per_block, 0,
                                device.stream(), config.virtual_thread_count, n,
                                lu_factor.data(), info));
  }
};

template struct CheckLUDiagonalForZerosFunctor<GPUDevice, float>;
template struct CheckLUDiagonalForZerosFunctor<GPUDevice, double>;
template struct CheckLUDiagonalForZerosFunctor<GPUDevice, complex64>;
template struct CheckLUDiagonalForZerosFunctor<GPUDevice, complex128>;

}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
