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

template <typename Scalar>
__global__ void EyeKernel(Cuda3DLaunchConfig config, int batch_size, int m,
                          int n, Scalar* matrix_batch_ptr) {
  const int matrix_size = m * n;
  const Scalar one = Scalar(1);
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
    const int batch_size = matrix_batch.dimension(0);
    const int m = matrix_batch.dimension(1);
    const int n = matrix_batch.dimension(2);
    Cuda3DLaunchConfig config = GetCuda3DLaunchConfig(batch_size, m, n, device,
                                                      EyeKernel<Scalar>, 0, 0);
    EyeKernel<<<config.block_count, config.thread_per_block, 0,
                device.stream()>>>(config, batch_size, m, n,
                                   matrix_batch.data());
  }
};

template struct EyeFunctor<GPUDevice, float>;
template struct EyeFunctor<GPUDevice, double>;
template struct EyeFunctor<GPUDevice, complex64>;
template struct EyeFunctor<GPUDevice, complex128>;

}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
