/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/matrix_set_diag_op.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

template <typename Scalar>
__global__ void MatrixSetDiagKernel(const int num_threads, const int m,
                                    const int n, const int minsize,
                                    const Scalar* diag_ptr,
                                    Scalar* output_ptr) {
  CUDA_1D_KERNEL_LOOP(index, num_threads) {
    const int batch = index / minsize;
    const int col = index - batch * minsize;
    const int out_index = batch * m * n + (n + 1) * col;
    output_ptr[out_index] = diag_ptr[index];
  }
}

template <typename Scalar>
__global__ void MatrixCopyInputAndSetDiagKernel(
    const int num_threads, const int m, const int n, const int minsize,
    const Scalar* input_ptr, const Scalar* diag_ptr, Scalar* output_ptr) {
  CUDA_1D_KERNEL_LOOP(index, num_threads) {
    const int global_row = index / n;
    const int col = index - global_row * n;
    const int batch = global_row / m;
    const int row = global_row - batch * m;
    if (col == row) {
      // Because col = index % n, and row = (index / n) % m,
      // we know that col==row => col < minsize, so the following is safe:
      output_ptr[index] = diag_ptr[batch * minsize + col];
    } else {
      output_ptr[index] = input_ptr[index];
    }
  }
}

template <typename Scalar>
struct MatrixSetDiag<GPUDevice, Scalar> {
  static void Compute(OpKernelContext* context, const GPUDevice& device,
                      typename TTypes<Scalar, 3>::ConstTensor input,
                      typename TTypes<Scalar, 2>::ConstTensor diag,
                      typename TTypes<Scalar, 3>::Tensor output) {
    const int batch_size = input.dimension(0);
    const int m = input.dimension(1);
    const int n = input.dimension(2);
    const int minsize = std::min(m, n);
    CHECK_EQ(diag.dimension(1), minsize);
    if (batch_size == 0 || minsize == 0) return;
    if (input.data() == output.data()) {
      CudaLaunchConfig config =
          GetCudaLaunchConfig(batch_size * minsize, device);
      MatrixSetDiagKernel<Scalar>
          <<<config.block_count, config.thread_per_block, 0, device.stream()>>>(
              config.virtual_thread_count, m, n, minsize, diag.data(),
              output.data());
    } else {
      CudaLaunchConfig config = GetCudaLaunchConfig(batch_size * m * n, device);
      MatrixCopyInputAndSetDiagKernel<Scalar>
          <<<config.block_count, config.thread_per_block, 0, device.stream()>>>(
              config.virtual_thread_count, m, n, minsize, input.data(),
              diag.data(), output.data());
    }
  }
};

#define DEFINE_GPU_SPEC(T) template struct MatrixSetDiag<GPUDevice, T>;

TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_SPEC);
TF_CALL_bool(DEFINE_GPU_SPEC);
TF_CALL_complex64(DEFINE_GPU_SPEC);
TF_CALL_complex128(DEFINE_GPU_SPEC);

}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
