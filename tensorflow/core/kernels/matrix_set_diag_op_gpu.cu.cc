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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/matrix_set_diag_op.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

template <typename Scalar>
__global__ void MatrixSetDiagKernel(const int num_threads, const int m,
                                    const int n, const int num_diags,
                                    const int max_diag_len,
                                    const int upper_diag_index,
                                    const Scalar* diag_ptr,
                                    Scalar* output_ptr) {
  GPU_1D_KERNEL_LOOP(index, num_threads) {
    const int batch_and_diag_index = index / max_diag_len;
    const int index_in_the_diagonal =
        index - batch_and_diag_index * max_diag_len;
    const int batch = batch_and_diag_index / num_diags;
    const int diag_index_in_input = batch_and_diag_index - batch * num_diags;
    const int diag_index = upper_diag_index - diag_index_in_input;
    const int y_index = index_in_the_diagonal + max(0, -diag_index);
    const int x_index = index_in_the_diagonal + max(0, diag_index);
    const int out_index = batch * m * n + y_index * n + x_index;
    output_ptr[out_index] = diag_ptr[index];
  }
}

template <typename Scalar>
__global__ void MatrixCopyInputAndSetDiagKernel(
    const int num_threads, const int m, const int n, const int num_diags,
    const int max_diag_len, const int lower_diag_index,
    const int upper_diag_index, const Scalar* input_ptr, const Scalar* diag_ptr,
    Scalar* output_ptr) {
  GPU_1D_KERNEL_LOOP(index, num_threads) {
    const int batch_and_row_index = index / n;
    const int col = index - batch_and_row_index * n;
    const int batch = batch_and_row_index / m;
    const int row = batch_and_row_index - batch * m;
    const int d = col - row;
    const int diag_index_in_input = upper_diag_index - d;
    const int index_in_the_diagonal = col - max(d, 0);
    if (lower_diag_index <= d && d <= upper_diag_index) {
      output_ptr[index] =
          diag_ptr[batch * num_diags * max_diag_len +
                   diag_index_in_input * max_diag_len + index_in_the_diagonal];
    } else {
      output_ptr[index] = input_ptr[index];
    }
  }
}

template <typename Scalar>
struct MatrixSetDiag<GPUDevice, Scalar> {
  static void Compute(OpKernelContext* context, const GPUDevice& device,
                      typename TTypes<Scalar, 3>::ConstTensor& input,
                      typename TTypes<Scalar>::ConstTensor& diag,
                      typename TTypes<Scalar, 3>::Tensor& output,
                      const Eigen::Index lower_diag_index,
                      const Eigen::Index upper_diag_index,
                      const Eigen::Index max_diag_len) {
    const int batch_size = input.dimension(0);
    const int m = input.dimension(1);
    const int n = input.dimension(2);
    const int num_diags = upper_diag_index - lower_diag_index + 1;

    if (batch_size == 0 || max_diag_len == 0 || m == 0 || n == 0) return;
    if (input.data() == output.data()) {
      GpuLaunchConfig config =
          GetGpuLaunchConfig(batch_size * num_diags * max_diag_len, device);
      TF_CHECK_OK(GpuLaunchKernel(
          MatrixSetDiagKernel<Scalar>, config.block_count,
          config.thread_per_block, 0, device.stream(),
          config.virtual_thread_count, m, n, num_diags, max_diag_len,
          upper_diag_index, diag.data(), output.data()));
    } else {
      GpuLaunchConfig config = GetGpuLaunchConfig(batch_size * m * n, device);
      TF_CHECK_OK(GpuLaunchKernel(
          MatrixCopyInputAndSetDiagKernel<Scalar>, config.block_count,
          config.thread_per_block, 0, device.stream(),
          config.virtual_thread_count, m, n, num_diags, max_diag_len,
          lower_diag_index, upper_diag_index, input.data(), diag.data(),
          output.data()));
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

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
