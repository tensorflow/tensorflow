/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/linalg/matrix_band_part_op.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {
namespace functor {
typedef Eigen::GpuDevice GPUDevice;

template <typename Scalar>
__global__ void MatrixBandPartKernel(const int num_threads,
                                     const int batch_size, const int m,
                                     const int n, const int num_lower_diags,
                                     const int num_upper_diags,
                                     const Scalar* __restrict__ input_ptr,
                                     Scalar* __restrict__ output_ptr) {
  GPU_1D_KERNEL_LOOP(index, num_threads) {
    const int col = index % n;
    const int row = (index / n) % m;
    const int band_start = (num_lower_diags < 0 ? 0 : row - num_lower_diags);
    const int band_end = (num_upper_diags < 0 ? n : row + num_upper_diags + 1);
    if (col < band_start || col >= band_end) {
      output_ptr[index] = Scalar(0);
    } else {
      output_ptr[index] = input_ptr[index];
    }
  }
}

template <typename Scalar>
struct MatrixBandPartFunctor<GPUDevice, Scalar> {
  void operator()(OpKernelContext* context, const GPUDevice& device,
                  int num_lower_diags, int num_upper_diags,
                  typename TTypes<Scalar, 3>::ConstTensor input,
                  typename TTypes<Scalar, 3>::Tensor output) {
    const int batch_size = input.dimension(0);
    const int m = input.dimension(1);
    const int n = input.dimension(2);
    GpuLaunchConfig config = GetGpuLaunchConfig(batch_size * m * n, device);
    TF_CHECK_OK(GpuLaunchKernel(MatrixBandPartKernel<Scalar>,
                                config.block_count, config.thread_per_block, 0,
                                device.stream(), config.virtual_thread_count,
                                batch_size, m, n, num_lower_diags,
                                num_upper_diags, input.data(), output.data()));
  }
};

#define DEFINE_GPU_SPEC(T) template struct MatrixBandPartFunctor<GPUDevice, T>;

TF_CALL_GPU_ALL_TYPES(DEFINE_GPU_SPEC);

#undef DEFINE_GPU_SPEC
}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
