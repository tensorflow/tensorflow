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

#include <complex>

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/diag_op.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

template <typename T>
__global__ void DiagGpuKernel(const int num_threads, const int64 size,
                              const T* __restrict__ in, T* __restrict__ out) {
  GPU_1D_KERNEL_LOOP(index, num_threads) {
    // Fill the diagonal elements or set to zero in other place.
    if (index % (1 + size) == 0) {
      out[index] = in[index / (1 + size)];
    } else {
      out[index] = T(0);
    }
  }
}

template <typename T>
struct DiagFunctor<GPUDevice, T> {
  EIGEN_ALWAYS_INLINE Status operator()(OpKernelContext* context,
                                        const int64 size, const T* in, T* out) {
    // Empty tensor couldn't launch the kernel.
    if (size == 0) {
      return Status::OK();
    }

    // GpuLaunchConfig uses an int for virtual_thread_count,
    // so this may overflow for `size*size` in extreme cases,
    // here is checking the multiplication overflow for integer.
    if (size && (int(size * size) / size) != size) {
      return errors::Internal("DiagOp got input size too large.");
    }
    int virtual_thread_count = int(size * size);

    // Launch the GPU kernel.
    const GPUDevice& device = context->eigen_device<GPUDevice>();
    GpuLaunchConfig diag_config =
        GetGpuLaunchConfig(virtual_thread_count, device);
    TF_CHECK_OK(GpuLaunchKernel(
        DiagGpuKernel<T>, diag_config.block_count, diag_config.thread_per_block,
        0, device.stream(), diag_config.virtual_thread_count, size, in, out));

    return Status::OK();
  }
};

template struct DiagFunctor<GPUDevice, double>;
template struct DiagFunctor<GPUDevice, float>;
template struct DiagFunctor<GPUDevice, int32>;
template struct DiagFunctor<GPUDevice, int64>;
template struct DiagFunctor<GPUDevice, complex64>;
template struct DiagFunctor<GPUDevice, complex128>;

template <typename T>
__global__ void DiagPartGpuKernel(const int num_threads, const int64 size,
                                  const T* __restrict__ in,
                                  T* __restrict__ out) {
  GPU_1D_KERNEL_LOOP(index, num_threads) {
    out[index] = in[(1 + size) * index];
  }
}

template <typename T>
struct DiagPartFunctor<GPUDevice, T> {
  EIGEN_ALWAYS_INLINE Status operator()(OpKernelContext* context,
                                        const int64 size, const T* in, T* out) {
    // Empty tensor couldn't launch the kernel.
    if (size == 0) {
      return Status::OK();
    }
    const GPUDevice& device = context->eigen_device<GPUDevice>();

    // Extract the diagonal elements.
    GpuLaunchConfig diag_config = GetGpuLaunchConfig(size, device);
    TF_CHECK_OK(
        GpuLaunchKernel(DiagPartGpuKernel<T>, diag_config.block_count,
                        diag_config.thread_per_block, 0, device.stream(),
                        diag_config.virtual_thread_count, size, in, out));

    return Status::OK();
  }
};

template struct DiagPartFunctor<GPUDevice, double>;
template struct DiagPartFunctor<GPUDevice, float>;
template struct DiagPartFunctor<GPUDevice, int32>;
template struct DiagPartFunctor<GPUDevice, int64>;
template struct DiagPartFunctor<GPUDevice, complex64>;
template struct DiagPartFunctor<GPUDevice, complex128>;

}  // end namespace functor
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
