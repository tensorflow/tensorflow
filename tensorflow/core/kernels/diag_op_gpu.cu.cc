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

#include <complex>
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "tensorflow/core/kernels/diag_op.h"

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

template <typename T>
__global__ void DiagCudaKernel(const int num_threads,
                               const int64 size,
                               const T* in,
                               T* out) {
  CUDA_1D_KERNEL_LOOP(index, num_threads) {
    out[(1 + size) * index] = in[index];
  }
}

template <typename T>
__global__ void ZeroCudaKernel(const int num_threads,
                               T* out) {
  CUDA_1D_KERNEL_LOOP(index, num_threads) {
    out[index] = T(0);
  }
}

template <typename T>
struct DiagFunctor<GPUDevice, T> {
  void operator() (const GPUDevice& device, const int64 size,
                      const T* in, T* out) {
    // using CudaType = typename CUDAComplexT<T>::type;
    // const CudaType* input_ptr = reinterpret_cast<const CudaType*>(in);
    // CudaType* output_ptr = reinterpret_cast<CudaType*>(out);
    //
    CudaLaunchConfig zero_config = GetCudaLaunchConfig(size*size, device);
    ZeroCudaKernel<<<zero_config.block_count,
                     zero_config.thread_per_block,
                     0, device.stream()>>>(
        zero_config.virtual_thread_count, out);

    CudaLaunchConfig diag_config = GetCudaLaunchConfig(size, device);
    DiagCudaKernel<<<diag_config.block_count,
                     diag_config.thread_per_block,
                     0, device.stream()>>>(
        diag_config.virtual_thread_count, size, in, out);
  }
};

template struct DiagFunctor<GPUDevice, double>;
template struct DiagFunctor<GPUDevice, float>;
template struct DiagFunctor<GPUDevice, int32>;
template struct DiagFunctor<GPUDevice, int64>;
template struct DiagFunctor<GPUDevice, complex64>;
template struct DiagFunctor<GPUDevice, complex128>;


template <typename T>
__global__ void DiagPartCudaKernel(const int num_threads,
                                   const int64 size,
                                   const T* in,
                                   T* out) {
  CUDA_1D_KERNEL_LOOP(index, num_threads) {
    out[index] = in[(1 + size) * index];
  }
}

template <typename T>
struct DiagPartFunctor<GPUDevice, T> {
  void operator() (const GPUDevice& device, const int64 size,
                      const T* in, T* out) {
    // using CudaType = typename CUDAComplexT<T>::type;
    // const CudaType* input_ptr = reinterpret_cast<const CudaType*>(in);
    // CudaType* output_ptr = reinterpret_cast<CudaType*>(out);
    //

    CudaLaunchConfig diag_config = GetCudaLaunchConfig(size, device);
    DiagPartCudaKernel<<<diag_config.block_count,
                     diag_config.thread_per_block,
                     0, device.stream()>>>(
        diag_config.virtual_thread_count, size, in, out);
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

#endif  // GOOGLE_CUDA
