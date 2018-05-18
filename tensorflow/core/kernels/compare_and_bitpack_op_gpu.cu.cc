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

#include "tensorflow/core/kernels/compare_and_bitpack_op.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace functor {

template <typename T>
__global__ void CompareAndBitpackKernel(const int size, const T* threshold,
                                        const T* input, uint8* output) {
  // TODO(ebrevdo): Erich said: to get a better memory access pattern
  // you could have 8 threads load this data and do a comparison, then
  // use the ballot instruction to combine the values from each thread
  // in the warp in one instruction (so each thread will have the
  // result for 4 blocks) followed by an appropriate shift and mask to
  // get the 8-bits of interest.
  const T thresh = ldg(threshold);
  CUDA_1D_KERNEL_LOOP(i, size) {
    const T* block = input + 8 * i;
    output[i] =
        ((((ldg(block) > thresh) << 7)) | (((ldg(block + 1) > thresh) << 6)) |
         (((ldg(block + 2) > thresh) << 5)) |
         (((ldg(block + 3) > thresh) << 4)) |
         (((ldg(block + 4) > thresh) << 3)) |
         (((ldg(block + 5) > thresh) << 2)) |
         (((ldg(block + 6) > thresh) << 1)) | (((ldg(block + 7) > thresh))));
  }
}

template <>
__global__ void CompareAndBitpackKernel<bool>(const int size,
                                              const bool* threshold,
                                              const bool* input,
                                              uint8* output) {
  // TODO(ebrevdo): Erich said: I think you could again have multiple
  // threads work on one block and use the ballot instruction to the
  // bit packing in one instruction.
  CUDA_1D_KERNEL_LOOP(i, size) {
    const int64 block = ldg(reinterpret_cast<const int64*>(input + 8 * i));
    // NOTE(ebrevdo): This assumes memory is little-endian.
    output[i] =
        ((((block & (1LL << (7 * 8))) >> (7 * 8 - 0))) |
         (((block & (1LL << (6 * 8))) >> (6 * 8 - 1))) |
         (((block & (1LL << (5 * 8))) >> (5 * 8 - 2))) |
         (((block & (1LL << (4 * 8))) >> (4 * 8 - 3))) |
         (((block & (1LL << (3 * 8))) >> (3 * 8 - 4))) |
         (((block & (1LL << (2 * 8))) >> (2 * 8 - 5))) |
         (((block & (1LL << 8)) >> (1 * 8 - 6))) | (((block & (1LL)) << 7)));
  }
}

template <>
__global__ void CompareAndBitpackKernel<float>(const int size,
                                               const float* threshold,
                                               const float* input,
                                               uint8* output) {
  const float thresh = ldg(threshold);
  CUDA_1D_KERNEL_LOOP(i, size) {
    const float4 block0 = ldg(reinterpret_cast<const float4*>(input + 8 * i));
    const float4 block1 =
        ldg(reinterpret_cast<const float4*>(input + 8 * i + 4));
    output[i] = ((((block0.x > thresh) << 7)) | (((block0.y > thresh) << 6)) |
                 (((block0.z > thresh) << 5)) | (((block0.w > thresh) << 4)) |
                 (((block1.x > thresh) << 3)) | (((block1.y > thresh) << 2)) |
                 (((block1.z > thresh) << 1)) | (((block1.w > thresh))));
  }
}

template <>
__global__ void CompareAndBitpackKernel<double>(const int size,
                                                const double* threshold,
                                                const double* input,
                                                uint8* output) {
  const double thresh = ldg(threshold);
  CUDA_1D_KERNEL_LOOP(i, size) {
    const double2 block0 = ldg(reinterpret_cast<const double2*>(input + 8 * i));
    const double2 block1 =
        ldg(reinterpret_cast<const double2*>(input + 8 * i + 2));
    const double2 block2 =
        ldg(reinterpret_cast<const double2*>(input + 8 * i + 4));
    const double2 block3 =
        ldg(reinterpret_cast<const double2*>(input + 8 * i + 6));
    output[i] = ((((block0.x > thresh) << 7)) | (((block0.y > thresh) << 6)) |
                 (((block1.x > thresh) << 5)) | (((block1.y > thresh) << 4)) |
                 (((block2.x > thresh) << 3)) | (((block2.y > thresh) << 2)) |
                 (((block3.x > thresh) << 1)) | (((block3.y > thresh))));
  }
}

#define DEFINE_GPU_SPECS(T)                                               \
  template <>                                                             \
  void CompareAndBitpack<GPUDevice, T>::operator()(                       \
      OpKernelContext* c, typename TTypes<T>::ConstMatrix input,          \
      typename TTypes<T>::ConstScalar threshold,                          \
      TTypes<uint8>::Matrix output) {                                     \
    const GPUDevice& d = c->eigen_device<GPUDevice>();                    \
    int64 total_count = output.size();                                    \
    CudaLaunchConfig config = GetCudaLaunchConfig(total_count, d);        \
                                                                          \
    CompareAndBitpackKernel<T>                                            \
        <<<config.block_count, config.thread_per_block, 0, d.stream()>>>( \
            total_count, threshold.data(), input.data(), output.data());  \
  }

TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_SPECS)
TF_CALL_bool(DEFINE_GPU_SPECS)

#undef DECLARE_GPU_SPECS

}  // namespace functor

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
