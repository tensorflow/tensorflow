/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include <stdio.h>

#include "tensorflow/core/kernels/split_lib.h"

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {
namespace functor {

template <typename Device, typename T>
void Split<Device, T>::operator()(
    const Device& d, typename TTypes<T, 3>::Tensor output,
    typename TTypes<T, 3>::ConstTensor input,
    const Eigen::DSizes<Eigen::DenseIndex, 3>& slice_indices,
    const Eigen::DSizes<Eigen::DenseIndex, 3>& slice_sizes) {
  To32Bit(output).device(d) = To32Bit(input).slice(slice_indices, slice_sizes);
}

#define DEFINE_GPU_KERNELS(T) template struct Split<Eigen::GpuDevice, T>;

TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_KERNELS);

}  // namespace functor

namespace {

template <typename T>
__global__ void SplitOpKernel(const T* input, int32 num_split,
                              int32 prefix_dim_size, int32 split_dim_size,
                              int32 suffix_dim_size, T** output_ptrs) {
  eigen_assert(blockDim.y == 1);
  eigen_assert(blockDim.z == 1);
  eigen_assert(split_dim_size % num_split == 0);

  int32 size = prefix_dim_size * split_dim_size * suffix_dim_size;
  int32 piece_size = split_dim_size / num_split;

  CUDA_1D_KERNEL_LOOP(offset, size) {
    // Calculate the index into input from offset.
    int32 i = offset / (split_dim_size * suffix_dim_size);
    int32 j = (offset % (split_dim_size * suffix_dim_size)) / suffix_dim_size;
    int32 k = offset % suffix_dim_size;

    // Find the output buffer that should be written to.
    T* output_ptr = output_ptrs[j / piece_size];
    // output_ptr is pointing to an array of size
    //  [prefix_dim_size][piece_size][suffix_dim_size].
    //
    // output_ptr[i][j % piece_size][k] = input[offset];
    // Linearize (i, j % piece_size, k) into an offset.
    int32 output_offset = i * piece_size * suffix_dim_size +
                          (j % piece_size) * suffix_dim_size + k;
    *(output_ptr + output_offset) = ldg(input + offset);
  }
}

}  // namespace

template <typename T>
struct SplitOpGPULaunch {
  void Run(const Eigen::GpuDevice& d, const T* input, int32 num_split,
           int32 prefix_dim_size, int32 split_dim_size, int32 suffix_dim_size,
           T** output_ptrs) {
    CudaLaunchConfig config = GetCudaLaunchConfig(
        prefix_dim_size * split_dim_size * suffix_dim_size, d);

    SplitOpKernel<
        T><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
        input, num_split, prefix_dim_size, split_dim_size, suffix_dim_size,
        static_cast<T**>(output_ptrs));
  }
};

#define REGISTER_GPU_KERNEL(T) template struct SplitOpGPULaunch<T>;

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
