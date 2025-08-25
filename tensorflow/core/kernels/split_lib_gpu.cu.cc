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

#include "tensorflow/core/kernels/split_lib_gpu.h"

#include <stdio.h>

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/gpu_device_array_gpu.h"
#include "tensorflow/core/kernels/split_lib.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {
namespace functor {

template <typename Device, typename T, int NDims>
void Split<Device, T, NDims>::operator()(
    const Device& d, typename TTypes<T, NDims>::Tensor output,
    typename TTypes<T, NDims>::ConstTensor input,
    const Eigen::DSizes<Eigen::DenseIndex, NDims>& slice_indices,
    const Eigen::DSizes<Eigen::DenseIndex, NDims>& slice_sizes) {
  To32Bit(output).device(d) = To32Bit(input).slice(slice_indices, slice_sizes);
}

template <typename Device, typename T>
void SplitCustom<Device, T>::operator()(
    const Device& d, typename TTypes<T, 2>::Tensor output,
    typename TTypes<T, 2>::ConstTensor input,
    const Eigen::DSizes<Eigen::DenseIndex, 2>& slice_indices,
    const Eigen::DSizes<Eigen::DenseIndex, 2>& slice_sizes) {
  To32Bit(output).device(d) = To32Bit(input).slice(slice_indices, slice_sizes);
}

#define DEFINE_GPU_KERNELS(T)                    \
  template struct Split<Eigen::GpuDevice, T, 2>; \
  template struct Split<Eigen::GpuDevice, T, 3>;

TF_CALL_int64(DEFINE_GPU_KERNELS);
TF_CALL_uint8(DEFINE_GPU_KERNELS);
TF_CALL_GPU_ALL_TYPES(DEFINE_GPU_KERNELS);

#undef DEFINE_GPU_KERNELS
#define DEFINE_GPU_KERNELS(T) template struct SplitCustom<Eigen::GpuDevice, T>;

TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_KERNELS);
TF_CALL_COMPLEX_TYPES(DEFINE_GPU_KERNELS);

#undef DEFINE_GPU_KERNELS

}  // namespace functor

namespace {

template <typename T>
__global__ void SplitOpKernel(const T* __restrict__ input,
                              int32 prefix_dim_size, int32 split_dim_size,
                              int32 suffix_dim_size,
                              GpuDeviceArrayStruct<T*> output_ptr_data) {
  const int32 num_split = output_ptr_data.size;
  T** output_ptrs = GetGpuDeviceArrayOnDevice(&output_ptr_data);

  eigen_assert(blockDim.y == 1);
  eigen_assert(blockDim.z == 1);
  eigen_assert(split_dim_size % num_split == 0);

  int32 size = prefix_dim_size * split_dim_size * suffix_dim_size;
  int32 piece_size = split_dim_size / num_split;

  GPU_1D_KERNEL_LOOP(offset, size) {
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

// cannot be in anonymous namespace due to extern shared memory
// very similar to the concat kernel except the input/output logic
// is reversed
template <typename T, typename IntType, bool useSmem>
__global__ void split_v_kernel(const T* __restrict__ input_ptr,
                               GpuDeviceArrayStruct<IntType> output_scan,
                               IntType total_rows, IntType total_cols,
                               GpuDeviceArrayStruct<T*> output_ptr_data) {
  T** output_ptrs = GetGpuDeviceArrayOnDevice(&output_ptr_data);
  IntType* col_scan = GetGpuDeviceArrayOnDevice(&output_scan);

  // do upper_bound on col to find which pointer we should be using
  IntType gidx = blockIdx.x * blockDim.x + threadIdx.x;
  int num_outputs = output_ptr_data.size;

  // verbose declaration needed due to template
  GPU_DYNAMIC_SHARED_MEM_DECL(2, unsigned char, smem);
  IntType* smem_col_scan = reinterpret_cast<IntType*>(smem);

  if (useSmem) {
    IntType lidx = threadIdx.y * blockDim.x + threadIdx.x;
    IntType blockSize = blockDim.x * blockDim.y;

    for (IntType i = lidx; i < output_scan.size; i += blockSize) {
      smem_col_scan[i] = col_scan[i];
    }

    __syncthreads();

    col_scan = smem_col_scan;
  }

  // do an initial binary search and then scan linearly from there
  // works well when there are many small segments and when the
  // segments are much longer
  IntType segment =
      gpu_helper::upper_bound<IntType>(col_scan, num_outputs, gidx) - 1;

  IntType curr_offset = col_scan[segment];
  IntType curr_segment = segment;
  for (; gidx < total_cols; gidx += blockDim.x * gridDim.x) {
    IntType curr_col_offset;
    while ((curr_col_offset = col_scan[curr_segment + 1]) <= gidx) {
      curr_offset = curr_col_offset;
      ++curr_segment;
    }

    IntType local_col = gidx - curr_offset;
    IntType segment_width = curr_col_offset - curr_offset;
    T* output_ptr = output_ptrs[curr_segment];

    IntType gidy = blockIdx.y * blockDim.y + threadIdx.y;
    for (; gidy < total_rows; gidy += blockDim.y * gridDim.y)
      output_ptr[gidy * segment_width + local_col] =
          input_ptr[gidy * total_cols + gidx];
  }
}

// different from the original split implementation due to 2D vs 3D
// dimensions.  This version is likely faster due to less integer math.
template <typename T>
__global__ void SplitVOpKernel_fixed(const T* __restrict__ input,
                                     int32 prefix_dim_size,
                                     int32 suffix_dim_size,
                                     GpuDeviceArrayStruct<T*> output_ptr_data) {
  const int32 num_split = output_ptr_data.size;
  T** output_ptrs = GetGpuDeviceArrayOnDevice(&output_ptr_data);

  eigen_assert(blockDim.y == 1);
  eigen_assert(blockDim.z == 1);

  int32 size = prefix_dim_size * suffix_dim_size;
  int32 piece_size = suffix_dim_size / num_split;

  GPU_1D_KERNEL_LOOP(offset, size) {
    // Calculate the index into input from offset.
    int32 i = offset / suffix_dim_size;
    int32 j = offset % suffix_dim_size;

    // Find the output buffer that should be written to.
    T* output_ptr = output_ptrs[j / piece_size];
    int32 output_offset = i * piece_size + (j % piece_size);
    output_ptr[output_offset] = input[offset];
  }
}

template <typename T>
void SplitOpGPULaunch<T>::Run(const Eigen::GpuDevice& d, const T* input,
                              int32 prefix_dim_size, int32 split_dim_size,
                              int32 suffix_dim_size,
                              const GpuDeviceArrayStruct<T*>& output_ptr_data) {
  GpuLaunchConfig config =
      GetGpuLaunchConfig(prefix_dim_size * split_dim_size * suffix_dim_size, d);

  TF_CHECK_OK(GpuLaunchKernel(SplitOpKernel<T>, config.block_count,
                              config.thread_per_block, 0, d.stream(), input,
                              prefix_dim_size, split_dim_size, suffix_dim_size,
                              output_ptr_data));
}

template <typename T, typename IntType>
void SplitVOpGPULaunch<T, IntType>::Run(
    const Eigen::GpuDevice& gpu_device, bool fixed_size, const T* input_ptr,
    int total_rows, int total_cols,
    const GpuDeviceArrayStruct<IntType>& output_scan,
    const GpuDeviceArrayStruct<T*>& output_ptr_data) {
  if (fixed_size) {
    GpuLaunchConfig config =
        GetGpuLaunchConfig(total_rows * total_cols, gpu_device);

    TF_CHECK_OK(GpuLaunchKernel(SplitVOpKernel_fixed<T>, config.block_count,
                                config.thread_per_block, 0, gpu_device.stream(),
                                input_ptr, total_rows, total_cols,
                                output_ptr_data));
  } else {
    auto config = GetGpu2DLaunchConfig(total_cols, total_rows, gpu_device);
    IntType smem_max = gpu_device.sharedMemPerBlock();
    IntType smem_usage = output_scan.size * sizeof(IntType);
    // performance crossover is less than using maximum available shared
    // memory on most processors possibly due to decreasing occupancy
    // 4096 inputs is a lot, most code will take the smem path
    const int32 kMaxSmemBytesPerformance = 16384;
    if (smem_usage < smem_max && smem_usage < kMaxSmemBytesPerformance) {
      TF_CHECK_OK(GpuLaunchKernel(
          split_v_kernel<T, IntType, true>, config.block_count,
          config.thread_per_block, smem_usage, gpu_device.stream(), input_ptr,
          output_scan, total_rows, total_cols, output_ptr_data));
    } else {
      TF_CHECK_OK(GpuLaunchKernel(
          split_v_kernel<T, IntType, false>, config.block_count,
          config.thread_per_block, 0, gpu_device.stream(), input_ptr,
          output_scan, total_rows, total_cols, output_ptr_data));
    }
  }
}

#define REGISTER_GPU_KERNEL(T) template struct SplitOpGPULaunch<T>;

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNEL);
TF_CALL_COMPLEX_TYPES(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL
#define REGISTER_GPU_KERNEL(T)                 \
  template struct SplitVOpGPULaunch<T, int8>;  \
  template struct SplitVOpGPULaunch<T, int32>; \
  template struct SplitVOpGPULaunch<T, int64>;

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNEL);
TF_CALL_COMPLEX_TYPES(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
