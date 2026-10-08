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

#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/concat_lib_gpu.h"
#include "tensorflow/core/kernels/gpu_device_array_gpu.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace {

template <typename T, typename IntType>
__global__ void concat_fixed_kernel(
    GpuDeviceArrayStruct<const T*> input_ptr_data, IntType split_size,
    IntType total_rows, IntType total_cols, T* __restrict__ output) {
  const T** input_ptrs = GetGpuDeviceArrayOnDevice(&input_ptr_data);
  IntType gidx = static_cast<IntType>(blockIdx.x) * blockDim.x + threadIdx.x;

  for (; gidx < total_cols;
       gidx += static_cast<IntType>(blockDim.x) * gridDim.x) {
    IntType gidy = static_cast<IntType>(blockIdx.y) * blockDim.y + threadIdx.y;

    IntType split = gidx / split_size;
    const T* input_ptr = input_ptrs[split];
    IntType col_offset = gidx % split_size;
#pragma unroll
    for (; gidy < total_rows;
         gidy += static_cast<IntType>(blockDim.y) * gridDim.y) {
      output[gidy * total_cols + gidx] =
          input_ptr[gidy * split_size + col_offset];
    }
  }
}

}  // end namespace

// cannot be in anonymous namespace due to extern shared memory
template <typename T, typename IntType, bool useSmem>
__global__ void concat_variable_kernel(
    GpuDeviceArrayStruct<const T*> input_ptr_data,
    GpuDeviceArrayStruct<IntType> output_scan, IntType total_rows,
    IntType total_cols, T* output) {
  const T** input_ptrs = GetGpuDeviceArrayOnDevice(&input_ptr_data);
  IntType* col_scan = GetGpuDeviceArrayOnDevice(&output_scan);

  // do upper_bound on col to find which pointer we should be using
  IntType gidx = static_cast<IntType>(blockIdx.x) * blockDim.x + threadIdx.x;
  IntType num_inputs = input_ptr_data.size;

  // verbose declaration needed due to template
  constexpr size_t kAlignTI =
      (alignof(T) > alignof(IntType)) ? alignof(T) : alignof(IntType);
  constexpr size_t kAlign = (kAlignTI < 16) ? 16 : kAlignTI;
  GPU_DYNAMIC_SHARED_MEM_DECL(kAlign, unsigned char, smem);
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
      gpu_helper::upper_bound<IntType>(col_scan, num_inputs, gidx) - 1;

  IntType curr_offset = col_scan[segment];
  IntType curr_segment = segment;
  for (; gidx < total_cols;
       gidx += static_cast<IntType>(blockDim.x) * gridDim.x) {
    IntType curr_col_offset;
    while ((curr_col_offset = col_scan[curr_segment + 1]) <= gidx) {
      curr_offset = curr_col_offset;
      ++curr_segment;
    }

    IntType local_col = gidx - curr_offset;
    IntType segment_width = curr_col_offset - curr_offset;
    const T* input_ptr = input_ptrs[curr_segment];

    IntType gidy = static_cast<IntType>(blockIdx.y) * blockDim.y + threadIdx.y;
    for (; gidy < total_rows;
         gidy += static_cast<IntType>(blockDim.y) * gridDim.y)
      output[gidy * total_cols + gidx] =
          input_ptr[gidy * segment_width + local_col];
  }
}

template <typename T, typename IntType>
struct ConcatBufferInfo {
  const T* ptr;
  IntType start_offset;
  IntType num_elements;
};

template <typename T, typename IntType, int MaxInputs = 32>
struct ConcatContiguousParams {
  int num_inputs;
  ConcatBufferInfo<T, IntType> inputs[MaxInputs];
};

template <typename T, typename IntType, int MaxInputs = 32>
__global__ void ConcatContiguousKernel(
    ConcatContiguousParams<T, IntType, MaxInputs> params,
    T* __restrict__ output) {
  for (int i = 0; i < params.num_inputs; ++i) {
    const auto& info = params.inputs[i];
    const T* const __restrict__ src = info.ptr;
    T* dst = output + info.start_offset;
    IntType count = info.num_elements;
    if (count <= 0) continue;

    for (IntType idx : GpuGridRangeX<IntType>(count)) {
      dst[idx] = src[idx];
    }
  }
}

template <typename T, typename IntType>
void ConcatGPUContiguous(
    const Eigen::GpuDevice& gpu_device,
    const std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>&
        inputs_flat,
    typename TTypes<T, 2>::Matrix* output) {
  constexpr int kBatchSize = 32;
  const int total_inputs = inputs_flat.size();
  IntType running_offset = 0;

  for (int start_idx = 0; start_idx < total_inputs; start_idx += kBatchSize) {
    ConcatContiguousParams<T, IntType, kBatchSize> params{};
    const int count = std::min(kBatchSize, total_inputs - start_idx);
    params.num_inputs = count;
    IntType max_elements = 0;

    for (int i = 0; i < count; ++i) {
      const int input_idx = start_idx + i;
      DCHECK_EQ(inputs_flat[input_idx]->dimension(0), 1);
      params.inputs[i].ptr = inputs_flat[input_idx]->data();
      params.inputs[i].start_offset = running_offset;
      params.inputs[i].num_elements =
          static_cast<IntType>(inputs_flat[input_idx]->dimension(1));
      running_offset += params.inputs[i].num_elements;
      max_elements = std::max(max_elements, params.inputs[i].num_elements);
    }

    if (max_elements <= 0) continue;

    int block_count = 0;
    int thread_per_block = 0;
    if constexpr (sizeof(IntType) == 8) {
      auto config_or = GetGpuLaunchConfig64(
          static_cast<int64_t>(max_elements), gpu_device);
      TF_CHECK_OK(config_or.status());
      block_count = config_or->block_count;
      thread_per_block = config_or->thread_per_block;
    } else {
      GpuLaunchConfig config = GetGpuLaunchConfig(
          static_cast<int>(max_elements), gpu_device);
      block_count = config.block_count;
      thread_per_block = config.thread_per_block;
    }

    TF_CHECK_OK(GpuLaunchKernel(
        ConcatContiguousKernel<T, IntType, kBatchSize>, block_count,
        thread_per_block, 0, gpu_device.stream(), params,
        output->data()));
  }
}

template <typename T, typename IntType>
void ConcatGPUSlice(
    const Eigen::GpuDevice& gpu_device,
    const std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>&
        inputs_flat,
    typename TTypes<T, 2>::Matrix* output) {
  Eigen::array<IntType, 2> offset{0, 0};
  for (int i = 0; i < inputs_flat.size(); ++i) {
    Eigen::array<IntType, 2> size;
    size[0] = inputs_flat[i]->dimension(0);
    size[1] = inputs_flat[i]->dimension(1);
    if (std::is_same<IntType, int32_t>::value) {
      To32Bit(*output).slice(offset, size).device(gpu_device) =
          To32Bit(*inputs_flat[i]);
    } else {
      output->slice(offset, size).device(gpu_device) = *inputs_flat[i];
    }

    offset[1] += size[1];
  }
}

template <typename T, typename IntType>
void ConcatGPUImpl(const Eigen::GpuDevice& gpu_device,
                   const GpuDeviceArrayStruct<const T*>& input_ptrs,
                   const GpuDeviceArrayStruct<IntType>& output_scan,
                   bool fixed_size, IntType split_size,
                   typename TTypes<T, 2>::Matrix* output) {
  auto config = GetGpu2DLaunchConfig(
      std::min<int64_t>(output->dimension(1),
                        std::numeric_limits<int32_t>::max()),
      std::min<int64_t>(output->dimension(0),
                        std::numeric_limits<int32_t>::max()),
      gpu_device);

  if (fixed_size) {
    TF_CHECK_OK(GpuLaunchKernel(
        concat_fixed_kernel<T, IntType>, config.block_count,
        config.thread_per_block, 0, gpu_device.stream(), input_ptrs, split_size,
        static_cast<IntType>(output->dimension(0)),
        static_cast<IntType>(output->dimension(1)), output->data()));
  } else {
    IntType smem_max = gpu_device.sharedMemPerBlock();
    IntType smem_usage = output_scan.size * sizeof(IntType);
    // performance crossover is less than using maximum available shared memory
    // on most processors
    // possibly due to decreasing occupancy
    // 4096 inputs is a lot, most code will take the smem path
    const int32_t kMaxSmemBytesPerformance = 16384;
    if (smem_usage < smem_max && smem_usage < kMaxSmemBytesPerformance) {
      TF_CHECK_OK(GpuLaunchKernel(
          concat_variable_kernel<T, IntType, true>, config.block_count,
          config.thread_per_block, smem_usage, gpu_device.stream(), input_ptrs,
          output_scan, static_cast<IntType>(output->dimension(0)),
          static_cast<IntType>(output->dimension(1)), output->data()));
    } else {
      TF_CHECK_OK(GpuLaunchKernel(
          concat_variable_kernel<T, IntType, false>, config.block_count,
          config.thread_per_block, 0, gpu_device.stream(), input_ptrs,
          output_scan, static_cast<IntType>(output->dimension(0)),
          static_cast<IntType>(output->dimension(1)), output->data()));
    }
  }
}

#define REGISTER_GPUCONCAT_CONTIGUOUS32(T)                                    \
  template void ConcatGPUContiguous<T, int32>(                                \
      const Eigen::GpuDevice& gpu_device,                                     \
      const std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>& \
          inputs_flat,                                                        \
      typename TTypes<T, 2>::Matrix* output);

#define REGISTER_GPUCONCAT_CONTIGUOUS64(T)                                    \
  template void ConcatGPUContiguous<T, int64>(                                \
      const Eigen::GpuDevice& gpu_device,                                     \
      const std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>& \
          inputs_flat,                                                        \
      typename TTypes<T, 2>::Matrix* output);

#define REGISTER_GPUCONCAT32(T)                                               \
  template void ConcatGPUSlice<T, int32>(                                     \
      const Eigen::GpuDevice& gpu_device,                                     \
      const std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>& \
          inputs_flat,                                                        \
      typename TTypes<T, 2>::Matrix* output);

#define REGISTER_GPUCONCAT64(T)                                               \
  template void ConcatGPUSlice<T, int64>(                                     \
      const Eigen::GpuDevice& gpu_device,                                     \
      const std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>& \
          inputs_flat,                                                        \
      typename TTypes<T, 2>::Matrix* output);

#define REGISTER_GPU32(T)                                              \
  template void ConcatGPUImpl<T, int32>(                               \
      const Eigen::GpuDevice& d,                                       \
      const GpuDeviceArrayStruct<const T*>& input_ptrs,                \
      const GpuDeviceArrayStruct<int32>& ptr_offsets, bool fixed_size, \
      int32 split_size, typename TTypes<T, 2>::Matrix* output);

#define REGISTER_GPU64(T)                                                \
  template void ConcatGPUImpl<T, int64>(                                 \
      const Eigen::GpuDevice& d,                                         \
      const GpuDeviceArrayStruct<const T*>& input_ptrs,                  \
      const GpuDeviceArrayStruct<int64_t>& ptr_offsets, bool fixed_size, \
      int64_t split_size, typename TTypes<T, 2>::Matrix* output);

TF_CALL_INTEGRAL_TYPES(REGISTER_GPUCONCAT_CONTIGUOUS32);
TF_CALL_GPU_ALL_TYPES(REGISTER_GPUCONCAT_CONTIGUOUS32);
TF_CALL_float8_e5m2(REGISTER_GPUCONCAT_CONTIGUOUS32);
TF_CALL_float8_e4m3fn(REGISTER_GPUCONCAT_CONTIGUOUS32);

TF_CALL_INTEGRAL_TYPES(REGISTER_GPUCONCAT_CONTIGUOUS64);
TF_CALL_GPU_ALL_TYPES(REGISTER_GPUCONCAT_CONTIGUOUS64);
TF_CALL_float8_e5m2(REGISTER_GPUCONCAT_CONTIGUOUS64);
TF_CALL_float8_e4m3fn(REGISTER_GPUCONCAT_CONTIGUOUS64);

TF_CALL_INTEGRAL_TYPES(REGISTER_GPUCONCAT32);  // int32 Needed for TensorLists.
TF_CALL_GPU_ALL_TYPES(REGISTER_GPUCONCAT32);
TF_CALL_float8_e5m2(REGISTER_GPUCONCAT32);
TF_CALL_float8_e4m3fn(REGISTER_GPUCONCAT32);

TF_CALL_INTEGRAL_TYPES(REGISTER_GPUCONCAT64);  // int32 Needed for TensorLists.
TF_CALL_GPU_ALL_TYPES(REGISTER_GPUCONCAT64);
TF_CALL_float8_e5m2(REGISTER_GPUCONCAT64);
TF_CALL_float8_e4m3fn(REGISTER_GPUCONCAT64);

TF_CALL_INTEGRAL_TYPES(REGISTER_GPU32);  // int32 Needed for TensorLists.
TF_CALL_GPU_ALL_TYPES(REGISTER_GPU32);
TF_CALL_float8_e5m2(REGISTER_GPU32);
TF_CALL_float8_e4m3fn(REGISTER_GPU32);

TF_CALL_INTEGRAL_TYPES(REGISTER_GPU64);  // int32 Needed for TensorLists.
TF_CALL_GPU_ALL_TYPES(REGISTER_GPU64);
TF_CALL_float8_e5m2(REGISTER_GPU64);
TF_CALL_float8_e4m3fn(REGISTER_GPU64);

#undef REGISTER_GPUCONCAT_CONTIGUOUS32
#undef REGISTER_GPUCONCAT_CONTIGUOUS64
#undef REGISTER_GPUCONCAT32
#undef REGISTER_GPUCONCAT64
#undef REGISTER_GPU32
#undef REGISTER_GPU64

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
