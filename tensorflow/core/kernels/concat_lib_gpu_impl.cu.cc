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

#include <memory>
#include <vector>

#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/cuda_device_array_gpu.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace {

template <typename T, typename IntType>
__global__ void concat_fixed_kernel(
    CudaDeviceArrayStruct<const T*> input_ptr_data, int split_size,
    int total_rows, int total_cols, T* output) {
  const T** input_ptrs = GetCudaDeviceArrayOnDevice(&input_ptr_data);
  IntType gidx = blockIdx.x * blockDim.x + threadIdx.x;

  for (; gidx < total_cols; gidx += blockDim.x * gridDim.x) {
    IntType gidy = blockIdx.y * blockDim.y + threadIdx.y;

    IntType split = gidx / split_size;
    const T* input_ptr = input_ptrs[split];
    IntType col_offset = gidx % split_size;
#pragma unroll
    for (; gidy < total_rows; gidy += blockDim.y * gridDim.y) {
      output[gidy * total_cols + gidx] =
          input_ptr[gidy * split_size + col_offset];
    }
  }
}

}  // end namespace

// cannot be in anonymous namespace due to extern shared memory
template <typename T, typename IntType, bool useSmem>
__global__ void concat_variable_kernel(
    CudaDeviceArrayStruct<const T*> input_ptr_data,
    CudaDeviceArrayStruct<IntType> output_scan, IntType total_rows,
    IntType total_cols, T* output) {
  const T** input_ptrs = GetCudaDeviceArrayOnDevice(&input_ptr_data);
  IntType* col_scan = GetCudaDeviceArrayOnDevice(&output_scan);

  // do upper_bound on col to find which pointer we should be using
  IntType gidx = blockIdx.x * blockDim.x + threadIdx.x;
  IntType num_inputs = input_ptr_data.size;

  // verbose declaration needed due to template
  extern __shared__ __align__(sizeof(T)) unsigned char smem[];
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
      cuda_helper::upper_bound<IntType>(col_scan, num_inputs, gidx) - 1;

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
    const T* input_ptr = input_ptrs[curr_segment];

    IntType gidy = blockIdx.y * blockDim.y + threadIdx.y;
    for (; gidy < total_rows; gidy += blockDim.y * gridDim.y)
      output[gidy * total_cols + gidx] =
          input_ptr[gidy * segment_width + local_col];
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
    if (std::is_same<IntType, int32>::value) {
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
                   const CudaDeviceArrayStruct<const T*>& input_ptrs,
                   const CudaDeviceArrayStruct<IntType>& output_scan,
                   bool fixed_size, int split_size,
                   typename TTypes<T, 2>::Matrix* output) {
  auto config = GetCuda2DLaunchConfig(output->dimension(1),
                                      output->dimension(0), gpu_device);

  if (fixed_size) {
    concat_fixed_kernel<T, IntType>
        <<<config.block_count, config.thread_per_block, 0,
           gpu_device.stream()>>>(input_ptrs, split_size, output->dimension(0),
                                  output->dimension(1), output->data());
  } else {
    IntType smem_max = gpu_device.sharedMemPerBlock();
    IntType smem_usage = output_scan.size * sizeof(IntType);
    // performance crossover is less than using maximum available shared memory
    // on most processors
    // possibly due to decreasing occupancy
    // 4096 inputs is a lot, most code will take the smem path
    const int32 kMaxSmemBytesPerformance = 16384;
    if (smem_usage < smem_max && smem_usage < kMaxSmemBytesPerformance)
      concat_variable_kernel<T, IntType, true>
          <<<config.block_count, config.thread_per_block, smem_usage,
             gpu_device.stream()>>>(input_ptrs, output_scan,
                                    output->dimension(0), output->dimension(1),
                                    output->data());
    else
      concat_variable_kernel<T, IntType, false>
          <<<config.block_count, config.thread_per_block, 0,
             gpu_device.stream()>>>(input_ptrs, output_scan,
                                    output->dimension(0), output->dimension(1),
                                    output->data());
  }
}

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

#define REGISTER_GPU32(T)                                               \
  template void ConcatGPUImpl<T, int32>(                                \
      const Eigen::GpuDevice& d,                                        \
      const CudaDeviceArrayStruct<const T*>& input_ptrs,                \
      const CudaDeviceArrayStruct<int32>& ptr_offsets, bool fixed_size, \
      int split_size, typename TTypes<T, 2>::Matrix* output);

#define REGISTER_GPU64(T)                                               \
  template void ConcatGPUImpl<T, int64>(                                \
      const Eigen::GpuDevice& d,                                        \
      const CudaDeviceArrayStruct<const T*>& input_ptrs,                \
      const CudaDeviceArrayStruct<int64>& ptr_offsets, bool fixed_size, \
      int split_size, typename TTypes<T, 2>::Matrix* output);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPUCONCAT32);
TF_CALL_complex64(REGISTER_GPUCONCAT32);
TF_CALL_complex128(REGISTER_GPUCONCAT32);
TF_CALL_int64(REGISTER_GPUCONCAT32);
REGISTER_GPUCONCAT32(bfloat16);
REGISTER_GPUCONCAT32(bool);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPUCONCAT64);
TF_CALL_complex64(REGISTER_GPUCONCAT64);
TF_CALL_complex128(REGISTER_GPUCONCAT64);
TF_CALL_int64(REGISTER_GPUCONCAT64);
REGISTER_GPUCONCAT64(bfloat16);
REGISTER_GPUCONCAT64(bool);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU32);
TF_CALL_complex64(REGISTER_GPU32);
TF_CALL_complex128(REGISTER_GPU32);
TF_CALL_int64(REGISTER_GPU32);
REGISTER_GPU32(bfloat16);
REGISTER_GPU32(bool);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU64);
TF_CALL_complex64(REGISTER_GPU64);
TF_CALL_complex128(REGISTER_GPU64);
TF_CALL_int64(REGISTER_GPU64);
REGISTER_GPU64(bfloat16);
REGISTER_GPU64(bool);

#undef REGISTER_GPUCONCAT32
#undef REGISTER_GPUCONCAT64
#undef REGISTER_GPU32
#undef REGISTER_GPU64

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
