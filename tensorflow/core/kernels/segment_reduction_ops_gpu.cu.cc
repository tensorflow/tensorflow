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

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/segment_reduction_ops.h"

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {

using GPUDevice = Eigen::GpuDevice;

// Helper for UnusortedSegmentSumCustomKernel that adds value into dest
// atomically.
template <typename T>
static __device__ __forceinline__ void AccumulateInto(T* dest, const T& value) {
  CudaAtomicAdd(dest, value);
}

// Specializations of AccumulateInto for complex types, which CudaAtomicAdd does
// not support. We treat a std::complex<T>* as a T* (the C++ standard section
// 26.4.4 allows this explicitly) and atomic add the real and imaginary
// components individually. The operation as a whole is not atomic, but we can
// safely treat the components independently for the purpose of accumulating.
template <>
__device__ __forceinline__ void AccumulateInto(
    std::complex<float>* dest, const std::complex<float>& value) {
  auto dest_scalar = reinterpret_cast<float*>(dest);
  CudaAtomicAdd(dest_scalar, value.real());
  CudaAtomicAdd(dest_scalar + 1, value.imag());
}

template <>
__device__ __forceinline__ void AccumulateInto(
    std::complex<double>* dest, const std::complex<double>& value) {
  auto dest_scalar = reinterpret_cast<double*>(dest);
  CudaAtomicAdd(dest_scalar, value.real());
  CudaAtomicAdd(dest_scalar + 1, value.imag());
}

// SortedSegmentSumFunctor kernel reduces input data just as
// UnsortedSegmentSumCustomKernel does except that input data
// is partitioned along the outer reduction dimension. This is
// because consecutive rows (elements in a row share the same
// outer dimension index) in the flattened 2D input data likely
// belong to the same segment in sorted segment sum operation.
// Therefore such partitioning strategy has two advantages over
// the UnsortedSegmentSumFunctor kernel:
// 1. Each thread reduces across multiple rows before writing
// answers to the global memory, we can therefore
// write reduction results to global memory less often.
// 2. We may know that the current thread is the only contributor
// to an output element because of the increasing nature of segment
// ids. In such cases, we do not need to use atomic operations
// to write results to global memory.
// In the flattened view of input data (with only outer and inner
// dimension), every thread processes a strip of input data of
// size OuterDimTileSize x 1. This strip runs across multiple
// rows of input data and all reduction elements share one inner
// dimension index.
template <typename T, typename Index, int OuterDimTileSize>
__global__ void SortedSegmentSumCustomKernel(const Index input_outer_dim_size,
                                             const Index inner_dim_size,
                                             const Index output_outer_dim_size,
                                             const Index* segment_ids,
                                             const T* input, T* output,
                                             const Index total_stripe_count) {
  CUDA_1D_KERNEL_LOOP(stripe_index, total_stripe_count) {
    const Index segment_offset = stripe_index % inner_dim_size;
    const Index input_outer_dim_index_base =
        stripe_index / inner_dim_size * Index(OuterDimTileSize);

    T sum = T(0);
    Index first_segment_id = segment_ids[input_outer_dim_index_base];
    Index last_output_segment_id = output_outer_dim_size;

    const Index actual_stripe_height =
        min(Index(OuterDimTileSize),
            input_outer_dim_size - input_outer_dim_index_base);
    for (Index j = 0; j < actual_stripe_height; j++) {
      Index current_output_segment_id =
          segment_ids[input_outer_dim_index_base + j];
      // Decide whether to write result to global memory.
      // Result is only written to global memory if we move
      // to another segment. Otherwise we can keep accumulating
      // locally.
      if (current_output_segment_id > last_output_segment_id) {
        const Index output_index =
            last_output_segment_id * inner_dim_size + segment_offset;
        // decide whether to write result to global memory using atomic
        // operations
        if (last_output_segment_id == first_segment_id) {
          AccumulateInto<T>(output + output_index, sum);
        } else {
          *(output + output_index) = sum;
        }
        sum = T(0);
      }
      sum += ldg(input + (input_outer_dim_index_base + j) * inner_dim_size +
                 segment_offset);
      last_output_segment_id = current_output_segment_id;
    }
    // For the last result in a strip, always write using atomic operations
    // due to possible race conditions with threads computing
    // the following strip.
    const Index output_index =
        last_output_segment_id * inner_dim_size + segment_offset;
    AccumulateInto<T>(output + output_index, sum);
  }
}

// UnsortedSegmentSumFunctor kernel processes 'input_total_size' elements.
// Each element is mapped from input to output by a combination of its
// 'segment_ids' mapping and 'inner_dim_size'.
template <typename T, typename Index>
__global__ void UnsortedSegmentSumCustomKernel(
    const Index input_outer_dim_size, const Index inner_dim_size,
    const Index output_outer_dim_size, const Index* segment_ids, const T* input,
    T* output) {
  const Index input_total_size = input_outer_dim_size * inner_dim_size;
  const Index output_total_size = output_outer_dim_size * inner_dim_size;
  CUDA_1D_KERNEL_LOOP(input_index, input_total_size) {
    const Index input_segment_index = input_index / inner_dim_size;
    const Index segment_offset = input_index % inner_dim_size;
    const Index output_segment_index = segment_ids[input_segment_index];

    if (output_segment_index < 0 || output_segment_index >= output_total_size) {
      continue;
    }
    const Index output_index =
        output_segment_index * inner_dim_size + segment_offset;
    AccumulateInto<T>(output + output_index, ldg(input + input_index));
  }
}

namespace functor {

template <typename T, typename Index>
void SegmentSumFunctor<T, Index>::operator()(
    OpKernelContext* ctx, const GPUDevice& d, const Index output_rows,
    const TensorShape& segment_ids_shape,
    typename TTypes<Index>::ConstFlat segment_ids, const Index data_size,
    const T* data, typename TTypes<T, 2>::Tensor output) {
  if (output.size() == 0) {
    return;
  }
  // Set 'output' to zeros.
  CudaLaunchConfig config = GetCudaLaunchConfig(output.size(), d);
  SetZero<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
      output.size(), output.data());
  if (data_size == 0 || segment_ids_shape.num_elements() == 0) {
    return;
  }

  // Launch kernel to compute sorted segment sum.
  // Notes:
  // *) 'input_total_size' is the total number of elements to process.
  // *) 'segment_ids.shape' is a prefix of data's shape.
  // *) 'input_outer_dim_size' is the total number of segments to process.
  const Index input_total_size = data_size;
  const Index input_outer_dim_size = segment_ids.dimension(0);
  const Index input_inner_dim_size = input_total_size / input_outer_dim_size;

  const int OuterDimTileSize = 8;

  const Index input_outer_dim_num_stripe =
      Eigen::divup(input_outer_dim_size, Index(OuterDimTileSize));

  const Index total_stripe_count =
      input_inner_dim_size * input_outer_dim_num_stripe;

  config = GetCudaLaunchConfig(total_stripe_count, d);
  SortedSegmentSumCustomKernel<T, Index, OuterDimTileSize>
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          input_outer_dim_size, input_inner_dim_size, output_rows,
          segment_ids.data(), data, output.data(), total_stripe_count);
};

// UnsortedSegmentSumFunctor implementation for GPUDevice.
template <typename T, typename Index>
struct UnsortedSegmentSumFunctor<GPUDevice, T, Index>: UnsortedSegmentBaseFunctor<GPUDevice, T, Index> {
  void operator()(OpKernelContext* ctx, const GPUDevice& d,
                  const Index output_rows, const TensorShape& segment_ids_shape,
                  typename TTypes<Index>::ConstFlat segment_ids,
                  const Index data_size, const T* data,
                  typename TTypes<T, 2>::Tensor output) override {
    if (output.size() == 0) {
      return;
    }
    // Set 'output' to zeros.
    CudaLaunchConfig config = GetCudaLaunchConfig(output.size(), d);
    SetZero<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
        output.size(), output.data());
    if (data_size == 0 || segment_ids_shape.num_elements() == 0) {
      return;
    }

    // Launch kernel to compute unsorted segment sum.
    // Notes:
    // *) 'input_total_size' is the total number of elements to process.
    // *) 'segment_ids.shape' is a prefix of data's shape.
    // *) 'input_outer_dim_size' is the total number of segments to process.
    const Index input_total_size = data_size;
    const Index input_outer_dim_size = segment_ids.dimension(0);
    const Index input_inner_dim_size = input_total_size / input_outer_dim_size;

    config = GetCudaLaunchConfig(input_total_size, d);
    UnsortedSegmentSumCustomKernel<
        T,
        Index><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
        input_outer_dim_size, input_inner_dim_size, output_rows,
        segment_ids.data(), data, output.data());
  }
};

#define DEFINE_SORTED_GPU_SPECS_INDEX(T, Index) \
  template struct SegmentSumFunctor<T, Index>

#define DEFINE_SORTED_GPU_SPECS(T)         \
  DEFINE_SORTED_GPU_SPECS_INDEX(T, int32); \
  DEFINE_SORTED_GPU_SPECS_INDEX(T, int64);

TF_CALL_GPU_NUMBER_TYPES(DEFINE_SORTED_GPU_SPECS);

#define DEFINE_GPU_SPECS_INDEX(T, Index) \
  template struct UnsortedSegmentSumFunctor<GPUDevice, T, Index>

#define DEFINE_GPU_SPECS(T)         \
  DEFINE_GPU_SPECS_INDEX(T, int32); \
  DEFINE_GPU_SPECS_INDEX(T, int64);

TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_SPECS);
TF_CALL_complex64(DEFINE_GPU_SPECS);
TF_CALL_complex128(DEFINE_GPU_SPECS);

#undef DEFINE_GPU_SPECS
#undef DEFINE_GPU_SPECS_INDEX

}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
