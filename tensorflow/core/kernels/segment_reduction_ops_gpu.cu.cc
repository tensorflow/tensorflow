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

// We need to include gpu_kernel_helper.h before segment_reduction_ops.h
// See comment in segment_reduction_ops.h for more details.
// clang-format off
#include "tensorflow/core/util/gpu_kernel_helper.h"
// clang-format on

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/segment_reduction_ops.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/util/gpu_device_functions.h"

namespace tensorflow {

using GPUDevice = Eigen::GpuDevice;

// SortedSegmentReductionFunctor kernel reduces input data just as
// UnsortedSegmentReductionCustomKernel does except that input data
// is partitioned along the outer reduction dimension. This is
// because consecutive rows (elements in a row share the same
// outer dimension index) in the flattened 2D input data likely
// belong to the same segment in sorted segment sum operation.
// Therefore such partitioning strategy has two advantages over
// the UnsortedSegmentReductionFunctor kernel:
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
template <typename T, typename Index, int OuterDimTileSize, typename ReductionF,
          typename AtomicReductionF>
__global__ void SortedSegmentReductionCustomKernel(
    const Index input_outer_dim_size, const Index inner_dim_size,
    const Index output_outer_dim_size, const Index* __restrict__ segment_ids,
    const T* __restrict__ input, T* __restrict__ output,
    const Index total_stripe_count, const T initial_value) {
  for (int stripe_index : GpuGridRangeX(total_stripe_count)) {
    const Index segment_offset = stripe_index % inner_dim_size;
    const Index input_outer_dim_index_base =
        stripe_index / inner_dim_size * Index(OuterDimTileSize);

    T reduce_res = initial_value;
    Index first_segment_id = segment_ids[input_outer_dim_index_base];
    Index last_output_segment_id = output_outer_dim_size;

    const Index actual_stripe_height =
        min(Index(OuterDimTileSize),
            input_outer_dim_size - input_outer_dim_index_base);
    for (Index j = 0; j < actual_stripe_height; j++) {
      Index current_output_segment_id =
          segment_ids[input_outer_dim_index_base + j];
      // Decide whether to write result to global memory. Result is only written
      // to global memory if we move to another segment. Otherwise we can keep
      // accumulating locally.
      if (current_output_segment_id > last_output_segment_id) {
        const Index output_index =
            last_output_segment_id * inner_dim_size + segment_offset;
        // Decide whether to write result to global memory using atomic
        // operations.
        if (last_output_segment_id == first_segment_id) {
          AtomicReductionF()(output + output_index, reduce_res);
        } else {
          ReductionF()(output + output_index, reduce_res);
        }
        reduce_res = initial_value;
      }
      ReductionF()(
          &reduce_res,
          ldg(input + (input_outer_dim_index_base + j) * inner_dim_size +
              segment_offset));
      last_output_segment_id = current_output_segment_id;
    }
    // For the last result in a strip, always write using atomic operations
    // due to possible race conditions with threads computing
    // the following strip.
    const Index output_index =
        last_output_segment_id * inner_dim_size + segment_offset;
    AtomicReductionF()(output + output_index, reduce_res);
  }
}

// UnsortedSegmentSumKernel processes 'input_total_size' elements.
// Each element is mapped from input to output by a combination of its
// 'segment_ids' mapping and 'inner_dim_size'.
template <typename T, typename Index, typename KernelReductionFunctor>
__global__ void UnsortedSegmentCustomKernel(
    const int64 input_outer_dim_size, const int64 inner_dim_size,
    const int64 output_outer_dim_size, const Index* __restrict__ segment_ids,
    const T* __restrict__ input, T* __restrict__ output) {
  const int64 input_total_size = input_outer_dim_size * inner_dim_size;
  for (int64 input_index : GpuGridRangeX(input_total_size)) {
    const int64 input_segment_index = input_index / inner_dim_size;
    const int64 segment_offset = input_index % inner_dim_size;
    const Index output_segment_index = segment_ids[input_segment_index];
    if (output_segment_index < 0 ||
        output_segment_index >= output_outer_dim_size) {
      continue;
    }
    const int64 output_index =
        output_segment_index * inner_dim_size + segment_offset;
    KernelReductionFunctor()(output + output_index, ldg(input + input_index));
  }
}

// TODO(duncanriach): move this into a utility and share it
bool RequireDeterminism() {
  static bool require_determinism = [] {
    bool deterministic_ops = false;
    TF_CHECK_OK(tensorflow::ReadBoolFromEnvVar("TF_DETERMINISTIC_OPS",
                                               /*default_val=*/false,
                                               &deterministic_ops));
    return deterministic_ops;
  }();
  return require_determinism;
}

bool DisableSegmentReductionOpDeterminismExceptions() {
  static bool cached_disable = [] {
    bool disable = false;
    TF_CHECK_OK(tensorflow::ReadBoolFromEnvVar(
        "TF_DISABLE_SEGMENT_REDUCTION_OP_DETERMINISM_EXCEPTIONS",
        /*default_val=*/false, &disable));
    return disable;
  }();
  return cached_disable;
}

namespace functor {

template <typename T, typename Index, typename InitialValueF,
          typename ReductionF, typename AtomicReductionF>
void SegmentReductionFunctor<
    T, Index, InitialValueF, ReductionF,
    AtomicReductionF>::operator()(OpKernelContext* ctx, const GPUDevice& d,
                                  const Index output_rows,
                                  const TensorShape& segment_ids_shape,
                                  typename TTypes<Index>::ConstFlat segment_ids,
                                  const Index data_size, const T* data,
                                  typename TTypes<T, 2>::Tensor output) {
  if (output.size() == 0) {
    return;
  }

  // Set 'output' to initial value.
  GpuLaunchConfig config = GetGpuLaunchConfig(output.size(), d);
  const T InitialValue = InitialValueF()();
  TF_CHECK_OK(GpuLaunchKernel(SetToValue<T>, config.block_count,
                              config.thread_per_block, 0, d.stream(),
                              output.size(), output.data(), InitialValue));
  if (data_size == 0 || segment_ids_shape.num_elements() == 0) {
    return;
  }

  // Launch kernel to compute sorted segment reduction.
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

  config = GetGpuLaunchConfig(total_stripe_count, d);
  TF_CHECK_OK(GpuLaunchKernel(
      SortedSegmentReductionCustomKernel<T, Index, OuterDimTileSize, ReductionF,
                                         AtomicReductionF>,
      config.block_count, config.thread_per_block, 0, d.stream(),
      input_outer_dim_size, input_inner_dim_size, output_rows,
      segment_ids.data(), data, output.data(), total_stripe_count,
      InitialValue));
}

template <typename T, typename Index, typename InitialValueF,
          typename ReductionF>
struct UnsortedSegmentFunctor<GPUDevice, T, Index, InitialValueF, ReductionF> {
  void operator()(OpKernelContext* ctx, const TensorShape& segment_ids_shape,
                  typename TTypes<Index>::ConstFlat segment_ids,
                  typename TTypes<T, 2>::ConstTensor data,
                  typename TTypes<T, 2>::Tensor output) {
    if (output.size() == 0) {
      return;
    }

    bool determinism_requirement_met =
        ReductionF::is_associative || !RequireDeterminism() ||
        DisableSegmentReductionOpDeterminismExceptions();
    OP_REQUIRES(
        ctx, determinism_requirement_met,
        errors::Unimplemented(
            "Deterministic GPU implementation of unsorted segment reduction op"
            " not available."));

    // Set 'output' to initial value.
    GPUDevice d = ctx->template eigen_device<GPUDevice>();
    GpuLaunchConfig config = GetGpuLaunchConfig(output.size(), d);
    TF_CHECK_OK(GpuLaunchKernel(
        SetToValue<T>, config.block_count, config.thread_per_block, 0,
        d.stream(), output.size(), output.data(), InitialValueF()()));
    const int64 data_size = data.size();
    if (data_size == 0 || segment_ids_shape.num_elements() == 0) {
      return;
    }
    // Launch kernel to compute unsorted segment reduction.
    // Notes:
    // *) 'data_size' is the total number of elements to process.
    // *) 'segment_ids.shape' is a prefix of data's shape.
    // *) 'input_outer_dim_size' is the total number of segments to process.
    const int64 input_outer_dim_size = segment_ids.dimension(0);
    const int64 input_inner_dim_size = data.dimension(1);
    const int64 output_outer_dim_size = output.dimension(0);
    config = GetGpuLaunchConfig(data_size, d);

    TF_CHECK_OK(GpuLaunchKernel(
        UnsortedSegmentCustomKernel<T, Index, ReductionF>, config.block_count,
        config.thread_per_block, 0, d.stream(), input_outer_dim_size,
        input_inner_dim_size, output_outer_dim_size, segment_ids.data(),
        data.data(), output.data()));
  }
};

#define DEFINE_SORTED_GPU_SPECS_INDEX(T, Index)                           \
  template struct SegmentReductionFunctor<T, Index, functor::Zero<T>,     \
                                          functor::NonAtomicSumOpGpu<T>,  \
                                          functor::AtomicSumOpGpu<T>>;    \
  template struct SegmentReductionFunctor<T, Index, functor::One<T>,      \
                                          functor::NonAtomicProdOpGpu<T>, \
                                          functor::AtomicProdOpGpu<T>>;   \
  template struct SegmentReductionFunctor<T, Index, functor::Highest<T>,  \
                                          functor::NonAtomicMinOpGpu<T>,  \
                                          functor::AtomicMinOpGpu<T>>;    \
  template struct SegmentReductionFunctor<T, Index, functor::Lowest<T>,   \
                                          functor::NonAtomicMaxOpGpu<T>,  \
                                          functor::AtomicMaxOpGpu<T>>;

#define DEFINE_SORTED_GPU_SPECS(T)         \
  DEFINE_SORTED_GPU_SPECS_INDEX(T, int32); \
  DEFINE_SORTED_GPU_SPECS_INDEX(T, int64);

TF_CALL_GPU_NUMBER_TYPES(DEFINE_SORTED_GPU_SPECS);

#define DEFINE_REAL_UNSORTED_GPU_SPECS_INDEX(T, Index)                         \
  template struct UnsortedSegmentFunctor<                                      \
      GPUDevice, T, Index, functor::Lowest<T>, functor::AtomicMaxOpGpu<T>>;    \
  template struct UnsortedSegmentFunctor<                                      \
      GPUDevice, T, Index, functor::Highest<T>, functor::AtomicMinOpGpu<T>>;   \
  template struct UnsortedSegmentFunctor<GPUDevice, T, Index, functor::One<T>, \
                                         functor::AtomicProdOpGpu<T>>;

// Sum is the only op that supports all input types currently.
#define DEFINE_SUM_UNSORTED_GPU_SPECS_INDEX(T, Index) \
  template struct UnsortedSegmentFunctor<             \
      GPUDevice, T, Index, functor::Zero<T>, functor::AtomicSumOpGpu<T>>;

#define DEFINE_REAL_GPU_SPECS(T)                  \
  DEFINE_REAL_UNSORTED_GPU_SPECS_INDEX(T, int32); \
  DEFINE_REAL_UNSORTED_GPU_SPECS_INDEX(T, int64);

#define DEFINE_SUM_GPU_SPECS(T)                  \
  DEFINE_SUM_UNSORTED_GPU_SPECS_INDEX(T, int32); \
  DEFINE_SUM_UNSORTED_GPU_SPECS_INDEX(T, int64);

TF_CALL_GPU_NUMBER_TYPES(DEFINE_REAL_GPU_SPECS);
TF_CALL_int32(DEFINE_REAL_GPU_SPECS);
TF_CALL_GPU_NUMBER_TYPES(DEFINE_SUM_GPU_SPECS);
TF_CALL_int32(DEFINE_SUM_GPU_SPECS);

// TODO(rocm): support atomicAdd for complex numbers on ROCm
#if GOOGLE_CUDA
TF_CALL_COMPLEX_TYPES(DEFINE_SUM_GPU_SPECS);
#endif

#undef DEFINE_SORTED_GPU_SPECS_INDEX
#undef DEFINE_SORTED_GPU_SPECS
#undef DEFINE_REAL_UNSORTED_GPU_SPECS_INDEX
#undef DEFINE_SUM_UNSORTED_GPU_SPECS_INDEX
#undef DEFINE_REAL_GPU_SPECS
#undef DEFINE_SUM_GPU_SPECS

}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
