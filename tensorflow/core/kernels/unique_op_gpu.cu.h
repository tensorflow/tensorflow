/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_UNIQUE_OP_GPU_CU_H_
#define TENSORFLOW_CORE_KERNELS_UNIQUE_OP_GPU_CU_H_

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/gpu_prim.h"
#include "tensorflow/core/kernels/gpu_prim_helpers.h"
#include "tensorflow/core/lib/core/bits.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_solvers.h"  // For ScratchSpace

#if GOOGLE_CUDA
#include "xla/stream_executor/gpu/scoped_activate_context.h"
#elif TENSORFLOW_USE_ROCM
#include "tensorflow/core/platform/rocm.h"
#endif

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace unique_op_gpu {

// Returns true iff index is at the end of a segment (which is equivalent to the
// beginning of the next segment).
template <typename T, typename TIndex>
struct SegmentIndicatorFunctor {
  const T* __restrict__ sorted_input_ptr_;
  SegmentIndicatorFunctor(const T* sorted_input_ptr)
      : sorted_input_ptr_(sorted_input_ptr) {}
  __device__ bool operator()(const TIndex& i) const {
    return i > 0 && sorted_input_ptr_[i] != sorted_input_ptr_[i - 1];
  }
};

template <typename TIndex>
__global__ void ExtractFirstOccurrenceIndicesKernel(
    int64_t input_size, int64_t uniq_size,
    const TIndex* __restrict__ sorted_input_inds,
    const TIndex* __restrict__ sorted_input_unique_ids,
    TIndex* __restrict__ unique_input_inds, TIndex* __restrict__ segment_ends) {
  GPU_1D_KERNEL_LOOP(i, input_size) {
    TIndex sorted_input_unique_id = sorted_input_unique_ids[i];
    if (i == 0 || sorted_input_unique_id != sorted_input_unique_ids[i - 1]) {
      unique_input_inds[sorted_input_unique_id] = sorted_input_inds[i];
      if (segment_ends) {
        if (i == 0) {
          // First thread writes the last element.
          segment_ends[uniq_size - 1] = input_size;
        } else {
          segment_ends[sorted_input_unique_id - 1] = i;
        }
      }
    }
  }
}

// Scatters the index of the first occurrence of each unique input value to
// unique_input_inds.
// If segment_ends is not nullptr, it is filled with the end index of each
// unique value's range in the sorted input (the last element is always set
// to input_size).
template <typename TIndex>
Status ExtractFirstOccurrenceIndices(const GPUDevice& d, int64_t input_size,
                                     int64_t uniq_size,
                                     const TIndex* sorted_input_inds,
                                     const TIndex* sorted_input_unique_ids,
                                     TIndex* unique_input_inds,
                                     TIndex* segment_ends) {
  CHECK_GT(input_size, 0);  // Crash OK
  GpuLaunchConfig config = GetGpuLaunchConfig(
      input_size, d, &ExtractFirstOccurrenceIndicesKernel<TIndex>,
      /*dynamic_shared_memory_size=*/0, /*block_size_limit=*/0);
  return GpuLaunchKernel(ExtractFirstOccurrenceIndicesKernel<TIndex>,
                         config.block_count, config.thread_per_block, 0,
                         d.stream(), input_size, uniq_size, sorted_input_inds,
                         sorted_input_unique_ids, unique_input_inds,
                         segment_ends);
}

template <typename T, typename TIndex>
__global__ void GatherOutputsAndInvertPermutationKernel(
    int64_t uniq_size, const T* __restrict__ input,
    const TIndex* __restrict__ sorted_unique_input_inds,
    const TIndex* __restrict__ sorted_unique_perm,
    const TIndex* __restrict__ segment_ends, T* __restrict__ output,
    TIndex* __restrict__ inv_sorted_unique_perm, TIndex* __restrict__ count) {
  GPU_1D_KERNEL_LOOP(i, uniq_size) {
    output[i] = input[sorted_unique_input_inds[i]];
    auto j = sorted_unique_perm[i];
    inv_sorted_unique_perm[j] = i;
    if (count) {
      TIndex beg = j == 0 ? 0 : segment_ends[j - 1];
      TIndex end = segment_ends[j];
      count[i] = end - beg;
    }
  }
}

// Gathers input values using sorted_unique_input_inds, and inverts the
// permutation specified by sorted_unique_perm.
template <typename T, typename TIndex>
Status GatherOutputsAndInvertPermutation(const GPUDevice& d, int64_t uniq_size,
                                         const T* input,
                                         const TIndex* sorted_unique_input_inds,
                                         const TIndex* sorted_unique_perm,
                                         const TIndex* segment_ends, T* output,
                                         TIndex* inv_sorted_unique_perm,
                                         TIndex* count) {
  if (uniq_size == 0) return OkStatus();
  GpuLaunchConfig config = GetGpuLaunchConfig(
      uniq_size, d, &GatherOutputsAndInvertPermutationKernel<T, TIndex>,
      /*dynamic_shared_memory_size=*/0, /*block_size_limit=*/0);
  return GpuLaunchKernel(GatherOutputsAndInvertPermutationKernel<T, TIndex>,
                         config.block_count, config.thread_per_block, 0,
                         d.stream(), uniq_size, input, sorted_unique_input_inds,
                         sorted_unique_perm, segment_ends, output,
                         inv_sorted_unique_perm, count);
}

template <typename TIndex>
__global__ void LookupAndScatterUniqueIdsKernel(
    int64_t input_size, const TIndex* sorted_input_inds,
    const TIndex* __restrict__ sorted_input_unique_ids,
    const TIndex* __restrict__ inv_sorted_unique_perm,
    TIndex* __restrict__ idx) {
  GPU_1D_KERNEL_LOOP(i, input_size) {
    idx[sorted_input_inds[i]] =
        inv_sorted_unique_perm[sorted_input_unique_ids[i]];
  }
}

// Maps the values of sorted_input_unique_ids and scatters them to idx using
// sorted_input_inds.
template <typename TIndex>
Status LookupAndScatterUniqueIds(const GPUDevice& d, int64_t input_size,
                                 const TIndex* sorted_input_inds,
                                 const TIndex* sorted_input_unique_ids,
                                 const TIndex* inv_sorted_unique_perm,
                                 TIndex* idx) {
  CHECK_GT(input_size, 0);  // Crash OK
  GpuLaunchConfig config = GetGpuLaunchConfig(
      input_size, d, &LookupAndScatterUniqueIdsKernel<TIndex>,
      /*dynamic_shared_memory_size=*/0, /*block_size_limit=*/0);
  return GpuLaunchKernel(LookupAndScatterUniqueIdsKernel<TIndex>,
                         config.block_count, config.thread_per_block, 0,
                         d.stream(), input_size, sorted_input_inds,
                         sorted_input_unique_ids, inv_sorted_unique_perm, idx);
}

}  // namespace unique_op_gpu

// This only supports Unique[WithCounts], not Unique[WithCounts]V2.
template <typename T, typename TIndex>
class UniqueOpGPU : public AsyncOpKernel {
 public:
  explicit UniqueOpGPU(OpKernelConstruction* context)
      : AsyncOpKernel(context) {}

  template <typename U>
  void AllocateTemp(OpKernelContext* context, int64_t size, Tensor* tensor,
                    U** tensor_data, DoneCallback done) const {
    OP_REQUIRES_OK_ASYNC(context,
                         context->allocate_temp(DataTypeToEnum<U>::value,
                                                TensorShape({size}), tensor),
                         done);
    *tensor_data = tensor->flat<U>().data();
  }

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    const Tensor& input = context->input(0);
    // TODO(dga):  Make unique polymorphic for returning int32 and int64
    // vectors to support large tensors.
    OP_REQUIRES_ASYNC(context,
                      input.NumElements() <= std::numeric_limits<int32>::max(),
                      errors::InvalidArgument(
                          "unique does not support input tensors larger than ",
                          std::numeric_limits<int32>::max(), " elements"),
                      done);

    OP_REQUIRES_ASYNC(context, TensorShapeUtils::IsVector(input.shape()),
                      errors::InvalidArgument("unique expects a 1D vector."),
                      done);

    se::Stream* stream = context->op_device_context()->stream();
    OP_REQUIRES_ASYNC(context, stream,
                      errors::Internal("No GPU stream available."), done);

    int64_t input_size = input.NumElements();
    bool has_count_output = num_outputs() > 2;
    if (input_size == 0) {
      // Early exit for trivial case.
      Tensor* t = nullptr;
      OP_REQUIRES_OK_ASYNC(
          context, context->allocate_output(0, TensorShape({0}), &t), done);
      OP_REQUIRES_OK_ASYNC(
          context, context->allocate_output(1, TensorShape({0}), &t), done);
      if (has_count_output) {
        OP_REQUIRES_OK_ASYNC(
            context, context->allocate_output(2, TensorShape({0}), &t), done);
      }
      done();
      return;
    }

    // The algorithm implemented here is as follows:
    // input = [3, 5, 3, 4, 1, 4, 9, 8, 6, 3, 5, 7, 8, 8, 4, 6, 4, 2, 5, 6]
    // 1) Sort the input to group equal values together in segments.
    //      sorted_input, sorted_input_inds = sort(input)
    // sorted_input:
    //   [1, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 8, 8, 8, 9]
    // sorted_input_inds:
    //   [4, 17, 0, 2, 9, 3, 5, 14, 16, 1, 10, 18, 8, 15, 19, 11, 7, 12, 13, 6]
    // 2) Identify the boundaries between segments and use prefix sum to
    //    compute the unique ID for each sorted value.
    //      sorted_input_unique_ids = prefix_sum(indicator(sorted_input))
    // indicator(sorted_input):
    //   [0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1]
    // sorted_input_unique_ids:
    //   [0, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 7, 7, 7, 8]
    // 3) Extract the input index of the first occurrence of each unique value.
    //    If counts are required, also extract the end index of each segment.
    //      unique_input_inds[sorted_input_unique_ids] =
    //          sorted_input_inds (@ indicator)
    //      segment_ends[sorted_input_unique_ids[i] - 1] = i (@ indicator)
    // unique_input_inds: [4, 17, 0, 3, 1, 8, 11, 7, 6]
    // segment_ends: [1, 2, 5, 9, 12, 15, 16, 19, 20]
    // 4) Sort the extracted unique input indices to put them in order of
    //    first appearance.
    //      sorted_unique_input_inds, sorted_unique_perm =
    //          sort(unique_input_inds)
    // sorted_unique_input_inds: [0, 1, 3, 4, 6, 7, 8, 11, 17]
    // sorted_unique_perm: [2, 4, 3, 0, 8, 7, 5, 6, 1]
    // 5) Gather the sorted unique input values to produce output, and invert
    //    the second sort permutation to produce an inverse ID mapping. If
    //    counts are required, also take the adjacent difference between
    //    segment_ends indices to produce counts.
    //      output = input[sorted_unique_input_inds]
    //      inv_sorted_unique_perm[sorted_unique_perm[i]] = i
    //      counts = adjacent_difference(segment_ends)
    // output: [3, 5, 4, 1, 9, 8, 6, 7, 2]
    // inv_sorted_unique_perm: [3, 8, 0, 2, 1, 6, 7, 5, 4]
    // counts: [3, 3, 4, 1, 1, 3, 3, 1, 1]
    // 6) Look up unique IDs via the inverse ID mapping and scatter them using
    //    the original sort permutation to produce the indices output.
    //      idx[sorted_input_inds] =
    //          inv_sorted_unique_perm[sorted_input_unique_ids]
    // idx: [0, 1, 0, 2, 3, 2, 4, 5, 6, 0, 1, 7, 5, 5, 2, 6, 2, 8, 1, 6]

    Tensor sorted_input_inds;
    TIndex* sorted_input_inds_ptr = nullptr;
    AllocateTemp(context, input_size, &sorted_input_inds,
                 &sorted_input_inds_ptr, done);
    if (!context->status().ok()) return;

    Tensor sorted_input;
    T* sorted_input_ptr = nullptr;
    AllocateTemp(context, input_size, &sorted_input, &sorted_input_ptr, done);
    if (!context->status().ok()) return;

    const T* input_ptr = input.flat<T>().data();
    OP_REQUIRES_OK_ASYNC(
        context,
        GpuRadixSort(context, input_size, /*keys_in=*/input_ptr,
                     /*keys_out=*/sorted_input_ptr,
                     /*indices_in=*/static_cast<const TIndex*>(nullptr),
                     /*indices_out=*/sorted_input_inds_ptr),
        done);

    using namespace unique_op_gpu;

    // Create a fancy input iterator to indicate segment boundaries.
    gpuprim::CountingInputIterator<TIndex> counting_iter(0);
    gpuprim::TransformInputIterator<TIndex, SegmentIndicatorFunctor<T, TIndex>,
                                    gpuprim::CountingInputIterator<TIndex>>
        segment_indicator_iter(counting_iter, {sorted_input_ptr});

    Tensor sorted_input_unique_ids;
    TIndex* sorted_input_unique_ids_ptr = nullptr;
    AllocateTemp(context, input_size, &sorted_input_unique_ids,
                 &sorted_input_unique_ids_ptr, done);
    if (!context->status().ok()) return;

    OP_REQUIRES_OK_ASYNC(
        context,
        GpuInclusivePrefixSum(context, input_size, segment_indicator_iter,
                              sorted_input_unique_ids_ptr),
        done);

    // Copy the last element of sorted_input_unique_ids back to the host to
    // obtain uniq_size.
    ScratchSpace<TIndex> last_idx_host(context, 1, /*on_host=*/true);
    OP_REQUIRES_OK_ASYNC(
        context,
        stream->Memcpy(last_idx_host.mutable_data(),
                       se::DeviceMemoryBase(
                           const_cast<TIndex*>(sorted_input_unique_ids_ptr) +
                               (input_size - 1),
                           sizeof(*last_idx_host.data())),
                       sizeof(*last_idx_host.data())),
        done);

    auto async_finish_computation = [this, context, input_size, input_ptr,
                                     sorted_input_inds, sorted_input_inds_ptr,
                                     sorted_input_unique_ids,
                                     sorted_input_unique_ids_ptr, last_idx_host,
                                     has_count_output, done]() -> void {
      const GPUDevice& device = context->eigen_gpu_device();
      int64 uniq_size = (*last_idx_host.data()) + 1;

      se::gpu::ScopedActivateContext scoped_activation{
          context->op_device_context()->stream()->parent()};

      Tensor unique_input_inds;
      TIndex* unique_input_inds_ptr = nullptr;
      AllocateTemp(context, uniq_size, &unique_input_inds,
                   &unique_input_inds_ptr, done);
      if (!context->status().ok()) return;

      Tensor segment_ends;
      TIndex* segment_ends_ptr = nullptr;
      if (has_count_output) {
        AllocateTemp(context, uniq_size, &segment_ends, &segment_ends_ptr,
                     done);
        if (!context->status().ok()) return;
      }

      OP_REQUIRES_OK_ASYNC(
          context,
          ExtractFirstOccurrenceIndices(
              device, input_size, uniq_size, sorted_input_inds_ptr,
              sorted_input_unique_ids_ptr, unique_input_inds_ptr,
              segment_ends_ptr),
          done);

      Tensor sorted_unique_input_inds;
      TIndex* sorted_unique_input_inds_ptr = nullptr;
      AllocateTemp(context, uniq_size, &sorted_unique_input_inds,
                   &sorted_unique_input_inds_ptr, done);
      if (!context->status().ok()) return;

      Tensor sorted_unique_perm;
      TIndex* sorted_unique_perm_ptr = nullptr;
      AllocateTemp(context, uniq_size, &sorted_unique_perm,
                   &sorted_unique_perm_ptr, done);
      if (!context->status().ok()) return;

      // Sort by input index so that output is in order of appearance.
      OP_REQUIRES_OK_ASYNC(
          context,
          GpuRadixSort(context, uniq_size,
                       /*keys_in=*/unique_input_inds_ptr,
                       /*keys_out=*/sorted_unique_input_inds_ptr,
                       /*indices_in=*/static_cast<const TIndex*>(nullptr),
                       /*indices_out=*/sorted_unique_perm_ptr,
                       /*num_bits=*/Log2Ceiling(input_size)),
          done);

      // Free temporary tensor that is no longer needed.
      unique_input_inds = Tensor();
      unique_input_inds_ptr = nullptr;

      Tensor* output = nullptr;
      OP_REQUIRES_OK_ASYNC(
          context,
          context->allocate_output(0, TensorShape({uniq_size}), &output), done);
      T* output_ptr = output->flat<T>().data();

      Tensor inv_sorted_unique_perm;
      TIndex* inv_sorted_unique_perm_ptr = nullptr;
      AllocateTemp(context, uniq_size, &inv_sorted_unique_perm,
                   &inv_sorted_unique_perm_ptr, done);
      if (!context->status().ok()) return;

      TIndex* count_ptr = nullptr;
      if (has_count_output) {
        Tensor* count = nullptr;
        OP_REQUIRES_OK_ASYNC(
            context,
            context->allocate_output(2, TensorShape({uniq_size}), &count),
            done);
        count_ptr = count->flat<TIndex>().data();
      }

      // Compute output and counts (if necessary).
      OP_REQUIRES_OK_ASYNC(
          context,
          GatherOutputsAndInvertPermutation(
              device, uniq_size, input_ptr, sorted_unique_input_inds_ptr,
              sorted_unique_perm_ptr, segment_ends_ptr, output_ptr,
              inv_sorted_unique_perm_ptr, count_ptr),
          done);

      // Free temporary tensors that are no longer needed.
      sorted_unique_perm = Tensor();
      sorted_unique_perm_ptr = nullptr;
      sorted_unique_input_inds = Tensor();
      sorted_unique_input_inds_ptr = nullptr;
      segment_ends = Tensor();
      segment_ends_ptr = nullptr;

      Tensor* idx = nullptr;
      OP_REQUIRES_OK_ASYNC(
          context, context->allocate_output(1, TensorShape({input_size}), &idx),
          done);
      TIndex* idx_ptr = idx->flat<TIndex>().data();

      // Compute indices output.
      OP_REQUIRES_OK_ASYNC(
          context,
          LookupAndScatterUniqueIds(device, input_size, sorted_input_inds_ptr,
                                    sorted_input_unique_ids_ptr,
                                    inv_sorted_unique_perm_ptr, idx_ptr),
          done);

      done();
    };

    context->device()
        ->tensorflow_accelerator_device_info()
        ->event_mgr->ThenExecute(stream, async_finish_computation);
  }
};

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA

#endif  // TENSORFLOW_CORE_KERNELS_UNIQUE_OP_GPU_CU_H_
