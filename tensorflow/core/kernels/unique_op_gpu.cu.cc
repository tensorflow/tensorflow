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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/gpu_prim.h"
#include "tensorflow/core/kernels/gpu_prim_helpers.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#if GOOGLE_CUDA
#include "tensorflow/core/util/cuda_solvers.h"  // For ScratchSpace
#elif TENSORFLOW_USE_ROCM
#include "tensorflow/core/util/rocm_solvers.h"
#endif

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace {

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

// Scatters the unique input values to output and the unique indexes to idx.
// If provided, segment_ends is filled with the end position of each unique
// value's span in the sorted array (the last element is not written as it is
// always equal to input_size; so segment_ends only needs to have space for
// uniq_size - 1 elements).
template <typename T, typename TIndex>
__global__ void ScatterToOutputsKernel(
    int64 input_size, const T* __restrict__ sorted_input,
    const TIndex* __restrict__ sorted_idx,
    const TIndex* __restrict__ sort_permutation, T* __restrict__ output,
    TIndex* __restrict__ idx, TIndex* __restrict__ segment_ends) {
  GPU_1D_KERNEL_LOOP(i, input_size) {
    TIndex sorted_idx_i = sorted_idx[i];
    if (i == 0 || sorted_idx_i != sorted_idx[i - 1]) {
      output[sorted_idx_i] = sorted_input[i];
      if (segment_ends && i > 0) {
        segment_ends[sorted_idx_i - 1] = i;
      }
    }
    idx[sort_permutation[i]] = sorted_idx_i;
  }
}

template <typename T, typename TIndex>
Status ScatterToOutputs(const GPUDevice& d, int64 input_size,
                        const T* sorted_input, const TIndex* sorted_idx,
                        const TIndex* sort_permutation, T* output, TIndex* idx,
                        TIndex* segment_ends) {
  if (input_size == 0) return Status::OK();
  GpuLaunchConfig config = GetGpuLaunchConfig(
      input_size, d, &ScatterToOutputsKernel<T, TIndex>,
      /*dynamic_shared_memory_size=*/0, /*block_size_limit=*/0);
  return GpuLaunchKernel(ScatterToOutputsKernel<T, TIndex>, config.block_count,
                         config.thread_per_block, 0, d.stream(), input_size,
                         sorted_input, sorted_idx, sort_permutation, output,
                         idx, segment_ends);
}

// Computes value counts by taking adjacent differences of segment_ends.
template <typename TIndex>
__global__ void ComputeCountsKernel(int64 uniq_size, int64 input_size,
                                    const TIndex* __restrict__ segment_ends,
                                    TIndex* __restrict__ count) {
  GPU_1D_KERNEL_LOOP(i, uniq_size) {
    TIndex beg = i == 0 ? 0 : segment_ends[i - 1];
    TIndex end = i < uniq_size - 1 ? segment_ends[i] : input_size;
    count[i] = end - beg;
  }
}

template <typename TIndex>
Status ComputeCounts(const GPUDevice& d, int64 uniq_size, int64 input_size,
                     const TIndex* __restrict__ segment_ends,
                     TIndex* __restrict__ count) {
  if (input_size == 0) return Status::OK();
  GpuLaunchConfig config = GetGpuLaunchConfig(
      uniq_size, d, &ComputeCountsKernel<TIndex>,
      /*dynamic_shared_memory_size=*/0, /*block_size_limit=*/0);
  return GpuLaunchKernel(ComputeCountsKernel<TIndex>, config.block_count,
                         config.thread_per_block, 0, d.stream(), uniq_size,
                         input_size, segment_ends, count);
}

}  // namespace

// This only supports Unique[WithCounts], not Unique[WithCounts]V2.
template <typename T, typename TIndex>
class UniqueOpGPU : public AsyncOpKernel {
 public:
  explicit UniqueOpGPU(OpKernelConstruction* context)
      : AsyncOpKernel(context) {}

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
    OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));

    int64 input_size = input.NumElements();

    // The algorithm implemented here is as follows:
    // 1) Sort input to get sorted_input and sort_permutation.
    // 2) Construct an indicator array (0's with a 1 at the end of each segment)
    //    from sorted_input and scan it to produce sorted_idx.
    // 3) Use sorted_idx to scatter each unique value in sorted_input to output,
    //    and use sort_permutation to scatter sorted_idx to idx.
    //    If counts are required, also scatter array indices to segment_ends.
    // 4) If counts are required, take the adjacent difference between values in
    //    segment_ends to produce counts.

    Tensor sorted_input;
    OP_REQUIRES_OK_ASYNC(
        context,
        context->allocate_temp(DataTypeToEnum<T>::value,
                               TensorShape({input_size}), &sorted_input),
        done);
    T* sorted_input_ptr = sorted_input.flat<T>().data();
    Tensor sort_permutation;
    OP_REQUIRES_OK_ASYNC(
        context,
        context->allocate_temp(DataTypeToEnum<TIndex>::value,
                               TensorShape({input_size}), &sort_permutation),
        done);
    TIndex* sort_permutation_ptr = sort_permutation.flat<TIndex>().data();
    const T* input_ptr = input.flat<T>().data();
    OP_REQUIRES_OK_ASYNC(
        context,
        (GpuRadixSort(context, input_size, input_ptr, sorted_input_ptr,
                      /*indices = */ static_cast<const TIndex*>(nullptr),
                      sort_permutation_ptr)),
        done);

    // Create a fancy input iterator to indicate segment boundaries.
    gpuprim::TransformInputIterator<bool, SegmentIndicatorFunctor<T, TIndex>,
                                    gpuprim::CountingInputIterator<TIndex>>
        segment_indicator_iter(0, {sorted_input_ptr});

    Tensor sorted_idx;
    OP_REQUIRES_OK_ASYNC(
        context,
        context->allocate_temp(DataTypeToEnum<TIndex>::value,
                               TensorShape({input_size}), &sorted_idx),
        done);
    TIndex* sorted_idx_ptr = sorted_idx.flat<TIndex>().data();
    OP_REQUIRES_OK_ASYNC(
        context,
        GpuInclusivePrefixSum(context, input_size, segment_indicator_iter,
                              sorted_idx_ptr),
        done);

    // Copy the last element of sorted_idx back to the host to obtain uniq_size.
    ScratchSpace<TIndex> last_idx_host(context, 1, /* on_host */ true);
    if (input_size > 0) {
      OP_REQUIRES_ASYNC(
          context,
          stream
              ->ThenMemcpy(
                  last_idx_host.mutable_data(),
                  se::DeviceMemoryBase(
                      const_cast<TIndex*>(sorted_idx_ptr) + (input_size - 1),
                      sizeof(*last_idx_host.data())),
                  sizeof(*last_idx_host.data()))
              .ok(),
          errors::Internal("Failed to copy last_idx to host"), done);
    } else {
      *last_idx_host.mutable_data() = -1;
    }

    bool has_count_output = num_outputs() > 2;
    auto async_finish_computation =
        [context, input_size, sorted_input, sorted_idx, sort_permutation,
         last_idx_host, has_count_output, done]() -> void {
      const GPUDevice& device = context->eigen_gpu_device();
      int64 uniq_size = (*last_idx_host.data()) + 1;

      // Allocate output for unique values.
      Tensor* output = nullptr;
      OP_REQUIRES_OK_ASYNC(
          context,
          context->allocate_output(0, TensorShape({uniq_size}), &output), done);
      T* output_ptr = output->flat<T>().data();

      // Allocate output for indices.
      Tensor* idx = nullptr;
      OP_REQUIRES_OK_ASYNC(
          context, context->allocate_output(1, TensorShape({input_size}), &idx),
          done);
      TIndex* idx_ptr = idx->flat<TIndex>().data();

      Tensor* count = nullptr;
      Tensor segment_ends;
      TIndex* segment_ends_ptr = nullptr;
      if (has_count_output) {
        // Allocate temp space and output for counts.
        OP_REQUIRES_OK_ASYNC(
            context,
            context->allocate_output(2, TensorShape({uniq_size}), &count),
            done);
        OP_REQUIRES_OK_ASYNC(
            context,
            context->allocate_temp(
                DataTypeToEnum<TIndex>::value,
                TensorShape({std::max(uniq_size - 1, int64(0))}),
                &segment_ends),
            done);
        segment_ends_ptr = segment_ends.flat<TIndex>().data();
      }

      // Compute output and idx.
      const T* sorted_input_ptr = sorted_input.flat<T>().data();
      const TIndex* sorted_idx_ptr = sorted_idx.flat<TIndex>().data();
      const TIndex* sort_permutation_ptr =
          sort_permutation.flat<TIndex>().data();
      OP_REQUIRES_OK_ASYNC(
          context,
          ScatterToOutputs(device, input_size, sorted_input_ptr, sorted_idx_ptr,
                           sort_permutation_ptr, output_ptr, idx_ptr,
                           segment_ends_ptr),
          done);

      if (has_count_output) {
        TIndex* count_ptr = count->flat<TIndex>().data();
        OP_REQUIRES_OK_ASYNC(context,
                             ComputeCounts(device, uniq_size, input_size,
                                           segment_ends_ptr, count_ptr),
                             done);
      }

      done();
    };

    context->device()->tensorflow_gpu_device_info()->event_mgr->ThenExecute(
        stream, async_finish_computation);
  }
};

#define REGISTER_UNIQUE_GPU(type)                                \
  REGISTER_KERNEL_BUILDER(Name("Unique")                         \
                              .Device(DEVICE_GPU)                \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int32>("out_idx"), \
                          UniqueOpGPU<type, int32>);             \
  REGISTER_KERNEL_BUILDER(Name("Unique")                         \
                              .Device(DEVICE_GPU)                \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int64>("out_idx"), \
                          UniqueOpGPU<type, int64>);             \
  REGISTER_KERNEL_BUILDER(Name("UniqueWithCounts")               \
                              .Device(DEVICE_GPU)                \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int32>("out_idx"), \
                          UniqueOpGPU<type, int32>);             \
  REGISTER_KERNEL_BUILDER(Name("UniqueWithCounts")               \
                              .Device(DEVICE_GPU)                \
                              .TypeConstraint<type>("T")         \
                              .TypeConstraint<int64>("out_idx"), \
                          UniqueOpGPU<type, int64>)

TF_CALL_REAL_NUMBER_TYPES(REGISTER_UNIQUE_GPU);
REGISTER_UNIQUE_GPU(bool);

#undef REGISTER_UNIQUE_GPU

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
