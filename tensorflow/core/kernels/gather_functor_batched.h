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

#ifndef TENSORFLOW_CORE_KERNELS_GATHER_FUNCTOR_BATCHED_H_
#define TENSORFLOW_CORE_KERNELS_GATHER_FUNCTOR_BATCHED_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/platform/prefetch.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

// Helper method to copy using memcpy.
template <typename T, typename Index, typename SliceIndex,
          SliceIndex static_slice_elems>
SliceIndex HandleCopiesBatched(OpKernelContext* ctx,
                               typename TTypes<T, 4>::ConstTensor params,
                               typename TTypes<Index>::ConstFlat indices,
                               SliceIndex slice_elems,
                               typename TTypes<T, 4>::Tensor out) {
  const SliceIndex batch_size = static_cast<SliceIndex>(params.dimension(0));
  const SliceIndex outer_size = static_cast<SliceIndex>(params.dimension(1));
  const SliceIndex indices_size =
      static_cast<SliceIndex>(indices.dimension(0)) / batch_size;

  const Index limit = static_cast<Index>(params.dimension(2));
  if (static_slice_elems >= 0) {
    // Give compiler static knowledge of the number of elements/bytes
    slice_elems = static_slice_elems;
  }
  // Compute slice_bytes here so that static knowledge is available
  const size_t slice_bytes = slice_elems * sizeof(T);
  auto* worker_threads = ctx->device()->tensorflow_cpu_worker_threads();
  mutex mu;
  // Store the value of invalidate index for printing error information, it's a
  // shared variable.
  SliceIndex result = -1;
  auto work = [&](int64 start, int64 end) {
    const int64 r_start = start % (outer_size * indices_size);
    SliceIndex batch_idx = static_cast<SliceIndex>(
        start / (outer_size * indices_size));
    SliceIndex outer_idx = static_cast<SliceIndex>(r_start / indices_size);
    SliceIndex indices_idx = static_cast<SliceIndex>(r_start % indices_size);

    SliceIndex batch_offset = batch_idx * indices_size;
    for (; start < end; ++start) {
      SliceIndex i_next = indices_idx + 1;
      SliceIndex o_next = outer_idx;
      SliceIndex b_next = batch_idx;
      SliceIndex b_offset_next = batch_offset;

      if (i_next >= indices_size) {
        i_next = 0;
        if (++o_next >= outer_size) {
          o_next = 0;
          ++b_next;
          b_offset_next += indices_size;
        }
      }
      if (start + 1 < end) {
        port::prefetch<port::PREFETCH_HINT_T0>(
            &params(b_next, o_next, indices(b_offset_next + i_next), 0));
        port::prefetch<port::PREFETCH_HINT_T0>(&out(b_next, o_next, i_next, 0));
      }
      const Index index = internal::SubtleMustCopy(
          indices(batch_offset + indices_idx));
      if (!FastBoundsCheck(index, limit)) {
        mutex_lock l(mu);
        result = batch_offset + indices_idx;
        return;
      }

      // Copy using memcpy if possible, otherwise an Eigen loop
      // TODO(cwhipkey): avoid linking to framework to get Allocator (to improve
      // ahead-of-time compilation binary size).
      if (is_simple_type<T>::value) {
        // Avoid auto-promotion to Index from SliceIndex by casting.
        memcpy(
            &out(batch_idx, outer_idx, indices_idx, 0),
            &params(batch_idx, outer_idx, static_cast<SliceIndex>(index), 0),
            slice_bytes);
      } else {
        // For non-"simple" types (e.g. strings).
        out.template chip<0>(batch_idx)
            .template chip<0>(outer_idx)
            .template chip<0>(indices_idx) =
            params.template chip<0>(batch_idx)
                .template chip<0>(outer_idx)
                .template chip<0>(static_cast<SliceIndex>(index));
      }

      indices_idx = i_next;
      outer_idx = o_next;
      batch_idx = b_next;
      batch_offset = b_offset_next;
    }
  };

  Shard(worker_threads->num_threads, worker_threads->workers,
        batch_size * outer_size * indices_size, slice_elems * sizeof(T), work);
  return result;
}

template <typename T, typename Index>
struct GatherFunctorBatchedCPU {
  int64 operator()(OpKernelContext* ctx,
                   typename TTypes<T, 4>::ConstTensor params,
                   typename TTypes<Index>::ConstFlat indices,
                   typename TTypes<T, 4>::Tensor out) {
    const int64 indices_size = indices.size();  // Includes the batch_size.
    const int64 slice_size = out.dimension(3);
    int64 bad_i;

    const int64 batch_size = params.dimension(0);
    const int64 outer_size = params.dimension(1);

    bool use_large = (slice_size > std::numeric_limits<int32>::max() ||
                      params.size() > std::numeric_limits<int32>::max() ||
                      indices_size > std::numeric_limits<int32>::max() ||
                      batch_size * outer_size * indices_size * slice_size >
                          std::numeric_limits<int32>::max());
#define CALL(elems)                                                      \
  do {                                                                   \
    if (use_large) {                                                     \
      bad_i = HandleCopiesBatched<T, Index, int64, elems>(               \
          ctx, params, indices, slice_size, out);                        \
    } else {                                                             \
      const int32 small_slice = static_cast<int32>(slice_size);          \
      bad_i = HandleCopiesBatched<T, Index, int32, elems>(               \
          ctx, params, indices, small_slice, out);                       \
    }                                                                    \
  } while (0)

    // TODO(rmlarsen): Investigate whether these specializations are still
    // needed and, if yes, whether the slice sizes are apropriate.
    if (slice_size == 10)
      CALL(10);
    else if (slice_size == 20)
      CALL(20);
    else
      CALL(-1);
#undef CALL

    return bad_i;
  }
};

template <typename Device, typename T, typename Index>
struct GatherFunctorBatched {
  int64 operator()(OpKernelContext* ctx,
                   typename TTypes<T, 4>::ConstTensor params,
                   typename TTypes<Index>::ConstFlat indices,
                   typename TTypes<T, 4>::Tensor out);
};

template <typename T, typename Index>
struct GatherFunctorBatched<CPUDevice, T, Index> {
  int64 operator()(OpKernelContext* ctx,
                   typename TTypes<T, 4>::ConstTensor params,
                   typename TTypes<Index>::ConstFlat indices,
                   typename TTypes<T, 4>::Tensor out) {
    return GatherFunctorBatchedCPU<T, Index>()(ctx, params, indices, out);
  }
};

template <typename Index>
struct GatherFunctorBatched<GPUDevice, Variant, Index> {
  int64 operator()(OpKernelContext* ctx,
                   typename TTypes<Variant, 4>::ConstTensor params,
                   typename TTypes<Index>::ConstFlat indices,
                   typename TTypes<Variant, 4>::Tensor out) {
    return GatherFunctorBatchedCPU<Variant, Index>()(ctx, params, indices, out);
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_GATHER_FUNCTOR_BATCHED_H_
