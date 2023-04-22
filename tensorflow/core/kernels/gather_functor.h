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

#ifndef TENSORFLOW_CORE_KERNELS_GATHER_FUNCTOR_H_
#define TENSORFLOW_CORE_KERNELS_GATHER_FUNCTOR_H_

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
SliceIndex HandleCopies(OpKernelContext* ctx,
                        typename TTypes<T, 3>::ConstTensor params,
                        typename TTypes<Index>::ConstFlat indices,
                        SliceIndex slice_elems,
                        typename TTypes<T, 3>::Tensor out) {
  const SliceIndex indices_size = static_cast<SliceIndex>(indices.dimension(0));
  const SliceIndex batch_size = static_cast<SliceIndex>(params.dimension(0));
  const Index limit = static_cast<Index>(params.dimension(1));
  T* out_base = out.data();
  const T* params_base = params.data();
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
    SliceIndex batch_idx = static_cast<SliceIndex>(start / indices_size);
    SliceIndex indices_idx = static_cast<SliceIndex>(start % indices_size);
    SliceIndex batch_idx_end = static_cast<SliceIndex>(end / indices_size);
    SliceIndex indices_idx_end = static_cast<SliceIndex>(end % indices_size);

    while ((batch_idx < batch_idx_end) ||
           (batch_idx == batch_idx_end && indices_idx < indices_idx_end)) {
      SliceIndex i_next = indices_idx + 1;
      SliceIndex b_next = batch_idx + 1;
      const Index index = internal::SubtleMustCopy(indices(indices_idx));
      if (!FastBoundsCheck(index, limit)) {
        mutex_lock l(mu);
        result = indices_idx;
        return;
      }
      if ((batch_idx == batch_idx_end && i_next < indices_idx_end) ||
          (i_next < indices_size)) {
        port::prefetch<port::PREFETCH_HINT_T0>(
            &params(batch_idx, indices(i_next), 0));
        port::prefetch<port::PREFETCH_HINT_T0>(&out(batch_idx, i_next, 0));
        b_next = batch_idx;
      } else if (b_next <= batch_idx_end) {
        port::prefetch<port::PREFETCH_HINT_T0>(&params(b_next, indices(0), 0));
        port::prefetch<port::PREFETCH_HINT_T0>(&out(b_next, 0, 0));
        i_next = 0;
      }
      // Copy using memcpy if possible, otherwise an Eigen loop
      // TODO(cwhipkey): avoid linking to framework to get Allocator (to improve
      // ahead-of-time compilation binary size).
      if (is_simple_type<T>::value) {
        // Avoid auto-promotion to Index from SliceIndex by casting.
        memcpy(
            out_base + (batch_idx * indices_size + indices_idx) * slice_elems,
            params_base + (batch_idx * static_cast<SliceIndex>(limit) +
                           static_cast<SliceIndex>(index)) *
                              slice_elems,
            slice_bytes);
      } else {
        // For non-"simple" types (e.g. strings).
        out.template chip<0>(batch_idx).template chip<0>(indices_idx) =
            params.template chip<0>(batch_idx).template chip<0>(index);
      }
      indices_idx = i_next;
      batch_idx = b_next;
    }
  };

  Shard(worker_threads->num_threads, worker_threads->workers,
        batch_size * indices_size, slice_elems * sizeof(T), work);
  return result;
}

template <typename T, typename Index>
struct GatherFunctorCPU {
  int64 operator()(OpKernelContext* ctx,
                   typename TTypes<T, 3>::ConstTensor params,
                   typename TTypes<Index>::ConstFlat indices,
                   typename TTypes<T, 3>::Tensor out) {
    const int64 indices_size = indices.size();
    const int64 slice_size = out.dimension(2);
    int64 bad_i;

    const int64 batch_size = params.dimension(0);

    bool use_large = (slice_size > std::numeric_limits<int32>::max() ||
                      params.size() > std::numeric_limits<int32>::max() ||
                      indices_size > std::numeric_limits<int32>::max() ||
                      batch_size * indices_size * slice_size >
                          std::numeric_limits<int32>::max());
#define CALL(elems)                                                      \
  do {                                                                   \
    if (use_large) {                                                     \
      bad_i = HandleCopies<T, Index, int64, elems>(ctx, params, indices, \
                                                   slice_size, out);     \
    } else {                                                             \
      const int32 small_slice = static_cast<int32>(slice_size);          \
      bad_i = HandleCopies<T, Index, int32, elems>(ctx, params, indices, \
                                                   small_slice, out);    \
    }                                                                    \
  } while (0)

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
struct GatherFunctor {
  int64 operator()(OpKernelContext* ctx,
                   typename TTypes<T, 3>::ConstTensor params,
                   typename TTypes<Index>::ConstFlat indices,
                   typename TTypes<T, 3>::Tensor out);
};

template <typename T, typename Index>
struct GatherFunctor<CPUDevice, T, Index> {
  int64 operator()(OpKernelContext* ctx,
                   typename TTypes<T, 3>::ConstTensor params,
                   typename TTypes<Index>::ConstFlat indices,
                   typename TTypes<T, 3>::Tensor out) {
    return GatherFunctorCPU<T, Index>()(ctx, params, indices, out);
  }
};

template <typename Index>
struct GatherFunctor<GPUDevice, Variant, Index> {
  int64 operator()(OpKernelContext* ctx,
                   typename TTypes<Variant, 3>::ConstTensor params,
                   typename TTypes<Index>::ConstFlat indices,
                   typename TTypes<Variant, 3>::Tensor out) {
    return GatherFunctorCPU<Variant, Index>()(ctx, params, indices, out);
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_GATHER_FUNCTOR_H_
