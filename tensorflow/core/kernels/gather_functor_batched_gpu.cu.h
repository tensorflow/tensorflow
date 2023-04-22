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

#ifndef TENSORFLOW_CORE_KERNELS_GATHER_FUNCTOR_BATCHED_GPU_CU_H_
#define TENSORFLOW_CORE_KERNELS_GATHER_FUNCTOR_BATCHED_GPU_CU_H_

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/gather_functor_batched.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

template <typename ValueOrVec, typename Index, bool is_axis_zero,
          bool is_batch_dims_zero>
__global__ void GatherOpKernel(const ValueOrVec* __restrict__ params,
                               const Index* __restrict__ indices,
                               ValueOrVec* __restrict__ out, int64 outer_size,
                               int64 gather_dim_size, int64 indices_size,
                               int64 slice_size, int64 out_size) {
  // params is a tensor of shape
  // [batch_size, outer_size, gather_dim_size, slice_size].
  GPU_1D_KERNEL_LOOP(i, out_size) {
    Index batch_i = 0;  // The batch index into params to use for i.
    Index outer_i = 0;  // The outer index into params to use for i.
    Index indices_i = 0;  // The index into indices to use for i.
    Index slice_i = 0;  // Index into the current slice in params to use for i.

    const Index slices_count = i / slice_size;
    if (is_batch_dims_zero) {
      if (is_axis_zero) {
        indices_i = slices_count;
      } else {
        outer_i = slices_count / indices_size;
        indices_i = slices_count - outer_i * indices_size;
      }
    } else {
      const Index entries_count = slices_count / indices_size;
      if (is_axis_zero) {
        batch_i = entries_count;
      } else {
        batch_i = entries_count / outer_size;
        outer_i = entries_count - batch_i * outer_size;
      }
      indices_i = slices_count - entries_count * indices_size;
    }
    slice_i = i - slices_count * slice_size;

    // Index into the gather axis to use for i.
    Index gather_i = ldg(indices + batch_i * indices_size + indices_i);

    // Check gather_i is in [0, gather_dim_size).
    if (!FastBoundsCheck(gather_i, gather_dim_size)) {
      // Set indices out of range to zero
      // TODO(fpmc): Log an error for transfer back to host.
      out[i] = ValueOrVec(0);
    } else {
      // Read params[batch_i, outer_i, gather_i, slice_i] and write it to the
      // i'th position in out.
      Index params_i = (
          (batch_i * outer_size + outer_i) * gather_dim_size + gather_i
      ) * slice_size + slice_i;
      out[i] = params[params_i];
    }
  }
}

namespace detail {

template <bool is_axis_zero, bool is_batch_dims_zero>
struct LaunchGatherKernelVectorized {
  template <int vec_size>
  struct Impl {
    template <typename T, typename Index>
    Status operator()(const GPUDevice& d, const T* params, const Index* indices,
                      T* out, int64 outer_size, int64 gather_dim_size,
                      int64 indices_size, int64 slice_size, int64 out_size) {
      DCHECK_EQ(slice_size % vec_size, 0);
      DCHECK_EQ(out_size % vec_size, 0);
      DCHECK_EQ(reinterpret_cast<std::uintptr_t>(params) % vec_size, 0);
      DCHECK_EQ(reinterpret_cast<std::uintptr_t>(out) % vec_size, 0);
      int64 out_size_vec = out_size / vec_size;
      int64 slice_size_vec = slice_size / vec_size;
      using Tvec = AlignedVector<T, vec_size>;
      const Tvec* params_vec = reinterpret_cast<const Tvec*>(params);
      Tvec* out_vec = reinterpret_cast<Tvec*>(out);

      GpuLaunchConfig config = GetGpuLaunchConfig(
          out_size_vec, d,
          &GatherOpKernel<Tvec, Index, is_axis_zero, is_batch_dims_zero>,
          /*dynamic_shared_memory_size=*/0, /*block_size_limit=*/0);
      return GpuLaunchKernel(
          GatherOpKernel<Tvec, Index, is_axis_zero, is_batch_dims_zero>,
          config.block_count, config.thread_per_block, 0, d.stream(),
          params_vec, indices, out_vec, outer_size, gather_dim_size,
          indices_size, slice_size_vec, out_size_vec);
    }
  };
};

}  // namespace detail

template <bool is_axis_zero, bool is_batch_dims_zero, typename T,
          typename Index>
Status LaunchGatherKernel(const GPUDevice& d, const T* params,
                          const Index* indices, T* out, int64 outer_size,
                          int64 gather_dim_size, int64 indices_size,
                          int64 slice_size, int64 out_size) {
  // Note that the GPU memory allocator always returns aligned buffers, so the
  // alignment of data pointers is expected to be deterministic.
  // There will be performance cliffs when slice_size is not aligned, but there
  // is no easy way to handle the misalignment because each row will be aligned
  // differently.
  return DispatchToVectorized<
      T, detail::LaunchGatherKernelVectorized<
             is_axis_zero, is_batch_dims_zero>::template Impl>(
      MinAlignmentOf(params, out, slice_size), d, params, indices, out,
      outer_size, gather_dim_size, indices_size, slice_size, out_size);
}

namespace functor {
template <typename T, typename Index>
struct GatherFunctorBatched<GPUDevice, T, Index> {
  int64 operator()(OpKernelContext* ctx,
                   typename TTypes<T, 4>::ConstTensor params,
                   typename TTypes<Index>::ConstFlat indices,
                   typename TTypes<T, 4>::Tensor out) {
    const GPUDevice& d = ctx->eigen_gpu_device();
    const int64 out_size = out.size();
    if (out_size == 0) {
      // We need a check here since the CPU version does useful error checking
      // work if there are nonempty indices but empty slices, so the kernel is
      // executed in that case.  In the GPU case we don't know how to do error
      // checking, so we skip the loop entirely.
      return -1;
    }
    const bool is_batch_dims_zero = params.dimension(0) == 1;
    const bool is_axis_zero = params.dimension(1) == 1;
    const int64 outer_size = params.dimension(1);
    const int64 gather_dim_size = params.dimension(2);
    const int64 indices_size = indices.size() / params.dimension(0);
    const int64 slice_size = params.dimension(3);

    const auto function =
        is_axis_zero
            ? (is_batch_dims_zero ? LaunchGatherKernel<true, true, T, Index>
                                  : LaunchGatherKernel<true, false, T, Index>)
            : (is_batch_dims_zero ? LaunchGatherKernel<false, true, T, Index>
                                  : LaunchGatherKernel<false, false, T, Index>);
    TF_CHECK_OK(function(d, params.data(), indices.data(), out.data(),
                         outer_size, gather_dim_size, indices_size, slice_size,
                         out_size));
    // TODO(fpmc): enable indices validation on GPU.
    // Right now checking for indices out of bound in the kernel would
    // require copying code between GPU/CPU, and thus slow.
    return -1;
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#endif  // TENSORFLOW_CORE_KERNELS_GATHER_FUNCTOR_BATCHED_GPU_CU_H_
