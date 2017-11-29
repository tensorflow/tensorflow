/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/scatter_nd_op.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

template <typename T, typename Index, scatter_nd_op::UpdateOp op, int IXDIM>
__global__ void ScatterNdOpKernel(
    const Index* indices, const T* updates, T* out,
    const Eigen::array<Eigen::DenseIndex, IXDIM> output_shape_prefix,
    const Eigen::array<int64, IXDIM> batch_strides, const int64 num_indices,
    const Index slice_size) {
#define ASSIGN(dst, src) (*(dst) = src)

#define OP_OVER_SLICE(op)                                       \
  for (int si = 0; si < slice_size; si++) {                     \
    op(out + i + si, ldg(updates + (index * slice_size + si))); \
  }
  CUDA_1D_KERNEL_LOOP(index, num_indices) {
    Index i = 0;
    bool out_of_bounds = false;
#pragma unroll
    for (int dim = 0; dim < IXDIM; ++dim) {
      int offset = (IXDIM * index + dim);
      const Index ix_d = internal::SubtleMustCopy(ldg(indices + offset));
      out_of_bounds |= !FastBoundsCheck(ix_d, output_shape_prefix[dim]);
      i += ix_d * batch_strides[dim] * slice_size;
    }
    if (!out_of_bounds) {
      switch (op) {
        case scatter_nd_op::UpdateOp::ASSIGN:
#pragma unroll
          OP_OVER_SLICE(ASSIGN);
          break;
        case scatter_nd_op::UpdateOp::ADD:
#pragma unroll
          OP_OVER_SLICE(CudaAtomicAdd);
          break;
        case scatter_nd_op::UpdateOp::SUB:
#pragma unroll
          OP_OVER_SLICE(CudaAtomicSub);
          break;
        case scatter_nd_op::UpdateOp::MUL:
#pragma unroll
          OP_OVER_SLICE(CudaAtomicMul);
          break;
        case scatter_nd_op::UpdateOp::DIV:
#pragma unroll
          OP_OVER_SLICE(CudaAtomicDiv);
          break;
      }
    }
  }
#undef OP_OVER_SLICE
#undef ASSIGN
}

namespace functor {

// Functor used by ScatterOp to do the computations.
template <typename T, typename Index, scatter_nd_op::UpdateOp op, int IXDIM>
struct ScatterNdFunctor<GPUDevice, T, Index, op, IXDIM> {
  Index operator()(
      const GPUDevice& d, const Index slice_size,
      const Eigen::array<Eigen::DenseIndex, IXDIM> output_shape_prefix,
      typename TTypes<T, 2>::Tensor Tparams,
      typename TTypes<Index, 2>::ConstTensor Tindices,
      typename TTypes<T, 2>::ConstTensor Tupdates,
      typename TTypes<T, 2>::Tensor Toutput) {
    const Eigen::DenseIndex batch_size = Tindices.dimension(0);

    // Index batch_strides[IXDIM];
    Eigen::array<int64, IXDIM> batch_strides;
    for (int dim = IXDIM - 1; dim >= 0; --dim) {
      if (dim == IXDIM - 1) {
        batch_strides[dim] = 1;
      } else {
        batch_strides[dim] =
            batch_strides[dim + 1] * output_shape_prefix[dim + 1];
      }
    }

    CudaLaunchConfig config = GetCudaLaunchConfig(Toutput.size(), d);
    // clang-format off
    ScatterNdOpKernel<T, Index, op, IXDIM>
    <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
      Tindices.data(), Tupdates.data(), Toutput.data(), output_shape_prefix,
      batch_strides, batch_size, slice_size);
    // clang-format on

    return -1;
  }
};

}  // namespace functor

#define DECLARE_GPU_SPECS_INDEX_OP_IXDIM(T, Index, op, IXDIM) \
  template struct functor::ScatterNdFunctor<GPUDevice, T, Index, op, IXDIM>;

#define DECLARE_GPU_SPECS_INDEX_OP(T, Index, op)     \
  DECLARE_GPU_SPECS_INDEX_OP_IXDIM(T, Index, op, 1); \
  DECLARE_GPU_SPECS_INDEX_OP_IXDIM(T, Index, op, 2); \
  DECLARE_GPU_SPECS_INDEX_OP_IXDIM(T, Index, op, 3); \
  DECLARE_GPU_SPECS_INDEX_OP_IXDIM(T, Index, op, 4); \
  DECLARE_GPU_SPECS_INDEX_OP_IXDIM(T, Index, op, 5)

#define DECLARE_GPU_SPECS_INDEX(T, Index)                                \
  DECLARE_GPU_SPECS_INDEX_OP(T, Index, scatter_nd_op::UpdateOp::ASSIGN); \
  DECLARE_GPU_SPECS_INDEX_OP(T, Index, scatter_nd_op::UpdateOp::ADD);    \
  DECLARE_GPU_SPECS_INDEX_OP(T, Index, scatter_nd_op::UpdateOp::SUB)

#define DECLARE_GPU_SPECS(T)         \
  DECLARE_GPU_SPECS_INDEX(T, int32); \
  DECLARE_GPU_SPECS_INDEX(T, int64)

TF_CALL_GPU_NUMBER_TYPES_NO_HALF(DECLARE_GPU_SPECS);

#undef DECLARE_GPU_SPECS
#undef DECLARE_GPU_SPECS_INDEX
#undef DECLARE_GPU_SPECS_INDEX_OP

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
