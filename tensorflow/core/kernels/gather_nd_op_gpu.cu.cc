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

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/gather_nd_op.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

template <typename T, typename Index, int IXDIM>
__global__ void GatherSliceOpKernel(
    const T* params, const Index* indices, T* out,
    const Eigen::array<int64, IXDIM> batch_strides,
    const Eigen::array<int64, IXDIM> batch_indices, const int64 indices_size,
    const int64 slice_size, const int64 out_size) {
  // TODO(ebrevdo): reduce inner loop into two loops:
  // one over the number of locs, and one over the offsets inside the locs.
  CUDA_1D_KERNEL_LOOP(i, out_size) {
    const Index loc = i / slice_size;
    const auto indices_i = indices + IXDIM * loc;
    bool out_of_bounds = false;
    Index offset = 0;
#pragma unroll
    for (int j = 0; j < IXDIM; ++j) {
      const Index index_j = ldg(indices_i + j);
      out_of_bounds |= !FastBoundsCheck(index_j, batch_indices[j]);
      offset += batch_strides[j] * index_j;
    }
    // TODO(ebrevdo):
    // This is the only part that depends on the offset.  The part
    // above does not need to be executed for every index i.
    // Is there a way to break the outer loop into two loops?  One
    // that determines how many slice_size-length locs are iterated
    // over, and another that iterates over slice_size iterations for
    // the correct indices?
    const Index loc_offset = i - loc * slice_size;
    out[i] = (out_of_bounds) ? T(0) : ldg(params + offset + loc_offset);
  }
}

namespace functor {

template <typename T, typename Index, int IXDIM>
struct GatherNdSlice<GPUDevice, T, Index, IXDIM> {
  Index operator()(const GPUDevice& d, const Index unused_slice_size,
                   typename TTypes<int32>::Scalar Tscratch,
                   typename TTypes<T, IXDIM + 1>::ConstTensor Tparams,
                   typename TTypes<Index>::ConstMatrix Tindices,
                   typename TTypes<T>::Matrix Tout) {
    const int64 indices_size = Tindices.dimension(1);
    const int64 out_size = Tout.size();
    int64 s_size = Tout.dimension(1);
    Eigen::array<int64, IXDIM> batch_strides;
    Eigen::array<int64, IXDIM> batch_indices;
    if (IXDIM > 0) {
      batch_strides[IXDIM - 1] = s_size;
      batch_indices[IXDIM - 1] = Tparams.dimension(IXDIM - 1);
    }
    for (int i = IXDIM - 1; i > 0; --i) {
      batch_indices[i - 1] = Tparams.dimension(i - 1);
      batch_strides[i - 1] = batch_strides[i] * Tparams.dimension(i);
    }
    CudaLaunchConfig config = GetCudaLaunchConfig(out_size, d);

    // clang-format off
    GatherSliceOpKernel<T, Index, IXDIM>
        <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
            Tparams.data(), Tindices.data(), Tout.data(), batch_strides,
            batch_indices, indices_size, s_size, out_size);
    // clang-format on

    // TODO(ebrevdo): enable indices validation on GPU.
    // Right now checking for indices out of bound in the kernel would
    // require copying code between GPU/CPU, and is too slow.
    return -1;
  }
};

}  // namespace functor

#define DEFINE_GPU_SPECS_INDEX_NDIM(T, Index, NDIM) \
  template struct functor::GatherNdSlice<GPUDevice, T, Index, NDIM>;

#define DEFINE_GPU_SPECS_INDEX(T, Index)    \
  DEFINE_GPU_SPECS_INDEX_NDIM(T, Index, 0); \
  DEFINE_GPU_SPECS_INDEX_NDIM(T, Index, 1); \
  DEFINE_GPU_SPECS_INDEX_NDIM(T, Index, 2); \
  DEFINE_GPU_SPECS_INDEX_NDIM(T, Index, 3); \
  DEFINE_GPU_SPECS_INDEX_NDIM(T, Index, 4); \
  DEFINE_GPU_SPECS_INDEX_NDIM(T, Index, 5);

#define DEFINE_GPU_SPECS(T)         \
  DEFINE_GPU_SPECS_INDEX(T, int32); \
  DEFINE_GPU_SPECS_INDEX(T, int64);

TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_SPECS);

#undef DEFINE_GPU_SPECS
#undef DEFINE_GPU_SPECS_INDEX

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
