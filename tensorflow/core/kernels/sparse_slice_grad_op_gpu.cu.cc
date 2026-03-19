/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/gpu_prim.h"
#include "tensorflow/core/kernels/sparse_slice_grad_op.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/gpu_device_functions.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace {

// Helper that wraps a multi-dimensional index and provides a comparison
// operator that shifts and then compares against another index. This is
// used for searching for an input index within the output indices.
struct MultiIndexComparator {
  EIGEN_DEVICE_FUNC MultiIndexComparator(int rank, const int64_t* indices,
                                         const int64_t* input_start)
      : rank_(rank), indices_(indices), input_start_(input_start) {}

  // Note: gpu_helper::lower_bound always compares `haystack < needle`,
  // so the argument here is always the needle (the input index) and *this
  // is the haystack (the output index).
  EIGEN_DEVICE_FUNC bool operator<(const MultiIndexComparator& input) const {
    for (int d = 0; d < rank_; ++d) {
      // Shift output index by the slice start.
      int64_t output_index_i = indices_[d] + input_start_[d];
      int64_t input_index_i = input.indices_[d];
      // Lexicographically compare the indexes.
      if (output_index_i < input_index_i) return true;
      if (output_index_i > input_index_i) return false;
    }
    return false;
  }

  EIGEN_DEVICE_FUNC bool operator==(const MultiIndexComparator& input) const {
    for (int d = 0; d < rank_; ++d) {
      // Shift output index by the slice start.
      int64_t output_index_i = indices_[d] + input_start_[d];
      int64_t input_index_i = input.indices_[d];
      if (output_index_i != input_index_i) return false;
    }
    return true;
  }

 private:
  int rank_;
  const int64_t* __restrict__ indices_;
  const int64_t* __restrict__ input_start_;
};

struct MultiIndexSearchFunctor {
  MultiIndexSearchFunctor(int rank, const int64_t* indices,
                          const int64_t* input_start)
      : rank_(rank), indices_(indices), input_start_(input_start) {}

  EIGEN_DEVICE_FUNC MultiIndexComparator operator()(int64_t i) const {
    return {rank_, indices_ + i * rank_, input_start_};
  }

 private:
  int rank_;
  const int64_t* indices_;
  const int64_t* input_start_;
};

using IndexIterator = gpuprim::CountingInputIterator<int64_t, int64_t>;
using MultiIndexSearchIterator =
    gpuprim::TransformInputIterator<MultiIndexComparator,
                                    MultiIndexSearchFunctor, IndexIterator>;

template <typename T>
__global__ void SparseSliceGradKernel(int64_t input_nnz, int64_t output_nnz,
                                      MultiIndexSearchIterator input_indices,
                                      MultiIndexSearchIterator output_indices,
                                      const T* backprop_val_grad, T* val_grad) {
  for (int64 input_nz : GpuGridRangeX<int64>(input_nnz)) {
    // Search for the input index in the output indices.
    // Note: It would be faster to first directly test if the input index is
    // within the slice volume (and also to use flattened indexes), but
    // unfortunately we don't have the dense shapes so we can't do that.
    const int64 output_nz = gpu_helper::lower_bound(output_indices, output_nnz,
                                                    input_indices[input_nz]);
    if (output_nz < output_nnz &&
        output_indices[output_nz] == input_indices[input_nz]) {
      // Found the input index in the output, so copy its gradient value.
      val_grad[input_nz] = backprop_val_grad[output_nz];
    } else {
      // Not found, meaning it was not within the slice volume.
      val_grad[input_nz] = T(0);
    }
  }
}

}  // namespace

namespace functor {

template <typename T>
struct SparseSliceGradFunctor<GPUDevice, T> {
  void operator()(OpKernelContext* ctx,
                  typename TTypes<T>::ConstFlat backprop_val_grad,
                  typename TTypes<int64_t>::ConstMatrix input_indices_mat,
                  typename TTypes<int64_t>::ConstFlat input_start_flat,
                  typename TTypes<int64_t>::ConstMatrix output_indices_mat,
                  typename TTypes<T>::Flat val_grad) const {
    const int rank = input_indices_mat.dimension(1);

    MultiIndexSearchIterator input_indices_iter(
        IndexIterator(int64_t(0)),
        MultiIndexSearchFunctor(rank, input_indices_mat.data(),
                                /*input_start=*/nullptr));
    MultiIndexSearchIterator output_indices_iter(
        IndexIterator(int64_t(0)),
        MultiIndexSearchFunctor(rank, output_indices_mat.data(),
                                input_start_flat.data()));

    const int64_t input_nnz = input_indices_mat.dimension(0);
    const int64_t output_nnz = output_indices_mat.dimension(0);

    const GPUDevice& device = ctx->eigen_gpu_device();

    GpuLaunchConfig config =
        GetGpuLaunchConfig(input_nnz, device, &SparseSliceGradKernel<T>,
                           /*dynamic_shared_memory_size=*/0,
                           /*block_size_limit=*/0);
    OP_REQUIRES_OK(
        ctx,
        GpuLaunchKernel(SparseSliceGradKernel<T>, config.block_count,
                        config.thread_per_block, 0, device.stream(), input_nnz,
                        output_nnz, input_indices_iter, output_indices_iter,
                        backprop_val_grad.data(), val_grad.data()));
  }
};

}  // namespace functor

#define DEFINE_SPARSE_SLICE_GRAD(T) \
  template struct functor::SparseSliceGradFunctor<GPUDevice, T>;
TF_CALL_NUMBER_TYPES(DEFINE_SPARSE_SLICE_GRAD);
#undef DEFINE_SPARSE_SLICE_GRAD

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
