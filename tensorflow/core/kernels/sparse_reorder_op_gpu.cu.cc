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

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/gpu_prim.h"
#include "tensorflow/core/kernels/gpu_prim_helpers.h"
#include "tensorflow/core/kernels/sparse_reorder_op.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

namespace tensorflow {

namespace {

__global__ void IndicesFlattenKernel(const int64* __restrict__ indices,
                                     const int64 nnz,
                                     const int64* __restrict__ dims,
                                     const int64 ndims,
                                     int64* __restrict__ flat_indices) {
  GPU_1D_KERNEL_LOOP(thread_idx, nnz) {
    eigen_assert(ndims >= 1);
    int64 output_idx = indices[thread_idx * ndims + ndims - 1];
    int64 strides = 1;
    for (int i = ndims - 2; i >= 0; i--) {
      strides *= dims[i + 1];
      output_idx += indices[thread_idx * ndims + i] * strides;
    }
    flat_indices[thread_idx] = output_idx;
  }
}

template <typename T>
__global__ void PermuteIndicesAndValuesKernel(
    const int64* __restrict__ indices, const T* __restrict__ values,
    const int64 nnz, const int64 ndims, const int64* __restrict__ permutation,
    int64* reordered_indices, T* reordered_values) {
  GPU_1D_KERNEL_LOOP(thread_idx, nnz) {
    for (int i = 0; i < ndims; i++) {
      reordered_indices[thread_idx * ndims + i] =
          indices[permutation[thread_idx] * ndims + i];
    }
    reordered_values[thread_idx] = values[permutation[thread_idx]];
  }
}

}  // namespace

using GPUDevice = Eigen::GpuDevice;

namespace functor {

template <typename T>
struct SparseReorderFunctor<GPUDevice, T> {
  void operator()(OpKernelContext* c, const Tensor& input_ind,
                  const Tensor& input_val, const Tensor& input_shape_in) {
    const Eigen::GpuDevice& d = c->eigen_gpu_device();

    const int64 num_elems = input_ind.dims() > 0 ? input_ind.dim_size(0) : 1;
    const int64 num_dims = input_ind.dims() > 1 ? input_ind.dim_size(1) : 1;

    auto indices = input_ind.template flat<int64>().data();
    auto values = input_val.template flat<T>().data();
    auto dims = input_shape_in.template flat<int64>().data();

    if (num_elems == 0) {
      c->set_output(0, input_ind);
      c->set_output(1, input_val);
      return;
    }

    Tensor flat_indices_tensor;
    OP_REQUIRES_OK(c, c->allocate_temp(DT_INT64, TensorShape({num_elems}),
                                       &flat_indices_tensor));
    auto flat_indices = flat_indices_tensor.template flat<int64>().data();

    GpuLaunchConfig config = GetGpuLaunchConfig(num_elems, d);
    OP_REQUIRES_OK(
        c, GpuLaunchKernel(IndicesFlattenKernel, config.block_count,
                           config.thread_per_block, 0, d.stream(), indices,
                           num_elems, dims, num_dims, flat_indices));

    Tensor permutation_tensor;
    OP_REQUIRES_OK(
        c, c->allocate_temp(DT_INT64, {num_elems}, &permutation_tensor));
    auto permutation_data = permutation_tensor.template flat<int64>().data();

    OP_REQUIRES_OK(
        c, GpuRadixSort(c, num_elems, /*keys_in=*/flat_indices,
                        /*keys_out=*/static_cast<int64*>(nullptr),
                        /*indices_in=*/static_cast<const int64*>(nullptr),
                        /*indices_out=*/permutation_data));

    // Free temporary tensor that is no longer needed.
    flat_indices_tensor = Tensor();
    flat_indices = nullptr;

    Tensor* reordered_ind_tensor = nullptr;
    Tensor* reordered_val_tensor = nullptr;
    OP_REQUIRES_OK(
        c, c->allocate_output(0, input_ind.shape(), &reordered_ind_tensor));
    OP_REQUIRES_OK(
        c, c->allocate_output(1, input_val.shape(), &reordered_val_tensor));
    auto reordered_ind_data =
        reordered_ind_tensor->template flat<int64>().data();
    auto reordered_val_data = reordered_val_tensor->template flat<T>().data();

    OP_REQUIRES_OK(
        c, GpuLaunchKernel(PermuteIndicesAndValuesKernel<T>, config.block_count,
                           config.thread_per_block, 0, d.stream(), indices,
                           values, num_elems, num_dims, permutation_data,
                           reordered_ind_data, reordered_val_data));
  }
};

}  // namespace functor

#define DEFINE_GPU_SPEC(T) \
  template struct functor::SparseReorderFunctor<GPUDevice, T>;

TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_SPEC);
TF_CALL_INTEGRAL_TYPES(DEFINE_GPU_SPEC);
DEFINE_GPU_SPEC(bool);

}  // namespace tensorflow
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
