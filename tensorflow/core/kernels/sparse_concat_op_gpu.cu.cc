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
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/gpu_device_array.h"
#include "tensorflow/core/kernels/gpu_device_array_gpu.h"
#include "tensorflow/core/kernels/gpu_prim_helpers.h"
#include "tensorflow/core/kernels/sparse_concat_op.h"
#include "tensorflow/core/lib/core/bits.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace functor {

namespace {

template <typename T>
__global__ void SparseConcatKernel(
    int64 output_nnz, int rank, int concat_dim, bool need_to_sort,
    GpuDeviceArrayStruct<const int64*> ind_ptrs_data,
    GpuDeviceArrayStruct<const T*> val_ptrs_data,
    GpuDeviceArrayStruct<int64_t> nnz_scan_data,
    GpuDeviceArrayStruct<int64_t> concat_size_scan_data,
    GpuDeviceArrayStruct<int64_t> output_shape_data,
    int64* __restrict__ output_inds, T* __restrict__ output_vals,
    int64* __restrict__ output_flat_inds) {
  const int64* __restrict__* __restrict__ ind_ptrs =
      GetGpuDeviceArrayOnDevice(&ind_ptrs_data);
  const T* __restrict__* __restrict__ val_ptrs =
      GetGpuDeviceArrayOnDevice(&val_ptrs_data);
  const int64* __restrict__ nnz_scan =
      GetGpuDeviceArrayOnDevice(&nnz_scan_data);
  const int64* __restrict__ concat_size_scan =
      GetGpuDeviceArrayOnDevice(&concat_size_scan_data);
  const int64* __restrict__ output_shape =
      GetGpuDeviceArrayOnDevice(&output_shape_data);
  const int64 num_inputs = ind_ptrs_data.size;

  for (int64 nz : GpuGridRangeX<int64_t>(output_nnz)) {
    const int64 input_num =
        gpu_helper::upper_bound<int64_t>(nnz_scan, num_inputs, nz) - 1;
    const int64 input_nz = nz - nnz_scan[input_num];
    const int64 ind_offset = concat_size_scan[input_num];
    if (!need_to_sort) {
      output_vals[nz] = val_ptrs[input_num][input_nz];
    }
    int64 flat_ind = 0;
    for (int j = 0; j < rank; ++j) {
      const int64 output_ind = ind_ptrs[input_num][input_nz * rank + j] +
                               (j == concat_dim ? ind_offset : 0);
      if (!need_to_sort) {
        output_inds[nz * rank + j] = output_ind;
      } else {
        flat_ind = flat_ind * output_shape[j] + output_ind;
        output_flat_inds[nz] = flat_ind;
      }
    }
  }
}

template <typename T>
__global__ void SparseConcatPermuteKernel(
    int64 output_nnz, int rank, GpuDeviceArrayStruct<const T*> val_ptrs_data,
    GpuDeviceArrayStruct<int64_t> nnz_scan_data,
    GpuDeviceArrayStruct<int64_t> output_shape_data,
    const int64* __restrict__ output_flat_inds,
    const int64* __restrict__ permutation, int64* __restrict__ output_inds,
    T* __restrict__ output_vals) {
  const T* __restrict__* __restrict__ val_ptrs =
      GetGpuDeviceArrayOnDevice(&val_ptrs_data);
  const int64* __restrict__ nnz_scan =
      GetGpuDeviceArrayOnDevice(&nnz_scan_data);
  const int64* __restrict__ output_shape =
      GetGpuDeviceArrayOnDevice(&output_shape_data);
  const int64 num_inputs = val_ptrs_data.size;

  for (int64 nz : GpuGridRangeX<int64_t>(output_nnz)) {
    const int64 permuted_nz = permutation[nz];
    const int64 input_num =
        gpu_helper::upper_bound<int64_t>(nnz_scan, num_inputs, permuted_nz) - 1;
    const int64 input_nz = permuted_nz - nnz_scan[input_num];
    output_vals[nz] = val_ptrs[input_num][input_nz];
    int64 output_flat_ind = output_flat_inds[permuted_nz];
    for (int j = rank - 1; j >= 0; --j) {
      const int64 output_dim_size = output_shape[j];
      output_inds[nz * rank + j] = output_flat_ind % output_dim_size;
      output_flat_ind /= output_dim_size;
    }
  }
}

}  // namespace

template <typename T>
struct SparseConcatFunctor<GPUDevice, T> {
  void operator()(OpKernelContext* context, const OpInputList& inds,
                  const OpInputList& vals, const OpInputList& shapes,
                  int concat_dim) {
    const int N = inds.size();
    const TensorShape input_shape0(shapes[0].vec<int64_t>());
    const int rank = input_shape0.dims();

    // The input non-zeros are assumed to be sorted by increasing dimension
    // number (i.e., row-major order), so if the concatenation is along the
    // first dimension then they remain in order and we can directly compute the
    // output indices and values. To concatenate along other dimensions, we
    // first compute the flattened (1D) row-major output indices, then sort
    // these to obtain the required permutation, and finally gather the permuted
    // input values.

    GpuDeviceArrayOnHost<const int64*> ind_ptrs(context, N);
    GpuDeviceArrayOnHost<const T*> val_ptrs(context, N);
    GpuDeviceArrayOnHost<int64_t> nnz_scan(context, N + 1);
    GpuDeviceArrayOnHost<int64_t> concat_size_scan(context, N + 1);
    OP_REQUIRES_OK(context, ind_ptrs.Init());
    OP_REQUIRES_OK(context, val_ptrs.Init());
    OP_REQUIRES_OK(context, nnz_scan.Init());
    OP_REQUIRES_OK(context, concat_size_scan.Init());
    int64 nnz_sum = 0;
    int64 concat_size_sum = 0;
    nnz_scan.Set(0, nnz_sum);
    concat_size_scan.Set(0, concat_size_sum);
    for (int i = 0; i < N; ++i) {
      ind_ptrs.Set(i, inds[i].matrix<int64_t>().data());
      val_ptrs.Set(i, vals[i].vec<T>().data());
      nnz_sum += inds[i].dim_size(0);
      nnz_scan.Set(i + 1, nnz_sum);
      const TensorShape current_shape(shapes[i].vec<int64_t>());
      concat_size_sum += current_shape.dim_size(concat_dim);
      concat_size_scan.Set(i + 1, concat_size_sum);
    }
    OP_REQUIRES_OK(context, ind_ptrs.Finalize());
    OP_REQUIRES_OK(context, val_ptrs.Finalize());
    OP_REQUIRES_OK(context, nnz_scan.Finalize());
    OP_REQUIRES_OK(context, concat_size_scan.Finalize());
    const int64 output_nnz = nnz_sum;
    const int64 output_concat_size = concat_size_sum;

    const bool need_to_sort = concat_dim != 0;

    GpuDeviceArrayOnHost<int64_t> output_shape(context, rank);
    int64 output_dense_elements;
    if (need_to_sort) {
      OP_REQUIRES_OK(context, output_shape.Init());
      output_dense_elements = 1;
      for (int j = 0; j < rank; ++j) {
        int64 output_dim_size =
            j == concat_dim ? output_concat_size : input_shape0.dim_size(j);
        output_shape.Set(j, output_dim_size);
        output_dense_elements *= output_dim_size;
      }
      OP_REQUIRES_OK(context, output_shape.Finalize());
    }

    int64* output_inds_ptr = nullptr;
    T* output_vals_ptr = nullptr;
    int64* output_flat_inds_ptr = nullptr;
    Tensor output_flat_inds;
    if (need_to_sort) {
      // SparseConcatKernel will (only) produce output_flat_inds.
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DT_INT64, TensorShape({output_nnz}),
                                            &output_flat_inds));
      output_flat_inds_ptr = output_flat_inds.vec<int64_t>().data();
    } else {
      OP_REQUIRES_OK(
          context, allocate_outputs(context, rank, output_nnz, &output_inds_ptr,
                                    &output_vals_ptr));
    }

    const GPUDevice& device = context->eigen_gpu_device();

    GpuLaunchConfig config = GetGpuLaunchConfig(
        output_nnz, device, &SparseConcatKernel<T>,
        /*dynamic_shared_memory_size=*/0, /*block_size_limit=*/0);
    OP_REQUIRES_OK(
        context, GpuLaunchKernel(
                     SparseConcatKernel<T>, config.block_count,
                     config.thread_per_block, 0, device.stream(), output_nnz,
                     rank, concat_dim, need_to_sort, ind_ptrs.data(),
                     val_ptrs.data(), nnz_scan.data(), concat_size_scan.data(),
                     (need_to_sort ? output_shape.data()
                                   : GpuDeviceArrayStruct<int64_t>()),
                     output_inds_ptr, output_vals_ptr, output_flat_inds_ptr));

    if (!need_to_sort) return;

    OP_REQUIRES_OK(context,
                   allocate_outputs(context, rank, output_nnz, &output_inds_ptr,
                                    &output_vals_ptr));

    Tensor permutation;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DT_INT64, TensorShape({output_nnz}),
                                          &permutation));
    int64* permutation_ptr = permutation.vec<int64_t>().data();
    OP_REQUIRES_OK(
        context,
        GpuRadixSort(context, /*size=*/output_nnz,
                     /*keys_in=*/output_flat_inds_ptr,
                     /*keys_out=*/static_cast<int64*>(nullptr),
                     /*indices_in=*/static_cast<const int64*>(nullptr),
                     /*indices_out=*/permutation_ptr,
                     /*num_bits=*/Log2Ceiling64(output_dense_elements)));

    config = GetGpuLaunchConfig(
        output_nnz, device, &SparseConcatPermuteKernel<T>,
        /*dynamic_shared_memory_size=*/0, /*block_size_limit=*/0);
    OP_REQUIRES_OK(
        context,
        GpuLaunchKernel(SparseConcatPermuteKernel<T>, config.block_count,
                        config.thread_per_block, 0, device.stream(), output_nnz,
                        rank, val_ptrs.data(), nnz_scan.data(),
                        output_shape.data(), output_flat_inds_ptr,
                        permutation_ptr, output_inds_ptr, output_vals_ptr));
  }

 private:
  Status allocate_outputs(OpKernelContext* context, int rank, int64 output_nnz,
                          int64** output_inds_ptr, T** output_vals_ptr) const {
    Tensor* output_inds = nullptr;
    TF_RETURN_IF_ERROR(context->allocate_output(
        0, TensorShape({output_nnz, rank}), &output_inds));
    *output_inds_ptr = output_inds->matrix<int64_t>().data();
    Tensor* output_vals = nullptr;
    TF_RETURN_IF_ERROR(
        context->allocate_output(1, TensorShape({output_nnz}), &output_vals));
    *output_vals_ptr = output_vals->vec<T>().data();
    return OkStatus();
  }
};

#define DEFINE_SPARSE_CONCAT_FUNCTOR(T) \
  template struct SparseConcatFunctor<GPUDevice, T>;
TF_CALL_POD_TYPES(DEFINE_SPARSE_CONCAT_FUNCTOR);

#undef DEFINE_SPARSE_CONCAT_FUNCTOR

}  // namespace functor

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
