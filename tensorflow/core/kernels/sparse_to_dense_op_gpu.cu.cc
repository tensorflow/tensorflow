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

#include "tensorflow/core/kernels/sparse_to_dense_op_gpu.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/stream_executor/gpu/gpu_activation.h"

namespace tensorflow {

namespace {

template <typename T, typename Index>
__global__ void SparseToDenseKernel(const Index* __restrict__ indices,
                                    const T* __restrict__ vals, const int nnz,
                                    const int num_vals,
                                    const Index* __restrict__ dims,
                                    const int ndims, T* __restrict__ dense) {
  GPU_1D_KERNEL_LOOP(thread_idx, nnz) {
    eigen_assert(ndims >= 1);
    int64 output_idx = indices[thread_idx * ndims + ndims - 1];
    Index strides = 1;
    for (int i = ndims - 2; i >= 0; i--) {
      strides *= dims[i + 1];
      output_idx += indices[thread_idx * ndims + i] * strides;
    }
    // If num_vals == 1, broadcast the scalar to the positions for non-zeros.
    dense[output_idx] = vals[(num_vals == 1) ? 0 : thread_idx];
  }
}

template <typename T, typename Index>
__global__ void SetDefaultValue(const T default_value, const int64 dense_size,
                                T* __restrict__ dense) {
  GPU_1D_KERNEL_LOOP(thread_idx, dense_size) {
    dense[thread_idx] = default_value;
  }
}

template <typename Index>
__global__ void CheckIndicesValid(const Index* __restrict__ indices,
                                  const int nnz, const Index* __restrict__ dims,
                                  const int ndims, int* __restrict__ status) {
  GPU_1D_KERNEL_LOOP(thread_idx, nnz) {
    bool increasing = true;
    bool different = false;
    bool valid = true;

    if (thread_idx == 0) {
      for (int di = 0; di < ndims; di++) {
        Index curr_idx = indices[thread_idx * ndims + di];
        if (curr_idx < 0 || curr_idx >= dims[di]) valid = false;
      }
      different = true;
    } else {
      for (int di = 0; di < ndims; di++) {
        Index curr_idx = indices[thread_idx * ndims + di];
        if (curr_idx < 0 || curr_idx >= dims[di]) valid = false;
        Index prev_idx = indices[(thread_idx - 1) * ndims + di];
        Index diff = curr_idx - prev_idx;
        if (diff > 0) different = true;
        if (!different && diff < 0) increasing = false;
      }
    }

    if (!valid) {
      atomicMin(detail::ToCudaSupportedPtr(&status[0]), thread_idx);
    }
    if (!increasing) {
      atomicMin(detail::ToCudaSupportedPtr(&status[1]), thread_idx);
    }
    if (!different) {
      atomicMin(detail::ToCudaSupportedPtr(&status[2]), thread_idx);
    }
  }
}

// IndicesValidStatus contains three status for the out-of-bound check, the
// sorted check, and the repeat check. If the value equals to INT_MAX, the
// check passes. Otherwise, it represents the first detected position of the
// invalid index for the check.
struct IndicesValidStatus {
  int valid;
  int increasing;
  int different;
};

template <typename T, typename Index>
Status LaunchComputeKernels(OpKernelContext* c, const int64 dense_size,
                            const T default_value, const Index* indices,
                            const T* values, const int num_elems,
                            const int num_values, const Index* shape,
                            const int num_dims, T* dense) {
  const Eigen::GpuDevice& d = c->eigen_gpu_device();
  if (dense_size > 0) {
    GpuLaunchConfig config0 = GetGpuLaunchConfig(dense_size, d);
    // The template type T might not necessarily be 32bit, and therefore, we use
    // SetDefaultValue instead of memset32.
    TF_RETURN_IF_ERROR(GpuLaunchKernel(SetDefaultValue<T, Index>,
                                       config0.block_count,
                                       config0.thread_per_block, 0, d.stream(),
                                       default_value, dense_size, dense));
  }

  if (num_elems > 0) {
    GpuLaunchConfig config1 = GetGpuLaunchConfig(num_elems, d);
    TF_RETURN_IF_ERROR(
        GpuLaunchKernel(SparseToDenseKernel<T, Index>, config1.block_count,
                        config1.thread_per_block, 0, d.stream(), indices,
                        values, num_elems, num_values, shape, num_dims, dense));
  }
  return Status::OK();
}

}  // namespace

namespace functor {

template <typename T, typename Index>
void LaunchSparseToDense<T, Index>::operator()(
    OpKernelContext* c, AsyncOpKernel::DoneCallback done, AsyncOpKernel* op,
    bool validate_indices, const Tensor& indices, const Tensor& values,
    const Tensor& shape, const T default_value, Tensor* dense) {
  auto* stream = c->op_device_context()->stream();
  const Eigen::GpuDevice& d = c->eigen_gpu_device();

  const Index* indices_ptr = indices.flat<Index>().data();
  const T* values_ptr = values.flat<T>().data();
  const Index* shape_ptr = shape.flat<Index>().data();
  T* dense_ptr = dense->flat<T>().data();
  const int64 dense_size = dense->NumElements();
  const int64 num_values = values.NumElements();
  const int64 num_elems = indices.dims() > 0 ? indices.dim_size(0) : 1;
  const int64 num_dims = indices.dims() > 1 ? indices.dim_size(1) : 1;
  if (validate_indices && num_elems != 0) {
    VLOG(1) << "SparseToDense will be performed on GPUs. For performance "
               "reasons, it is suggested to pass False to validate_indices.";

    IndicesValidStatus valid_status;
    int valid_status_size = sizeof(valid_status) / sizeof(int);
    int valid_status_bytes = sizeof(valid_status);

    Tensor valid_status_tensor;
    OP_REQUIRES_OK_ASYNC(
        c,
        c->allocate_temp(DT_INT32, TensorShape({valid_status_size}),
                         &valid_status_tensor),
        done);

    auto status_ptr = valid_status_tensor.template flat<int>().data();
    se::DeviceMemoryBase valid_status_ptr(status_ptr, valid_status_bytes);

    GpuLaunchConfig config = GetGpuLaunchConfig(num_elems, d);
    stream->ThenMemset32(&valid_status_ptr, INT_MAX, valid_status_bytes);
    OP_REQUIRES_OK_ASYNC(
        c,
        GpuLaunchKernel(CheckIndicesValid<Index>, config.block_count,
                        config.thread_per_block, 0, d.stream(), indices_ptr,
                        num_elems, shape_ptr, num_dims, status_ptr),
        done);
    stream->ThenMemcpy(reinterpret_cast<int*>(&valid_status), valid_status_ptr,
                       valid_status_bytes);

    // We capture 'shape' instead of 'shape_ptr' since this lambda outlives
    // the 'shape' tensor.
    auto check_status_and_compute = [op, c, valid_status, dense_size,
                                     default_value, indices_ptr, values_ptr,
                                     num_elems, num_values, shape, num_dims,
                                     dense_ptr, done]() {
      // Ensure that within the callback, the proper GPU settings are
      // configured.
      auto stream = c->op_device_context()->stream();
      se::gpu::ScopedActivateExecutorContext scoped_activation{
          stream->parent()};

      OP_REQUIRES_ASYNC(c, valid_status.valid == INT_MAX,
                        errors::InvalidArgument("indices[", valid_status.valid,
                                                "] is out of bounds."),
                        done);

      OP_REQUIRES_ASYNC(c, valid_status.increasing == INT_MAX,
                        errors::InvalidArgument(
                            "indices[", valid_status.increasing,
                            "] is out of "
                            "order. Many sparse ops require sorted indices.\n"
                            "  Use `tf.sparse.reorder` to create a correctly "
                            "ordered copy.\n\n"),
                        done);

      OP_REQUIRES_ASYNC(
          c, valid_status.different == INT_MAX,
          errors::InvalidArgument("indices[", valid_status.different,
                                  "] is "
                                  "repeated."),
          done);

      OP_REQUIRES_OK_ASYNC(
          c,
          LaunchComputeKernels(c, dense_size, default_value, indices_ptr,
                               values_ptr, num_elems, num_values,
                               shape.flat<Index>().data(), num_dims, dense_ptr),
          done);
      done();
    };

    c->device()->tensorflow_gpu_device_info()->event_mgr->ThenExecute(
        stream, check_status_and_compute);
  } else {
    OP_REQUIRES_OK_ASYNC(
        c,
        LaunchComputeKernels(c, dense_size, default_value, indices_ptr,
                             values_ptr, num_elems, num_values, shape_ptr,
                             num_dims, dense_ptr),
        done);
    done();
  }
}

}  // namespace functor

#define DEFINE_GPU_SPEC(T)                                \
  template struct functor::LaunchSparseToDense<T, int64>; \
  template struct functor::LaunchSparseToDense<T, int32>;

TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_SPEC)
TF_CALL_INTEGRAL_TYPES(DEFINE_GPU_SPEC)
DEFINE_GPU_SPEC(bool)

}  // namespace tensorflow
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
