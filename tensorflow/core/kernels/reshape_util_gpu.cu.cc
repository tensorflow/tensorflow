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

#include "xla/tsl/platform/statusor.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/kernels/reshape_util.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

using GPUDevice = Eigen::GpuDevice;

namespace {

template <typename Tindex>
__global__ void ReshapeSparseTensorKernel(
    const Tindex nnz, const Tindex input_rank, const Tindex output_rank,
    const Tindex* __restrict__ input_shape,
    const Tindex* __restrict__ output_shape,
    const Tindex* __restrict__ input_indices,
    Tindex* __restrict__ output_indices,
    Tindex* __restrict__ first_invalid_row) {
  GPU_1D_KERNEL_LOOP(sparse_index, nnz, Tindex) {
    const Tindex* input_index = &input_indices[sparse_index * input_rank];
    Tindex* output_index = &output_indices[sparse_index * output_rank];
    int64_t dense_index = 0;  // int64 to avoid overflow if Tindex is int32
    bool valid = true;
    // Flatten input index from slowest- to fastest-changing dimension.
    for (int i = 0; i < input_rank; ++i) {
      const Tindex index = input_index[i];
      if (!FastBoundsCheck(index, input_shape[i])) {
        valid = false;
        break;
      }
      dense_index = dense_index * input_shape[i] + index;
    }
    if (!valid) {
      // Report the invalid row instead of computing an overflowing dense index.
      GpuAtomicMin(first_invalid_row, sparse_index);
      continue;
    }
    // Compute output index from fastest- to slowest-changing dimension.
    for (int i = output_rank - 1; i >= 0; --i) {
      Tindex output_size = output_shape[i];
      output_index[i] = dense_index % output_size;
      dense_index /= output_size;
    }
  }
}

}  // namespace

namespace functor {

absl::Status ReshapeSparseTensorFunctor<GPUDevice>::operator()(
    OpKernelContext* context, const TensorShape& input_shape,
    const TensorShape& output_shape,
    typename TTypes<int64_t>::ConstMatrix input_indices,
    typename TTypes<int64_t>::Matrix output_indices,
    typename TTypes<int64_t>::Scalar first_invalid_row, Tensor* input_shape_dev,
    Tensor* output_shape_dev) const {
  const int64_t input_rank = input_shape.dims();
  const int64_t output_rank = output_shape.dims();
  const int64_t nnz = input_indices.dimension(0);
  // We copy input_shape and output_shape to the GPU and then launch a kernel
  // to compute output_indices. The caller keeps the device copies alive until
  // the kernel has completed.
  TF_RETURN_IF_ERROR(context->allocate_temp(DT_INT64, TensorShape({input_rank}),
                                            input_shape_dev));
  auto input_shape_gpu = input_shape_dev->flat<int64_t>();
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DT_INT64, TensorShape({output_rank}), output_shape_dev));
  auto output_shape_gpu = output_shape_dev->flat<int64_t>();
  se::Stream* stream = context->op_device_context()->stream();
  if (!stream) return absl::InternalError("No GPU stream available.");
  stream_executor::DeviceAddressBase input_shape_gpu_mem(
      input_shape_gpu.data(), input_rank * sizeof(int64_t));
  TF_RETURN_IF_ERROR(stream->Memcpy(&input_shape_gpu_mem,
                                    input_shape.dim_sizes().data(),
                                    input_rank * sizeof(int64_t)));
  stream_executor::DeviceAddressBase output_shape_gpu_mem(
      output_shape_gpu.data(), output_rank * sizeof(int64_t));
  TF_RETURN_IF_ERROR(stream->Memcpy(&output_shape_gpu_mem,
                                    output_shape.dim_sizes().data(),
                                    output_rank * sizeof(int64_t)));
  const GPUDevice& device = context->template eigen_device<GPUDevice>();
  // All rows are valid until the kernel finds an invalid one.
  TF_RETURN_IF_ERROR(GpuLaunchKernel(SetToValue<int64_t>, 1, 1, 0,
                                     device.stream(), /*count=*/1,
                                     first_invalid_row.data(), nnz));
  TF_ASSIGN_OR_RETURN(GpuLaunchConfig64 config,
                      GetGpuLaunchConfig64(nnz, device));
  return GpuLaunchKernel(ReshapeSparseTensorKernel<int64_t>, config.block_count,
                         config.thread_per_block, 0, device.stream(), nnz,
                         /*input_rank=*/input_rank,
                         /*output_rank=*/output_rank,
                         /*input_shape=*/input_shape_gpu.data(),
                         /*output_shape=*/output_shape_gpu.data(),
                         /*input_indices=*/input_indices.data(),
                         /*output_indices=*/output_indices.data(),
                         /*first_invalid_row=*/first_invalid_row.data());
}

}  // namespace functor

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
