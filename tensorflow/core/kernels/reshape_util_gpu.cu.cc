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

#include <limits>

#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/reshape_util.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

using GPUDevice = Eigen::GpuDevice;

namespace {

template <typename Tindex>
__global__ void ReshapeSparseTensorKernel(
    const int64_t nnz, const int32_t input_rank, const int32_t output_rank,
    const Tindex* __restrict__ input_shape,
    const Tindex* __restrict__ output_shape,
    const Tindex* __restrict__ input_indices,
    Tindex* __restrict__ output_indices) {
  GPU_1D_KERNEL_LOOP(sparse_index, nnz) {
    const Tindex* input_index = &input_indices[sparse_index * input_rank];
    Tindex* output_index = &output_indices[sparse_index * output_rank];
    int64_t dense_index = 0;  // int64 to avoid overflow if Tindex is int32/int16
    // Flatten input index from slowest- to fastest-changing dimension.
    for (int i = 0; i < input_rank; ++i) {
      dense_index = dense_index * input_shape[i] + input_index[i];
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

namespace {
// Uploads shape dims (int64 on host) to a GPU tensor of type Tindices.
// cpu_shape_t is allocated with on_host+gpu_compatible attributes so TF's
// allocator keeps the pinned buffer valid until stream ops complete.
template <typename Tindices>
absl::Status UploadShapeToGPU(OpKernelContext* context,
                               const TensorShape& shape,
                               se::Stream* stream,
                               Tensor* gpu_shape_t,
                               Tensor* cpu_shape_t) {
  const int64_t rank = shape.dims();
  const int64_t max_index =
      static_cast<int64_t>(std::numeric_limits<Tindices>::max());
  for (int i = 0; i < rank; ++i) {
    if (shape.dim_size(i) > max_index) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Shape dimension ", i, " (", shape.dim_size(i),
          ") exceeds the maximum value representable by the index type (",
          max_index, ")"));
    }
  }
  TF_RETURN_IF_ERROR(context->allocate_temp(DataTypeToEnum<Tindices>::value,
                                            TensorShape({rank}), gpu_shape_t));
  AllocatorAttributes alloc_attr;
  alloc_attr.set_on_host(true);
  alloc_attr.set_gpu_compatible(true);
  TF_RETURN_IF_ERROR(context->allocate_temp(DataTypeToEnum<Tindices>::value,
                                            TensorShape({rank}), cpu_shape_t,
                                            alloc_attr));
  auto cpu_shape_flat = cpu_shape_t->flat<Tindices>();
  for (int i = 0; i < rank; ++i) {
    cpu_shape_flat(i) = static_cast<Tindices>(shape.dim_size(i));
  }
  stream_executor::DeviceAddressBase gpu_mem(
      gpu_shape_t->flat<Tindices>().data(), rank * sizeof(Tindices));
  return stream->Memcpy(&gpu_mem, cpu_shape_flat.data(),
                        rank * sizeof(Tindices));
}

template <typename Tindices>
absl::Status ReshapeSparseTensorFunctorGPUImpl(
    OpKernelContext* context, const TensorShape& input_shape,
    const TensorShape& output_shape,
    typename TTypes<Tindices>::ConstMatrix input_indices,
    typename TTypes<Tindices>::Matrix output_indices) {
  const int64_t input_rank = input_shape.dims();
  const int64_t output_rank = output_shape.dims();
  const int64_t nnz = input_indices.dimension(0);
  se::Stream* stream = context->op_device_context()->stream();
  if (!stream) return absl::InternalError("No GPU stream available.");
  Tensor input_shape_gpu_t, output_shape_gpu_t;
  Tensor input_shape_cpu_t, output_shape_cpu_t;
  TF_RETURN_IF_ERROR(UploadShapeToGPU<Tindices>(context, input_shape, stream,
                                                &input_shape_gpu_t,
                                                &input_shape_cpu_t));
  TF_RETURN_IF_ERROR(UploadShapeToGPU<Tindices>(context, output_shape, stream,
                                                &output_shape_gpu_t,
                                                &output_shape_cpu_t));
  const GPUDevice& device = context->template eigen_device<GPUDevice>();
  auto config = GetGpuLaunchConfig(nnz, device);
  return GpuLaunchKernel(
      ReshapeSparseTensorKernel<Tindices>, config.block_count,
      config.thread_per_block, 0, device.stream(), nnz,
      static_cast<int32_t>(input_rank), static_cast<int32_t>(output_rank),
      input_shape_gpu_t.flat<Tindices>().data(),
      output_shape_gpu_t.flat<Tindices>().data(), input_indices.data(),
      output_indices.data());
}
}  // namespace

template <>
absl::Status ReshapeSparseTensorFunctor<GPUDevice, int64_t>::operator()(
    OpKernelContext* context, const TensorShape& input_shape,
    const TensorShape& output_shape,
    typename TTypes<int64_t>::ConstMatrix input_indices,
    typename TTypes<int64_t>::Matrix output_indices) const {
  return ReshapeSparseTensorFunctorGPUImpl<int64_t>(
      context, input_shape, output_shape, input_indices, output_indices);
}

template <>
absl::Status ReshapeSparseTensorFunctor<GPUDevice, int32_t>::operator()(
    OpKernelContext* context, const TensorShape& input_shape,
    const TensorShape& output_shape,
    typename TTypes<int32_t>::ConstMatrix input_indices,
    typename TTypes<int32_t>::Matrix output_indices) const {
  return ReshapeSparseTensorFunctorGPUImpl<int32_t>(
      context, input_shape, output_shape, input_indices, output_indices);
}

template <>
absl::Status ReshapeSparseTensorFunctor<GPUDevice, int16_t>::operator()(
    OpKernelContext* context, const TensorShape& input_shape,
    const TensorShape& output_shape,
    typename TTypes<int16_t>::ConstMatrix input_indices,
    typename TTypes<int16_t>::Matrix output_indices) const {
  return ReshapeSparseTensorFunctorGPUImpl<int16_t>(
      context, input_shape, output_shape, input_indices, output_indices);
}

}  // namespace functor

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
