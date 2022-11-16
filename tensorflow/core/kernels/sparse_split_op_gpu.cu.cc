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
#include "tensorflow/core/kernels/sparse_split_op.h"
#include "tensorflow/core/lib/core/bits.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_solvers.h"  // For ScratchSpace

#if GOOGLE_CUDA
#include "tensorflow/compiler/xla/stream_executor/cuda/cuda_activation.h"
using stream_executor::cuda::ScopedActivateExecutorContext;
#elif TENSORFLOW_USE_ROCM
#include "tensorflow/compiler/xla/stream_executor/rocm/rocm_activation.h"
using stream_executor::rocm::ScopedActivateExecutorContext;
#endif

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace functor {

namespace {

template <typename Index>
inline __device__ Index GetSliceIndex(const Index index, const Index split_size,
                                      const Index residual) {
  if (residual == 0) return index / split_size;
  const Index offset = residual * (split_size + Index(1));
  if (index < offset) {
    return index / (split_size + Index(1));
  } else {
    return residual + ((index - offset) / split_size);
  }
}

template <typename Index>
inline __device__ Index GetDimensionInSlice(const Index index,
                                            const Index split_size,
                                            const Index residual) {
  if (residual == 0) return index % split_size;
  const Index offset = residual * (split_size + 1);
  if (index < offset) {
    return index % (split_size + 1);
  } else {
    return (index - offset) % split_size;
  }
}

template <typename Index>
inline Index GetSliceShape(const Index slice_index, const Index split_size,
                           const Index residual) {
  if (residual == 0) return split_size;
  if (slice_index < residual) {
    return split_size + 1;
  } else {
    return split_size;
  }
}

template <typename Index>
struct SliceIndexer {
  SliceIndexer(const Index split_dim_size, const Index num_split)
      : split_size_(split_dim_size / num_split),
        residual_(split_dim_size % num_split) {}

  inline __device__ Index GetSliceIndex(const Index index) const {
    return tensorflow::functor::GetSliceIndex(index, split_size_, residual_);
  }

  inline __device__ Index GetIndexInSlice(const Index index) const {
    return GetDimensionInSlice(index, split_size_, residual_);
  }

  inline __host__ Index GetSliceSize(const Index slice_index) const {
    return GetSliceShape(slice_index, split_size_, residual_);
  }

 private:
  const Index split_size_;
  const Index residual_;
};

template <typename Index>
__global__ void SparseSplitSliceIndexesKernel(
    Index input_nnz, int rank, int axis, SliceIndexer<Index> slice_indexer,
    const Index* __restrict__ input_indices, int* __restrict__ slice_indexes) {
  for (Index input_nz : GpuGridRangeX<Index>(input_nnz)) {
    slice_indexes[input_nz] =
        slice_indexer.GetSliceIndex(input_indices[input_nz * rank + axis]);
  }
}

template <typename Index>
Status LaunchSparseSplitSliceIndexesKernel(const GPUDevice& device,
                                           Index input_nnz, int num_split,
                                           int rank, int axis,
                                           SliceIndexer<Index> slice_indexer,
                                           const Index* input_indices,
                                           int* slice_indexes) {
  if (input_nnz == 0) return OkStatus();
  GpuLaunchConfig config = GetGpuLaunchConfig(
      input_nnz, device, &SparseSplitSliceIndexesKernel<Index>,
      /*dynamic_shared_memory_size=*/0, /*block_size_limit=*/0);
  return GpuLaunchKernel(SparseSplitSliceIndexesKernel<Index>,
                         config.block_count, config.thread_per_block, 0,
                         device.stream(), input_nnz, rank, axis, slice_indexer,
                         input_indices, slice_indexes);
}

template <typename Index>
__global__ void SparseSplitFindSliceEndsKernel(
    Index input_nnz, int num_split,
    const int* __restrict__ sorted_slice_indexes,
    Index* __restrict__ slice_ends) {
  for (int slice_index : GpuGridRangeX<int>(num_split)) {
    slice_ends[slice_index] =
        gpu_helper::upper_bound(sorted_slice_indexes, input_nnz, slice_index);
  }
}

template <typename Index>
Status LaunchSparseSplitFindSliceEndsKernel(const GPUDevice& device,
                                            Index input_nnz, int num_split,
                                            const int* sorted_slice_indexes,
                                            Index* slice_ends) {
  GpuLaunchConfig config = GetGpuLaunchConfig(
      num_split, device, &SparseSplitFindSliceEndsKernel<Index>,
      /*dynamic_shared_memory_size=*/0, /*block_size_limit=*/0);
  return GpuLaunchKernel(SparseSplitFindSliceEndsKernel<Index>,
                         config.block_count, config.thread_per_block, 0,
                         device.stream(), input_nnz, num_split,
                         sorted_slice_indexes, slice_ends);
}

// Scatters (and offsets) input indices and values to the outputs.
template <typename T, typename Index>
__global__ void SparseSplitScatterKernel(
    Index input_nnz, int rank, int axis, SliceIndexer<Index> slice_indexer,
    const Index* __restrict__ sort_permutation,
    const Index* __restrict__ slice_ends,
    const Index* __restrict__ input_indices, const T* __restrict__ input_values,
    GpuDeviceArrayStruct<Index*> output_indices_data,
    GpuDeviceArrayStruct<T*> output_values_data) {
  Index* __restrict__* __restrict__ output_indices =
      GetGpuDeviceArrayOnDevice(&output_indices_data);
  T* __restrict__* __restrict__ output_values =
      GetGpuDeviceArrayOnDevice(&output_values_data);

  for (Index sorted_input_nz : GpuGridRangeX<Index>(input_nnz)) {
    Index input_nz = sort_permutation[sorted_input_nz];
    int slice_index =
        slice_indexer.GetSliceIndex(input_indices[input_nz * rank + axis]);
    Index slice_nz =
        sorted_input_nz -
        (slice_index == 0 ? Index(0) : slice_ends[slice_index - 1]);
    output_values[slice_index][slice_nz] = input_values[input_nz];
    for (int dim = 0; dim < rank; ++dim) {
      Index input_index = input_indices[input_nz * rank + dim];
      output_indices[slice_index][slice_nz * rank + dim] =
          (dim == axis) ? slice_indexer.GetIndexInSlice(input_index)
                        : input_index;
    }
  }
}

template <typename T, typename Index>
Status LaunchSparseSplitScatterKernel(
    const GPUDevice& device, Index input_nnz, int rank, int axis,
    SliceIndexer<Index> slice_indexer, const Index* sort_permutation,
    const Index* slice_ends, const Index* input_indices, const T* input_values,
    GpuDeviceArrayStruct<Index*> output_indices_data,
    GpuDeviceArrayStruct<T*> output_values_data) {
  if (input_nnz == 0) return OkStatus();
  GpuLaunchConfig config = GetGpuLaunchConfig(
      input_nnz, device, &SparseSplitScatterKernel<T, Index>,
      /*dynamic_shared_memory_size=*/0, /*block_size_limit=*/0);
  return GpuLaunchKernel(SparseSplitScatterKernel<T, Index>, config.block_count,
                         config.thread_per_block, 0, device.stream(), input_nnz,
                         rank, axis, slice_indexer, sort_permutation,
                         slice_ends, input_indices, input_values,
                         output_indices_data, output_values_data);
}

}  // namespace

template <typename T>
struct SparseSplitFunctor<GPUDevice, T> {
  void operator()(OpKernelContext* context, const Tensor& input_indices,
                  const Tensor& input_values, const TensorShape& dense_shape,
                  const int64_t axis, const int num_split,
                  typename AsyncOpKernel::DoneCallback done) {
    using Index = int64_t;

    const Index input_nnz = input_indices.dim_size(0);
    const Index split_dim_size = dense_shape.dim_size(static_cast<int>(axis));
    const int rank = dense_shape.dims();

    const Index* input_indices_ptr = input_indices.matrix<Index>().data();
    const T* input_values_ptr = input_values.vec<T>().data();

    const SliceIndexer<Index> slice_indexer(split_dim_size, num_split);

    const GPUDevice& device = context->eigen_gpu_device();
    se::Stream* stream = context->op_device_context()->stream();
    OP_REQUIRES_ASYNC(context, stream,
                      errors::Internal("No GPU stream available."), done);

    Tensor sort_permutation;
    OP_REQUIRES_OK_ASYNC(
        context,
        context->allocate_temp(DataTypeToEnum<Index>::value,
                               TensorShape({input_nnz}), &sort_permutation),
        done);
    Index* sort_permutation_ptr = sort_permutation.vec<Index>().data();

    Tensor slice_ends;
    OP_REQUIRES_OK_ASYNC(
        context,
        context->allocate_temp(DataTypeToEnum<Index>::value,
                               TensorShape({num_split}), &slice_ends),
        done);
    Index* slice_ends_ptr = slice_ends.vec<Index>().data();

    // First we compute the slice index for each element, sort them, and use a
    // binary search to find the end of each slice.
    {
      Tensor slice_indexes;
      OP_REQUIRES_OK_ASYNC(
          context,
          context->allocate_temp(DT_INT32, TensorShape({input_nnz}),
                                 &slice_indexes),
          done);
      int* slice_indexes_ptr = slice_indexes.vec<int>().data();

      OP_REQUIRES_OK_ASYNC(
          context,
          LaunchSparseSplitSliceIndexesKernel(
              device, input_nnz, num_split, rank, axis, slice_indexer,
              input_indices_ptr, slice_indexes_ptr),
          done);

      Tensor sorted_slice_indexes;
      OP_REQUIRES_OK_ASYNC(
          context,
          context->allocate_temp(DT_INT32, TensorShape({input_nnz}),
                                 &sorted_slice_indexes),
          done);
      int* sorted_slice_indexes_ptr = sorted_slice_indexes.vec<int>().data();
      OP_REQUIRES_OK_ASYNC(
          context,
          GpuRadixSort(context, /*size=*/input_nnz,
                       /*keys_in=*/slice_indexes_ptr,
                       /*keys_out=*/sorted_slice_indexes_ptr,
                       /*indices_in=*/static_cast<const Index*>(nullptr),
                       /*indices_out=*/sort_permutation_ptr,
                       /*num_bits=*/Log2Ceiling(num_split)),
          done);

      OP_REQUIRES_OK_ASYNC(context,
                           LaunchSparseSplitFindSliceEndsKernel(
                               device, input_nnz, num_split,
                               sorted_slice_indexes_ptr, slice_ends_ptr),
                           done);
    }

    // Copy the slice ends to the host so that we can compute the output shapes.
    ScratchSpace<Index> slice_ends_host(context, num_split, /*on_host=*/true);
    OP_REQUIRES_ASYNC(
        context,
        stream
            ->ThenMemcpy(
                slice_ends_host.mutable_data(),
                se::DeviceMemoryBase(slice_ends_ptr,
                                     num_split * sizeof(*slice_ends_ptr)),
                num_split * sizeof(*slice_ends_ptr))
            .ok(),
        errors::Internal("Failed to copy slice_ends to host"), done);

    auto async_finish_computation =
        [this, context, input_nnz, num_split, rank, axis, dense_shape,
         slice_indexer, slice_ends_host, input_indices, input_indices_ptr,
         input_values, input_values_ptr, sort_permutation, sort_permutation_ptr,
         slice_ends, slice_ends_ptr, done]() -> void {
      // Ensure that within the callback, the proper GPU settings are
      // configured.
      auto stream = context->op_device_context()->stream();
      ScopedActivateExecutorContext scoped_activation{stream->parent()};

      GpuDeviceArrayOnHost<Index*> output_indices(context, num_split);
      GpuDeviceArrayOnHost<T*> output_values(context, num_split);
      OP_REQUIRES_OK_ASYNC(
          context,
          AllocateOutputs(context, num_split, rank, axis, dense_shape,
                          slice_indexer, slice_ends_host.data(),
                          &output_indices, &output_values),
          done);

      const GPUDevice& device = context->eigen_device<GPUDevice>();

      // Finally, scatter (and offset) input indices and values to the outputs.
      OP_REQUIRES_OK_ASYNC(
          context,
          LaunchSparseSplitScatterKernel(
              device, input_nnz, rank, axis, slice_indexer,
              sort_permutation_ptr, slice_ends_ptr, input_indices_ptr,
              input_values_ptr, output_indices.data(), output_values.data()),
          done);

      done();
    };

    context->device()
        ->tensorflow_accelerator_device_info()
        ->event_mgr->ThenExecute(stream, async_finish_computation);
  }

 private:
  template <typename Index>
  Status AllocateOutputs(OpKernelContext* context, int num_split, int rank,
                         int axis, const TensorShape& dense_shape,
                         const SliceIndexer<Index>& slice_indexer,
                         const Index* slice_ends_host,
                         GpuDeviceArrayOnHost<Index*>* output_indices,
                         GpuDeviceArrayOnHost<T*>* output_values) const {
    TF_RETURN_IF_ERROR(output_indices->Init());
    TF_RETURN_IF_ERROR(output_values->Init());
    for (int slice_index = 0; slice_index < num_split; ++slice_index) {
      Index slice_nnz =
          slice_ends_host[slice_index] -
          (slice_index == 0 ? Index(0) : slice_ends_host[slice_index - 1]);
      Tensor* output_inds = nullptr;
      TF_RETURN_IF_ERROR(context->allocate_output(
          slice_index, {slice_nnz, rank}, &output_inds));
      output_indices->Set(slice_index, output_inds->matrix<Index>().data());
      Tensor* output_vals = nullptr;
      TF_RETURN_IF_ERROR(context->allocate_output(num_split + slice_index,
                                                  {slice_nnz}, &output_vals));
      output_values->Set(slice_index, output_vals->vec<T>().data());
      Tensor* output_shape = nullptr;
      TF_RETURN_IF_ERROR(context->allocate_output(num_split * 2 + slice_index,
                                                  {rank}, &output_shape));
      for (int dim = 0; dim < rank; ++dim) {
        output_shape->vec<int64_t>()(dim) =
            (dim == axis) ? slice_indexer.GetSliceSize(slice_index)
                          : dense_shape.dim_size(dim);
      }
    }
    TF_RETURN_IF_ERROR(output_indices->Finalize());
    TF_RETURN_IF_ERROR(output_values->Finalize());
    return OkStatus();
  }
};

#define DEFINE_SPARSE_SPLIT_FUNCTOR(T) \
  template struct SparseSplitFunctor<GPUDevice, T>;
TF_CALL_POD_TYPES(DEFINE_SPARSE_SPLIT_FUNCTOR);

#undef DEFINE_SPARSE_SPLIT_FUNCTOR

}  // namespace functor

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
