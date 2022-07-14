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

// The algorithm for dynamic partition has the following steps:
// 1. Let N be the size of partitions. We initialize a new vector indices_in
//    with the values 0, 1, 2, ..., N-1.
// 2. We apply gpuprim::DeviceRadixSort::SortPairs to the key - value pairs
//    given by partitions and indices_in. This will result in two new vectors
//    partitions_out and indices_out, with partitions_out sorted.
// 3. The first dimension of outputs[i] is equal to the number of i-values in
//    partitions_out. We determine it in two steps:
//    - apply gpuprim::DeviceReduce::ReduceByKey to count how many times each
//      value appears in partitions_out,
//    - move the results to partition_count. This handles missing values
//      (corresponding to empty parts).
// 4. Because partition_count is on the GPU, we bring it asynchronously to
//    the CPU. Then we can allocate the output tensors.
// 5. Finally, we use indices_out and the gather functor to collect the output.
//    This works, because for each interval of i-values, indices_out points
//    to the slices which should form output[i].

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_reference.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/gather_functor_gpu.cu.h"
#include "tensorflow/core/kernels/gpu_prim.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/transform_output_iterator.h"

#if GOOGLE_CUDA
#include "tensorflow/stream_executor/cuda/cuda_activation.h"
using stream_executor::cuda::ScopedActivateExecutorContext;
#elif TENSORFLOW_USE_ROCM
#include "tensorflow/core/platform/rocm.h"
using stream_executor::rocm::ScopedActivateExecutorContext;
#endif  // GOOGLE_CUDA

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace {

template <typename T>
__global__ void RangeInitKernel(const T start, const T delta, const int32 size,
                                T* out) {
  GPU_1D_KERNEL_LOOP(i, size) { out[i] = start + i * delta; }
}

__global__ void MoveValuesKernel(const int32* keys, const int32* values,
                                 const int32* size, int32 out_size,
                                 int32* out) {
  int32 N = min(ldg(size), out_size);
  GPU_1D_KERNEL_LOOP(i, N) {
    int32 key = ldg(keys + i);
    int32 value = ldg(values + i);
    if (FastBoundsCheck(key, out_size)) out[key] = value;
  }
}

// Initialize out with range start, start + delta, start + 2 * delta, ...
// This is needed because tf.range has no GPU implementation.
template <typename T>
void RangeInit(const GPUDevice& d, const T start, const T delta,
               const int32 size, typename TTypes<T>::Flat out) {
  GpuLaunchConfig config = GetGpuLaunchConfig(size, d);
  TF_CHECK_OK(GpuLaunchKernel(RangeInitKernel<T>, config.block_count,
                              config.thread_per_block, 0, d.stream(), start,
                              delta, size, out.data()));
}

// Given *num_runs pairs (key, value), this function moves the value
// corresponding to key i at position i in the array out.
void MoveValues(const GPUDevice& d, int32* keys, int32* values, int32* num_runs,
                int32 out_size, int32* out) {
  // Because num_runs is located on the GPU, we can not access it directly.
  // So we launch the kernel with size = out_size.
  // This is valid for correct inputs, because then out_size >= *num_runs.
  // For wrong inputs, we may have out_size < *num_runs. In this case we will
  // only handle the first out_size values.
  GpuLaunchConfig config = GetGpuLaunchConfig(out_size, d);
  TF_CHECK_OK(GpuLaunchKernel(MoveValuesKernel, config.block_count,
                              config.thread_per_block, 0, d.stream(), keys,
                              values, num_runs, out_size, out));
}

struct IdentityOp {
  __device__ int32 __forceinline__ operator()(const int32& a) const {
    return a;
  }
};

// Define an output iterator that only allows assignment to
// positions between [base, base + limit).
class BoundedOutputIterator
    : public TransformOutputIterator<int32, int32, IdentityOp> {
 private:
  int32 limit;
  int32* base;

  struct BoundedReference : Reference {
    int32 limit;
    int32* base;
    // Constructor
    __host__ __device__ __forceinline__
    BoundedReference(int32* __restrict__ ptr, int32* __restrict__ base,
                     IdentityOp op, int32 limit)
        : Reference(ptr, op), limit(limit), base(base) {}

    // Assignment
    __host__ __device__ __forceinline__ int32 operator=(int32 val) {
      if (ptr - base < limit && ptr - base >= 0) *ptr = val;
      return val;
    }
  };

 public:
  typedef BoundedOutputIterator self_type;
  typedef BoundedReference reference;

  __host__ __device__ __forceinline__
  BoundedOutputIterator(int32* __restrict__ ptr, IdentityOp op, int32 size)
      : TransformOutputIterator(ptr, op), limit(size), base(ptr) {}

  __host__ __device__ __forceinline__
  BoundedOutputIterator(int32* __restrict__ ptr, int32* __restrict__ base,
                        IdentityOp op, int32 size)
      : TransformOutputIterator(ptr, op), limit(size), base(base) {}

  // Indirection
  __host__ __device__ __forceinline__ reference operator*() const {
    return BoundedReference(ptr, base, conversion_op, limit);
  }

  // Array subscript
  __host__ __device__ __forceinline__ reference operator[](int32 n) const {
    return BoundedReference(ptr + n, base, conversion_op, limit);
  }

  // Addition
  __host__ __device__ __forceinline__ self_type operator+(int32 n) const {
    self_type retval(ptr + n, base, conversion_op, limit);
    return retval;
  }

  // Subtraction
  __host__ __device__ __forceinline__ self_type operator-(int32 n) const {
    self_type retval(ptr - n, base, conversion_op, limit);
    return retval;
  }
};

}  // namespace

// The current implementation has memory cost on GPU
// I + P + max(3N + R + P, O + N), where:
// I - the size of the input
// N - the size of the partitions tensor
// R - the temporary storage used by gpuprim::RadixSort, about 2N
// P - the number of partitions
// O - the size of the output
// So roughly the cost is I + P + max(5N, O + N).
template <typename T>
class DynamicPartitionOpGPU : public AsyncOpKernel {
 public:
  explicit DynamicPartitionOpGPU(OpKernelConstruction* c) : AsyncOpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("num_partitions", &num_partitions_));
    OP_REQUIRES(c, num_partitions_ >= 1,
                errors::InvalidArgument("num_partitions must be at least 1"));
  }

  void AllocateTempSpace(OpKernelContext* c, int32 N, Tensor* indices_in,
                         Tensor* partitions_out, Tensor* indices_out,
                         DoneCallback done) {
    int32 M = std::max(N, num_partitions_);
    // indices_in will be made slightly larger to accommodate
    // later computations.
    OP_REQUIRES_OK_ASYNC(
        c, c->allocate_temp(DT_INT32, TensorShape({M}), indices_in), done);
    OP_REQUIRES_OK_ASYNC(
        c, c->allocate_temp(DT_INT32, TensorShape({N}), partitions_out), done);
    OP_REQUIRES_OK_ASYNC(
        c, c->allocate_temp(DT_INT32, TensorShape({N}), indices_out), done);
  }

  void AllocateOutputs(OpKernelContext* c, const Tensor* data,
                       const Tensor* partitions, const Tensor* partition_count,
                       OpOutputList* Tout, DoneCallback done) {
    auto e_part_count = partition_count->flat<int32>();
    // Allocate output tensors of the right size
    OP_REQUIRES_OK_ASYNC(c, c->output_list("outputs", Tout), done);
    for (int p = 0; p < num_partitions_; p++) {
      TensorShape shape;
      shape.AddDim(e_part_count(p));
      for (int i = partitions->dims(); i < data->dims(); i++) {
        shape.AddDim(data->dim_size(i));
      }
      Tensor* out;
      OP_REQUIRES_OK_ASYNC(c, Tout->allocate(p, shape, &out), done);
    }
  }

  void ComputeAsync(OpKernelContext* c, DoneCallback done) {
    const Tensor& data = c->input(0);
    const Tensor& partitions = c->input(1);

    OP_REQUIRES_ASYNC(
        c, TensorShapeUtils::StartsWith(data.shape(), partitions.shape()),
        errors::InvalidArgument(
            "data.shape must start with partitions.shape, ",
            "got data.shape = ", data.shape().DebugString(),
            ", partitions.shape = ", partitions.shape().DebugString()),
        done);

    Tensor partition_count;

    // We must handle the case of empty partitions separately,
    // because kernels don't work with 0-sized tensors.
    if (partitions.NumElements() == 0) {
      AllocatorAttributes alloc_attr;
      alloc_attr.set_on_host(true);
      OP_REQUIRES_OK_ASYNC(
          c,
          c->allocate_temp(DT_INT32, TensorShape({num_partitions_}),
                           &partition_count, alloc_attr),
          done);
      auto e_part_count = partition_count.flat<int32>();
      for (int i = 0; i < num_partitions_; i++) e_part_count(i) = 0;
      OpOutputList outputs;
      this->AllocateOutputs(c, &data, &partitions, &partition_count, &outputs,
                            done);
      if (c->status().ok()) done();
      return;
    }

    // Prepare for counting.
    OP_REQUIRES_OK_ASYNC(
        c,
        c->allocate_temp(DT_INT32, TensorShape({num_partitions_}),
                         &partition_count),
        done);
    Tensor indices_out;
    // Count how many times each partition index occurs.
    // Also sort the info in partitions and output it in indices_out,
    // in preparation for the next step.
    this->CountAndSortParts(c, &partitions, &partition_count, &indices_out,
                            done);
    if (!c->status().ok()) return;

    // In order to allocate the output tensor we have to move partition_count
    // to CPU.
    auto* stream = c->op_device_context()->stream();
    OP_REQUIRES_ASYNC(c, stream, errors::Internal("No GPU stream available."),
                      done);
    Tensor cpu_tensor;
    AllocatorAttributes alloc_attr;
    alloc_attr.set_on_host(true);
    alloc_attr.set_gpu_compatible(true);
    OP_REQUIRES_OK_ASYNC(
        c,
        c->allocate_temp(partition_count.dtype(), partition_count.shape(),
                         &cpu_tensor, alloc_attr),
        done);
    se::DeviceMemoryBase wrapped(partition_count.flat<int32>().data(),
                                 num_partitions_ * sizeof(int32));
    const bool status =
        stream
            ->ThenMemcpy(cpu_tensor.flat<int32>().data(), wrapped,
                         num_partitions_ * sizeof(int32))
            .ok();
    OP_REQUIRES_ASYNC(
        c, status,
        errors::Internal("Failed to launch copy from device to host."), done);

    // Keep a reference to partition_count so that the buffer
    // is not deallocated at the end of the function, before
    // memcpy is completed.
    TensorReference partition_ref(partition_count);
    auto wrapped_callback = [this, c, &data, &partitions, indices_out,
                             partition_ref, cpu_tensor, done]() {
      auto stream = c->op_device_context()->stream();
      ScopedActivateExecutorContext scoped_activation{stream->parent()};

      OpOutputList outputs;
      this->AllocateOutputs(c, &data, &partitions, &cpu_tensor, &outputs, done);
      if (!c->status().ok()) {
        partition_ref.Unref();
        return;
      }
      int32 N = partitions.NumElements();
      int64 slice_size = data.NumElements() / N;
      this->GatherSlices(c, &data, &indices_out, N, slice_size, outputs);
      partition_ref.Unref();
      done();
    };

    c->device()->tensorflow_accelerator_device_info()->event_mgr->ThenExecute(
        stream, wrapped_callback);
  }

 protected:
  void RadixSort(OpKernelContext* c, const Tensor* partitions,
                 Tensor* indices_in, Tensor* partitions_out,
                 Tensor* indices_out, DoneCallback done) {
    int32 N = partitions->NumElements();
    const GPUDevice& device = c->eigen_device<GPUDevice>();
    const auto& cu_stream = GetGpuStream(c);

    // Initialize the indices_in tensor using the Range GPU kernel.
    RangeInit(device, 0, 1, N, indices_in->flat<int32>());
    // Obtain the pointers to inner buffers.
    const int32* partitions_ptr = partitions->flat<int32>().data();
    int32* partitions_out_ptr = partitions_out->flat<int32>().data();
    int32* indices_in_ptr = indices_in->flat<int32>().data();
    int32* indices_out_ptr = indices_out->flat<int32>().data();
    // Determine temporary device storage requirements.
    Tensor cub_temp_storage;
    size_t temp_storage_bytes = 0;
    gpuprim::DeviceRadixSort::SortPairs(
        NULL, temp_storage_bytes, partitions_ptr, partitions_out_ptr,
        indices_in_ptr, indices_out_ptr, N, 0, sizeof(int32) * 8, cu_stream);
    // Allocate temporary storage.
    OP_REQUIRES_OK_ASYNC(
        c,
        c->allocate_temp(
            DT_INT8, TensorShape({static_cast<int64_t>(temp_storage_bytes)}),
            &cub_temp_storage),
        done);
    // Radix-sort the partition information.
    gpuprim::DeviceRadixSort::SortPairs(
        cub_temp_storage.flat<int8>().data(), temp_storage_bytes,
        partitions_ptr, partitions_out_ptr, indices_in_ptr, indices_out_ptr, N,
        0, sizeof(int32) * 8, cu_stream);
  }  // At this point cub_temp_storage will be marked for deallocation.

  void CountAndSortParts(OpKernelContext* c, const Tensor* partitions,
                         Tensor* partition_count, Tensor* indices_out,
                         DoneCallback done) {
    const GPUDevice& device = c->eigen_device<GPUDevice>();
    const auto& cu_stream = GetGpuStream(c);
    int32 N = partitions->NumElements();
    Tensor indices_in;
    Tensor partitions_out;
    Tensor aggregates_out;

    // Allocate memory for Radix-Sort.
    this->AllocateTempSpace(c, N, &indices_in, &partitions_out, indices_out,
                            done);
    if (!c->status().ok()) return;
    this->RadixSort(c, partitions, &indices_in, &partitions_out, indices_out,
                    done);
    if (!c->status().ok()) return;
    // We will now apply a reduce operation to count how many times
    // each index appears in partitions.

    // Zero-out the partition_count tensor.
    functor::SetZeroFunctor<GPUDevice, int32> zero_functor;
    zero_functor(device, partition_count->flat<int32>());
    // Allocate memory for aggregates_out.
    OP_REQUIRES_OK_ASYNC(
        c,
        c->allocate_temp(DT_INT32, TensorShape({num_partitions_}),
                         &aggregates_out),
        done);
    // Obtain the pointers to inner buffers.
    int32* keys_in_ptr = partitions_out.flat<int32>().data();
    // Here we reuse the indices_in tensor for the unique keys output.
    int32* unique_out_ptr = indices_in.flat<int32>().data();
    int32* aggregates_out_ptr = aggregates_out.flat<int32>().data();
    // We wrap the pointers in bounded output iterators to guard against
    // wrong inputs (more than num_partitions distinct indices).
    IdentityOp id_op;
    BoundedOutputIterator unique_out_it(unique_out_ptr, id_op, num_partitions_);
    BoundedOutputIterator aggregates_out_it(aggregates_out_ptr, id_op,
                                            num_partitions_);

#if GOOGLE_CUDA
    cub::ConstantInputIterator<int32> values_in(1);
#elif TENSORFLOW_USE_ROCM
    using ConstantInputIterator =
        ::rocprim::constant_iterator<int32, ptrdiff_t>;
    ConstantInputIterator values_in(1);
#endif
    gpuprim::Sum reduction_op;

    // Allocate space on GPU for the number of runs. This is required by CUB.
    Tensor num_runs;
    OP_REQUIRES_OK_ASYNC(
        c, c->allocate_temp(DT_INT32, TensorShape({1}), &num_runs), done);
    int32* num_runs_ptr = num_runs.flat<int32>().data();

    // Determine temporary device storage requirements
    Tensor cub_temp_storage;
    size_t temp_storage_bytes = 0;
    gpuprim::DeviceReduce::ReduceByKey(
        NULL, temp_storage_bytes, keys_in_ptr, unique_out_it, values_in,
        aggregates_out_it, num_runs_ptr, reduction_op, N, cu_stream);
    // Allocate temporary storage.
    OP_REQUIRES_OK_ASYNC(
        c,
        c->allocate_temp(
            DT_INT8, TensorShape({static_cast<int64_t>(temp_storage_bytes)}),
            &cub_temp_storage),
        done);
    // Run reduce-by-key. The effect is that we count how many times
    // each index appears in partitions. The distinct indices are stored
    // in unique_out, while the count is stored in aggregates_out.
    // The total number of distinct indices is stored in num_runs.
    gpuprim::DeviceReduce::ReduceByKey(
        cub_temp_storage.flat<int8>().data(), temp_storage_bytes, keys_in_ptr,
        unique_out_it, values_in, aggregates_out_it, num_runs_ptr, reduction_op,
        N, cu_stream);
    // We are not done yet. unique_out only contains the indices that appeared
    // at least once in partitions. We move each value from aggregates_out
    // to the corresponding position in partition_count. This will handle
    // possibly empty parts.
    MoveValues(device, unique_out_ptr, aggregates_out_ptr, num_runs_ptr,
               num_partitions_, partition_count->flat<int32>().data());
  }  // At this point indices_in, partitions_out, aggregates_out
     // and cub_temp_storage will be marked for deallocation.

  void GatherSlices(OpKernelContext* c, const Tensor* data,
                    const Tensor* indices, int32 N, int64 slice_size,
                    OpOutputList& outs) {
    const GPUDevice& device = c->eigen_device<GPUDevice>();
    const int32* ind_base = indices->flat<int32>().data();
    const T* data_base = data->flat<T>().data();

    for (int p = 0; p < num_partitions_; p++) {
      int32 indices_size = outs[p]->dim_size(0);
      int64 out_size = outs[p]->NumElements();
      T* out_base = outs[p]->flat<T>().data();
      if (out_size > 0)
        TF_CHECK_OK(LaunchGatherKernel</*is_axis_zero = */ true>(
            device, data_base, ind_base, out_base, N, indices_size, slice_size,
            out_size));
      ind_base += indices_size;
    }
  }

  int32 num_partitions_;
};

#define REGISTER_DYNAMIC_PARTITION_GPU(T)                                 \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("DynamicPartition").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      DynamicPartitionOpGPU<T>)

TF_CALL_int32(REGISTER_DYNAMIC_PARTITION_GPU);
TF_CALL_int64(REGISTER_DYNAMIC_PARTITION_GPU);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_DYNAMIC_PARTITION_GPU);
TF_CALL_COMPLEX_TYPES(REGISTER_DYNAMIC_PARTITION_GPU);
#undef REGISTER_DYNAMIC_PARTITION_GPU

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
