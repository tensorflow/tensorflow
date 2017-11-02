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
// 2. We apply cub::DeviceRadixSort::SortPairs to the key - value pairs given
//    by partitions and indices_in. This will result in two new vectors
//    partitions_out and indices_out, with partitions_out sorted.
// 3. The first dimension of outputs[i] is equal to the length of the interval
//    of i-values in partitions_out. We determine it in two steps:
//    - compute the starting and ending point of each interval,
//    - subtract the starting and ending points to find the length.
//    The result is placed in partition_count.
// 4. Because partition_count is on the GPU, we bring it asynchronously to
//    the CPU. Then we can allocate the output tensors.
// 5. Finally, we use indices_out and the gather functor to collect the output.
//    This works, because for each interval of i-values, indices_out points
//    to the slices which should form output[i].

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "external/cub_archive/cub/device/device_radix_sort.cuh"
#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/gather_functor_gpu.cu.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace {

template <typename T>
__global__ void RangeInitKernel(const T start, const T delta, const int32 size,
                                T* out) {
  CUDA_1D_KERNEL_LOOP(i, size) { out[i] = start + i * delta; }
}

__global__ void FindEndpointsKernel(const int32* partitions, int32 size,
                                    int32 nump, int32* start, int32* end) {
  CUDA_1D_KERNEL_LOOP(i, size) {
    int32 current = ldg(partitions + i);
    if (FastBoundsCheck(current, nump)) {
      if (i == 0)
        start[current] = i;
      else {
        int32 before = ldg(partitions + i - 1);
        if (before != current) start[current] = i;
      }
      if (i == size - 1)
        end[current] = i + 1;
      else {
        int32 after = ldg(partitions + i + 1);
        if (after != current) end[current] = i + 1;
      }
    }
  }
}

// We create a local version of subtract, because the tf.subtract kernel
// is not defined for int32. We use it to compute the length of an interval
// by subtracting the endpoints.
__global__ void IntervalLengthKernel(int32* start, int32 size, int32* end) {
  CUDA_1D_KERNEL_LOOP(i, size) {
    int32 start_point = ldg(start + i);
    end[i] = end[i] - start_point;
  }
}

// Initialize out with range start, start + delta, start + 2 * delta, ...
// This is needed because tf.range has no GPU implementation.
template <typename T>
void RangeInit(const GPUDevice& d, const T start, const T delta,
               const int32 size, typename TTypes<T>::Flat out) {
  CudaLaunchConfig config = GetCudaLaunchConfig(size, d);
  RangeInitKernel<
      T><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
      start, delta, size, out.data());
}

// Partitions is a sorted vector of N non-negative integer numbers.
// This function computes the starting and ending points of each interval
// of values.
void ComputeIntervals(const GPUDevice& d, Tensor* partitions, int32 N,
                      int32 nump, int32* start_ptr, int32* end_ptr) {
  CudaLaunchConfig config = GetCudaLaunchConfig(N, d);
  FindEndpointsKernel<<<config.block_count, config.thread_per_block, 0,
                        d.stream()>>>(partitions->flat<int32>().data(), N, nump,
                                      start_ptr, end_ptr);
}

// Subtract the ending points of each interval to obtain the interval length.
void ComputeItvLength(const GPUDevice& d, int32 num, int32* start_ptr,
                      int32* end_ptr) {
  CudaLaunchConfig config = GetCudaLaunchConfig(num, d);
  IntervalLengthKernel<<<config.block_count, config.thread_per_block, 0,
                         d.stream()>>>(start_ptr, num, end_ptr);
}

template <typename T>
void CallGatherKernel(const GPUDevice& d, const T* params, const int32* indices,
                      T* out, int64 gather_dim_size, int64 indices_size,
                      int64 slice_size, int64 out_size) {
  CudaLaunchConfig config = GetCudaLaunchConfig(out_size, d);
  GatherOpKernel<
      T, int32,
      true><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
      params, indices, out, gather_dim_size, indices_size, slice_size,
      out_size);
}

}  // namespace

// The current implementation has memory cost on GPU
// I + P + max(3N + R, O + N), where:
// I - the size of the input
// N - the size of the partitions tensor
// R - the temporary storage used by cub::RadixSort, about 2N
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
    // indices_in will be made slightly larger to accomodate
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
        errors::InvalidArgument("data.shape must start with partitions.shape, ",
                                "got data.shape = ", data.shape().DebugString(),
                                ", partitions.shape = ",
                                partitions.shape().DebugString()),
        done);

    Tensor partition_count;

    // We must handle the case of empty partitions separately,
    // because kernels don't work with 0-sized tensors.
    if (partitions.NumElements() == 0) {
      AllocatorAttributes alloc_attr;
      alloc_attr.set_on_host(true);
      OP_REQUIRES_OK_ASYNC(
          c, c->allocate_temp(DT_INT32, TensorShape({num_partitions_}),
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
        c, c->allocate_temp(DT_INT32, TensorShape({num_partitions_}),
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
        c, c->allocate_temp(partition_count.dtype(), partition_count.shape(),
                            &cpu_tensor, alloc_attr),
        done);
    perftools::gputools::DeviceMemoryBase wrapped(
        partition_count.flat<int32>().data(), num_partitions_ * sizeof(int32));
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

    c->device()->tensorflow_gpu_device_info()->event_mgr->ThenExecute(
        stream, wrapped_callback);
  }

 protected:
  void RadixSort(OpKernelContext* c, const Tensor* partitions,
                 Tensor* indices_in, Tensor* partitions_out,
                 Tensor* indices_out, DoneCallback done) {
    int32 N = partitions->NumElements();
    const GPUDevice& device = c->eigen_device<GPUDevice>();
    const cudaStream_t& cu_stream = GetCudaStream(c);

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
    cub::DeviceRadixSort::SortPairs(
        NULL, temp_storage_bytes, partitions_ptr, partitions_out_ptr,
        indices_in_ptr, indices_out_ptr, N, 0, sizeof(int32) * 8, cu_stream);
    // Allocate temporary storage.
    OP_REQUIRES_OK_ASYNC(
        c, c->allocate_temp(
               DT_INT8, TensorShape({static_cast<int64>(temp_storage_bytes)}),
               &cub_temp_storage),
        done);
    // Radix-sort the partition information.
    cub::DeviceRadixSort::SortPairs(
        cub_temp_storage.flat<int8>().data(), temp_storage_bytes,
        partitions_ptr, partitions_out_ptr, indices_in_ptr, indices_out_ptr, N,
        0, sizeof(int32) * 8, cu_stream);
  }  // At this point cub_temp_storage will be marked for deallocation.

  void CountAndSortParts(OpKernelContext* c, const Tensor* partitions,
                         Tensor* partition_count, Tensor* indices_out,
                         DoneCallback done) {
    const GPUDevice& device = c->eigen_device<GPUDevice>();
    int32 N = partitions->NumElements();
    Tensor indices_in;
    Tensor partitions_out;

    // Allocate memory for Radix-Sort.
    this->AllocateTempSpace(c, N, &indices_in, &partitions_out, indices_out,
                            done);
    if (!c->status().ok()) return;
    this->RadixSort(c, partitions, &indices_in, &partitions_out, indices_out,
                    done);
    if (!c->status().ok()) return;
    // We still need a little bit of additional memory. However,
    // we can reuse the indices_in tensor. We could also use atomic
    // operations and no additional memory, but this approach seems faster.

    // Zero-out the allocated memory.
    functor::SetZeroFunctor<GPUDevice, int32> zero_functor;
    zero_functor(device, partition_count->flat<int32>());
    zero_functor(device, indices_in.flat<int32>());
    // Obtain the pointers to inner buffers.
    int32* start_ptr = indices_in.flat<int32>().data();
    int32* end_ptr = partition_count->flat<int32>().data();
    // Obtain the starting and ending points of each interval.
    ComputeIntervals(device, &partitions_out, N, num_partitions_, start_ptr,
                     end_ptr);
    // Subtract to compute the number of appearances of each id.
    ComputeItvLength(device, num_partitions_, start_ptr, end_ptr);
  }  // At this point indices_in and partitions_out will be marked
     // for deallocation.

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
        CallGatherKernel<T>(device, data_base, ind_base, out_base, N,
                            indices_size, slice_size, out_size);
      ind_base += indices_size;
    }
  }

  int num_partitions_;
};

#define REGISTER_DYNAMIC_PARTITION_GPU(T)                                 \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("DynamicPartition").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      DynamicPartitionOpGPU<T>)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_DYNAMIC_PARTITION_GPU);
TF_CALL_complex64(REGISTER_DYNAMIC_PARTITION_GPU);
TF_CALL_complex128(REGISTER_DYNAMIC_PARTITION_GPU);
#undef REGISTER_DYNAMIC_PARTITION_GPU

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
