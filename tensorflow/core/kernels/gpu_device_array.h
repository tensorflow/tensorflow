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
#ifndef TENSORFLOW_CORE_KERNELS_GPU_DEVICE_ARRAY_H_
#define TENSORFLOW_CORE_KERNELS_GPU_DEVICE_ARRAY_H_

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)

#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_reference.h"
#include "tensorflow/core/kernels/gpu_device_array_gpu.h"

namespace tensorflow {

// Create an array of value on the host, to be sent to kernel using
// GpuDeviceArrayStruct.
//
// Usage:
//   int size = ...;
//   GpuDeviceArrayOnHost ptrs(context, size);
//   OP_REQUIRES_OK(ptrs.Init());
//   for (int i = 0; i < size; ++i) {
//     ptrs.Set(i, ...);
//   }
//   OP_REQUIRES_OK(ptrs.Finalize());
//   launchKernel(..., ptrs.data, ...);
//
// ValueType must be memcopyable.
template <typename ValueType, int MaxInlineValues = 8>
class GpuDeviceArrayOnHost {
 public:
  GpuDeviceArrayOnHost(OpKernelContext* context, int32_t size)
      : context_(context),
        total_bytes_(static_cast<int64_t>(size) * sizeof(ValueType)) {
    data_.size = size;
  }

  Status Init() {
    if (inlined()) {
      values_ = data_.inline_values;
      return OkStatus();
    }

    // Out-of-line: allocate data that will be memcopied.
    AllocatorAttributes attr;
    attr.set_on_host(true);
    attr.set_gpu_compatible(true);
    TF_RETURN_IF_ERROR(
        context_->allocate_temp(DT_INT8, TensorShape{total_bytes_},
                                &out_of_line_values_on_host_, attr));
    values_ = reinterpret_cast<ValueType*>(
        out_of_line_values_on_host_.flat<int8>().data());
    return OkStatus();
  }

  void Set(int index, ValueType val) {
    DCHECK(values_);  // ensure Init was called.
    DCHECK_LT(index, data_.size);
    *(values_ + index) = val;
  }

  Status Finalize() {
    if (inlined()) {
      return OkStatus();
    }

    // Out-of-line - copy pointers to device.
    auto stream = context_->op_device_context()->stream();
    TensorReference tensor_ref(out_of_line_values_on_host_);
    TF_RETURN_IF_ERROR(context_->allocate_temp(
        DT_INT8, TensorShape{total_bytes_}, &out_of_line_values_on_gpu_));
    se::DeviceMemoryBase output_values_base{
        out_of_line_values_on_gpu_.flat<int8>().data(),
        static_cast<uint64>(total_bytes_)};
    stream->ThenMemcpy(&output_values_base,
                       out_of_line_values_on_host_.flat<int8>().data(),
                       total_bytes_);
    context_->device()
        ->tensorflow_accelerator_device_info()
        ->event_mgr->ThenExecute(stream,
                                 [tensor_ref]() { tensor_ref.Unref(); });
    data_.out_of_line_values = reinterpret_cast<ValueType*>(
        out_of_line_values_on_gpu_.flat<int8>().data());
    return OkStatus();
  }

  const GpuDeviceArrayStruct<ValueType, MaxInlineValues>& data() const {
    // Ensure Finalize is called.
    DCHECK(inlined() || out_of_line_values_on_gpu_.IsInitialized());
    return data_;
  }

 private:
  bool inlined() const { return data_.size <= MaxInlineValues; }

  OpKernelContext* const context_;
  const int64_t total_bytes_;  // total size of all pointers.
  ValueType* values_ = nullptr;
  GpuDeviceArrayStruct<ValueType, MaxInlineValues> data_;

  Tensor out_of_line_values_on_host_;
  Tensor out_of_line_values_on_gpu_;

  GpuDeviceArrayOnHost(const GpuDeviceArrayOnHost&) = delete;
  void operator=(const GpuDeviceArrayOnHost&) = delete;
};

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#endif  // TENSORFLOW_CORE_KERNELS_GPU_DEVICE_ARRAY_H_
