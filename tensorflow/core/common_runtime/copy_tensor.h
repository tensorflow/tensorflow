/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_COPY_TENSOR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_COPY_TENSOR_H_

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class CopyTensor {
 public:
  typedef void (*CopyFunction)(
      DeviceContext* send_dev_context, DeviceContext* recv_dev_context,
      Device* src, Device* dst, const AllocatorAttributes src_alloc_attr,
      const AllocatorAttributes dst_alloc_attr, const Tensor* input,
      Tensor* output, int dev_to_dev_stream_index, StatusCallback done);

  // Copies "input" to "output" between devices accessible to the
  // local process via some DMA-like method.  "edge_name" is the name
  // of the tensor being copied, for debugging purposes. Depending on
  // the type of devices and memory in use, the copy may be performed
  // synchronously or asynchronously.  'done' will be invoked only
  // after the copy is actually complete.
  static void ViaDMA(StringPiece edge_name, DeviceContext* send_dev_context,
                     DeviceContext* recv_dev_context, Device* src, Device* dst,
                     const AllocatorAttributes src_alloc_attr,
                     const AllocatorAttributes dst_alloc_attr,
                     const Tensor* input, Tensor* output,
                     int dev_to_dev_stream_index, StatusCallback done,
                     bool sync_dst_compute = true);

  // Object used to call Register() at static-initialization time.
  // Note: This should only ever be used as a global-static object; no stack
  // or heap instances.
  class Registration {
   public:
    Registration(DeviceType sender_device_type, DeviceType receiver_device_type,
                 CopyFunction copy_function) {
      TF_QCHECK_OK(Register(sender_device_type, receiver_device_type,
                            copy_function, /*is_pluggable_device=*/false));
    }
  };

  // Register a function for copying between two specific DeviceTypes.
  // Note: This should only be called via the constructor of
  // CopyTensor::Registration or from PluggableDevice implementation.
  static absl::Status Register(DeviceType sender_device_type,
                               DeviceType receiver_device_type,
                               CopyFunction copy_function,
                               bool is_pluggable_device);
};

void CopyDeviceToHost(const Tensor* input, Allocator* cpu_allocator,
                      Allocator* out_allocator, StringPiece edge_name,
                      Device* src, Tensor* output,
                      DeviceContext* send_dev_context, StatusCallback done);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_COPY_TENSOR_H_
