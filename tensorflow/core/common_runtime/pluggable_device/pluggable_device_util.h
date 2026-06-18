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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_PLUGGABLE_DEVICE_PLUGGABLE_DEVICE_UTIL_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_PLUGGABLE_DEVICE_PLUGGABLE_DEVICE_UTIL_H_

#include "absl/status/status.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/stream_executor.h"

namespace tensorflow {

class RecvTensorResponse;
class TensorProto;

class PluggableDeviceUtil {
 public:
  // Copies the data in 'device_tensor' into 'cpu_tensor'.
  // 'device_tensor''s backing memory must be on 'device' and
  // 'cpu_tensor' must be allocated to be of the same size as
  // 'device_tensor'. Synchronous: may block.
  static void CopyPluggableDeviceTensorToCPU(
      Device* device, const DeviceContext* device_context,
      const Tensor* device_tensor, Tensor* cpu_tensor, StatusCallback done);
  // Blocks until all operations queued on the stream associated with
  // 'device' at the time of the call have completed. Returns any
  // error pending on the stream at completion.
  static absl::Status Sync(Device* device);

  // Blocks until all operations queued on all streams associated with the
  // corresponding 'device' at the time of call have completed.
  // Returns any error pending on the stream at completion.
  static absl::Status SyncAll(Device* device);

  static void CopyCPUTensorToPluggableDevice(
      const Tensor* cpu_tensor, const DeviceContext* device_context,
      Device* device, Tensor* device_tensor, StatusCallback done,
      bool sync_dst_compute);

  static void DeviceToDeviceCopy(
      DeviceContext* send_dev_context, DeviceContext* recv_dev_context,
      Device* src, Device* dst, AllocatorAttributes src_alloc_attr,
      AllocatorAttributes dst_alloc_attr, const Tensor* input, Tensor* output,
      int dev_to_dev_stream_index, StatusCallback done);

  // Deep-copying of PluggableDevice tensor on the same device.
  // 'src_device_tensor''s and 'dst_device_tensor''s backing memory must be on
  // 'device' and 'dst_cpu_tensor' must be allocated to be of the same
  // size as 'src_device_tensor'.
  static void CopyPluggableDeviceTensorToSameDevice(
      Device* device, const DeviceContext* device_context,
      const Tensor* src_device_tensor, Tensor* dst_device_tensor,
      StatusCallback done);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_PLUGGABLE_DEVICE_PLUGGABLE_DEVICE_UTIL_H_
