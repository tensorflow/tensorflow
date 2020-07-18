/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_STREAM_EXECUTOR_TPU_DEVICE_MEMORY_BASE_HELPER_H_
#define TENSORFLOW_STREAM_EXECUTOR_TPU_DEVICE_MEMORY_BASE_HELPER_H_

#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/tpu/tpu_executor_c_api.h"

class DeviceMemoryBaseHelper {
 public:
  static stream_executor::DeviceMemoryBase
  SE_DeviceMemoryBaseToDeviceMemoryBase(SE_DeviceMemoryBase se_base) {
    stream_executor::DeviceMemoryBase base(se_base.opaque, se_base.size);
    base.SetPayload(se_base.payload);
    return base;
  }

  static SE_DeviceMemoryBase DeviceMemoryBaseToSE_DeviceMemoryBase(
      const stream_executor::DeviceMemoryBase& base) {
    SE_DeviceMemoryBase se_base;
    se_base.opaque = const_cast<void*>(base.opaque());
    se_base.payload = base.payload();
    se_base.size = base.size();
    return se_base;
  }
};

#endif  // TENSORFLOW_STREAM_EXECUTOR_TPU_DEVICE_MEMORY_BASE_HELPER_H_
