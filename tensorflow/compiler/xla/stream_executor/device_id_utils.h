
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

#ifndef TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_DEVICE_ID_UTILS_H_
#define TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_DEVICE_ID_UTILS_H_

#include "tensorflow/compiler/xla/stream_executor/platform.h"
#include "tensorflow/compiler/xla/stream_executor/stream_executor.h"
#include "tensorflow/tsl/framework/device_id.h"
#include "tensorflow/tsl/framework/device_id_manager.h"

namespace stream_executor {

// Utility methods for getting the associated executor given a TfDeviceId
// or PlatformDeviceId.
class DeviceIdUtil {
 public:
  static tsl::StatusOr<StreamExecutor*> ExecutorForPlatformDeviceId(
      Platform* device_manager, tsl::PlatformDeviceId platform_device_id) {
    return device_manager->ExecutorForDevice(platform_device_id.value());
  }
  static tsl::StatusOr<StreamExecutor*> ExecutorForTfDeviceId(
      const tsl::DeviceType& type, Platform* device_manager,
      tsl::TfDeviceId tf_device_id) {
    tsl::PlatformDeviceId platform_device_id;
    TF_RETURN_IF_ERROR(tsl::DeviceIdManager::TfToPlatformDeviceId(
        type, tf_device_id, &platform_device_id));
    return ExecutorForPlatformDeviceId(device_manager, platform_device_id);
  }
};

}  // namespace stream_executor

#endif  // TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_DEVICE_ID_UTILS_H_
