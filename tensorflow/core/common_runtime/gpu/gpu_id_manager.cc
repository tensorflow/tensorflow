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

#include "tensorflow/core/common_runtime/gpu/gpu_id_manager.h"

#include "tensorflow/core/common_runtime/device/device_id_manager.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {

Status GpuIdManager::InsertTfPlatformDeviceIdPair(
    TfDeviceId tf_device_id, PlatformDeviceId platform_device_id) {
  return DeviceIdManager::InsertTfPlatformDeviceIdPair(DEVICE_GPU, tf_device_id,
                                                       platform_device_id);
}

Status GpuIdManager::TfToPlatformDeviceId(
    TfDeviceId tf_device_id, PlatformDeviceId* platform_device_id) {
  return DeviceIdManager::TfToPlatformDeviceId(DEVICE_GPU, tf_device_id,
                                               platform_device_id);
}

void GpuIdManager::TestOnlyReset() { DeviceIdManager::TestOnlyReset(); }

}  // namespace tensorflow
