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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_ID_MANAGER_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_ID_MANAGER_H_

#include "tensorflow/core/lib/core/status.h"
#include "tsl/framework/device_id.h"

namespace tensorflow {

// Class that maintains a map from TfDeviceId to PlatformDeviceId, and manages
// the translation between them.
class GpuIdManager {
 public:
  // Adds a mapping from tf_device_id to platform_device_id.
  static Status InsertTfPlatformDeviceIdPair(
      tsl::TfDeviceId tf_device_id, tsl::PlatformDeviceId platform_device_id);

  // Gets the platform_device_id associated with tf_device_id. Returns OK if
  // found.
  static Status TfToPlatformDeviceId(tsl::TfDeviceId tf_device_id,
                                     tsl::PlatformDeviceId* platform_device_id);

  // Clears the map. Used in unit tests only.
  static void TestOnlyReset();
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_ID_MANAGER_H_
