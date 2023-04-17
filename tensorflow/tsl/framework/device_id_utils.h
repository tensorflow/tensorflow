/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_TSL_FRAMEWORK_DEVICE_ID_UTILS_H_
#define TENSORFLOW_TSL_FRAMEWORK_DEVICE_ID_UTILS_H_

#include <string>
#include <vector>

#include "tensorflow/tsl/framework/device_id.h"
#include "tensorflow/tsl/framework/device_type.h"
#include "tensorflow/tsl/platform/status.h"

namespace tsl {

// Utility methods for translation between TensorFlow device ids and platform
// device ids.

// Verify that the platform_device_id associated with a TfDeviceId is
// legitimate.
void CheckValidTfDeviceId(const DeviceType& type, int visible_device_count,
                          TfDeviceId tf_device_id);

// Parse `visible_device_list` into a list of platform Device ids.
Status ParseVisibleDeviceList(
    const std::string& visible_device_list, int visible_device_count,
    std::vector<PlatformDeviceId>* visible_device_order);

}  // namespace tsl

#endif  // TENSORFLOW_TSL_FRAMEWORK_DEVICE_ID_UTILS_H_
