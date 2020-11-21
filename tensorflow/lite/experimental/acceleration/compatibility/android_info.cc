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
#include "tensorflow/lite/experimental/acceleration/compatibility/android_info.h"

#include <string>

#include "absl/status/status.h"

#ifdef __ANDROID__
#include <sys/system_properties.h>
#endif  // __ANDROID__

namespace {
std::string GetPropertyValue(const std::string& property) {
#ifdef __ANDROID__
  char value[PROP_VALUE_MAX];
  __system_property_get(property.c_str(), value);
  return std::string(value);
#else   // !__ANDROID__
  return std::string();
#endif  // __ANDROID__
}
}  // namespace

namespace tflite {
namespace acceleration {

absl::Status RequestAndroidInfo(AndroidInfo* info_out) {
  if (!info_out) {
    return absl::InvalidArgumentError("info_out may not be null");
  }
  info_out->android_sdk_version = GetPropertyValue("ro.build.version.sdk");
  info_out->device = GetPropertyValue("ro.product.device");
  info_out->model = GetPropertyValue("ro.product.model");
  info_out->manufacturer = GetPropertyValue("ro.product.manufacturer");
  return absl::OkStatus();
}

}  // namespace acceleration
}  // namespace tflite
