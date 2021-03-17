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

#include <iostream>
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
#ifdef __ANDROID__
  // Based on
  // https://github.com/flutter/plugins/blob/master/packages/device_info/device_info/android/src/main/java/io/flutter/plugins/deviceinfo/MethodCallHandlerImpl.java
  // + QUMA detection (system properties return empty) and qemu detection
  // (ro.kernel.qemu).
  std::string brand = GetPropertyValue("ro.product.brand");
  const std::string& device = info_out->device;
  std::string fingerprint = GetPropertyValue("ro.build.fingerprint");
  std::string hardware = GetPropertyValue("ro.hardware");
  const std::string& model = info_out->model;
  const std::string& manufacturer = info_out->manufacturer;
  std::string product = GetPropertyValue("ro.build.product");
  std::string ro_kernel_qemu = GetPropertyValue("ro.kernel.qemu");
  info_out->is_emulator =
      ((brand.find("generic") == 0 && device.find("generic") == 0) ||  // NOLINT
       fingerprint.find("generic") == 0 ||                             // NOLINT
       fingerprint.find("unknown") == 0 ||                             // NOLINT
       hardware.find("goldfish") != std::string::npos ||               // NOLINT
       hardware.find("ranchu") != std::string::npos ||                 // NOLINT
       model.find("google_sdk") != std::string::npos ||                // NOLINT
       model.find("Emulator") != std::string::npos ||                  // NOLINT
       model.find("Android SDK built for x86") !=                      // NOLINT
           std::string::npos ||                                        // NOLINT
       manufacturer.find("Genymotion") != std::string::npos ||         // NOLINT
       product.find("sdk_google") != std::string::npos ||              // NOLINT
       product.find("google_sdk") != std::string::npos ||              // NOLINT
       product.find("sdk") != std::string::npos ||                     // NOLINT
       product.find("sdk_x86") != std::string::npos ||                 // NOLINT
       product.find("vbox86p") != std::string::npos ||                 // NOLINT
       product.find("emulator") != std::string::npos ||                // NOLINT
       product.find("simulator") != std::string::npos ||               // NOLINT
       ro_kernel_qemu == "1" ||                                        // NOLINT
       info_out->android_sdk_version.empty());                         // NOLINT
#else
  info_out->is_emulator = false;
#endif

  return absl::OkStatus();
}

}  // namespace acceleration
}  // namespace tflite
