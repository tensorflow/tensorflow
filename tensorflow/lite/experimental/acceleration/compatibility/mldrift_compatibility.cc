/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/experimental/acceleration/compatibility/mldrift_compatibility.h"

#include <cstddef>

#include "tensorflow/lite/delegates/gpu/common/gpu_info.h"
#include "tensorflow/lite/experimental/acceleration/compatibility/android_info.h"
#include "tensorflow/lite/experimental/acceleration/compatibility/gpu_compatibility.h"
#include "tensorflow/lite/experimental/acceleration/compatibility/mldrift_compatibility_binary.h"
#include "tensorflow/lite/logger.h"
#include "tensorflow/lite/minimal_logging.h"

#if defined(__ANDROID__)
#include <memory>
#include <thread>  // NOLINT: only used on Android, where std::thread is allowed

#include "tensorflow/lite/delegates/gpu/gl/egl_environment.h"
#endif  // defined(__ANDROID__)

namespace tflite {
namespace acceleration {

bool IsMlDriftGpuSupported(const unsigned char* compatibility_binary,
                           size_t compatibility_binary_len,
                           const AndroidInfo& android_info,
                           const ::tflite::gpu::GpuInfo& gpu_info) {
  if (!compatibility_binary || compatibility_binary_len == 0) {
    return false;
  }
  auto compatibility_list = GPUCompatibilityList::Create(
      compatibility_binary, compatibility_binary_len);
  if (!compatibility_list) {
    TFLITE_LOG_PROD(
        TFLITE_LOG_WARNING,
        "IsMlDriftGpuSupported: Failed to parse compatibility binary.");
    return false;
  }

  return compatibility_list->Includes(android_info, gpu_info);
}

bool IsMlDriftGpuSupported(const AndroidInfo& android_info,
                           const ::tflite::gpu::GpuInfo& gpu_info) {
  return IsMlDriftGpuSupported(
      g_tflite_acceleration_mldrift_compatibility_binary,
      g_tflite_acceleration_mldrift_compatibility_binary_len, android_info,
      gpu_info);
}

bool IsMlDriftGpuSupportedOnThisDevice() {
#if defined(__ANDROID__)
  static const bool is_supported = []() {
    AndroidInfo android_info;
    auto android_status = RequestAndroidInfo(&android_info);
    if (!android_status.ok()) {
      TFLITE_LOG_PROD(
          TFLITE_LOG_WARNING,
          "IsMlDriftGpuSupportedOnThisDevice: RequestAndroidInfo failed: %s",
          android_status.ToString().c_str());
      return false;
    }

    ::tflite::gpu::GpuInfo gpu_info;
    std::thread([&gpu_info]() {
      std::unique_ptr<::tflite::gpu::gl::EglEnvironment> env;
      auto egl_status =
          ::tflite::gpu::gl::EglEnvironment::NewEglEnvironment(&env);
      if (egl_status.ok()) {
        gpu_info = env->gpu_info();
      } else {
        TFLITE_LOG_PROD(
            TFLITE_LOG_WARNING,
            "IsMlDriftGpuSupportedOnThisDevice: NewEglEnvironment failed: %s",
            egl_status.ToString().c_str());
      }
    }).join();

    return IsMlDriftGpuSupported(android_info, gpu_info);
  }();
  return is_supported;
#else
  return true;
#endif  // defined(__ANDROID__)
}

}  // namespace acceleration
}  // namespace tflite
