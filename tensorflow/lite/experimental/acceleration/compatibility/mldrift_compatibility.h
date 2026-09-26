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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_COMPATIBILITY_MLDRIFT_COMPATIBILITY_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_COMPATIBILITY_MLDRIFT_COMPATIBILITY_H_

#include <cstddef>

#include "tensorflow/lite/delegates/gpu/common/gpu_info.h"
#include "tensorflow/lite/experimental/acceleration/compatibility/android_info.h"

namespace tflite {
namespace acceleration {

// Returns true if the specified device and GPU are supported for ML Drift GPU
// acceleration in the given compatibility binary.
bool IsMlDriftGpuSupported(const unsigned char* compatibility_binary,
                           size_t compatibility_binary_len,
                           const AndroidInfo& android_info,
                           const ::tflite::gpu::GpuInfo& gpu_info);

// Returns true if the specified device and GPU are supported for ML Drift GPU
// acceleration.
bool IsMlDriftGpuSupported(const AndroidInfo& android_info,
                           const ::tflite::gpu::GpuInfo& gpu_info);

// Returns true if the current device is supported for ML Drift GPU
// acceleration.
bool IsMlDriftGpuSupportedOnThisDevice();

}  // namespace acceleration
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_COMPATIBILITY_MLDRIFT_COMPATIBILITY_H_
