/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_METAL_ENVIRONMENT_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_METAL_ENVIRONMENT_H_

#include <string>

namespace tflite {
namespace gpu {
namespace metal {

enum class Vendor {
  kUnknown,
  kApple,
  kIntel,
  kAMD,
};

enum class AppleGPU {
  kUnknown,
  kA7,
  kA8,
  kA8X,
  kA9,
  kA9X,
  kA10,
  kA10X,
  kA11,
  kA12,
  kA12X,
  kA12Z,
  kA13,
};

struct AppleGPUInfo {
  AppleGPUInfo() = default;
  explicit AppleGPUInfo(const std::string& device_name);
  AppleGPU gpu_type;

  bool IsLocalMemoryPreferredOverGlobal() const;

  bool IsBionic() const;

  // floating point rounding mode
  bool IsRoundToNearestSupported() const;

  // returns true if device have fixed wave size equal to 32
  bool IsWaveSizeEqualTo32() const;

  int GetComputeUnitsCount() const;
};

struct DeviceInfo {
  DeviceInfo() = default;
  explicit DeviceInfo(const std::string& device_name);

  Vendor vendor;

  AppleGPUInfo apple_info;

  bool IsIntelGPU() const;
  bool IsAppleGPU() const;
  bool IsAMDGPU() const;

  // floating point rounding mode
  bool IsRoundToNearestSupported() const;

  // returns true if device have fixed wave size equal to 32
  bool IsWaveSizeEqualTo32() const;

  int GetComputeUnitsCount() const;
};

}  // namespace metal
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_METAL_ENVIRONMENT_H_
