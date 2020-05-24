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

#include "tensorflow/lite/delegates/gpu/metal/environment.h"

#include <map>
#include <string>

namespace tflite {
namespace gpu {
namespace metal {
namespace {
Vendor GetVendorFromString(const std::string& device_name) {
  const std::map<std::string, Vendor> kMapping = {
    {"Apple", Vendor::kApple},
    {"Intel", Vendor::kIntel},
    {"AMD", Vendor::kAMD},
  };
  for (auto v : kMapping) {
    if (device_name.find(v.first) != std::string::npos) {
      return v.second;
    }
  }
  return Vendor::kUnknown;
}
}  // namespace

AppleGPUInfo::AppleGPUInfo(const std::string& device_name) {
  const std::map<std::string, AppleGPU> kMapping = {
    {"Apple A7 GPU", AppleGPU::kA7},
    {"Apple A8 GPU", AppleGPU::kA8},
    {"Apple A8X GPU", AppleGPU::kA8X},
    {"Apple A9 GPU", AppleGPU::kA9},
    {"Apple A9X GPU", AppleGPU::kA9X},
    {"Apple A10 GPU", AppleGPU::kA10},
    {"Apple A10X GPU", AppleGPU::kA10X},
    {"Apple A11 GPU", AppleGPU::kA11},
    {"Apple A12 GPU", AppleGPU::kA12},
    {"Apple A12X GPU", AppleGPU::kA12X},
    {"Apple A12Z GPU", AppleGPU::kA12Z},
    {"Apple A13 GPU", AppleGPU::kA13},
  };
  auto it = kMapping.find(device_name);
  if (it != kMapping.end()) {
    gpu_type = it->second;
  } else {
    gpu_type = AppleGPU::kUnknown;
  }
}

bool AppleGPUInfo::IsLocalMemoryPreferredOverGlobal() const {
  return gpu_type == AppleGPU::kA7 ||
         gpu_type == AppleGPU::kA8 ||
         gpu_type == AppleGPU::kA8X;
}

bool AppleGPUInfo::IsBionic() const {
  return gpu_type == AppleGPU::kA11 ||
         gpu_type == AppleGPU::kA12 ||
         gpu_type == AppleGPU::kA12X ||
         gpu_type == AppleGPU::kA12Z ||
         gpu_type == AppleGPU::kA13;
}

bool AppleGPUInfo::IsRoundToNearestSupported() const {
  return IsBionic();
}

bool AppleGPUInfo::IsWaveSizeEqualTo32() const {
  return true;
}

int AppleGPUInfo::GetComputeUnitsCount() const {
  switch (gpu_type) {
    case AppleGPU::kA7:
      return 4;
    case AppleGPU::kA8:
      return 4;
    case AppleGPU::kA8X:
      return 8;
    case AppleGPU::kA9:
      return 6;
    case AppleGPU::kA9X:
      return 12;
    case AppleGPU::kA10:
      return 6;
    case AppleGPU::kA10X:
      return 12;
    case AppleGPU::kA11:
      return 3;
    case AppleGPU::kA12:
      return 4;
    case AppleGPU::kA12X:
      return 7;
    case AppleGPU::kA12Z:
      return 8;
    case AppleGPU::kA13:
      return 4;
    case AppleGPU::kUnknown:
      return 1;
  }
}

DeviceInfo::DeviceInfo(const std::string& device_name) : vendor(GetVendorFromString(device_name)) {
  if (vendor == Vendor::kApple) {
    apple_info = AppleGPUInfo(device_name);
  }
}

bool DeviceInfo::IsIntelGPU() const {
  return vendor == Vendor::kIntel;
}

bool DeviceInfo::IsAppleGPU() const {
  return vendor == Vendor::kApple;
}

bool DeviceInfo::IsAMDGPU() const {
  return vendor == Vendor::kAMD;
}

bool DeviceInfo::IsRoundToNearestSupported() const {
  if (vendor == Vendor::kApple) {
    return apple_info.IsRoundToNearestSupported();
  } else {
    return true;
  }
}

bool DeviceInfo::IsWaveSizeEqualTo32() const {
  if (vendor == Vendor::kApple) {
    return apple_info.IsWaveSizeEqualTo32();
  } else {
    return false;
  }
}

int DeviceInfo::GetComputeUnitsCount() const {
  if (vendor == Vendor::kApple) {
    return apple_info.GetComputeUnitsCount();
  } else {
    return 1;
  }
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
