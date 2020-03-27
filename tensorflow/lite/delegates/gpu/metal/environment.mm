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

#import <Metal/Metal.h>

#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/lite/delegates/gpu/metal/common.h"

namespace tflite {
namespace gpu {
namespace metal {

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

DeviceInfo::DeviceInfo(const std::string& device_name) : apple_info(device_name) {}

bool DeviceInfo::IsRoundToNearestSupported() const {
  return apple_info.IsRoundToNearestSupported();
}

int DeviceInfo::GetComputeUnitsCount() const {
  return apple_info.GetComputeUnitsCount();
}

GpuType GetGpuType() {
  int max_feature_set = 0;
#if defined(__IPHONE_9_0) && __IPHONE_OS_VERSION_MIN_REQUIRED >= __IPHONE_9_0
  std::vector<std::pair<MTLFeatureSet, int>> features;
  if (@available(iOS 8.0, *)) {
    features.emplace_back(MTLFeatureSet_iOS_GPUFamily1_v1, 7);
    features.emplace_back(MTLFeatureSet_iOS_GPUFamily2_v1, 8);
  }
  if (@available(iOS 9.0, *)) {
    features.emplace_back(MTLFeatureSet_iOS_GPUFamily1_v2, 7);
    features.emplace_back(MTLFeatureSet_iOS_GPUFamily2_v2, 8);
    features.emplace_back(MTLFeatureSet_iOS_GPUFamily3_v1, 9);
  }
  if (@available(iOS 10.0, *)) {
    features.emplace_back(MTLFeatureSet_iOS_GPUFamily1_v3, 7);
    features.emplace_back(MTLFeatureSet_iOS_GPUFamily2_v3, 8);
    features.emplace_back(MTLFeatureSet_iOS_GPUFamily3_v2, 9);
  }
  if (@available(iOS 11.0, *)) {
    features.emplace_back(MTLFeatureSet_iOS_GPUFamily2_v4, 8);
    features.emplace_back(MTLFeatureSet_iOS_GPUFamily3_v3, 9);
    features.emplace_back(MTLFeatureSet_iOS_GPUFamily4_v1, 11);
  }
  if (@available(iOS 12.0, *)) {
    features.emplace_back(MTLFeatureSet_iOS_GPUFamily1_v5, 7);
    features.emplace_back(MTLFeatureSet_iOS_GPUFamily2_v5, 8);
    features.emplace_back(MTLFeatureSet_iOS_GPUFamily3_v4, 9);
    features.emplace_back(MTLFeatureSet_iOS_GPUFamily4_v2, 11);
    features.emplace_back(MTLFeatureSet_iOS_GPUFamily5_v1, 12);
  }
  id<MTLDevice> device = GetBestSupportedMetalDevice();
  for (auto &type : features) {
    if ([device supportsFeatureSet:type.first]) {
      max_feature_set = std::max(max_feature_set, type.second);
    }
  }
#elif defined(__MAC_10_5) && __MAC_OS_X_VERSION_MIN_REQUIRED >= __MAC_10_5
  std::vector<std::pair<MTLFeatureSet, int>> features;
  if (@available(macOS 10.15, *)) {
    features.emplace_back(MTLFeatureSet_macOS_GPUFamily2_v1, 12);
  }
  id<MTLDevice> device = GetBestSupportedMetalDevice();
  for (auto &type : features) {
    if ([device supportsFeatureSet:type.first]) {
      max_feature_set = std::max(max_feature_set, type.second);
    }
  }
#endif
  switch (max_feature_set) {
    case 7:
      return GpuType::kA7;
    case 8:
      return GpuType::kA8;
    case 9:
      return GpuType::kA9;
    case 10:
      return GpuType::kA10;
    case 11:
      return GpuType::kA11;
    case 12:
      return GpuType::kA12;
    default:
      return GpuType::kUnknown;
  };
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
