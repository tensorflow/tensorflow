/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/metal/metal_device.h"

#import <sys/utsname.h>

#include <string>

namespace tflite {
namespace gpu {
namespace metal {
namespace {
GpuInfo CreateGpuInfoFromMetalDevice(id<MTLDevice> device) {
  std::string device_name = std::string([[device name] UTF8String]);
  GpuInfo gpu_info;
  GetGpuInfoFromDeviceDescription(device_name, GpuApi::kMetal, &gpu_info);

  if (gpu_info.apple_info.gpu_type == AppleGpu::kA15) {
    struct utsname system_info;
    uname(&system_info);
    const std::string gadget_name(system_info.machine);
    // iPhone 13 mini(iPhone14,4) and iPhone 13(iPhone14,5) have A15 with 4 core
    // GPU.
    // In general A15 GPU has 5 cores.
    if (gadget_name == "iPhone14,4" || gadget_name == "iPhone14,5") {
      gpu_info.apple_info.SetComputeUnits(4);
    }
  }

  const bool family_apple1_or_2 =
      gpu_info.IsApple() &&
      gpu_info.apple_info.IsFamilyOrLower(AppleInfo::Family::kApple2);
  gpu_info.metal_info.image2d_max_width =
      family_apple1_or_2 ? 1024 * 8 : 1024 * 16;
  gpu_info.metal_info.image2d_max_height =
      family_apple1_or_2 ? 1024 * 8 : 1024 * 16;
  gpu_info.metal_info.image_array_max_layers = 2048;
  gpu_info.metal_info.image3d_max_width = 2048;
  gpu_info.metal_info.image3d_max_height = 2048;
  gpu_info.metal_info.image3d_max_depth = 2048;

  if (@available(macOS 10.11, iOS 9.0, tvOS 9.0, *)) {
    MTLSize threadsPerGroup = [device maxThreadsPerThreadgroup];
    gpu_info.metal_info.max_work_group_size_x = threadsPerGroup.width;
    gpu_info.metal_info.max_work_group_size_y = threadsPerGroup.height;
    gpu_info.metal_info.max_work_group_size_z = threadsPerGroup.depth;
  } else {
    gpu_info.metal_info.max_work_group_size_x = 256;
    gpu_info.metal_info.max_work_group_size_y = 256;
    gpu_info.metal_info.max_work_group_size_z = 64;
  }

  if (@available(macOS 10.14, iOS 12.0, tvOS 12.0, *)) {
    gpu_info.metal_info.buffer_max_size = [device maxBufferLength];
  } else {
    // 256 MB
    gpu_info.metal_info.buffer_max_size = 256 * 1024 * 1024;
  }

  if (@available(macOS 11.0, iOS 14.0, tvOS 14.0, *)) {
    gpu_info.metal_info.language_version = MetalLanguageVersion::kMetal2_3;
  } else if (@available(macOS 10.15, iOS 13.0, tvOS 13.0, *)) {
    gpu_info.metal_info.language_version = MetalLanguageVersion::kMetal2_2;
  } else if (@available(macOS 10.14, iOS 12.0, tvOS 12.0, *)) {
    gpu_info.metal_info.language_version = MetalLanguageVersion::kMetal2_1;
  } else if (@available(macOS 10.13, iOS 11.0, tvOS 11.0, *)) {
    gpu_info.metal_info.language_version = MetalLanguageVersion::kMetal2_0;
  } else if (@available(macOS 10.12, iOS 10.0, tvOS 10.0, *)) {
    gpu_info.metal_info.language_version = MetalLanguageVersion::kMetal1_2;
  } else if (@available(macOS 10.11, iOS 9.0, tvOS 9.0, *)) {
    gpu_info.metal_info.language_version = MetalLanguageVersion::kMetal1_1;
  } else {
    gpu_info.metal_info.language_version = MetalLanguageVersion::kMetal1_0;
  }

  return gpu_info;
}
}  // namespace

MetalDevice::MetalDevice() : device_(MTLCreateSystemDefaultDevice()) {
  info_ = CreateGpuInfoFromMetalDevice(device_);
}
MetalDevice::MetalDevice(id<MTLDevice> device) : device_(device) {
  info_ = CreateGpuInfoFromMetalDevice(device_);
}

bool MetalDevice::IsLanguageVersion2orHigher() const {
  auto version = info_.metal_info.language_version;
  return version != MetalLanguageVersion::kMetal1_0 &&
         version != MetalLanguageVersion::kMetal1_1 &&
         version != MetalLanguageVersion::kMetal1_2;
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
