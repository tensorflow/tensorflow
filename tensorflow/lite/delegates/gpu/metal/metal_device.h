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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_METAL_METAL_DEVICE_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_METAL_METAL_DEVICE_H_

#import <Metal/Metal.h>

#include <string>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/gpu_info.h"

namespace tflite {
namespace gpu {
namespace metal {

// A wrapper around metal device
class MetalDevice {
 public:
  MetalDevice();
  MetalDevice(id<MTLDevice> device);

  MetalDevice(MetalDevice&& device) = default;
  MetalDevice& operator=(MetalDevice&& device) = default;
  MetalDevice(const MetalDevice&) = delete;
  MetalDevice& operator=(const MetalDevice&) = delete;

  ~MetalDevice() = default;

  id<MTLDevice> device() const { return device_; }

  const GpuInfo& GetInfo() const { return info_; }

  bool IsLanguageVersion2orHigher() const;

 private:
  id<MTLDevice> device_ = nullptr;
  GpuInfo info_;
};

}  // namespace metal
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_METAL_METAL_DEVICE_H_
