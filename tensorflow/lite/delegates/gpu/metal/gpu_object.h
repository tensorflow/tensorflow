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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_METAL_GPU_OBJECT_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_METAL_GPU_OBJECT_H_

#import <Metal/Metal.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/access_type.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_object_desc.h"

namespace tflite {
namespace gpu {
namespace metal {

struct GPUResourcesWithValue {
  GenericGPUResourcesWithValue generic;

  struct BufferParameter {
    id<MTLBuffer> handle;
    uint64_t offset;
  };
  std::vector<std::pair<std::string, BufferParameter>> buffers;
  std::vector<std::pair<std::string, id<MTLTexture>>> images2d;
  std::vector<std::pair<std::string, id<MTLTexture>>> image2d_arrays;
  std::vector<std::pair<std::string, id<MTLTexture>>> images3d;
  std::vector<std::pair<std::string, id<MTLTexture>>> image_buffers;

  void AddFloat(const std::string& name, float value) {
    generic.AddFloat(name, value);
  }
  void AddInt(const std::string& name, int value) {
    generic.AddInt(name, value);
  }
};

class GPUObject {
 public:
  GPUObject() = default;
  // Move only
  GPUObject(GPUObject&& obj_desc) = default;
  GPUObject& operator=(GPUObject&& obj_desc) = default;
  GPUObject(const GPUObject&) = delete;
  GPUObject& operator=(const GPUObject&) = delete;
  virtual ~GPUObject() = default;
  virtual absl::Status GetGPUResources(
      const GPUObjectDescriptor* obj_ptr,
      GPUResourcesWithValue* resources) const = 0;
};

using GPUObjectPtr = std::unique_ptr<GPUObject>;

}  // namespace metal
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_METAL_GPU_OBJECT_H_
