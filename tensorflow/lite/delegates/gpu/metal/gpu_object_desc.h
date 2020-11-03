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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_METAL_GPU_OBJECT_DESC_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_METAL_GPU_OBJECT_DESC_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/access_type.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace metal {

enum class MemoryType { GLOBAL, CONSTANT, LOCAL };

std::string MemoryTypeToMetalType(MemoryType type);

struct GPUBufferDescriptor {
  DataType data_type;
  AccessType access_type;
  int element_size;
  MemoryType memory_type = MemoryType::GLOBAL;
  std::vector<std::string> attributes;
};

struct GPUResources {
  std::vector<std::string> ints;
  std::vector<std::string> floats;
  std::vector<std::pair<std::string, GPUBufferDescriptor>> buffers;

  std::vector<std::string> GetNames() const {
    std::vector<std::string> names = ints;
    names.insert(names.end(), floats.begin(), floats.end());
    for (const auto& obj : buffers) {
      names.push_back(obj.first);
    }
    return names;
  }
};

class GPUObjectDescriptor {
 public:
  GPUObjectDescriptor() = default;
  GPUObjectDescriptor(const GPUObjectDescriptor&) = default;
  GPUObjectDescriptor& operator=(const GPUObjectDescriptor&) = default;
  GPUObjectDescriptor(GPUObjectDescriptor&& obj_desc) = default;
  GPUObjectDescriptor& operator=(GPUObjectDescriptor&& obj_desc) = default;

  virtual ~GPUObjectDescriptor() = default;

  void SetStateVar(const std::string& key, const std::string& value) const {
    state_vars_[key] = value;
  }

  virtual std::string PerformConstExpr(const std::string& const_expr) const {
    return "";
  }

  virtual absl::Status PerformSelector(
      const std::string& selector, const std::vector<std::string>& args,
      const std::vector<std::string>& template_args,
      std::string* result) const {
    *result = "";
    return absl::OkStatus();
  }
  virtual GPUResources GetGPUResources() const { return GPUResources(); }

  virtual void Release() {}

  void SetAccess(AccessType access_type) { access_type_ = access_type; }
  AccessType GetAccess() const { return access_type_; }

 protected:
  mutable std::map<std::string, std::string> state_vars_;
  AccessType access_type_;
};

using GPUObjectDescriptorPtr = std::unique_ptr<GPUObjectDescriptor>;

}  // namespace metal
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_METAL_GPU_OBJECT_DESC_H_
