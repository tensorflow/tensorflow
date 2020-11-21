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

#include "tensorflow/lite/delegates/gpu/api.h"

namespace tflite {
namespace gpu {
namespace {

struct ObjectTypeGetter {
  ObjectType operator()(absl::monostate) const { return ObjectType::UNKNOWN; }
  ObjectType operator()(OpenGlBuffer) const { return ObjectType::OPENGL_SSBO; }
  ObjectType operator()(OpenGlTexture) const {
    return ObjectType::OPENGL_TEXTURE;
  }
  ObjectType operator()(OpenClBuffer) const {
    return ObjectType::OPENCL_BUFFER;
  }
  ObjectType operator()(OpenClTexture) const {
    return ObjectType::OPENCL_TEXTURE;
  }
  ObjectType operator()(VulkanBuffer) const {
    return ObjectType::VULKAN_BUFFER;
  }
  ObjectType operator()(VulkanTexture) const {
    return ObjectType::VULKAN_TEXTURE;
  }
  ObjectType operator()(CpuMemory) const { return ObjectType::CPU_MEMORY; }
};

struct ObjectValidityChecker {
  bool operator()(absl::monostate) const { return false; }
  bool operator()(OpenGlBuffer obj) const { return obj.id != GL_INVALID_INDEX; }
  bool operator()(OpenGlTexture obj) const {
    return obj.id != GL_INVALID_INDEX && obj.format != GL_INVALID_ENUM;
  }
  bool operator()(OpenClBuffer obj) const { return obj.memobj; }
  bool operator()(OpenClTexture obj) const { return obj.memobj; }
  bool operator()(VulkanBuffer obj) const { return obj.memory; }
  bool operator()(VulkanTexture obj) const { return obj.memory; }
  bool operator()(CpuMemory obj) const {
    return obj.data != nullptr && obj.size_bytes > 0 &&
           (data_type == DataType::UNKNOWN ||
            obj.size_bytes % SizeOf(data_type) == 0);
  }
  DataType data_type;
};

}  // namespace

bool IsValid(const ObjectDef& def) {
  return def.data_type != DataType::UNKNOWN &&
         def.data_layout != DataLayout::UNKNOWN &&
         def.object_type != ObjectType::UNKNOWN;
}

ObjectType GetType(const TensorObject& object) {
  return absl::visit(ObjectTypeGetter{}, object);
}

bool IsValid(const TensorObjectDef& def) { return IsValid(def.object_def); }

bool IsValid(const TensorObjectDef& def, const TensorObject& object) {
  return GetType(object) == def.object_def.object_type &&
         absl::visit(ObjectValidityChecker{def.object_def.data_type}, object);
}

bool IsObjectPresent(ObjectType type, const TensorObject& obj) {
  switch (type) {
    case ObjectType::CPU_MEMORY:
      return absl::holds_alternative<CpuMemory>(obj);
    case ObjectType::OPENGL_SSBO:
      return absl::holds_alternative<OpenGlBuffer>(obj);
    case ObjectType::OPENGL_TEXTURE:
      return absl::holds_alternative<OpenGlTexture>(obj);
    case ObjectType::OPENCL_BUFFER:
      return absl::holds_alternative<OpenClBuffer>(obj);
    case ObjectType::OPENCL_TEXTURE:
      return absl::holds_alternative<OpenClTexture>(obj);
    case ObjectType::VULKAN_BUFFER:
      return absl::holds_alternative<VulkanBuffer>(obj);
    case ObjectType::VULKAN_TEXTURE:
      return absl::holds_alternative<VulkanTexture>(obj);
    case ObjectType::UNKNOWN:
      return false;
  }
}

uint32_t NumElements(const TensorObjectDef& def) {
  const auto& d = def.dimensions;
  switch (def.object_def.data_layout) {
    case DataLayout::BHWC:
      return d.product();
    case DataLayout::HWDC4:
    case DataLayout::HDWC4:
    case DataLayout::DHWC4:
      return d.b * d.h * d.w * AlignByN(d.c, 4);
    case DataLayout::UNKNOWN:
      return 0;
  }
  return 0;
}

int GetPosition(const InferenceOptions& options, InferencePriority p) {
  if (options.priority1 == p) return 1;
  if (options.priority2 == p) return 2;
  if (options.priority3 == p) return 3;
  return 4;  // least important
}

PriorityImportance GetRelativeImportance(const InferenceOptions& options,
                                         InferencePriority p1,
                                         InferencePriority p2) {
  int p1_position = GetPosition(options, p1);
  int p2_position = GetPosition(options, p2);
  if (p1_position == p2_position) return PriorityImportance::UNKNOWN;
  return p1_position < p2_position ? PriorityImportance::HIGHER
                                   : PriorityImportance::LOWER;
}

bool IsValid(const InferenceOptions& options) {
  if (options.usage == InferenceUsage::UNKNOWN) {
    return false;
  }
  if (options.priority1 == InferencePriority::UNKNOWN ||
      options.priority2 == InferencePriority::UNKNOWN ||
      options.priority3 == InferencePriority::UNKNOWN) {
    return false;
  }
  if (options.priority1 == InferencePriority::AUTO) {
    return false;
  }
  if (options.priority2 == InferencePriority::AUTO &&
      options.priority3 != InferencePriority::AUTO) {
    return false;
  }
  if (options.priority1 == options.priority2 ||
      options.priority1 == options.priority3) {
    return false;
  }
  if (options.priority2 == options.priority3 &&
      options.priority2 != InferencePriority::AUTO) {
    return false;
  }
  return true;
}

// Implementation note: this resolution logic is shared between GL and CL
// backends, but they might have own logic. Thus, the function is defined
// here just for code re-use purposes.
void ResolveAutoPriority(InferenceOptions* options) {
  // priority1 can not be AUTO as it would make options invalid.
  if (options->priority2 == InferencePriority::AUTO) {
    switch (options->priority1) {
      case InferencePriority::MIN_LATENCY:
        options->priority2 = InferencePriority::MIN_MEMORY_USAGE;
        options->priority3 = InferencePriority::MAX_PRECISION;
        return;
      case InferencePriority::MIN_MEMORY_USAGE:
        options->priority2 = InferencePriority::MAX_PRECISION;
        options->priority3 = InferencePriority::MIN_LATENCY;
        return;
      case InferencePriority::MAX_PRECISION:
        options->priority2 = InferencePriority::MIN_LATENCY;
        options->priority3 = InferencePriority::MIN_MEMORY_USAGE;
        return;
      case InferencePriority::UNKNOWN:
      case InferencePriority::AUTO:
        // Invalid and unreachable option.
        return;
    }
  }

  if (options->priority3 == InferencePriority::AUTO) {
    // Simply add missing priority
    if (GetPosition(*options, InferencePriority::MIN_LATENCY) == 4) {
      options->priority3 = InferencePriority::MIN_LATENCY;
    } else if (GetPosition(*options, InferencePriority::MAX_PRECISION) == 4) {
      options->priority3 = InferencePriority::MAX_PRECISION;
    } else if (GetPosition(*options, InferencePriority::MIN_MEMORY_USAGE) ==
               4) {
      options->priority3 = InferencePriority::MIN_MEMORY_USAGE;
    }
  }
}

}  // namespace gpu
}  // namespace tflite
