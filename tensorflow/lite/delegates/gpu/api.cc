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
      return absl::get_if<CpuMemory>(&obj);
    case ObjectType::OPENGL_SSBO:
      return absl::get_if<OpenGlBuffer>(&obj);
    case ObjectType::OPENGL_TEXTURE:
      return absl::get_if<OpenGlTexture>(&obj);
    case ObjectType::OPENCL_BUFFER:
      return absl::get_if<OpenClBuffer>(&obj);
    case ObjectType::OPENCL_TEXTURE:
      return absl::get_if<OpenClTexture>(&obj);
    case ObjectType::UNKNOWN:
      return false;
  }
}

}  // namespace gpu
}  // namespace tflite
