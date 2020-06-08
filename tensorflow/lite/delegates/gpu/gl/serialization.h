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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_GL_SERIALIZATION_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_GL_SERIALIZATION_H_

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/gl/compiled_model_generated.h"
#include "tensorflow/lite/delegates/gpu/gl/object.h"
#include "tensorflow/lite/delegates/gpu/gl/variable.h"

namespace tflite {
namespace gpu {
namespace gl {

struct CompiledModelOptions {
  // If true, a model was compiled with dynamic batch size and therefore,
  // a user may change BATCH dimension at runtime.
  bool dynamic_batch = false;
};

// Accumulates shaders and programs and stores it in FlatBuffer format.
class SerializedCompiledModelBuilder {
 public:
  SerializedCompiledModelBuilder() : builder_(32 * 1024) {}

  void AddShader(const std::string& shader_src);

  void AddProgram(const std::vector<Variable>& parameters,
                  const std::vector<Object>& objects,
                  const uint3& workgroup_size, const uint3& num_workgroups,
                  size_t shader_index);

  // Returns serialized data that will stay valid until this object is
  // destroyed.
  absl::Span<const uint8_t> Finalize(const CompiledModelOptions& options);

 private:
  std::vector<flatbuffers::Offset<flatbuffers::String>> shaders_;
  std::vector<flatbuffers::Offset<data::Program>> programs_;
  ::flatbuffers::FlatBufferBuilder builder_;
};

// Handles deserialization events. it is guaranteed that shaders will be called
// first in the appropriate order and programs come next.
class DeserializationHandler {
 public:
  virtual ~DeserializationHandler() = default;

  virtual absl::Status OnShader(absl::Span<const char> shader_src) = 0;

  virtual absl::Status OnProgram(const std::vector<Variable>& parameters,
                                 const std::vector<Object>& objects,
                                 const uint3& workgroup_size,
                                 const uint3& num_workgroups,
                                 size_t shader_index) = 0;

  virtual void OnOptions(const CompiledModelOptions& options) = 0;
};

absl::Status DeserializeCompiledModel(absl::Span<const uint8_t> serialized,
                                      DeserializationHandler* handler);

}  // namespace gl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_GL_SERIALIZATION_H_
