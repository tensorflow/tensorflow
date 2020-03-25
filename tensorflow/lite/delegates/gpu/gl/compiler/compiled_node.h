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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_GL_COMPILER_COMPILED_NODE_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_GL_COMPILER_COMPILED_NODE_H_

#include <vector>

#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/gl/node_shader.h"
#include "tensorflow/lite/delegates/gpu/gl/object.h"

namespace tflite {
namespace gpu {
namespace gl {

// Contains compiler internal attributes for each node after it was processed by
// NodeShader.
struct CompiledNodeAttributes {
  std::vector<Object> inputs;
  std::vector<Object> outputs;

  GeneratedCode code;

  // nodes that are covered by the provided shader.
  std::vector<NodeId> node_indices;
};

// Moves all code objects, parameters and node indices from attr to merged_attr.
// Parameters and objects in attr.code.source_code are renamed to ensure
// uniqueness.
absl::Status MergeCode(CompiledNodeAttributes* attr,
                       CompiledNodeAttributes* merged_attr);

}  // namespace gl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_GL_COMPILER_COMPILED_NODE_H_
