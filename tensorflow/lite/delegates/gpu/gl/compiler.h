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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_GL_COMPILER_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_GL_COMPILER_H_

#include <functional>
#include <memory>
#include <unordered_set>

#include "tensorflow/lite/delegates/gpu/common/gpu_info.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/gl/compiler/shader_code.h"
#include "tensorflow/lite/delegates/gpu/gl/compiler_options.h"
#include "tensorflow/lite/delegates/gpu/gl/node_shader.h"

namespace tflite {
namespace gpu {
namespace gl {

using ShaderCodeCallback = std::function<absl::Status(ShaderCode code)>;

class Compiler {
 public:
  virtual ~Compiler() = default;

  // Goes over a graph and generates OpenGL shaders for the given graph.
  // Callback is called for every generated shader. Callback may execute shaders
  // as they come or store them elsewhere to execute later.
  virtual absl::Status Compile(
      const GraphFloat32& graph,
      const std::unordered_set<int>& tflite_graph_io,  // NOLINT
      const ShaderCodeCallback& callback) = 0;
};

std::unique_ptr<Compiler> NewCompiler(
    const NodeShader* node_shader, const GpuInfo* gpu_info,
    const CompilationOptions& options);

}  // namespace gl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_GL_COMPILER_H_
