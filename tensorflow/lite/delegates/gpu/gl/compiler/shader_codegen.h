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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_GL_COMPILER_SHADER_CODEGEN_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_GL_COMPILER_SHADER_CODEGEN_H_

#include <string>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/gpu_info.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/gl/compiler/compiled_node.h"
#include "tensorflow/lite/delegates/gpu/gl/compiler/object_accessor.h"
#include "tensorflow/lite/delegates/gpu/gl/compiler/shader_code.h"
#include "tensorflow/lite/delegates/gpu/gl/compiler_options.h"
#include "tensorflow/lite/delegates/gpu/gl/object.h"

namespace tflite {
namespace gpu {
namespace gl {

// This class is responsible for assembling a shader by putting together
// objects, parameters declarations and main function.
class ShaderCodegen {
 public:
  ShaderCodegen(const CompilationOptions& options, const GpuInfo& gpu_info);

  // Builds final program representation.
  absl::Status Build(CompiledNodeAttributes attr,
                     ShaderCode* shader_code) const;

 private:
  const CompilationOptions options_;
  const GpuType gpu_type_;
};

}  // namespace gl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_GL_COMPILER_SHADER_CODEGEN_H_
