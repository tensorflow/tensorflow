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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_GL_COMPILER_SHADER_CODE_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_GL_COMPILER_SHADER_CODE_H_

#include <string>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/gl/object.h"
#include "tensorflow/lite/delegates/gpu/gl/variable.h"

namespace tflite {
namespace gpu {
namespace gl {

struct ShaderCode {
  ShaderCode() = default;
  ShaderCode(const std::vector<Variable>& in_parameters,
             const std::vector<Object>& in_objects, const uint3& in_workload,
             const uint3& in_recommended_workgroup,
             const std::string& in_source_code,
             const std::vector<NodeId>& in_node_indices)
      : parameters(in_parameters),
        objects(in_objects),
        workload(in_workload),
        recommended_workgroup(in_recommended_workgroup),
        source_code(in_source_code),
        node_indices(in_node_indices) {}

  // A list of uniform parameters to be set.
  std::vector<Variable> parameters;

  // A list of objects to bind to opengl program.
  std::vector<Object> objects;

  uint3 workload;

  // operation may specify recommended workgroup size
  uint3 recommended_workgroup;

  // Generated source code does not set local size, therefore it needs to be set
  // elsewhere.
  std::string source_code;

  // nodes of the graph that are covered by the shader.
  std::vector<NodeId> node_indices;
};

}  // namespace gl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_GL_COMPILER_SHADER_CODE_H_
