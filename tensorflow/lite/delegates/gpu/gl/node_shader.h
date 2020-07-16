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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_GL_NODE_SHADER_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_GL_NODE_SHADER_H_

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/types/any.h"
#include "tensorflow/lite/delegates/gpu/common/gpu_info.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/gl/compiler_options.h"
#include "tensorflow/lite/delegates/gpu/gl/object.h"
#include "tensorflow/lite/delegates/gpu/gl/variable.h"

namespace tflite {
namespace gpu {
namespace gl {

enum class IOStructure {
  // Source code uses standard inputs or outputs that should be generated from
  // node inputs/outputs. Compiler will generate them automatically as
  // 'input_data_N'/'output_data_N', where N is an index of the input/output.
  //
  // Generated code should not return input objects.
  ONLY_DEFINITIONS,

  // For inputs:
  //   Source code runs computations using 'vec4 value_N' declared by
  //   the compiler, where where N is an index of the input. Each value comes
  //   from inputs using coordinates set by GlobalInvocationID and a dispatch
  //   method, therefore, source code should not explicitly read values.
  //
  // For outputs:
  //   Source code runs computations and leaves results in 'vec4 value_N'
  //   declared by the compiler, where N is an index of the output. Value will
  //   be written to the output using coordinates set by GlobalInvocationID and
  //   a dispatch method. Therefore, source code should not explicitly write
  //   results.
  AUTO,
};

struct GeneratedCode {
  // A list of parameters to be set as uniform or hardcoded in a shader.
  std::vector<Variable> parameters;

  // A list of objects to bind before shader could be executed.
  std::vector<std::pair<std::string, Object>> objects;

  // A list of shared variables in the shader program.
  std::vector<Variable> shared_variables;

  // Compute shader operate on an abstract concept of work groups, each
  // three-dimensional. The number of work groups to be executed is defined by
  // workload tuple. Therefore,
  //   workload[x,y,z] := workgroup_size[x,y,z] X workgroup_count[x,y,z]
  // where 'X' is element-wise multiplication.
  //
  // Zero workload is calculated as PHWC4 based on output tensor.
  uint3 workload;

  // operation may specify recommended workgroup size. If not set, runtime will
  // figure it out automatically.
  uint3 workgroup;

  std::string source_code;

  // Parameters below reveal additional information about source_code.

  IOStructure input;
  IOStructure output;
};

// A class handles shader generation and setting runtime shader parameters.
class NodeShader {
 public:
  virtual ~NodeShader() = default;

  // A context for generating a code.
  struct GenerationContext {
    const GpuInfo* gpu_info;
    CompilationOptions compiler_options;

    // Information extracted & copied from compiled graph.
    const std::string& op_type;
    const absl::any& op_attr;
    // Do NOT use StrongShape<Layout::BHWC> in preparation for
    // RankedTensorType::getShape() which returns ArrayRef<int64_t>.
    std::vector<std::array<int64_t, 4>> input_shapes;
    std::vector<std::array<int64_t, 4>> output_shapes;
  };

  // Generates shader code for a node. The code should be just a function body.
  virtual absl::Status GenerateCode(const GenerationContext& ctx,
                                    GeneratedCode* generated_code) const = 0;

  // Limit the size of the const offsets array
  static constexpr int kMaxConstArraySize = 9;
};

}  // namespace gl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_GL_NODE_SHADER_H_
