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

#include "tensorflow/lite/delegates/gpu/gl/converters/phwc4_to_bhwc.h"

#include <algorithm>
#include <cstdint>
#include <string>

#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/gl/converters/util.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_program.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_shader.h"
#include "tensorflow/lite/delegates/gpu/gl/variable.h"

namespace tflite {
namespace gpu {
namespace gl {

absl::Status ConverterPhwc4ToBhwc::Create(ConverterPhwc4ToBhwc* converter) {
  uint3 workgroup_size = uint3(4, 4, 4);
  std::string shader_source = GetShaderHeader(workgroup_size) + R"(
    layout(std430) buffer;

    precision highp float;

    layout(binding = 0) readonly buffer B0 {
      vec4 elements[];
    } input_data;

    layout(binding = 1) writeonly buffer B1 {
      float elements[];
    } output_data;

    uniform ivec4 sizes_;

    void main() {
      ivec3 gid = ivec3(gl_GlobalInvocationID.xyz);
      if (gid.x >= sizes_.x || gid.y >= sizes_.y || gid.z >= sizes_.z) {
        return;
      }
      output_data.elements[(gid.y * sizes_.x + gid.x) * sizes_.z + gid.z] = input_data.elements[(gid.z / 4 * sizes_.y + gid.y) * sizes_.x + gid.x][gid.z % 4];
    })";

  GlShader shader;
  RETURN_IF_ERROR(
      GlShader::CompileShader(GL_COMPUTE_SHADER, shader_source, &shader));
  GlProgram program;
  RETURN_IF_ERROR(GlProgram::CreateWithShader(shader, &program));
  *converter = ConverterPhwc4ToBhwc(std::move(program), workgroup_size);
  return absl::OkStatus();
}

absl::Status ConverterPhwc4ToBhwc::Convert(const BHWC& shape,
                                           const GlBuffer& source,
                                           CommandQueue* command_queue,
                                           GlBuffer* destination) {
  if (source.bytes_size() < BytesForPHWC4(shape)) {
    return absl::InvalidArgumentError(
        "Phwc4ToBhwc: Input data size does not match expected size.");
  }
  if (destination->bytes_size() < BytesForBHWC(shape)) {
    return absl::InvalidArgumentError(
        "Phwc4ToBhwc: output data size does not match expected size.");
  }
  if (shape.b != 1) {
    return absl::UnimplementedError(
        "Phwc4ToBhwc: Batch size is not equal to 1.");
  }

  uint3 workload = uint3(shape.w, shape.h, shape.c);
  uint3 num_workgroups = DivideRoundUp(workload, workgroup_size_);

  // TODO(akulik): simply pass workload as soon as UniformParameter
  // supports uint3
  RETURN_IF_ERROR(program_.SetParameter(
      {"sizes_",
       int4(static_cast<int32_t>(workload.x), static_cast<int32_t>(workload.y),
            static_cast<int32_t>(workload.z), 0)}));
  RETURN_IF_ERROR(source.BindToIndex(0));
  RETURN_IF_ERROR(destination->BindToIndex(1));
  if (command_queue) {
    return command_queue->Dispatch(program_, num_workgroups);
  }
  return program_.Dispatch(num_workgroups);
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
