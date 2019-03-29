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

#include "tensorflow/lite/delegates/gpu/gl/kernels/pooling.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

Status GenerateMaxPoolingCode(const Pooling2DAttributes& attr,
                              const NodeShader::GenerationContext& ctx,
                              GeneratedCode* generated_code) {
  auto input = ctx.graph->FindInputs(ctx.node->id)[0];

  if (attr.padding.prepended.h > attr.kernel.h ||
      attr.padding.prepended.w > attr.kernel.w) {
    return InvalidArgumentError("Padding is bigger than kernel.");
  }

  std::vector<UniformParameter> parameters = {
      {"input_data_0_h", input->tensor.shape.h},
      {"input_data_0_w", input->tensor.shape.w},
      {"stride", int2(attr.strides.w, attr.strides.h)},
      {"offset", int2(attr.padding.prepended.w, attr.padding.prepended.h)},
      {"window_h", attr.kernel.h},
      {"window_w", attr.kernel.w},
  };

  // Per GLSL_ES 3.1 spec in Issue 13.4
  // "Floating Point Representation and Functionality" highp floats are
  // expected to behave as defined in IEEE 754. In particular, signed
  // infinities are mandated and defined as a number divided by 0.
  std::string source = R"(
  const highp float inf = -(1.0f / 0.0f);
  value_0 = vec4(inf);)";
  if (attr.output_indices) {
    source += R"(
  ivec4 value_1;
)";
  }
  source += R"(
  ivec2 base_coord = gid.xy * $stride$ - $offset$;
  for (int a = 0; a < $window_h$; ++a) {
    for (int b = 0; b < $window_w$; ++b) {
      ivec2 coord = base_coord + ivec2(b, a);
      if (coord.x < 0 || coord.y < 0 || coord.x >= $input_data_0_w$ || coord.y >= $input_data_0_h$) {
        continue;
      }
      vec4 input_ = $input_data_0[coord.x, coord.y, gid.z]$;)";
  if (attr.output_indices) {
    source += R"(
      int window_index = a * $window_w$ + b;
      if (input_.x > value_0.x) value_1.x = window_index;
      if (input_.y > value_0.y) value_1.y = window_index;
      if (input_.z > value_0.z) value_1.z = window_index;
      if (input_.w > value_0.w) value_1.w = window_index;)";
  }
  source += R"(
      value_0 = max(value_0, input_);
    }
  }
)";
  *generated_code = {
      /*parameters=*/std::move(parameters),
      /*objects=*/{},
      /*workload=*/uint3(),
      /*workgroup=*/uint3(),
      /*source_code=*/std::move(source),
      /*input=*/IOStructure::ONLY_DEFINITIONS,
      /*output=*/IOStructure::AUTO,
  };
  return OkStatus();
}

Status GenerateAveragePoolingCode(const Pooling2DAttributes& attr,
                                  const NodeShader::GenerationContext& ctx,
                                  GeneratedCode* generated_code) {
  auto input = ctx.graph->FindInputs(ctx.node->id)[0];

  std::vector<UniformParameter> parameters = {
      {"input_data_0_h", input->tensor.shape.h},
      {"input_data_0_w", input->tensor.shape.w},
      {"stride", int2(attr.strides.w, attr.strides.h)},
      {"offset", int2(attr.padding.prepended.w, attr.padding.prepended.h)},
      {"window_h", attr.kernel.h},
      {"window_w", attr.kernel.w},
      {"multiplier", 1.0f / static_cast<float>(attr.kernel.h * attr.kernel.w)},
  };

  std::string source = R"(
  for (int a = 0; a < $window_h$; ++a) {
    for (int b = 0; b < $window_w$; ++b) {
      ivec2 coord = gid.xy * $stride$ - $offset$ + ivec2(b, a);
      if (coord.x >= 0 && coord.y >= 0 && coord.x < $input_data_0_w$ && coord.y < $input_data_0_h$) {
        value_0 += $input_data_0[coord.x, coord.y, gid.z]$;
      }
    }
  }
  value_0 *= $multiplier$;
)";
  *generated_code = {
      /*parameters=*/std::move(parameters),
      /*objects=*/{},
      /*workload=*/uint3(),
      /*workgroup=*/uint3(),
      /*source_code=*/std::move(source),
      /*input=*/IOStructure::ONLY_DEFINITIONS,
      /*output=*/IOStructure::AUTO,
  };
  return OkStatus();
}

class Pooling : public NodeShader {
 public:
  Status GenerateCode(const GenerationContext& ctx,
                      GeneratedCode* generated_code) const final {
    const auto& attr =
        absl::any_cast<Pooling2DAttributes>(ctx.node->operation.attributes);
    switch (attr.type) {
      case PoolingType::AVERAGE:
        return GenerateAveragePoolingCode(attr, ctx, generated_code);
      case PoolingType::MAX:
        return GenerateMaxPoolingCode(attr, ctx, generated_code);
      default:
        return InvalidArgumentError("Incorrect attributes' type.");
    }
  }
};

}  // namespace

std::unique_ptr<NodeShader> NewPoolingNodeShader() {
  return absl::make_unique<Pooling>();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
