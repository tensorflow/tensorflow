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
#include "tensorflow/lite/delegates/gpu/gl/variable.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

absl::Status GenerateMaxPoolingCode(const Pooling2DAttributes& attr,
                                    const NodeShader::GenerationContext& ctx,
                                    GeneratedCode* generated_code) {
  if (attr.padding.prepended.h > attr.kernel.h ||
      attr.padding.prepended.w > attr.kernel.w) {
    return absl::InvalidArgumentError("Padding is bigger than kernel.");
  }

  std::vector<Variable> parameters = {
      {"input_data_0_h", static_cast<int>(ctx.input_shapes[0][1])},
      {"input_data_0_w", static_cast<int>(ctx.input_shapes[0][2])},
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
      /*shared_variables=*/{},
      /*workload=*/uint3(),
      /*workgroup=*/uint3(),
      /*source_code=*/std::move(source),
      /*input=*/IOStructure::ONLY_DEFINITIONS,
      /*output=*/IOStructure::AUTO,
  };
  return absl::OkStatus();
}

absl::Status GenerateAveragePoolingCode(
    const Pooling2DAttributes& attr, const NodeShader::GenerationContext& ctx,
    GeneratedCode* generated_code) {
  std::vector<Variable> parameters = {
      {"input_data_0_h", static_cast<int>(ctx.input_shapes[0][1])},
      {"input_data_0_w", static_cast<int>(ctx.input_shapes[0][2])},
      {"stride", int2(attr.strides.w, attr.strides.h)},
      {"offset", int2(attr.padding.prepended.w, attr.padding.prepended.h)},
      {"window_h", attr.kernel.h},
      {"window_w", attr.kernel.w},
  };

  // Bounds checking helper functions.
  auto x_in_bounds = [input_width = ctx.input_shapes[0][2],
                      kernel_width = attr.kernel.w](int64_t x) -> bool {
    return 0 <= x && x + kernel_width <= input_width;
  };
  auto y_in_bounds = [input_height = ctx.input_shapes[0][1],
                      kernel_height = attr.kernel.h](int64_t y) -> bool {
    return 0 <= y && y + kernel_height <= input_height;
  };

  // Only include a bounds check in the shader if it will actually be necessary
  // at run time.
  const int64_t output_shape_max_y = ctx.output_shapes[0][1] - 1;
  const int64_t output_shape_max_x = ctx.output_shapes[0][2] - 1;
  const int64_t base_x = -attr.padding.prepended.w;
  const int64_t base_y = -attr.padding.prepended.h;
  const bool bounds_check_necessary =
      !(x_in_bounds(base_x) &&
        x_in_bounds(base_x + output_shape_max_x * attr.strides.w) &&
        y_in_bounds(base_y) &&
        y_in_bounds(base_y + output_shape_max_y * attr.strides.h));

  std::string source = bounds_check_necessary ?
                                              R"(
  int window_size = 0;
  for (int a = 0; a < $window_h$; ++a) {
    for (int b = 0; b < $window_w$; ++b) {
      ivec2 coord = gid.xy * $stride$ - $offset$ + ivec2(b, a);
      if (coord.x >= 0 && coord.y >= 0 && coord.x < $input_data_0_w$ && coord.y < $input_data_0_h$) {
        value_0 += $input_data_0[coord.x, coord.y, gid.z]$;
        window_size++;
      }
    }
  }
  // If window_size==0, window covered nothing. This situation is a sign of
  // incorrectly constructed operation. NaNs are expected as output.
  value_0 /= float(window_size);
)"
                                              :
                                              R"(
  for (int a = 0; a < $window_h$; ++a) {
    for (int b = 0; b < $window_w$; ++b) {
      ivec2 coord = gid.xy * $stride$ - $offset$ + ivec2(b, a);
      value_0 += $input_data_0[coord.x, coord.y, gid.z]$;
    }
  }
  // If the denominator is 0, that is a sign of an incorrectly constructed
  // operation. NaNs are expected as output.
  value_0 /= float($window_h$ * $window_w$);
)";

  *generated_code = {
      /*parameters=*/std::move(parameters),
      /*objects=*/{},
      /*shared_variables=*/{},
      /*workload=*/uint3(),
      /*workgroup=*/uint3(),
      /*source_code=*/std::move(source),
      /*input=*/IOStructure::ONLY_DEFINITIONS,
      /*output=*/IOStructure::AUTO,
  };
  return absl::OkStatus();
}

class Pooling : public NodeShader {
 public:
  absl::Status GenerateCode(const GenerationContext& ctx,
                            GeneratedCode* generated_code) const final {
    const auto& attr = absl::any_cast<const Pooling2DAttributes&>(ctx.op_attr);
    switch (attr.type) {
      case PoolingType::AVERAGE:
        return GenerateAveragePoolingCode(attr, ctx, generated_code);
      case PoolingType::MAX:
        return GenerateMaxPoolingCode(attr, ctx, generated_code);
      default:
        return absl::InvalidArgumentError("Incorrect attributes' type.");
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
