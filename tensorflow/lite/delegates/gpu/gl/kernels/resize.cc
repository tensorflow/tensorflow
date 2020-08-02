/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/gl/kernels/resize.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

class Resize : public NodeShader {
 public:
  absl::Status GenerateCode(const GenerationContext& ctx,
                            GeneratedCode* generated_code) const final {
    const auto& attr = absl::any_cast<const Resize2DAttributes&>(ctx.op_attr);

    if (ctx.input_shapes[0][2] > ctx.output_shapes[0][2] ||
        ctx.input_shapes[0][1] > ctx.output_shapes[0][1]) {
      return absl::InvalidArgumentError("Output size is less than input size.");
    }
    if (ctx.output_shapes[0][2] != attr.new_shape.w ||
        ctx.output_shapes[0][1] != attr.new_shape.h) {
      return absl::InvalidArgumentError(
          "Output size does not match new_size in attributes.");
    }
    if (ctx.input_shapes[0][3] != ctx.output_shapes[0][3]) {
      return absl::InvalidArgumentError("Input/output channels mismatch.");
    }
    if (ctx.input_shapes[0][1] == 1 && ctx.input_shapes[0][2] == 1) {
      // Copy a single element from input.
      *generated_code = {
          /*parameters=*/{},
          /*objects=*/{},
          /*shared_variables=*/{},
          /*workload=*/uint3(),
          /*workgroup=*/uint3(),
          /*source_code=*/"value_0 = $input_data_0[0, 0, gid.z]$;",
          /*input=*/IOStructure::ONLY_DEFINITIONS,
          /*output=*/IOStructure::AUTO,
      };
      return absl::OkStatus();
    }
    std::vector<Variable> parameters = {
        {"input_data_0_h", static_cast<int>(ctx.input_shapes[0][1])},
        {"input_data_0_w", static_cast<int>(ctx.input_shapes[0][2])},
        {"scale_factor",
         float2(CalculateResizeScale(ctx.input_shapes[0][2],
                                     ctx.output_shapes[0][2], attr),
                CalculateResizeScale(ctx.input_shapes[0][1],
                                     ctx.output_shapes[0][1], attr))},
    };

    std::string source;
    if (attr.type == SamplingType::BILINEAR) {
      if (attr.half_pixel_centers) {
        source = "vec2 coord = (vec2(gid.xy) + 0.5) * $scale_factor$ - 0.5;";
      } else {
        source = "vec2 coord = vec2(gid.xy) * $scale_factor$;";
      }
      source += R"(
      vec2 coord_floor = floor(coord);
      ivec2 icoord_floor = ivec2(coord_floor);
      ivec2 borders = ivec2($input_data_0_w$, $input_data_0_h$) - ivec2(1, 1);
      ivec4 st;
      st.xy = max(icoord_floor, ivec2(0, 0));
      st.zw = min(icoord_floor + ivec2(1, 1), borders);

      vec2 t = coord - coord_floor; // interpolating factors

      vec4 tex11 = $input_data_0[st.x, st.y, gid.z]$;
      vec4 tex21 = $input_data_0[st.z, st.y, gid.z]$;
      vec4 tex12 = $input_data_0[st.x, st.w, gid.z]$;
      vec4 tex22 = $input_data_0[st.z, st.w, gid.z]$;

      value_0 = mix(mix(tex11, tex21, t.x), mix(tex12, tex22, t.x), t.y);)";
    } else if (attr.type == SamplingType::NEAREST) {
      std::string fxc;
      std::string fyc;
      if (attr.half_pixel_centers) {
        fxc = "(float(gid.x) + 0.5) * $scale_factor.x$";
        fyc = "(float(gid.y) + 0.5) * $scale_factor.y$";
      } else {
        fxc = "float(gid.x) * $scale_factor.x$";
        fyc = "float(gid.y) * $scale_factor.y$";
      }
      if (attr.align_corners) {
        fxc += " + 0.5";
        fyc += " + 0.5";
      }
      source += "  ivec2 coord;\n";
      source += "  coord.x = int(" + fxc + ");\n";
      source += "  coord.y = int(" + fyc + ");\n";
      source += "  coord.x = max(0, coord.x);\n";
      source += "  coord.y = max(0, coord.y);\n";
      source += "  coord.x = min(coord.x, $input_data_0_w$ - 1);\n";
      source += "  coord.y = min(coord.y, $input_data_0_h$ - 1);\n";
      source += R"(
      value_0 = $input_data_0[coord.x, coord.y, gid.z]$;
      )";
    } else {
      return absl::InvalidArgumentError("Unknown sampling type");
    }
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
};

}  // namespace

std::unique_ptr<NodeShader> NewResizeNodeShader() {
  return absl::make_unique<Resize>();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
