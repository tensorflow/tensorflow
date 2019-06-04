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

#include "tensorflow/lite/delegates/gpu/gl/kernels/transpose_conv.h"

#include <memory>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/lite/delegates/gpu/common/convert.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/gl/node_shader.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

class ConvolutionTransposedBuffers : public NodeShader {
 public:
  Status GenerateCode(const GenerationContext& ctx,
                      GeneratedCode* generated_code) const final {
    auto input = ctx.graph->FindInputs(ctx.node->id)[0];
    auto attr = absl::any_cast<const ConvolutionTransposedAttributes&>(
        ctx.node->operation.attributes);
    auto weights = attr.weights.shape;
    const int32_t inner_size_w = (weights.w - 1) / attr.stride.w + 1;
    const int32_t inner_size_h = (weights.h - 1) / attr.stride.h + 1;

    std::vector<UniformParameter> parameters = {
        {"input_data_0_h", input->tensor.shape.h},
        {"input_data_0_w", input->tensor.shape.w},
        {"src_depth", IntegralDivideRoundUp(weights.i, 4)},
        {"kernel_size", int2(weights.w, weights.h)},
        {"stride", int2(attr.stride.w, attr.stride.h)},
        {"padding", int2(attr.padding.prepended.w, attr.padding.prepended.h)},
        {"inner_size", int2(inner_size_w, inner_size_h)},
    };

    std::vector<std::pair<std::string, Object>> objects = {
        {"weights", MakeReadonlyObject(Get3DSizeForPHWO4I4(attr.weights.shape),
                                       ConvertToPHWO4I4(attr.weights))}};

    std::string source = R"(
    ivec2 kernel_offset = $kernel_size$ - ivec2(1,1);
    ivec2 offset = gid.xy + $padding$ - kernel_offset;
    offset %= $stride$;
    offset += $stride$;
    offset %= $stride$;
    ivec2 f_offset;
    f_offset.x = offset.x == 0 ? 0 : ($stride.x$ - offset.x);
    f_offset.y = offset.y == 0 ? 0 : ($stride.y$ - offset.y);
    for (int ky = 0; ky < $inner_size.y$; ++ky) {
      for (int kx = 0; kx < $inner_size.x$; ++kx) {
        ivec2 index = ivec2(kx, ky) * $stride$ + f_offset;
        bool inside_kernel = index.x < $kernel_size.x$ && index.y < $kernel_size.y$;
        ivec2 coord = (gid.xy + index + $padding$ - kernel_offset) / $stride$;
        bool outside = coord.x < 0 || coord.y < 0 ||
                       coord.x >= $input_data_0_w$ || coord.y >= $input_data_0_h$;
        if (inside_kernel && !outside) {
          index = kernel_offset - index;
          int i = index.y * $kernel_size.x$ + index.x;
          for (int l = 0; l < $src_depth$; ++l) {
            vec4 src_color = $input_data_0[coord.x, coord.y, l]$;
            value_0.x += dot(src_color, $weights[l * 4 + 0, i, gid.z]$);
            value_0.y += dot(src_color, $weights[l * 4 + 1, i, gid.z]$);
            value_0.z += dot(src_color, $weights[l * 4 + 2, i, gid.z]$);
            value_0.w += dot(src_color, $weights[l * 4 + 3, i, gid.z]$);
          }
        }
      }
    }
)";
    if (!attr.bias.data.empty()) {
      source += "value_0 += $bias[gid.z]$;\n";
      objects.push_back({"bias", MakeReadonlyObject(attr.bias.data)});
    }
    *generated_code = {
        /*parameters=*/std::move(parameters),
        /*objects=*/std::move(objects),
        /*workload=*/uint3(),
        /*workgroup=*/uint3(),
        /*source_code=*/source,
        /*input=*/IOStructure::ONLY_DEFINITIONS,
        /*output=*/IOStructure::AUTO,
    };
    return OkStatus();
  }
};

}  // namespace

std::unique_ptr<NodeShader> NewConvolutionTransposedNodeShader() {
  return absl::make_unique<ConvolutionTransposedBuffers>();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
