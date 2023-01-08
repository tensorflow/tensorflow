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

#include <any>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/lite/delegates/gpu/common/convert.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/gl/node_shader.h"
#include "tensorflow/lite/delegates/gpu/gl/variable.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

class ConvolutionTransposedBuffers : public NodeShader {
 public:
  absl::Status GenerateCode(const GenerationContext& ctx,
                            GeneratedCode* generated_code) const final {
    if (ctx.input_shapes.size() != 1) {
      return absl::UnimplementedError(
          "Convolution Transposed does not support more than 1 runtime tensor");
    }
    const auto& attr =
        std::any_cast<const ConvolutionTransposedAttributes&>(ctx.op_attr);
    auto weights = attr.weights.shape;

    std::vector<Variable> parameters = {
        {"input_data_0_h", static_cast<int>(ctx.input_shapes[0][1])},
        {"input_data_0_w", static_cast<int>(ctx.input_shapes[0][2])},
        {"src_depth", DivideRoundUp(weights.i, 4)},
        {"kernel_size", int2(weights.w, weights.h)},
        {"stride", int2(attr.stride.w, attr.stride.h)},
        {"padding", int2(weights.w - 1 - attr.padding.prepended.w,
                         weights.h - 1 - attr.padding.prepended.h)},
    };

    std::vector<std::pair<std::string, Object>> objects = {
        {"weights",
         MakeReadonlyObject(Get3DSizeForPHWO4I4(attr.weights.shape),
                            ConvertToPHWO4I4Transposed(attr.weights))}};

    std::string source = R"(
    #define IN_BOUNDS(p, p0, p1) (all(greaterThanEqual(p, p0)) && all(lessThan(p, p1)))

    ivec2 p0 = ($padding$ + $stride$ - gid.xy % $stride$) % $stride$;
    for (int y = p0.y; y < $kernel_size.y$; y += $stride.y$) {
      for (int x = p0.x; x < $kernel_size.x$; x += $stride.x$) {

        int i = int(float(y * $kernel_size.x$) + float(x));
        ivec2 idx = ivec2(vec2(gid.xy + ivec2(x, y)) - vec2($padding$));

        if (IN_BOUNDS(idx, ivec2(0), ivec2($input_data_0_w$, $input_data_0_h$) * $stride$)) {
          ivec2 coord = idx / $stride$;
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
        /*shared_variables=*/{},
        /*workload=*/uint3(),
        /*workgroup=*/uint3(),
        /*source_code=*/source,
        /*input=*/IOStructure::ONLY_DEFINITIONS,
        /*output=*/IOStructure::AUTO,
    };
    return absl::OkStatus();
  }
};

}  // namespace

std::unique_ptr<NodeShader> NewConvolutionTransposedNodeShader() {
  return std::make_unique<ConvolutionTransposedBuffers>();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
