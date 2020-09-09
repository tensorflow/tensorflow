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

#include "tensorflow/lite/delegates/gpu/gl/kernels/reshape.h"

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

class Reshape : public NodeShader {
 public:
  absl::Status GenerateCode(const GenerationContext& ctx,
                            GeneratedCode* generated_code) const final {
    if (ctx.input_shapes[0][1] * ctx.input_shapes[0][2] *
            ctx.input_shapes[0][3] !=
        ctx.output_shapes[0][1] * ctx.output_shapes[0][2] *
            ctx.output_shapes[0][3]) {
      return absl::InvalidArgumentError(
          "Number of elements in input & output tensors don't match.");
    }
    const auto& attr = absl::any_cast<const ReshapeAttributes&>(ctx.op_attr);
    if (attr.new_shape.h != ctx.output_shapes[0][1] ||
        attr.new_shape.w != ctx.output_shapes[0][2] ||
        attr.new_shape.c != ctx.output_shapes[0][3]) {
      return absl::InvalidArgumentError(
          "Dimensions for output does not match new_shape attribute");
    }

    std::string code = R"(
    int input_ch_w = $input_channels$ * $input_data_0_w$;
    int output_ch_w = $output_channels$ * $output_data_0_w$;
    for (int i = 0; i < 4; ++i) {
      int dst_channel = gid.z * 4 + i;
      if (dst_channel >= $output_channels$) {
        continue;
      }
      int p = dst_channel + $output_channels$ * gid.x + output_ch_w * gid.y;
      int src_y = p / input_ch_w;
      int src_x = (p % input_ch_w) / $input_channels$;
      int src_z = (p % input_ch_w) % $input_channels$;
      int src_layer = src_z / 4;
      int src_channel = src_z % 4;
      value_0[i] = $input_data_0[src_x, src_y, src_layer]$[src_channel];
    }
    )";
    *generated_code = {
        /*parameters=*/{
            {"input_data_0_w", static_cast<int>(ctx.input_shapes[0][2])},
            {"input_channels", static_cast<int>(ctx.input_shapes[0][3])},
            {"output_data_0_w", static_cast<int>(ctx.output_shapes[0][2])},
            {"output_channels", static_cast<int>(ctx.output_shapes[0][3])},
        },
        /*objects=*/{},
        /*shared_variables=*/{},
        /*workload=*/uint3(),
        /*workgroup=*/uint3(),
        /*source_code=*/std::move(code),
        /*input=*/IOStructure::ONLY_DEFINITIONS,
        /*output=*/IOStructure::AUTO,
    };
    return absl::OkStatus();
  }
};

}  // namespace

std::unique_ptr<NodeShader> NewReshapeNodeShader() {
  return absl::make_unique<Reshape>();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
