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

#include "tensorflow/lite/delegates/gpu/gl/kernels/pad.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/gl/variable.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

class Pad : public NodeShader {
 public:
  Status GenerateCode(const GenerationContext& ctx,
                      GeneratedCode* generated_code) const final {
    auto input = ctx.graph->FindInputs(ctx.node->id)[0];
    auto attr = absl::any_cast<PadAttributes>(ctx.node->operation.attributes);

    if (attr.type != PaddingContentType::ZEROS &&
        attr.type != PaddingContentType::REFLECT) {
      return UnimplementedError(
          "Only ZERO and REFLECT padding types are supported.");
    }
    if (attr.appended.h < 0 || attr.appended.w < 0 || attr.appended.c < 0 ||
        attr.prepended.h < 0 || attr.prepended.w < 0 || attr.prepended.c < 0) {
      return UnimplementedError("Negative padding is not supported.");
    }
    if (attr.appended.b != 0 || attr.prepended.b != 0) {
      return UnimplementedError("Padding for BATCH is not supported.");
    }
    std::vector<Variable> parameters = {
        {"input_data_0_h", input->tensor.shape.h},
        {"input_data_0_w", input->tensor.shape.w},
        {"input_data_0_c", input->tensor.shape.c},
        {"prepended",
         int4(attr.prepended.w, attr.prepended.h, attr.prepended.c, 0)},
    };
    std::string source;
    if (attr.type == PaddingContentType::REFLECT) {
      source = R"(
  int src_x = gid.x - $prepended.x$;
  src_x = abs(src_x);
  src_x = $input_data_0_w$ - 1 - abs(src_x - $input_data_0_w$ + 1);

  int src_y = gid.y - $prepended.y$;
  src_y = abs(src_y);
  src_y = $input_data_0_h$ - 1 - abs(src_y - $input_data_0_h$ + 1);
)";
      if (attr.prepended.c == 0 && attr.appended.c == 0) {
        // optimized case
        source += "  value_0 = $input_data_0[src_x, src_y, gid.z]$;\n";
      } else {
        source += R"(
  int start_channel = gid.z * 4;
  for (int i = 0; i < 4; ++i) {
    int channel = start_channel + i;
    int src_z = channel - $prepended.z$;
    src_z = abs(src_z);
    src_z = $input_data_0_c$ - 1 - abs(src_z - $input_data_0_c$ + 1);
    value_0[i] = $input_data_0[src_x, src_y, src_z / 4]$[src_z % 4];
  }
)";
      }
    } else {
      source = R"(
  int src_x = gid.x - $prepended.x$;
  int src_y = gid.y - $prepended.y$;
  if (src_x >= 0 && src_x < $input_data_0_w$ && src_y >= 0 && src_y < $input_data_0_h$) {
)";
      if (attr.prepended.c == 0 && attr.appended.c == 0) {
        // optimized case
        source += "    value_0 = $input_data_0[src_x, src_y, gid.z]$;\n";
      } else if (attr.prepended.c % 4 == 0) {
        parameters.push_back(
            {"src_slices", IntegralDivideRoundUp(input->tensor.shape.c, 4)});
        source += R"(
    int src_z = gid.z - $prepended.z$ / 4;
    if (src_z >= 0 && src_z < $src_slices$) {
      value_0 = $input_data_0[src_x, src_y, src_z]$;
    }
)";
      } else {
        source += R"(
    int start_channel = gid.z * 4;
    for (int i = 0; i < 4; ++i) {
      int channel = start_channel + i;
      int src_z = channel - $prepended.z$;
      if (src_z >= 0 && src_z < $input_data_0_c$) {
        value_0[i] = $input_data_0[src_x, src_y, src_z / 4]$[src_z % 4];
      }
    }
)";
      }
      source += "  }\n";
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
    return OkStatus();
  }
};

}  // namespace

std::unique_ptr<NodeShader> NewPadNodeShader() {
  return absl::make_unique<Pad>();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
