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

#include "tensorflow/lite/delegates/gpu/gl/kernels/slice.h"

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

class Slice : public NodeShader {
 public:
  absl::Status GenerateCode(const GenerationContext& ctx,
                            GeneratedCode* generated_code) const final {
    const auto& attr = absl::any_cast<const SliceAttributes&>(ctx.op_attr);

    const int4 channels(attr.starts.c, attr.strides.c, attr.ends.c, 0);
    const int4 heights(attr.starts.h, attr.strides.h, attr.ends.h, 0);
    const int4 widths(attr.starts.w, attr.strides.w, attr.ends.w, 0);

    std::vector<Variable> parameters = {
        {"channels", channels},
        {"heights", heights},
        {"widths", widths},
        {"dst_size", static_cast<int>(ctx.output_shapes[0][3])},
    };

    std::string code;
    code += "      ivec2 offset;\n";
    if (attr.strides.w > 0) {
      code += "      offset.x = $widths.x$;\n";
    } else {
      if (attr.ends.w > 0) {
        code += "      offset.x = $widths.z$;\n";
      } else {
        code += "      offset.x = $src_size.x$ + $widths.z$;\n";
      }
    }
    if (attr.strides.h > 0) {
      code += "      offset.y = $heights.x$;\n";
    } else {
      if (attr.ends.h > 0) {
        code += "      offset.y = $heights.z$;\n";
      } else {
        code += "      offset.y = src_height + $heights.z$;\n";
      }
    }
    code += "      ivec2 stride = ivec2($widths.y$, $heights.y$);\n";
    code += "      ivec2 coord = offset + ivec2(gid.xy) * stride;\n";
    code += "      bool outside = false;\n";
    code += "      int step = gid.z * 4;\n";
    code += "      int buffer_index = 0;\n";
    code += "      int addr = 0;\n";
    for (int i = 0; i < 4; i++) {
      code += "      addr = step * $channels.y$;\n";
      if (attr.strides.c > 0) {
        code += "      addr += $channels.x$;\n";
      } else {
        if (attr.ends.c > 0) {
          code += "      addr += $channels.z$;\n";
        } else {
          code += "      addr += src_channels + $channels.z$;\n";
        }
      }
      code += "      if (step < $dst_size$) {\n        value_0[" +
              std::to_string(i) +
              "] = $input_data_0[coord.x, coord.y, addr / 4]$[addr % 4];\n     "
              " }\n";
      if (i != 3) {
        code += "      step++;\n";
      }
    }

    *generated_code = {
        /*parameters=*/std::move(parameters),
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

std::unique_ptr<NodeShader> NewSliceNodeShader() {
  return absl::make_unique<Slice>();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
