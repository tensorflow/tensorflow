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
#include <any>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
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
    const auto& attr = std::any_cast<const SliceAttributes&>(ctx.op_attr);

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
    code += "      ivec3 offset;\n";
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
    if (attr.strides.c > 0) {
      code += "      offset.z = $channels.x$;\n";
    } else {
      if (attr.ends.c > 0) {
        code += "      offset.z = $channels.z$;\n";
      } else {
        code += "      offset.z = src_channels + $channels.z$;\n";
      }
    }
    code +=
        "      ivec3 stride = "
        "ivec3($widths.y$, $heights.y$, $channels.y$);\n";
    code += "      ivec3 coord;\n";
    code += "      coord.xy = offset.xy + ivec2(gid.xy) * stride.xy;\n";
    code += "      int step = gid.z * 4;\n";
    code += "      coord.z = offset.z + step * stride.z;\n";
    code +=
        "      if(step++ < $dst_size$) value_0[0] = "
        "$input_data_0[coord.x, coord.y, coord.z / 4]$[coord.z % 4];\n";
    code += "      coord.z += $channels.y$;\n";
    code +=
        "      if(step++ < $dst_size$) value_0[1] = "
        "$input_data_0[coord.x, coord.y, coord.z / 4]$[coord.z % 4];\n";
    code += "      coord.z += $channels.y$;\n";
    code +=
        "      if(step++ < $dst_size$) value_0[2] = "
        "$input_data_0[coord.x, coord.y, coord.z / 4]$[coord.z % 4];\n";
    code += "      coord.z += $channels.y$;\n";
    code +=
        "      if(step++ < $dst_size$) value_0[3] = "
        "$input_data_0[coord.x, coord.y, coord.z / 4]$[coord.z % 4];\n";

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
  return std::make_unique<Slice>();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
