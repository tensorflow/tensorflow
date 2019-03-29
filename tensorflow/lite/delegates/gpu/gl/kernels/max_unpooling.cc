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

#include "tensorflow/lite/delegates/gpu/gl/kernels/max_unpooling.h"

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

class MaxUnpooling : public NodeShader {
 public:
  Status GenerateCode(const GenerationContext& ctx,
                      GeneratedCode* generated_code) const final {
    auto attr = absl::any_cast<MaxUnpooling2DAttributes>(
        ctx.node->operation.attributes);
    std::vector<UniformParameter> parameters = {
        {"stride", int2(attr.strides.w, attr.strides.h)},
        {"offset", int2(attr.padding.prepended.w, attr.padding.prepended.h)},
        {"window_h", attr.kernel.h},
        {"window_w", attr.kernel.w},
    };

    std::string source = R"(
      ivec2 coord = (gid.xy + $offset$) / $stride$;
      ivec4 indices = $input_data_1[coord.x, coord.y, gid.z]$;
      vec4 input_ = $input_data_0[coord.x, coord.y, gid.z]$;
      coord = coord * $stride$ - $offset$;
      for (int i = 0; i < 4; ++i) {
        ivec2 t = coord + ivec2(indices[i] % $window_w$, indices[i] / $window_w$);
        if (t.x == gid.x && t.y == gid.y) {
          value_0[i] = input_[i];
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
};

}  // namespace

std::unique_ptr<NodeShader> NewMaxUnpoolingNodeShader() {
  return absl::make_unique<MaxUnpooling>();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
