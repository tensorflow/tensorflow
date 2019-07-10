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

#include "tensorflow/lite/delegates/gpu/gl/kernels/fully_connected.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/lite/delegates/gpu/common/convert.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

class FullyConnectedBuffers : public NodeShader {
 public:
  Status GenerateCode(const GenerationContext& ctx,
                      GeneratedCode* generated_code) const final {
    auto attr = absl::any_cast<const FullyConnectedAttributes&>(
        ctx.node->operation.attributes);

    // TODO(akulik): check that input has h,w == 1,1
    std::vector<UniformParameter> parameters = {
        {"src_depth", IntegralDivideRoundUp(attr.weights.shape.i, 4)},
    };

    // TODO(akulik): refactor indexed access to weights.
    std::vector<std::pair<std::string, Object>> objects = {
        {"weights", MakeReadonlyObject(ConvertToPHWO4I4(attr.weights))}};

    std::string source = R"(
  int offset = gid.z * $src_depth$ * 4;
  for (int d = 0; d < $src_depth$; ++d, offset += 4) {
      vec4 src = $input_data_0[0, 0, d]$;
      value_0.x += dot(src, $weights[offset]$);
      value_0.y += dot(src, $weights[offset + 1]$);
      value_0.z += dot(src, $weights[offset + 2]$);
      value_0.w += dot(src, $weights[offset + 3]$);
  }
)";
    if (!attr.bias.data.empty()) {
      source += "  value_0 += $bias[gid.z]$;\n";
      objects.push_back({"bias", MakeReadonlyObject(attr.bias.data)});
    }
    *generated_code = {
        /*parameters=*/std::move(parameters),
        /*objects=*/std::move(objects),
        /*workload=*/
        uint3(1, 1, IntegralDivideRoundUp(attr.weights.shape.o, 4)),
        /*workgroup=*/uint3(),
        /*source_code=*/std::move(source),
        /*input=*/IOStructure::ONLY_DEFINITIONS,
        /*output=*/IOStructure::AUTO,
    };
    return OkStatus();
  }
};

}  // namespace

std::unique_ptr<NodeShader> NewFullyConnectedNodeShader() {
  return absl::make_unique<FullyConnectedBuffers>();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
