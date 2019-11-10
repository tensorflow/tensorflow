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
#include "tensorflow/lite/delegates/gpu/gl/variable.h"

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

    const int src_depth = IntegralDivideRoundUp(attr.weights.shape.i, 4);
    const int dst_depth = IntegralDivideRoundUp(attr.weights.shape.o, 4);

    // This shader can work with any workgroup size, the values below work well
    // for OpenGL.
    constexpr int kWorkgroupHintX = 4;
    constexpr int kWorkgroupHintY = 4;

    // TODO(akulik): check that input has h,w == 1,1
    std::vector<Variable> parameters = {
        {"src_depth", src_depth},
        {"dst_depth", dst_depth},
    };

    // TODO(akulik): refactor indexed access to weights.
    std::vector<std::pair<std::string, Object>> objects = {
        {"weights", MakeReadonlyObject(ConvertToPHWO4I4(attr.weights))}};

    std::string source = R"(
  const int threads = int(gl_WorkGroupSize.y);
  const int workers = int(gl_WorkGroupSize.x);
  ivec3 tid = ivec3(gl_LocalInvocationID);

  if (gid.x < $dst_depth$) {
    int offset = 4 * gid.x * $src_depth$ + 4 * tid.y;
    int iterations = ($src_depth$ + threads-1) / threads;
    for (int d = 0; d < iterations; d++, offset += 4 * threads) {
      vec4 src = $input_data_0[0, 0, d * threads + tid.y]$;
      value_0.x += dot(src, $weights[offset + 0]$);
      value_0.y += dot(src, $weights[offset + 1]$);
      value_0.z += dot(src, $weights[offset + 2]$);
      value_0.w += dot(src, $weights[offset + 3]$);
    }
    sh_mem[workers * tid.y + tid.x] = value_0;
  }
  memoryBarrierShared();
  barrier();

  if (tid.y > 0 || gid.x >= $dst_depth$) {
    return;
  }

  for (int t = 1; t < threads; t++) {
    value_0 += sh_mem[workers * t + tid.x];
  }
)";
    if (!attr.bias.data.empty()) {
      source += "  value_0 += $bias[gid.x]$;\n";
      objects.push_back({"bias", MakeReadonlyObject(attr.bias.data)});
    }
    source += "  $output_data_0[0, 0, gid.x] = value_0$;";

    std::vector<Variable> shared_variables = {
        // The actual size of sh_mem depends on the WorkgroupSize
        {"sh_mem", std::vector<float4>(0)},
    };

    *generated_code = {
        /*parameters=*/std::move(parameters),
        /*objects=*/std::move(objects),
        /*shared_variables=*/std::move(shared_variables),
        /*workload=*/uint3(dst_depth, kWorkgroupHintY, 1),
        /*workgroup=*/uint3(kWorkgroupHintX, kWorkgroupHintY, 1),
        /*source_code=*/std::move(source),
        /*input=*/IOStructure::ONLY_DEFINITIONS,
        /*output=*/IOStructure::ONLY_DEFINITIONS,
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
