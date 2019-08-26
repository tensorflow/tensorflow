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

constexpr int kWorkPerThread = 4;
constexpr int kVectorizedWidth = 4;  // Also number of 'offsetN' in kernel.

class FullyConnectedBuffers : public NodeShader {
 public:
  Status GenerateCode(const GenerationContext& ctx,
                      GeneratedCode* generated_code) const final {
    auto attr = absl::any_cast<const FullyConnectedAttributes&>(
        ctx.node->operation.attributes);

    // Number of float4 chunks needed.
    const int src_depth = IntegralDivideRoundUp(attr.weights.shape.i, 4);
    const int dst_depth = IntegralDivideRoundUp(attr.weights.shape.o, 4);

    // TODO(akulik): check that input has h,w == 1,1
    std::vector<Variable> parameters = {
        {"src_depth", src_depth},
        {"src_depth_x4", IntegralDivideRoundUp(src_depth, kVectorizedWidth)},
        {"src_size", attr.weights.shape.i},
        {"dst_depth", dst_depth},
        {"dst_size", attr.weights.shape.o},
    };

    // TODO(akulik): refactor indexed access to weights.
    std::vector<std::pair<std::string, Object>> objects = {
        {"weights", MakeReadonlyObject(ConvertToPHWO4I4(attr.weights))}};

    std::string source = R"(
  // setup
  ivec2 tid = ivec2(gl_LocalInvocationID.xy);
  vec4 sum = vec4(0.0);  // accumulator
  int channel = int(tid.y);  // vector coord for every thread
  int work_per_thread = int(gl_WorkGroupSize.x);

  // matrix vector workgroup mul
  uint offset0 = uint(gid.x * $src_depth$ * 4 + tid.y * 4 + 0);
  uint offset1 = uint(gid.x * $src_depth$ * 4 + tid.y * 4 + 1);
  uint offset2 = uint(gid.x * $src_depth$ * 4 + tid.y * 4 + 2);
  uint offset3 = uint(gid.x * $src_depth$ * 4 + tid.y * 4 + 3);
  uint offset_stride = 16u;  // src_depth_x4 == (src_size / 16)
  for (int i = 0; i < $src_depth_x4$; ++i, channel += int(4)) {
    vec4 v = $input_data_0[0, 0, channel]$;
    vec4 m0 = $weights[ offset0 ]$;
    vec4 m1 = $weights[ offset1 ]$;
    vec4 m2 = $weights[ offset2 ]$;
    vec4 m3 = $weights[ offset3 ]$;
    offset0 += offset_stride;
    offset1 += offset_stride;
    offset2 += offset_stride;
    offset3 += offset_stride;
    sum.x += dot(v, m0);  // matrix * vector
    sum.y += dot(v, m1);
    sum.z += dot(v, m2);
    sum.w += dot(v, m3);
  }

  // accumulate local partial sums
  sh_mem[tid.x + tid.y * work_per_thread] = sum;
  memoryBarrierShared();
  barrier();

  // accumulate global sums, write results
  if (tid.y == 0 && gid.x < $dst_depth$) {
    /*sum+=sh_mem[tid.x + 0 * work_per_thread];*/  // current thread
    sum += sh_mem[tid.x + 1 * work_per_thread];
    sum += sh_mem[tid.x + 2 * work_per_thread];
    sum += sh_mem[tid.x + 3 * work_per_thread];
    vec4 r0 = sum;
)" + std::string(attr.bias.data.empty() ? R"( )" : R"(
    r0 += $bias[gid.x]$;  )") +
                         std::string(R"(
    $output_data_0[0, 0, gid.x] = r0$;
  }
)");
    if (!attr.bias.data.empty()) {
      objects.push_back({"bias", MakeReadonlyObject(attr.bias.data)});
    }

    std::vector<Variable> shared_variables = {
        {"sh_mem", std::vector<float4>(kWorkPerThread * kVectorizedWidth)},
    };

    *generated_code = {
        /*parameters=*/std::move(parameters),
        /*objects=*/std::move(objects),
        /*shared_variables=*/std::move(shared_variables),
        /*workload=*/uint3(dst_depth, kVectorizedWidth, 1),
        /*workgroup=*/uint3(kWorkPerThread, kVectorizedWidth, 1),
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
