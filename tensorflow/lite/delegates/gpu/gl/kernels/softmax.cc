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

#include "tensorflow/lite/delegates/gpu/gl/kernels/softmax.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/gl/variable.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

class Softmax : public NodeShader {
 public:
  Status GenerateCode(const GenerationContext& ctx,
                      GeneratedCode* generated_code) const final {
    const auto* input = ctx.graph->FindInputs(ctx.node->id)[0];
    const auto* output = ctx.graph->FindOutputs(ctx.node->id)[0];
    const auto& attr = absl::any_cast<const SoftmaxAttributes&>(
        ctx.node->operation.attributes);
    if (input->tensor.shape != output->tensor.shape) {
      return InvalidArgumentError("Input and output shape does not match");
    }
    if (attr.axis != Axis::CHANNELS) {
      return UnimplementedError("Softmax is only supported for channels axis.");
    }

    float4 mask(0.0f);
    const int channels = output->tensor.shape.c;
    const int reminder = (channels % 4 == 0) ? 4 : channels % 4;
    for (int i = 0; i < reminder; ++i) {
      mask[i] = 1.0f;
    }
    std::vector<Variable> parameters = {
        {"src_depth", IntegralDivideRoundUp(output->tensor.shape.c, 4)},
        {"mask", mask},
    };

    std::string source = R"(
  highp float sum = 0.0;
  for (int d = 0; d < $src_depth$ - 1; ++d) {
    highp vec4 v = $input_data_0[gid.x, gid.y, d]$;
    sum += dot(vec4(1.0), exp(v));
  }
  {
    int d = $src_depth$ - 1;
    highp vec4 v = $input_data_0[gid.x, gid.y, d]$;
    sum += dot($mask$, exp(v));
  }
  for (int d = 0; d < $src_depth$; ++d) {
    highp vec4 v = $input_data_0[gid.x, gid.y, d]$;
    vec4 temp_sum = exp(v) / sum;
    $output_data_0[gid.x, gid.y, d] = temp_sum$;
  }
)";
    *generated_code = {
        /*parameters=*/std::move(parameters),
        /*objects=*/{},
        /*shared_variables=*/{},
        /*workload=*/uint3(output->tensor.shape.w, output->tensor.shape.h, 1),
        /*workgroup=*/uint3(),
        /*source_code=*/std::move(source),
        /*input=*/IOStructure::ONLY_DEFINITIONS,
        /*output=*/IOStructure::ONLY_DEFINITIONS,
    };
    return OkStatus();
  }
};

}  // namespace

std::unique_ptr<NodeShader> NewSoftmaxNodeShader() {
  return absl::make_unique<Softmax>();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
