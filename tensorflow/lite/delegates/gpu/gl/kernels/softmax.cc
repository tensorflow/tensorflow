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

#include <memory>
#include <string>
#include <utility>
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

float4 GetMask(int num_channels) {
  float4 mask(0.0f);
  const int remainder = num_channels % 4 == 0 ? 4 : num_channels % 4;
  for (int i = 0; i < remainder; ++i) mask[i] = 1.0f;
  return mask;
}

class Softmax : public NodeShader {
 public:
  Status GenerateCode(const GenerationContext& ctx,
                      GeneratedCode* generated_code) const final {
    const auto* input = ctx.graph->FindInputs(ctx.node->id)[0];
    const auto* output = ctx.graph->FindOutputs(ctx.node->id)[0];
    const auto& attr = absl::any_cast<const SoftmaxAttributes&>(
        ctx.node->operation.attributes);
    if (input->tensor.shape != output->tensor.shape) {
      return InvalidArgumentError("Input and output shapes do not match.");
    }
    if (attr.axis != Axis::CHANNELS) {
      return UnimplementedError("Softmax is only supported for channels axis.");
    }
    return input->tensor.shape.h == 1 && input->tensor.shape.w == 1
               ? GenerateCodeFor1x1(ctx, generated_code)
               : GenerateCodeGeneral(ctx, generated_code);
  }

 private:
  Status GenerateCodeFor1x1(const GenerationContext& ctx,
                            GeneratedCode* generated_code) const {
    const auto* output = ctx.graph->FindOutputs(ctx.node->id)[0];
    const int depth = IntegralDivideRoundUp(output->tensor.shape.c, 4);
    std::vector<Variable> shared_variables = {
        {"partial_sum", std::vector<float4>(8)},
    };
    std::vector<Variable> uniform_parameters = {
        {"depth", depth},
        {"depth_div_32", IntegralDivideRoundUp(depth, 32)},
        {"mask", GetMask(output->tensor.shape.c)},
    };
    std::string source_code = R"(
  highp vec4 kOnes = vec4(1.0);
  highp float sum = 0.0;
  int offset = 0;
  int s = 0;
  int tid = int(gl_LocalInvocationID.x);
  do {
    int z = offset + tid;
    if (z < $depth$) {
      highp vec4 mask_temp = z == $depth$ - 1 ? $mask$ : kOnes;
      highp vec4 src = $input_data_0[0, 0, z]$;
      sum += dot(mask_temp, exp(src));
      offset += 32;
    }
    s++;
  } while (s < $depth_div_32$);

  partial_sum[tid / 4][tid % 4] = sum;

  memoryBarrierShared();
  barrier();

  if (tid == 0) {
    sum = dot(kOnes, partial_sum[0]);
    sum += dot(kOnes, partial_sum[1]);
    sum += dot(kOnes, partial_sum[2]);
    sum += dot(kOnes, partial_sum[3]);
    sum += dot(kOnes, partial_sum[4]);
    sum += dot(kOnes, partial_sum[5]);
    sum += dot(kOnes, partial_sum[6]);
    sum += dot(kOnes, partial_sum[7]);
    partial_sum[0][0] = 1.0 / sum;
  }

  memoryBarrierShared();
  barrier();

  sum = partial_sum[0][0];

  offset = 0;
  s = 0;
  do {
    int z = offset + tid;
    if (z < $depth$) {
      highp vec4 src = $input_data_0[0, 0, z]$;
      highp vec4 temp = exp(src) * sum;
      $output_data_0[0, 0, z]$ = temp;
      offset += 32;
    }
    s++;
  } while (s < $depth_div_32$);
)";
    *generated_code = {
        /*parameters=*/std::move(uniform_parameters),
        /*objects=*/{},
        /*shared_variables=*/std::move(shared_variables),
        /*workload=*/uint3(32, 1, 1),
        /*workgroup=*/uint3(32, 1, 1),
        /*source_code=*/std::move(source_code),
        /*input=*/IOStructure::ONLY_DEFINITIONS,
        /*output=*/IOStructure::ONLY_DEFINITIONS,
    };
    return OkStatus();
  }

  Status GenerateCodeGeneral(const GenerationContext& ctx,
                             GeneratedCode* generated_code) const {
    const auto* output = ctx.graph->FindOutputs(ctx.node->id)[0];
    std::vector<Variable> parameters = {
        {"src_depth", IntegralDivideRoundUp(output->tensor.shape.c, 4)},
        {"mask", GetMask(output->tensor.shape.c)},
    };

    std::string source_code = R"(
  highp vec4 kOnes = vec4(1.0);
  highp float sum = 0.0;
  for (int d = 0; d < $src_depth$ - 1; ++d) {
    highp vec4 src = $input_data_0[gid.x, gid.y, d]$;
    sum += dot(kOnes, exp(src));
  }
  {
    int d = $src_depth$ - 1;
    highp vec4 src = $input_data_0[gid.x, gid.y, d]$;
    sum += dot($mask$, exp(src));
  }
  for (int d = 0; d < $src_depth$; ++d) {
    highp vec4 src = $input_data_0[gid.x, gid.y, d]$;
    highp vec4 temp_sum = exp(src) / sum;
    $output_data_0[gid.x, gid.y, d] = temp_sum$;
  }
)";
    *generated_code = {
        /*parameters=*/std::move(parameters),
        /*objects=*/{},
        /*shared_variables=*/{},
        /*workload=*/uint3(output->tensor.shape.w, output->tensor.shape.h, 1),
        /*workgroup=*/uint3(),
        /*source_code=*/std::move(source_code),
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
