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
  absl::Status GenerateCode(const GenerationContext& ctx,
                            GeneratedCode* generated_code) const final {
    const auto& attr = absl::any_cast<const SoftmaxAttributes&>(ctx.op_attr);
    if (ctx.input_shapes[0] != ctx.output_shapes[0]) {
      return absl::InvalidArgumentError(
          "Input and output shapes do not match.");
    }
    if (attr.axis != Axis::CHANNELS) {
      return absl::UnimplementedError(
          "Softmax is only supported for channels axis.");
    }
    return ctx.input_shapes[0][1] == 1 && ctx.input_shapes[0][2] == 1
               ? GenerateCodeFor1x1(ctx, generated_code)
               : GenerateCodeGeneral(ctx, generated_code);
  }

 private:
  absl::Status GenerateCodeFor1x1(const GenerationContext& ctx,
                                  GeneratedCode* generated_code) const {
    const int depth = DivideRoundUp(ctx.output_shapes[0][3], 4);
    std::vector<Variable> shared_variables = {
        {"partial_sum", std::vector<float4>(8)},
    };
    std::vector<Variable> uniform_parameters = {
        {"depth", depth},
        {"mask", GetMask(ctx.output_shapes[0][3])},
    };
    std::string source_code = R"(
  highp vec4 kOnes = vec4(1.0);
  int tid = int(gl_LocalInvocationID.x);
  highp vec4 maxx4 = $input_data_0[0, 0, 0]$;
  maxx4.y = maxx4.x;
  maxx4.z = maxx4.x;
  maxx4.w = maxx4.x;
  for (int s = tid; s < $depth$; s += 32) {
    highp vec4 mask_a = s == $depth$ - 1 ? $mask$ : kOnes;
    highp vec4 mask_b = kOnes - mask_a;
    highp vec4 src = $input_data_0[0, 0, s]$;
    src = src * mask_a + mask_b * src.x;
    maxx4 = max(maxx4, src);
  }
  highp float maximum = max(maxx4.x, maxx4.y);
  maximum = max(maximum, maxx4.z);
  maximum = max(maximum, maxx4.w);
  partial_sum[tid / 4][tid % 4] = maximum;

  memoryBarrierShared();
  barrier();

  if (tid == 0) {
    maxx4 = max(partial_sum[0], partial_sum[1]);
    maxx4 = max(maxx4, partial_sum[2]);
    maxx4 = max(maxx4, partial_sum[3]);
    maxx4 = max(maxx4, partial_sum[4]);
    maxx4 = max(maxx4, partial_sum[5]);
    maxx4 = max(maxx4, partial_sum[6]);
    maxx4 = max(maxx4, partial_sum[7]);
    maximum = max(maxx4.x, maxx4.y);
    maximum = max(maximum, maxx4.z);
    maximum = max(maximum, maxx4.w);
    partial_sum[0][0] = maximum;
  }

  memoryBarrierShared();
  barrier();

  maximum = partial_sum[0][0];

  highp float sum = 0.0;
  for (int s = tid; s < $depth$; s += 32) {
    highp vec4 mask_temp = s == $depth$ - 1 ? $mask$ : kOnes;
    highp vec4 src = $input_data_0[0, 0, s]$ - vec4(maximum);
    sum += dot(mask_temp, exp(src));
  }

  memoryBarrierShared();
  barrier();

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

  int dst_s = int(gl_GlobalInvocationID.x);
  if (dst_s < $depth$) {
    highp vec4 src = $input_data_0[0, 0, dst_s]$ - vec4(maximum);
    highp vec4 temp = exp(src) * sum;
    $output_data_0[0, 0, dst_s] = temp$;
  }
)";

    *generated_code = {
        /*parameters=*/std::move(uniform_parameters),
        /*objects=*/{},
        /*shared_variables=*/std::move(shared_variables),
        /*workload=*/uint3(depth, 1, 1),
        /*workgroup=*/uint3(32, 1, 1),
        /*source_code=*/std::move(source_code),
        /*input=*/IOStructure::ONLY_DEFINITIONS,
        /*output=*/IOStructure::ONLY_DEFINITIONS,
    };
    return absl::OkStatus();
  }

  absl::Status GenerateCodeGeneral(const GenerationContext& ctx,
                                   GeneratedCode* generated_code) const {
    std::vector<Variable> parameters = {
        {"src_depth",
         DivideRoundUp(static_cast<int>(ctx.output_shapes[0][3]), 4)},
        {"mask", GetMask(ctx.output_shapes[0][3])},
    };

    std::string source_code = R"(
  highp vec4 kOnes = vec4(1.0);
  highp float sum = 0.0;
  highp float maximum = $input_data_0[gid.x, gid.y, 0]$.x;
  for (int d = 0; d < $src_depth$; ++d) {
    highp vec4 mask_a = d == $src_depth$ - 1 ? $mask$ : kOnes;
    highp vec4 mask_b = kOnes - mask_a;
    highp vec4 src = $input_data_0[gid.x, gid.y, d]$;
    src = src * mask_a + mask_b * src.x;
    maximum = max(maximum, src.x);
    maximum = max(maximum, src.y);
    maximum = max(maximum, src.z);
    maximum = max(maximum, src.w);
  }
  for (int d = 0; d < $src_depth$; ++d) {
    highp vec4 mask_temp = d == $src_depth$ - 1 ? $mask$ : kOnes;
    highp vec4 src = $input_data_0[gid.x, gid.y, d]$ - vec4(maximum);
    sum += dot(mask_temp, exp(src));
  }
  for (int d = 0; d < $src_depth$; ++d) {
    highp vec4 src = $input_data_0[gid.x, gid.y, d]$ - vec4(maximum);
    highp vec4 temp_sum = exp(src) / sum;
    $output_data_0[gid.x, gid.y, d] = temp_sum$;
  }
)";
    *generated_code = {
        /*parameters=*/std::move(parameters),
        /*objects=*/{},
        /*shared_variables=*/{},
        /*workload=*/
        uint3(static_cast<int>(ctx.output_shapes[0][2]),
              static_cast<int>(ctx.output_shapes[0][1]), 1),
        /*workgroup=*/uint3(),
        /*source_code=*/std::move(source_code),
        /*input=*/IOStructure::ONLY_DEFINITIONS,
        /*output=*/IOStructure::ONLY_DEFINITIONS,
    };
    return absl::OkStatus();
  }
};

}  // namespace

std::unique_ptr<NodeShader> NewSoftmaxNodeShader() {
  return absl::make_unique<Softmax>();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
