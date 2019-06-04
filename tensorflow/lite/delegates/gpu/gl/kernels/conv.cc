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

#include "tensorflow/lite/delegates/gpu/gl/kernels/conv.h"

#include <memory>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/lite/delegates/gpu/common/convert.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/gl/node_shader.h"
#include "tensorflow/lite/delegates/gpu/gl/workgroups/ideal_workgroup_picker.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

class Convolution : public NodeShader {
 public:
  Status GenerateCode(const GenerationContext& ctx,
                      GeneratedCode* generated_code) const final {
    auto input = ctx.graph->FindInputs(ctx.node->id)[0];
    auto attr = absl::any_cast<const Convolution2DAttributes&>(
        ctx.node->operation.attributes);
    auto weights = attr.weights.shape;
    const int offsets_count = weights.h * weights.w;
    std::vector<int2> offsets;
    for (int h = 0; h < weights.h; ++h) {
      for (int w = 0; w < weights.w; ++w) {
        offsets.emplace_back(w * attr.dilations.w - attr.padding.prepended.w,
                             h * attr.dilations.h - attr.padding.prepended.h);
      }
    }
    std::vector<UniformParameter> parameters = {
        {"input_data_0_h", input->tensor.shape.h},
        {"input_data_0_w", input->tensor.shape.w},
        {"offsets_count", offsets_count},
        {"offsets", offsets},
        {"src_depth", IntegralDivideRoundUp(weights.i, 4)},
        {"stride", int2(attr.strides.w, attr.strides.h)},
    };

    // at least one padding is not empty
    bool non_empty_padding =
        attr.padding.appended.h != 0 || attr.padding.appended.w != 0 ||
        attr.padding.prepended.h != 0 || attr.padding.prepended.w != 0;

    std::vector<std::pair<std::string, Object>> objects = {
        {"weights", MakeReadonlyObject(Get3DSizeForPHWO4I4(attr.weights.shape),
                                       ConvertToPHWO4I4(attr.weights))}};

    std::string source = R"(
      for (int i = 0; i < $offsets_count$; ++i) {
        ivec2 coord = gid.xy * $stride$ + $offsets[i]$;)";
    if (non_empty_padding) {
      source += R"(
        if (coord.x < 0 || coord.y < 0 || coord.x >= $input_data_0_w$ || coord.y >= $input_data_0_h$) {
          continue;
        })";
    }
    source += R"(
        for (int l = 0; l < $src_depth$; ++l) {
          highp vec4 input_ = $input_data_0[coord.x, coord.y, l]$;
          value_0.x += dot(input_, $weights[l * 4 + 0, i, gid.z]$);
          value_0.y += dot(input_, $weights[l * 4 + 1, i, gid.z]$);
          value_0.z += dot(input_, $weights[l * 4 + 2, i, gid.z]$);
          value_0.w += dot(input_, $weights[l * 4 + 3, i, gid.z]$);
        }
      }
    )";
    if (!attr.bias.data.empty()) {
      source += "value_0 += $bias[gid.z]$;\n";
      objects.push_back({"bias", MakeReadonlyObject(attr.bias.data)});
    }

    // This is a hotfix for special convolution, which worked 10ms on
    // textures16. With this fix it works 4ms.
    // TODO(eignasheva): fix this problem in the proper way
    uint3 workgroup = uint3(0, 0, 0);
    if (weights.h == 7 && weights.w == 7 && attr.strides.h == 4 &&
        attr.strides.w == 4) {
      workgroup = uint3(8, 8, 8);
    }

    *generated_code = {
        /*parameters=*/std::move(parameters),
        /*objects=*/std::move(objects),
        /*workload=*/uint3(),
        /*workgroup=*/
        GetIdealWorkgroupIfPossible(
            ctx.gpu_info->gpu_model, OperationType::CONVOLUTION_2D,
            HW(weights.h, weights.w), attr.strides, workgroup,
            OHWI(weights.o, input->tensor.shape.h, input->tensor.shape.w,
                 input->tensor.shape.c)),
        /*source_code=*/std::move(source),
        /*input=*/IOStructure::ONLY_DEFINITIONS,
        /*output=*/IOStructure::AUTO,
    };
    return OkStatus();
  }
};

int SelectMultiplier(int32_t input_width,
                     const NodeShader::GenerationContext& ctx) {
  std::vector<int> multipliers = {4, 2};
  if (!ctx.compiler_options.allow_precision_loss &&
      ctx.gpu_info->type == GpuType::MALI) {
    multipliers = {2};
  }
  for (int i : multipliers) {
    if (input_width % i == 0) {
      return i;
    }
  }
  return 1;
}

class Convolution1x1 : public NodeShader {
 public:
  Status GenerateCode(const GenerationContext& ctx,
                      GeneratedCode* generated_code) const final {
    auto input = ctx.graph->FindInputs(ctx.node->id)[0];
    auto output = ctx.graph->FindOutputs(ctx.node->id)[0];
    auto attr = absl::any_cast<const Convolution2DAttributes&>(
        ctx.node->operation.attributes);
    if (attr.weights.shape.h != 1 || attr.weights.shape.w != 1) {
      return UnimplementedError("Height and width should be 1.");
    }
    if (attr.dilations.h != 1 || attr.dilations.w != 1) {
      return UnimplementedError("Dilations are not supported.");
    }
    if (attr.strides.h != 1 || attr.strides.w != 1) {
      return UnimplementedError("Strides are not supported.");
    }
    if (attr.padding.appended.h != 0 || attr.padding.appended.w != 0 ||
        attr.padding.prepended.h != 0 || attr.padding.prepended.w != 0) {
      return UnimplementedError("Padding is not supported.");
    }

    int multiplier = SelectMultiplier(input->tensor.shape.w, ctx);

    std::vector<UniformParameter> parameters = {
        {"src_depth", IntegralDivideRoundUp(input->tensor.shape.c, 4)},
    };

    std::vector<std::pair<std::string, Object>> objects = {
        {"weights", MakeReadonlyObject(
                        uint3(4, IntegralDivideRoundUp(attr.weights.shape.i, 4),
                              IntegralDivideRoundUp(attr.weights.shape.o, 4)),
                        ConvertToPHWO4I4(attr.weights))}};
    std::string source;
    for (int i = 0; i < multiplier; i++) {
      absl::StrAppend(&source, "highp vec4 result", i, " = vec4(0);\n");
    }
    absl::StrAppend(&source, "vec4 f;\n");
    absl::StrAppend(&source, "for (int l = 0; l < $src_depth$; ++l) {\n");
    for (int i = 0; i < multiplier; i++) {
      absl::StrAppend(&source, "  vec4 input", i, " = $input_data_0[gid.x * ",
                      multiplier, " + ", i, ",gid.y,l]$;\n");
    }
    for (int k = 0; k < 4; k++) {
      absl::StrAppend(&source, "  f = $weights[", k, ", l, gid.z]$;\n");
      for (int i = 0; i < multiplier; i++) {
        absl::StrAppend(&source, "  result", i, "[", k, "] += dot(input", i,
                        ", f);\n");
      }
    }
    absl::StrAppend(&source, "}\n");
    if (!attr.bias.data.empty()) {
      objects.push_back({"bias", MakeReadonlyObject(attr.bias.data)});
      absl::StrAppend(&source, "vec4 b = $bias[gid.z]$;\n");
      for (int i = 0; i < multiplier; i++) {
        absl::StrAppend(&source, "result", i, " += b;\n");
      }
    }
    if (multiplier != 1) {
      for (int i = 0; i < multiplier; i++) {
        absl::StrAppend(&source, "$inplace_update:result", i, "$\n");
        absl::StrAppend(&source, "$output_data_0[gid.x * ", multiplier, " + ",
                        i, ",gid.y,gid.z] = result", i, "$;\n");
      }
    } else {
      absl::StrAppend(&source, "value_0 = result0;\n");
    }

    auto dst_depth = IntegralDivideRoundUp(output->tensor.shape.c, 4);
    uint3 workgroup = uint3(16, 16, 1);
    if (ctx.gpu_info->type == GpuType::ADRENO) {
      if (dst_depth >= 2) {
        workgroup = uint3(8, 8, 2);
      }
      if (dst_depth >= 4) {
        workgroup = uint3(4, 8, 4);
      }
      if (dst_depth >= 8) {
        workgroup = uint3(4, 4, 8);
      }
      if (dst_depth >= 32) {
        workgroup = uint3(4, 4, 16);
      }
      if (dst_depth >= 64) {
        workgroup = uint3(2, 8, 16);
      }
    } else {
      if (dst_depth >= 2) {
        workgroup = uint3(16, 8, 2);
      }
      if (dst_depth >= 4) {
        workgroup = uint3(16, 4, 4);
      }
      if (dst_depth >= 8) {
        workgroup = uint3(8, 4, 8);
      }
      if (dst_depth >= 32) {
        workgroup = uint3(8, 4, 8);
      }
      if (dst_depth >= 64) {
        workgroup = uint3(8, 4, 8);
      }
    }
    *generated_code = {
        /*parameters=*/std::move(parameters),
        /*objects=*/std::move(objects),
        /*workload=*/
        uint3(output->tensor.shape.w / multiplier, output->tensor.shape.h,
              IntegralDivideRoundUp(output->tensor.shape.c, 4)),
        /*workgroup=*/
        GetIdealWorkgroupIfPossible(
            ctx.gpu_info->gpu_model, OperationType::CONVOLUTION_2D,
            HW(attr.weights.shape.h, attr.weights.shape.w), attr.strides,
            workgroup,
            OHWI(attr.weights.shape.o, input->tensor.shape.h,
                 input->tensor.shape.w, input->tensor.shape.c)),
        /*source_code=*/std::move(source),
        /*input=*/IOStructure::ONLY_DEFINITIONS,
        /*output=*/multiplier == 1 ? IOStructure::AUTO
                                   : IOStructure::ONLY_DEFINITIONS,
    };
    return OkStatus();
  }
};

}  // namespace

std::unique_ptr<NodeShader> NewConvolutionNodeShader() {
  return absl::make_unique<Convolution>();
}

std::unique_ptr<NodeShader> NewConvolution1x1NodeShader() {
  return absl::make_unique<Convolution1x1>();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
