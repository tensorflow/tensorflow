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

#include "tensorflow/lite/delegates/gpu/gl/kernels/mul.h"

#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/strings/str_cat.h"
#include "tensorflow/lite/delegates/gpu/common/convert.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {
namespace gl {

namespace {

// Returns the coordinate to iterate over the second runtime tensor.
absl::Status GetCoordinate(const NodeShader::GenerationContext& ctx, int dim,
                           const std::string& default_coord,
                           std::string* coord) {
  std::string result;
  if (ctx.input_shapes[1][dim] == 1 && ctx.input_shapes[0][dim] != 1) {
    result = "0";
  } else if (ctx.input_shapes[0][dim] == ctx.input_shapes[1][dim]) {
    result = default_coord;
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Second runtime tensor dimension ", dim,
                     " must either match "
                     "first tensor's dimensions or be 1."));
  }
  *coord = result;
  return absl::OkStatus();
}

absl::Status GenerateMultiplyRuntimeTensorCode(
    const NodeShader::GenerationContext& ctx, GeneratedCode* generated_code) {
  std::string x_coord, y_coord, z_coord;
  RETURN_IF_ERROR(
      GetCoordinate(ctx, /*dim=*/2, /*default_coord=*/"gid.x", &x_coord));
  RETURN_IF_ERROR(
      GetCoordinate(ctx, /*dim=*/1, /*default_coord=*/"gid.y", &y_coord));
  RETURN_IF_ERROR(
      GetCoordinate(ctx, /*dim=*/3, /*default_coord=*/"gid.z", &z_coord));

  std::string source =
      absl::StrCat("vec4 input1_value = $input_data_1[", x_coord, ", ", y_coord,
                   ", ", z_coord, "]$;");
  // Single channel mask support. Without this duplication, the rest of channels
  // will be zeros, which will make the mul operation produce incorrect result.
  if (ctx.input_shapes[1][3] == 1 && ctx.input_shapes[0][3] != 1) {
    absl::StrAppend(
        &source,
        "\ninput1_value = vec4(input1_value.x, input1_value.x, input1_value.x, "
        "input1_value.x);\n");
  }
  absl::StrAppend(
      &source, "value_0 = $input_data_0[gid.x, gid.y, gid.z]$ * input1_value;");

  *generated_code = {
      /*parameters=*/{},
      /*objects=*/{},
      /*shared_variables=*/{},
      /*workload=*/uint3(),
      /*workgroup=*/uint3(),
      /*source_code=*/std::move(source),
      /*input=*/IOStructure::ONLY_DEFINITIONS,
      /*output=*/IOStructure::AUTO,
  };
  return absl::OkStatus();
}

absl::Status GenerateMultiplyConstantTensorCode(
    const NodeShader::GenerationContext& ctx, GeneratedCode* generated_code) {
  const auto& attr = std::any_cast<const ElementwiseAttributes&>(ctx.op_attr);

  if (std::holds_alternative<float>(attr.param)) {
    *generated_code = {
        /*parameters=*/{{"scalar", std::get<float>(attr.param)}},
        /*objects=*/{},
        /*shared_variables=*/{},
        /*workload=*/uint3(),
        /*workgroup=*/uint3(),
        /*source_code=*/"value_0 *= $scalar$;",
        /*input=*/IOStructure::AUTO,
        /*output=*/IOStructure::AUTO,
    };
    return absl::OkStatus();
  }

  if (std::holds_alternative<Tensor<Linear, DataType::FLOAT32>>(attr.param)) {
    *generated_code = {
        /*parameters=*/{},
        /*objects=*/
        {{"mul_buffer",
          MakeReadonlyObject(
              std::get<Tensor<Linear, DataType::FLOAT32>>(attr.param).data)}},
        /*shared_variables=*/{},
        // Declare workload explicitly because shader depends on gid.z.
        /*workload=*/
        uint3(static_cast<int>(ctx.input_shapes[0][2]),
              static_cast<int>(ctx.input_shapes[0][1]),
              DivideRoundUp(static_cast<int>(ctx.input_shapes[0][3]), 4)),
        /*workgroup=*/uint3(),
        /*source_code=*/"value_0 *= $mul_buffer[gid.z]$;",
        /*input=*/IOStructure::AUTO,
        /*output=*/IOStructure::AUTO,
    };
    return absl::OkStatus();
  }

  if (std::holds_alternative<Tensor<HWC, DataType::FLOAT32>>(attr.param)) {
    std::string source;
    if (ctx.input_shapes[0][1] == 1 && ctx.input_shapes[0][2] == 1 &&
        ctx.input_shapes[0][3] == 1) {
      source = R"(
        value_0 = $input_data_0[0, 0, 0]$;
        value_0 = vec4(value_0.x, value_0.x, value_0.x, value_0.x);
      )";
    }

    auto param_shape =
        std::get<Tensor<HWC, DataType::FLOAT32>>(attr.param).shape;

    if (param_shape.c == 1) {
      if (param_shape.h == 1 && param_shape.w == 1) {
        absl::StrAppend(&source, "vec4 const_val = $hwc_buffer[0, 0, 0]$;");
      } else {
        absl::StrAppend(&source,
                        "vec4 const_val = $hwc_buffer[gid.x, gid.y, 0]$;");
      }
      absl::StrAppend(&source,
                      "const_val = vec4(const_val.x, const_val.x, const_val.x, "
                      "const_val.x);");
    } else {
      source += "vec4 const_val = $hwc_buffer[gid.x, gid.y, gid.z]$;";
    }

    absl::StrAppend(&source, "value_0 *= const_val;");

    *generated_code = {
        /*parameters=*/{},
        /*objects=*/
        {{"hwc_buffer",
          MakeReadonlyObject(
              uint3(param_shape.w, param_shape.h,
                    DivideRoundUp(param_shape.c, 4)),
              ConvertToPHWC4(
                  std::get<Tensor<HWC, DataType::FLOAT32>>(attr.param)))}},
        /*shared_variables=*/{},
        // Declare workload explicitly because shader depends on gid.z.
        /*workload=*/
        uint3(static_cast<int>(ctx.input_shapes[0][2]),
              static_cast<int>(ctx.input_shapes[0][1]),
              DivideRoundUp(static_cast<int>(ctx.input_shapes[0][3]), 4)),
        /*workgroup=*/uint3(),
        /*source_code=*/
        std::move(source),
        /*input=*/IOStructure::AUTO,
        /*output=*/IOStructure::AUTO,
    };
    return absl::OkStatus();
  }

  return absl::InvalidArgumentError("Unsupported Multiplication case.");
}

class Multiply : public NodeShader {
 public:
  absl::Status GenerateCode(const GenerationContext& ctx,
                            GeneratedCode* generated_code) const final {
    if (ctx.input_shapes.size() == 2) {
      return GenerateMultiplyRuntimeTensorCode(ctx, generated_code);
    } else {
      return GenerateMultiplyConstantTensorCode(ctx, generated_code);
    }
  }
};

}  // namespace

std::unique_ptr<NodeShader> NewMultiplyNodeShader() {
  return std::make_unique<Multiply>();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
