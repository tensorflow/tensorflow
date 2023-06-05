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

#include <algorithm>
#include <any>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/lite/delegates/gpu/common/convert.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace gl {

namespace {

bool IsApplyMaskSupported(const NodeShader::GenerationContext& ctx) {
  if (ctx.input_shapes.size() != 2) return false;

  // [H, W, C] x [H, W, 0][0]
  if (ctx.input_shapes[0][1] == ctx.input_shapes[1][1] &&
      ctx.input_shapes[0][2] == ctx.input_shapes[1][2] &&
      ctx.input_shapes[1][3] == 1) {
    return true;
  }

  // [H, W, C] x [H, W, C]
  if (ctx.input_shapes[0] == ctx.input_shapes[1]) return true;

  // [H, W, C] x [0, 0, C]
  return ctx.input_shapes[1][1] == 1 && ctx.input_shapes[1][2] == 1 &&
         ctx.input_shapes[0][3] == ctx.input_shapes[1][3];
}

absl::Status GenerateApplyMaskCode(const NodeShader::GenerationContext& ctx,
                                   GeneratedCode* generated_code) {
  std::string source = "value_0 = $input_data_0[gid.x, gid.y, gid.z]$ * ";
  if (ctx.input_shapes[1][3] == 1) {
    // [H, W, C] x [H, W, 0][0]
    absl::StrAppend(&source, "$input_data_1[gid.x, gid.y, 0]$.x;");
  } else if (ctx.input_shapes[0][1] == ctx.input_shapes[1][1] &&
             ctx.input_shapes[0][2] == ctx.input_shapes[1][2]) {
    // [H, W, C] x [H, W, C]
    absl::StrAppend(&source, "$input_data_1[gid.x, gid.y, gid.z]$;");
  } else {
    // [H, W, C] x [0, 0, C]
    absl::StrAppend(&source, "$input_data_1[0, 0, gid.z]$;");
  }

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

absl::Status GenerateMultiplyScalarCode(
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
    *generated_code = {
        /*parameters=*/{},
        /*objects=*/
        {{"hwc_buffer",
          MakeReadonlyObject(
              uint3(static_cast<int>(ctx.input_shapes[0][2]),
                    static_cast<int>(ctx.input_shapes[0][1]),
                    DivideRoundUp(static_cast<int>(ctx.input_shapes[0][3]), 4)),
              ConvertToPHWC4(
                  std::get<Tensor<HWC, DataType::FLOAT32>>(attr.param)))}},
        /*shared_variables=*/{},
        // Declare workload explicitly because shader depends on gid.z.
        /*workload=*/
        uint3(static_cast<int>(ctx.input_shapes[0][2]),
              static_cast<int>(ctx.input_shapes[0][1]),
              DivideRoundUp(static_cast<int>(ctx.input_shapes[0][3]), 4)),
        /*workgroup=*/uint3(),
        /*source_code=*/"value_0 *= $hwc_buffer[gid.x, gid.y, gid.z]$;",
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
    if (IsApplyMaskSupported(ctx)) {
      return GenerateApplyMaskCode(ctx, generated_code);
    } else {
      return GenerateMultiplyScalarCode(ctx, generated_code);
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
