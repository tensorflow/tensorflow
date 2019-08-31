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
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

class ApplyMask : public NodeShader {
 public:
  static bool IsSupported(const GenerationContext& ctx) {
    const auto inputs = ctx.graph->FindInputs(ctx.node->id);
    if (inputs.size() != 2) return false;
    const auto& shape0 = inputs[0]->tensor.shape;
    const auto& shape1 = inputs[1]->tensor.shape;

    // [H, W, C] x [H, W, 0][0]
    if (shape1.c == 1) return true;

    if (shape0.c != shape1.c) return false;

    // [H, W, C] x [H, W, C]
    if (shape0.h == shape1.h && shape0.w == shape1.w) return true;

    // [H, W, C] x [0, 0, C]
    return shape1.h == 1 && shape1.w == 1;
  }

  Status GenerateCode(const GenerationContext& ctx,
                      GeneratedCode* generated_code) const final {
    if (!IsSupported(ctx)) {
      return InvalidArgumentError(
          "This case is not supported by apply mask operation");
    }
    const auto inputs = ctx.graph->FindInputs(ctx.node->id);
    const auto& shape0 = inputs[0]->tensor.shape;
    const auto& shape1 = inputs[1]->tensor.shape;

    std::string source = "value_0 = $input_data_0[gid.x, gid.y, gid.z]$ * ";
    if (shape1.c == 1) {
      // [H, W, C] x [H, W, 0][0]
      absl::StrAppend(&source, "$input_data_1[gid.x, gid.y, 0]$.x;");
    } else if (shape0.h == shape1.h && shape0.w == shape1.w) {
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
    return OkStatus();
  }
};

class MultiplyScalar : public NodeShader {
 public:
  Status GenerateCode(const GenerationContext& ctx,
                      GeneratedCode* generated_code) const final {
    auto attr = absl::any_cast<MultiplyScalarAttributes>(
        ctx.node->operation.attributes);
    auto muls = absl::get_if<Tensor<Linear, DataType::FLOAT32>>(&attr.param);
    auto scalar = absl::get_if<float>(&attr.param);

    if (scalar) {
      *generated_code = {
          /*parameters=*/{{"scalar", *scalar}},
          /*objects=*/{},
          /*shared_variables=*/{},
          /*workload=*/uint3(),
          /*workgroup=*/uint3(),
          /*source_code=*/"value_0 *= $scalar$;",
          /*input=*/IOStructure::AUTO,
          /*output=*/IOStructure::AUTO,
      };
    } else {
      if (!muls) {
        return InvalidArgumentError("Empty parameters for Multiplication.");
      }
      auto shape = ctx.graph->FindInputs(ctx.node->id)[0]->tensor.shape;
      *generated_code = {
          /*parameters=*/{},
          /*objects=*/{{"mul_buffer", MakeReadonlyObject(muls->data)}},
          /*shared_variables=*/{},
          // Declare workload explicitly because shader depends on gid.z.
          /*workload=*/
          uint3(shape.w, shape.h, IntegralDivideRoundUp(shape.c, 4)),
          /*workgroup=*/uint3(),
          /*source_code=*/"value_0 *= $mul_buffer[gid.z]$;",
          /*input=*/IOStructure::AUTO,
          /*output=*/IOStructure::AUTO,
      };
    }

    return OkStatus();
  }
};

}  // namespace

std::unique_ptr<NodeShader> NewApplyMaskNodeShader() {
  return absl::make_unique<ApplyMask>();
}

std::unique_ptr<NodeShader> NewMultiplyScalarNodeShader() {
  return absl::make_unique<MultiplyScalar>();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
