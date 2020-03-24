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

#include "tensorflow/lite/delegates/gpu/gl/kernels/prelu.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/lite/delegates/gpu/common/convert.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

class PReLULinearAlpha : public NodeShader {
 public:
  Status GenerateCode(const GenerationContext& ctx,
                      GeneratedCode* generated_code) const final {
    auto output = ctx.graph->FindOutputs(ctx.node->id)[0];
    auto attr =
        absl::any_cast<const PReLUAttributes&>(ctx.node->operation.attributes);
    auto alpha = absl::get_if<Tensor<Linear, DataType::FLOAT32>>(&attr.alpha);
    if (!alpha) {
      return InvalidArgumentError("Alpha is missing");
    }
    if (alpha->shape.v != output->tensor.shape.c) {
      return InvalidArgumentError(
          "Alpha shape does not match the number of channels.");
    }

    auto shape = output->tensor.shape;

    *generated_code =
        attr.clip
            ? GeneratedCode{
                  /*parameters=*/{{"clip", attr.clip}},
                  /*objects=*/{{"alpha", MakeReadonlyObject(alpha->data)}},
                  /*shared_variables=*/{},
                  /*workload=*/uint3(),
                  /*workgroup=*/uint3(),
                  "value_0 = clamp(value_0, 0.0, $clip$) + $alpha[gid.z]$ * "
                  "min(value_0, 0.0);",
                  /*input=*/IOStructure::AUTO,
                  /*output=*/IOStructure::AUTO,
              }
            : GeneratedCode{
                  /*parameters=*/{},
                  /*objects=*/{{"alpha", MakeReadonlyObject(alpha->data)}},
                  /*shared_variables=*/{},
                  // Declare workload explicitly because shader depends on
                  // gid.z.
                  /*workload=*/
                  uint3(shape.w, shape.h, IntegralDivideRoundUp(shape.c, 4)),
                  /*workgroup=*/uint3(),
                  /*source_code=*/
                  "value_0 = max(value_0, 0.0) + $alpha[gid.z]$ * min(value_0, "
                  "0.0);",
                  /*input=*/IOStructure::AUTO,
                  /*output=*/IOStructure::AUTO,
              };
    return OkStatus();
  }
};

class PReLUFull : public NodeShader {
 public:
  Status GenerateCode(const GenerationContext& ctx,
                      GeneratedCode* generated_code) const final {
    auto output = ctx.graph->FindOutputs(ctx.node->id)[0];
    auto attr =
        absl::any_cast<const PReLUAttributes&>(ctx.node->operation.attributes);
    auto alpha = absl::get_if<Tensor<HWC, DataType::FLOAT32>>(&attr.alpha);
    if (!alpha) {
      return InvalidArgumentError("Alpha is missing");
    }
    if (alpha->shape.h != output->tensor.shape.h ||
        alpha->shape.w != output->tensor.shape.w ||
        alpha->shape.c != output->tensor.shape.c) {
      return InvalidArgumentError("Alpha shape does not match input shape.");
    }

    auto shape = output->tensor.shape;

    ObjectSize obj_size =
        uint3(shape.w, shape.h, IntegralDivideRoundUp(shape.c, 4));

    *generated_code =
        attr.clip
            ? GeneratedCode{
                  /*parameters=*/{{"clip", attr.clip}},
                  /*objects=*/
                  {{"alpha",
                    MakeReadonlyObject(obj_size, ConvertToPHWC4(*alpha))}},
                  /*shared_variables=*/{},
                  // Declare workload explicitly because shader
                  // depends on gid.z.
                  /*workload=*/
                  uint3(shape.w, shape.h, IntegralDivideRoundUp(shape.c, 4)),
                  /*workgroup=*/uint3(),
                  /*source_code=*/
                  "value_0 = clamp(value_0, 0.0, $clip$) + "
                  "$alpha[gid.x, gid.y, gid.z]$ * min(value_0, 0.0);",
                  /*input=*/IOStructure::AUTO,
                  /*output=*/IOStructure::AUTO,
              }
            : GeneratedCode{
                  /*parameters=*/{},
                  /*objects=*/
                  {{"alpha",
                    MakeReadonlyObject(obj_size, ConvertToPHWC4(*alpha))}},
                  /*shared_variables=*/{},
                  // Declare workload explicitly because shader depends on
                  // gid.z.
                  /*workload=*/
                  uint3(shape.w, shape.h, IntegralDivideRoundUp(shape.c, 4)),
                  /*workgroup=*/uint3(),
                  /*source_code=*/
                  "value_0 = max(value_0, 0.0) + $alpha[gid.x, gid.y, gid.z]$ "
                  "* min(value_0, 0.0);",
                  /*input=*/IOStructure::AUTO,
                  /*output=*/IOStructure::AUTO,
              };
    return OkStatus();
  }
};

class PReLU : public NodeShader {
 public:
  Status GenerateCode(const GenerationContext& ctx,
                      GeneratedCode* generated_code) const final {
    auto attr =
        absl::any_cast<const PReLUAttributes&>(ctx.node->operation.attributes);
    auto alpha = absl::get_if<Tensor<HWC, DataType::FLOAT32>>(&attr.alpha);
    return alpha ? full_.GenerateCode(ctx, generated_code)
                 : linear_.GenerateCode(ctx, generated_code);
  }

 private:
  PReLULinearAlpha linear_;
  PReLUFull full_;
};

}  // namespace

std::unique_ptr<NodeShader> NewPReLUNodeShader() {
  return absl::make_unique<PReLU>();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
