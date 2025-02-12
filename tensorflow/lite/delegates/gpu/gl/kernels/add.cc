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

#include "tensorflow/lite/delegates/gpu/gl/kernels/add.h"

#include <any>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/strings/str_cat.h"
#include "tensorflow/lite/delegates/gpu/common/convert.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

class Add : public NodeShader {
 public:
  absl::Status GenerateCode(const GenerationContext& ctx,
                            GeneratedCode* generated_code) const final {
    const auto& attr = std::any_cast<const ElementwiseAttributes&>(ctx.op_attr);
    auto adds = std::get_if<Tensor<Linear, DataType::FLOAT32>>(&attr.param);
    auto scalar = std::get_if<float>(&attr.param);

    const auto* hwc_tensor =
        std::get_if<Tensor<HWC, DataType::FLOAT32>>(&attr.param);

    if (hwc_tensor) {
      std::string code;
      const std::string x_coord = hwc_tensor->shape.w == 1 ? "0" : "gid.x";
      const std::string y_coord = hwc_tensor->shape.h == 1 ? "0" : "gid.y";
      const std::string s_coord = hwc_tensor->shape.c == 1 ? "0" : "gid.z";
      code = absl::StrCat("vec4 second_val = $hwc_buffer[", x_coord, ", ",
                          y_coord, ", ", s_coord, "]$;\n");
      if (hwc_tensor->shape.c == 1) {
        code += "  second_val.y = second_val.x;\n";
        code += "  second_val.z = second_val.x;\n";
        code += "  second_val.w = second_val.x;\n";
      }
      code += "  value_0 += second_val;\n";
      *generated_code = {
          /*parameters=*/{},
          /*objects=*/
          {{"hwc_buffer",
            MakeReadonlyObject(
                uint3(hwc_tensor->shape.w, hwc_tensor->shape.h,
                      DivideRoundUp(hwc_tensor->shape.c, 4)),
                ConvertToPHWC4(
                    std::get<Tensor<HWC, DataType::FLOAT32>>(attr.param)))}},
          /*shared_variables=*/{},
          // Declare workload explicitly because shader depends on gid.z.
          /*workload=*/
          uint3(static_cast<int>(ctx.input_shapes[0][2]),
                static_cast<int>(ctx.input_shapes[0][1]),
                DivideRoundUp(static_cast<int>(ctx.input_shapes[0][3]), 4)),
          /*workgroup=*/uint3(),
          /*source_code=*/std::move(code),
          /*input=*/IOStructure::AUTO,
          /*output=*/IOStructure::AUTO,
      };
      return absl::OkStatus();
    }

    if (!adds && !scalar) {
      // check if it is a broadcast
      if (ctx.input_shapes.size() == 2 &&
          ctx.input_shapes[0] != ctx.input_shapes[1] &&
          ctx.input_shapes[1][1] == 1 && ctx.input_shapes[1][2] == 1 &&
          ctx.input_shapes[0][3] == ctx.input_shapes[1][3]) {
        // TODO(b/147771327): investigate why input_data_1[gid.z] worked before
        *generated_code = {
            /*parameters=*/{},
            /*objects=*/{},
            /*shared_variables=*/{},
            /*workload=*/uint3(),
            /*workgroup=*/uint3(),
            /*source_code=*/
            "value_0 = $input_data_0[gid.x, gid.y, gid.z]$ + "
            "          $input_data_1[0, 0, gid.z]$;",
            /*input=*/IOStructure::ONLY_DEFINITIONS,
            /*output=*/IOStructure::AUTO,
        };
        return absl::OkStatus();
      }

      std::string code = "value_0 = value_0";
      for (int index = 1; index < ctx.input_shapes.size(); ++index) {
        if (ctx.input_shapes[index] != ctx.input_shapes[0]) {
          return absl::InvalidArgumentError("Shapes are not equal");
        }
        absl::StrAppend(&code, " + value_", index);
      }
      absl::StrAppend(&code, ";");
      *generated_code = {
          /*parameters=*/{},
          /*objects=*/{},
          /*shared_variables=*/{},
          /*workload=*/uint3(),
          /*workgroup=*/uint3(),
          /*source_code=*/std::move(code),
          /*input=*/IOStructure::AUTO,
          /*output=*/IOStructure::AUTO,
      };
      return absl::OkStatus();
    }

    if (scalar) {
      *generated_code = {
          /*parameters=*/{{"scalar", *scalar}},
          /*objects=*/{},
          /*shared_variables=*/{},
          /*workload=*/uint3(),
          /*workgroup=*/uint3(),
          /*source_code=*/"value_0 += $scalar$;",
          /*input=*/IOStructure::AUTO,
          /*output=*/IOStructure::AUTO,
      };
      return absl::OkStatus();
    }

    *generated_code = {
        /*parameters=*/{},
        /*objects=*/{{"add_buffer", MakeReadonlyObject(adds->data)}},
        /*shared_variables=*/{},
        // Declare workload explicitly because shader depends on gid.z.
        /*workload=*/
        uint3(ctx.input_shapes[0][2], ctx.input_shapes[0][1],
              DivideRoundUp(ctx.input_shapes[0][3], 4)),
        /*workgroup=*/uint3(),
        /*source_code=*/"value_0 += $add_buffer[gid.z]$;",
        /*input=*/IOStructure::AUTO,
        /*output=*/IOStructure::AUTO,
    };
    return absl::OkStatus();
  }
};

}  // namespace

std::unique_ptr<NodeShader> NewAddNodeShader() {
  return std::make_unique<Add>();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
