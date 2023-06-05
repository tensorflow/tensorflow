/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/gl/kernels/quantize_and_dequantize.h"

#include <any>
#include <memory>
#include <string>

#include "absl/memory/memory.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

class QuantizeAndDequantize : public NodeShader {
 public:
  absl::Status GenerateCode(const GenerationContext& ctx,
                            GeneratedCode* generated_code) const final {
    std::string code = R"(
value_0 = clamp(value_0, vec4($quant_min$), vec4($quant_max$));
value_0 = (value_0 - vec4($quant_min$)) / vec4($quant_scale$);
value_0 = floor(value_0 + vec4(0.5));
value_0 = value_0 * vec4($quant_scale$) + vec4($quant_min$);
)";

    const auto& attr =
        std::any_cast<const QuantizeAndDequantizeAttributes&>(ctx.op_attr);
    *generated_code = {
        /*parameters=*/{{"quant_min", attr.min},
                        {"quant_max", attr.max},
                        {"quant_scale", attr.scale}},
        /*objects=*/{},
        /*shared_variables=*/{},
        /*workload=*/uint3(),
        /*workgroup=*/uint3(),
        /*source_code=*/code,
        /*input=*/IOStructure::AUTO,
        /*output=*/IOStructure::AUTO,
    };
    return absl::OkStatus();
  }
};

}  // namespace

std::unique_ptr<NodeShader> NewQuantizeAndDequantizeNodeShader() {
  return std::make_unique<QuantizeAndDequantize>();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
