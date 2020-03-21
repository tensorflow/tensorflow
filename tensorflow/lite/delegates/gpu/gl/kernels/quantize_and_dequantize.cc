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
  Status GenerateCode(const GenerationContext& ctx,
                      GeneratedCode* generated_code) const final {
    std::string code;
    // Constants
    code += "vec4 scale = vec4($quant_scale$);";
    code += "vec4 min_bound = vec4($quant_min$);";
    code += "vec4 max_bound = vec4($quant_max$);";
    // Quantize
    code += "value_0 = clamp(value_0, min_bound, max_bound);";
    code += "value_0 = (value_0 - min_bound) / scale;";
    code += "value_0 = floor(value_0 + vec4(0.5));";
    // Dequantize
    code += "value_0 = value_0 * scale + min_bound;";

    auto attr = absl::any_cast<const QuantizeAndDequantizeAttributes&>(
        ctx.node->operation.attributes);
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
    return OkStatus();
  }
};

}  // namespace

std::unique_ptr<NodeShader> NewQuantizeAndDequantizeNodeShader() {
  return absl::make_unique<QuantizeAndDequantize>();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
