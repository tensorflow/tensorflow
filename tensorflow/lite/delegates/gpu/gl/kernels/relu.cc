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

#include "tensorflow/lite/delegates/gpu/gl/kernels/relu.h"

#include <algorithm>
#include <any>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/gl/variable.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

class ReLU : public NodeShader {
 public:
  absl::Status GenerateCode(const GenerationContext& ctx,
                            GeneratedCode* generated_code) const final {
    const auto& attr = std::any_cast<const ReLUAttributes&>(ctx.op_attr);
    // clamp(value, min(0, alpha * value), activation_max)
    std::vector<Variable> params;
    std::string min;
    if (attr.alpha == 0) {
      min = "vec4($activation_min$)";
      params.push_back({"activation_min", attr.activation_min});
    } else {
      min = "min($alpha$ * value_0, 0.0)";
      params.push_back({"alpha", attr.alpha});
    }
    std::string code;
    if (attr.activation_max == 0) {
      code = "value_0 = max(value_0, " + min + ");";
    } else {
      code = "value_0 = clamp(value_0, " + min + ", vec4($activation_max$));";
      params.push_back({"activation_max", attr.activation_max});
    }
    *generated_code = {
        /*parameters=*/std::move(params),
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
};

}  // namespace

std::unique_ptr<NodeShader> NewReLUNodeShader() {
  return std::make_unique<ReLU>();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
