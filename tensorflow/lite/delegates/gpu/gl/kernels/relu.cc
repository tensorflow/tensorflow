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
#include <cstdint>
#include <cstring>
#include <string>
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
  Status GenerateCode(const GenerationContext& ctx,
                      GeneratedCode* generated_code) const final {
    auto attr = absl::any_cast<ReLUAttributes>(ctx.node->operation.attributes);
    // clamp(value, min(0, alpha * value), clip)
    std::vector<Variable> params;
    std::string min;
    if (attr.alpha == 0) {
      min = "vec4(0.0)";
    } else {
      min = "min($alpha$ * value_0, 0.0)";
      params.push_back({"alpha", attr.alpha});
    }
    std::string code;
    if (attr.clip == 0) {
      code = "value_0 = max(value_0, " + min + ");";
    } else {
      code = "value_0 = clamp(value_0, " + min + ", vec4($clip$));";
      params.push_back({"clip", attr.clip});
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
    return OkStatus();
  }
};

}  // namespace

std::unique_ptr<NodeShader> NewReLUNodeShader() {
  return absl::make_unique<ReLU>();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
