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

#include "tensorflow/lite/delegates/gpu/gl/kernels/sub.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

class Subtract : public NodeShader {
 public:
  static bool IsSupported(const GenerationContext& ctx) {
    auto inputs = ctx.graph->FindInputs(ctx.node->id);

    // Implementation supports concatenation of 2 tensors only.
    if (inputs.size() != 2) {
      return false;
    }

    auto shape0 = inputs[0]->tensor.shape;
    auto shape1 = inputs[1]->tensor.shape;

    // Shapes must be the same
    if (shape0 != shape1) {
      return false;
    }

    return true;
  }

  Status GenerateCode(const GenerationContext& ctx,
                      GeneratedCode* generated_code) const final {
    if (!IsSupported(ctx)) {
      return InvalidArgumentError(
          "This case is not supported by subtract operation");
    }
    *generated_code = {
        /*parameters=*/{},
        /*objects=*/{},
        /*workload=*/uint3(),
        /*workgroup=*/uint3(),
        /*source_code=*/"value_0 -= value_1;",
        /*input=*/IOStructure::AUTO,
        /*output=*/IOStructure::AUTO,
    };
    return OkStatus();
  }
};

}  // namespace

std::unique_ptr<NodeShader> NewSubtractNodeShader() {
  return absl::make_unique<Subtract>();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
