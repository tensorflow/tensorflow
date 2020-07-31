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

#include "tensorflow/lite/delegates/gpu/gl/kernels/mean.h"

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

class Mean : public NodeShader {
 public:
  absl::Status GenerateCode(const GenerationContext& ctx,
                            GeneratedCode* generated_code) const final {
    const auto& attr = absl::any_cast<const MeanAttributes&>(ctx.op_attr);
    if (attr.dims != std::set<Axis>({Axis::HEIGHT, Axis::WIDTH})) {
      return absl::InvalidArgumentError(
          "Mean calculation is supported only for height and width.");
    }

    std::vector<Variable> parameters = {
        {"input_data_0_h", static_cast<int>(ctx.input_shapes[0][1])},
        {"input_data_0_w", static_cast<int>(ctx.input_shapes[0][2])}};

    std::string source = R"(
      // Shaders may be compiled with a precision hint mediump, which means that
      // GLSL compiler may drop the size of float data type from 32 to 16 bits.
      // If "sum" and "size" variables are 16bit floats, their values range
      // become not enough for providing a good results accuracy. That is why
      // their precision is forced to be 32bit by using highp qualifier.

      highp vec4 sum = vec4(0.0);
      highp float size = float($input_data_0_w$ * $input_data_0_h$);
      for (int w = 0; w < $input_data_0_w$; w++) {
        for (int h = 0; h < $input_data_0_h$; h++) {
          sum += $input_data_0[w, h, gid.z]$;
        }
      }
      value_0 = sum / size;
    )";
    *generated_code = {
        /*parameters=*/std::move(parameters),
        /*objects=*/{},
        /*shared_variables=*/{},
        /*workload=*/uint3(),
        /*workgroup=*/uint3(1, 1, 4),
        /*source_code=*/std::move(source),
        /*input=*/IOStructure::ONLY_DEFINITIONS,
        /*output=*/IOStructure::AUTO,
    };
    return absl::OkStatus();
  }
};

}  // namespace

std::unique_ptr<NodeShader> NewMeanNodeShader() {
  return absl::make_unique<Mean>();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
