/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/gl/kernels/tile.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/types/any.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/gl/node_shader.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

class Tile : public NodeShader {
 public:
  absl::Status GenerateCode(const GenerationContext& ctx,
                            GeneratedCode* generated_code) const final {
    std::string code = R"(
      for (int i = 0; i < 4; ++i) {
        int dst_c = 4 * gid.z + i;
        int src_x = gid.x % $input_data_w$;
        int src_y = gid.y % $input_data_h$;
        int src_c = dst_c % $input_data_c$;
        value_0[i] = $input_data_0[src_x, src_y, src_c / 4]$[src_c % 4];
      }
    )";

    *generated_code = {
        /*parameters=*/{
            {"input_data_h", static_cast<int>(ctx.input_shapes[0][1])},
            {"input_data_w", static_cast<int>(ctx.input_shapes[0][2])},
            {"input_data_c", static_cast<int>(ctx.input_shapes[0][3])},
        },
        /*objects=*/{},
        /*shared_variables=*/{},
        /*workload=*/uint3(),
        /*workgroup=*/uint3(),
        /*source_code=*/std::move(code),
        /*input=*/IOStructure::ONLY_DEFINITIONS,
        /*output=*/IOStructure::AUTO,
    };
    return absl::OkStatus();
  }
};
}  // namespace

std::unique_ptr<NodeShader> NewTileNodeShader() {
  return std::make_unique<Tile>();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
