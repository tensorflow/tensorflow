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

#include "tensorflow/lite/delegates/gpu/gl/kernels/resampler.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

class Resampler : public NodeShader {
 public:
  absl::Status GenerateCode(const GenerationContext& ctx,
                            GeneratedCode* generated_code) const final {
    std::vector<Variable> parameters = {
        {"src_height", static_cast<int>(ctx.input_shapes[0][1])},
        {"src_width", static_cast<int>(ctx.input_shapes[0][2])},
    };

    std::string source = R"(
  highp int X = int(gid.x);
  highp int Y = int(gid.y);
  highp int S = int(gid.z);
  highp vec2 f_coords = ($input_data_1[X, Y, 0]$).xy;
  highp vec2 f_coords_floor = floor(f_coords);
  highp ivec4 st;
  st.xy = ivec2(f_coords_floor.x, f_coords_floor.y);
  st.zw = st.xy + ivec2(1, 1);
  highp vec2 t = f_coords - f_coords_floor;
  bool stx_in = st.x >= 0 && st.x < $src_width$;
  bool stz_in = st.z >= 0 && st.z < $src_width$;
  bool sty_in = st.y >= 0 && st.y < $src_height$;
  bool stw_in = st.w >= 0 && st.w < $src_height$;
  vec4 src0 = (stx_in && sty_in) ? $input_data_0[st.x, st.y, S]$ : vec4(0.0);
  vec4 src1 = (stz_in && sty_in) ? $input_data_0[st.z, st.y, S]$ : vec4(0.0);
  vec4 src2 = (stx_in && stw_in) ? $input_data_0[st.x, st.w, S]$ : vec4(0.0);
  vec4 src3 = (stz_in && stw_in) ? $input_data_0[st.z, st.w, S]$ : vec4(0.0);
  value_0 = mix(mix(src0, src1, t.x), mix(src2, src3, t.x), t.y);
)";
    *generated_code = {
        /*parameters=*/std::move(parameters),
        /*objects=*/{},
        /*shared_variables=*/{},
        /*workload=*/uint3(),
        /*workgroup=*/uint3(),
        /*source_code=*/std::move(source),
        /*input=*/IOStructure::ONLY_DEFINITIONS,
        /*output=*/IOStructure::AUTO,
    };
    return absl::OkStatus();
  }
};

}  // namespace

std::unique_ptr<NodeShader> NewResamplerNodeShader() {
  return absl::make_unique<Resampler>();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
