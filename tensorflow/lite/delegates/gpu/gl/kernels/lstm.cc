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

#include "tensorflow/lite/delegates/gpu/gl/kernels/lstm.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/memory/memory.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/gl/node_shader.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

// Basic LSTMCell gates.
//
//  inputs:  0             1
//           activ_temp    prev_state
//                  \      /
//               [[LSTM gates]]
//                  /      \
//            new_state    activation
//  outputs:  0            1
//
// The size of activ_temp should be 4x size of new_state.
// The size of prev_state == new_state == activation.
//
class LstmNodeShader : public NodeShader {
 public:
  absl::Status GenerateCode(const GenerationContext& ctx,
                            GeneratedCode* generated_code) const final {
    std::string code = R"(
      vec4 prev_state  = $input_data_1[gid.x, gid.y, gid.z]$;

      int c0 = 0 * $workload_z$;
      int c1 = 1 * $workload_z$;
      int c2 = 2 * $workload_z$;
      int c3 = 3 * $workload_z$;

      // input, new, forget, output
      vec4 gate_0 = $input_data_0[gid.x, gid.y, gid.z + c0]$;
      vec4 gate_1 = $input_data_0[gid.x, gid.y, gid.z + c1]$;
      vec4 gate_2 = $input_data_0[gid.x, gid.y, gid.z + c2]$;
      vec4 gate_3 = $input_data_0[gid.x, gid.y, gid.z + c3]$;

      vec4 input_gate  = 1.0f / (1.0f + exp(-1.0 * gate_0));  // sig(x)
      vec4 new_input   = tanh(gate_1);                        // tanh(x)
      vec4 forget_gate = 1.0f / (1.0f + exp(-1.0 * gate_2));  // sig(x)
      vec4 output_gate = 1.0f / (1.0f + exp(-1.0 * gate_3));  // sig(x)

      vec4 new_state = input_gate * new_input + forget_gate * prev_state;
      vec4 activation = output_gate * tanh(new_state);

      value_0 = new_state;
      value_1 = activation;
    )";
    *generated_code = {
        /*parameters=*/{},
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

std::unique_ptr<NodeShader> NewLstmNodeShader() {
  return std::make_unique<LstmNodeShader>();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
