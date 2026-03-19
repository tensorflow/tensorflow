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

#include "tensorflow/lite/delegates/gpu/gl/kernels/concat.h"

#include <any>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/gl/variable.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

class AlignedConcatByChannels : public NodeShader {
 public:
  static bool IsSupported(const GenerationContext& ctx) {
    const auto& attr = std::any_cast<const ConcatAttributes&>(ctx.op_attr);

    // Implementation supports concatenation by channels only.
    if (attr.axis != Axis::CHANNELS) return false;

    // Implementation supports concatenation of 2 tensors only.
    if (ctx.input_shapes.size() != 2) return false;

    // H and W must be the same for every concatenated tensor.
    for (int i = 1; i < ctx.input_shapes.size(); i++) {
      if (ctx.input_shapes[0][1] != ctx.input_shapes[i][1] ||
          ctx.input_shapes[0][2] != ctx.input_shapes[i][2]) {
        return false;
      }
    }

    // Channels must be aligned by 4 for every concatenated tensor.
    for (const auto& shape : ctx.input_shapes) {
      if (shape[3] % 4 != 0) return false;
    }

    return true;
  }

  absl::Status GenerateCode(const GenerationContext& ctx,
                            GeneratedCode* generated_code) const final {
    if (!IsSupported(ctx)) {
      return absl::InvalidArgumentError(
          "This case is not supported by aligned concat");
    }

    // Shader below concatenates 2 tensors which channels are aligned by 4
    std::string source = R"(
      if (gid.z < $border$) {
        value_0 = $input_data_0[gid.x, gid.y, gid.z]$;
      } else {
        int z = gid.z - $border$;
        value_0 = $input_data_1[gid.x, gid.y, z]$;
      }
)";
    *generated_code = {
        /*parameters=*/{
            {"border", static_cast<int>(ctx.input_shapes[0][3]) / 4}},
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

class ConcatByAnyChannel : public NodeShader {
 public:
  static bool IsSupported(const GenerationContext& ctx) {
    const auto& attr = std::any_cast<const ConcatAttributes&>(ctx.op_attr);

    // Implementation supports concatenation by channels only.
    if (attr.axis != Axis::CHANNELS) return false;

    // Implementation supports concatenation of more that 1 tensors only.
    if (ctx.input_shapes.size() <= 1) return false;

    // H and W must be the same for every concatenated tensor.
    for (int i = 1; i < ctx.input_shapes.size(); i++) {
      if (ctx.input_shapes[0][1] != ctx.input_shapes[i][1] ||
          ctx.input_shapes[0][2] != ctx.input_shapes[i][2]) {
        return false;
      }
    }

    return true;
  }

  absl::Status GenerateCode(const GenerationContext& ctx,
                            GeneratedCode* generated_code) const final {
    if (!IsSupported(ctx)) {
      return absl::UnimplementedError("This case is not supported by concat");
    }

    std::string code = DeclareVariables();

    // "already_written" is used to keep the amount of already joined channels
    int already_written = 0;
    // "t" is an id of the next temp* variable.
    // Generally, temp* variables are used in macros
    // READ_BUFFER_VEC4(buff, addr, var).
    // This macros instantiate the variable "var" and
    // reads the value from buffer "buff" by address "addr"
    int t = 0;
    for (int current_input_id = 0; current_input_id < ctx.input_shapes.size();
         current_input_id++) {
      // Start joining next inout tensor

      // Grab channels amount
      int in_ch = ctx.input_shapes[current_input_id][3];
      code += PrintStartMessage(current_input_id, in_ch, already_written);

      // Construct the buffer name associated with this tensor
      std::string input = "input_data_" + std::to_string(current_input_id);

      // "reminder" shows us how many cells in 4-element vector are left after
      // the last write. As example, if we join two tensors both with
      // 3 channels, after joining the first one we come to this line again
      // and, when joining the second tensor, the reminder value
      // will be equal to 1
      int reminder = already_written % 4;

      if (reminder == 0) {
        code += AlignedCase(in_ch, input);
      } else {
        code += UnalignedCase(reminder, in_ch, input, &t);
      }
      already_written += in_ch;
    }

    *generated_code = {
        /*parameters=*/{},
        /*objects=*/{},
        /*shared_variables=*/{},
        /*workload=*/
        uint3(static_cast<int>(ctx.output_shapes[0][2]),
              static_cast<int>(ctx.output_shapes[0][1]), 1),
        /*workgroup=*/uint3(),
        /*source_code=*/std::move(code),
        /*input=*/IOStructure::ONLY_DEFINITIONS,
        /*output=*/IOStructure::ONLY_DEFINITIONS,
    };
    return absl::OkStatus();
  }

 private:
  // Utility function
  std::string temp(int t) const { return "temp" + std::to_string(t); }

  std::string DeclareVariables() const {
    // "val" is used to collect useful information before the next
    // upcoming write.
    return R"(
int z = gid.z;
vec4 val = vec4(0.0f);

)";
  }

  std::string PrintStartMessage(int current_input_id, int in_ch,
                                int already_written) const {
    return "//              Joining " + std::to_string(current_input_id) +
           " tensor with " + std::to_string(in_ch) +
           " channels\n//  * * * *\\n// Already wrote " +
           std::to_string(already_written) + " elements\n\n";
  }

  std::string AlignedCase(int in_ch, const std::string& input) const {
    std::string code;
    // This branch is for aligned reading and writing, when we can copy
    // all 4 components at once. Address of the first element to write
    // should be aligned.
    // Visual examples:
    // 1) when copy input_data_0
    //
    //       | * * * * | * * * @ | @ @ . . .
    //         ^
    // 2) when in the middle of joining process:
    //
    //       | X X X X | * * * @ | @ @ . . .
    //                   ^
    // Note that amount of * equals to the in_ch
    //
    // X - cells were written before
    // * - you are going to write into these cells
    // @ - you will fill these cells next cycles
    // ^ - first elem you start writing from
    int blocks_amount = DivideRoundUp<int>(in_ch, 4);
    code += "// Aligned case\n";
    code += "// I'm going to make " + std::to_string(blocks_amount) +
            " write(s)\n\n";
    for (int block = 0; block < blocks_amount; block++) {
      // Copy full 4-element vector
      code += "val = $" + input + "[gid.x, gid.y, " + std::to_string(block) +
              "]$;\n" +
              "$output_data_0[gid.x, gid.y, z] = val$;\n"
              // calculate next address to write
              + "z++; \n\n";
    }
    return code;
  }

  std::string UnalignedCase(int reminder, int in_ch, const std::string& input,
                            int* t) const {
    // This branch is for copying cell-by-cell. It will never start from the
    // first tensor input_data_0. This function is splitting in two stages:
    // 1) Copy the "leftovers" for the previous cells
    // 2) Copy all other
    // Visual examples:
    //
    //        Stage 1       Stage 2
    //        -----------   -------------------------
    // . . X | X  X  X *1 | *2 *2 *2  @ | @  @  . . .
    //               ^
    // . . X | X  X *1 *1 | *2 *2 *2 *2 | *2 *2 . . .
    //             ^
    // . . X | X *1 *1 *1 | *2  @  @  @ | @  @  . . .
    //           ^
    // Note that amount of * equals to the in_ch
    //
    // X - cells were written before
    // *1 - write there at the Stage 1
    // *2 - write there at the Stage 2
    // @ - you will fill these cells next cycles
    // ^ - first elem you start writing from

    std::string code = "// Unaligned case\n";

    // Variable "shift" showes how many "empty" cells are left after previous
    // write. Remember, that this case should is unaligned.
    // shift now can only be 1, 2 or 3
    int shift = 4 - reminder;
    if (shift > in_ch) {
      shift = in_ch;
    }
    code += "\n// Stage 1\n";
    code += "vec4 " + temp(*t) + " = $" + input + "[gid.x, gid.y, 0]$;\n";
    for (int i = 0; i < shift; i++) {
      // Note that reminder + i has implicitly added 1, cause
      // reminder by it's nature is an amount, not an index
      code += "val[" + std::to_string(reminder + i) + "] = " + temp(*t) + "[" +
              std::to_string(i) + "];\n";
    }
    // Rewrite previous value with updated last cells
    code += "$output_data_0[gid.x, gid.y, z - 1] = val$;\n";
    (*t)++;

    // "left_blocks" is equal to an amount of WRITE_BUFFER_VEC4 calls
    // which will are left for this input to be finally copied
    int left_blocks = (in_ch - shift) / 4;
    if ((in_ch - shift) % 4 != 0) {
      left_blocks++;
    }
    if (left_blocks) {
      code += "\n// Stage 2\n";
      for (int block = 0; block < left_blocks; block++) {
        for (int elem = 0; elem < 4; elem++) {
          if (shift % 4 == 0) {
            code += "vec4 " + temp(*t) + " = $" + input + "[gid.x, gid.y, " +
                    std::to_string(block + 1) + "]$;\n";
            (*t)++;
          }
          code += "val[" + std::to_string(elem) + "] = " + temp(*t - 1) + "[" +
                  std::to_string(shift % 4) + "];\n";
          if (shift == in_ch) {
            break;
          }
          shift++;
        }
        code += "$output_data_0[gid.x, gid.y, z] = val$;\n";
        code += "z++;\n";
      }
    } else {
      code += "// No Stage 2\n";
    }
    return code;
  }
};

class FlatConcatByHeight : public NodeShader {
 public:
  static bool IsSupported(const GenerationContext& ctx) {
    const auto& attr = std::any_cast<const ConcatAttributes&>(ctx.op_attr);

    // Implementation supports concatenation by height only.
    if (attr.axis != Axis::HEIGHT) return false;

    // Implementation supports concatenation of more that 1 tensors only.
    if (ctx.input_shapes.size() <= 1) return false;

    // C and W must be the same for every concatenated tensor.
    for (int i = 1; i < ctx.input_shapes.size(); i++) {
      if (ctx.input_shapes[0][3] != ctx.input_shapes[i][3] ||
          ctx.input_shapes[0][2] != ctx.input_shapes[i][2]) {
        return false;
      }
    }

    return true;
  }

  absl::Status GenerateCode(const GenerationContext& ctx,
                            GeneratedCode* generated_code) const final {
    std::string code;
    std::vector<Variable> params;
    for (int i = 0, shift = 0; i < ctx.input_shapes.size();
         shift += ctx.input_shapes[i][1], i++) {
      code += "if (";
      if (i != 0) {
        code += "$input_data_" + std::to_string(i - 1) + "_h$ <= gid.y && ";
      }
      code +=
          "gid.y < " + std::to_string(shift + ctx.input_shapes[i][1]) + ") {\n";
      code += "if (gid.y - " + std::to_string(shift) + " >= $input_data_" +
              std::to_string(i) + "_h$) return;\n";
      code += "value_0 = $input_data_" + std::to_string(i) +
              "[gid.x, gid.y - " + std::to_string(shift) + ", gid.z]$;\n}\n";
      if (i != ctx.input_shapes.size() - 1) {
        code += " else ";
      }
      params.push_back({"input_data_" + std::to_string(i) + "_h",
                        static_cast<int>(ctx.input_shapes[i][1])});
    }

    *generated_code = {
        /*parameters=*/std::move(params),
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

class FlatConcatByWidth : public NodeShader {
 public:
  static bool IsSupported(const GenerationContext& ctx) {
    const auto& attr = std::any_cast<const ConcatAttributes&>(ctx.op_attr);

    // Implementation supports concatenation by width only.
    if (attr.axis != Axis::WIDTH) return false;

    // Implementation supports concatenation of more that 1 tensors only.
    if (ctx.input_shapes.size() <= 1) return false;

    // C and H must be the same for every concatenated tensor.
    for (int i = 1; i < ctx.input_shapes.size(); i++) {
      if (ctx.input_shapes[0][3] != ctx.input_shapes[i][3] ||
          ctx.input_shapes[0][1] != ctx.input_shapes[i][1]) {
        return false;
      }
    }

    return true;
  }

  absl::Status GenerateCode(const GenerationContext& ctx,
                            GeneratedCode* generated_code) const final {
    std::string code;
    std::vector<Variable> params;
    for (int i = 0, shift = 0; i < ctx.input_shapes.size();
         shift += ctx.input_shapes[i][2], i++) {
      code += "if (";
      if (i != 0) {
        code += "$input_data_" + std::to_string(i - 1) + "_w$ <= gid.x && ";
      }
      code +=
          "gid.x < " + std::to_string(shift + ctx.input_shapes[i][2]) + ") {\n";
      code += "if (gid.x - " + std::to_string(shift) + " >= $input_data_" +
              std::to_string(i) + "_w$) return;\n";
      code += "value_0 = $input_data_" + std::to_string(i) + "[gid.x - " +
              std::to_string(shift) + ", gid.y, gid.z]$;\n}\n";
      if (i != ctx.input_shapes.size() - 1) {
        code += " else ";
      }
      params.push_back({"input_data_" + std::to_string(i) + "_w",
                        static_cast<int>(ctx.input_shapes[i][2])});
    }

    *generated_code = {
        /*parameters=*/std::move(params),
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

class FlatConcat : public NodeShader {
 public:
  absl::Status GenerateCode(const GenerationContext& ctx,
                            GeneratedCode* generated_code) const final {
    if (FlatConcatByHeight::IsSupported(ctx)) {
      return flat_concat_by_height_.GenerateCode(ctx, generated_code);
    }
    if (FlatConcatByWidth::IsSupported(ctx)) {
      return flat_concat_by_width_.GenerateCode(ctx, generated_code);
    }
    return absl::InvalidArgumentError(
        "This case is not supported by flat concat");
  }

 private:
  FlatConcatByHeight flat_concat_by_height_;
  FlatConcatByWidth flat_concat_by_width_;
};

}  // namespace

std::unique_ptr<NodeShader> NewAlignedConcatNodeShader() {
  return std::make_unique<AlignedConcatByChannels>();
}

std::unique_ptr<NodeShader> NewConcatNodeShader() {
  return std::make_unique<ConcatByAnyChannel>();
}

std::unique_ptr<NodeShader> NewFlatConcatNodeShader() {
  return std::make_unique<FlatConcat>();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
