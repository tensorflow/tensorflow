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

#include "tensorflow/compiler/xla/service/bitcast_dtypes_expander.h"

#include "absl/algorithm/container.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/lib/broadcast.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instructions.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/tsl/platform/logging.h"

namespace xla {

StatusOr<HloInstruction*> BitcastDtypesExpander::ExpandInstruction(
    HloInstruction* instruction) {
  HloInstruction* input = instruction->mutable_operand(0);
  const Shape& from_shape = input->shape();
  const Shape& to_shape = instruction->shape();

  int input_bit_width = primitive_util::BitWidth(from_shape.element_type());
  int output_bit_width = primitive_util::BitWidth(to_shape.element_type());

  PrimitiveType input_logical_type =
      primitive_util::UnsignedIntegralTypeForBitWidth(input_bit_width);
  PrimitiveType output_logical_type =
      primitive_util::UnsignedIntegralTypeForBitWidth(output_bit_width);

  if (input_bit_width == output_bit_width) {
    return instruction;
  }

  std::string name =
      absl::StrFormat("xla.bitcast_convert_%s_2_%s", from_shape.ToString(),
                      to_shape.ToString());

  // Note: we are duplicating a hack from `cholesky_expander` to build a
  // computation using XlaBuilder.
  HloModule* module = instruction->GetModule();
  HloComputation*& computation =
      computation_cache_.emplace(name, nullptr).first->second;
  if (!computation) {
    XlaBuilder b(name);
    XlaOp input = Parameter(&b, 0, instruction->operand(0)->shape(), "a");

    if (input_bit_width > output_bit_width) {
      std::vector<int64_t> broadcasted_input_shape(
          from_shape.dimensions().begin(), from_shape.dimensions().end());
      std::vector<int64_t> reshaped_input_shape(from_shape.dimensions().begin(),
                                                from_shape.dimensions().end());
      broadcasted_input_shape.push_back(input_bit_width / output_bit_width);
      reshaped_input_shape.push_back(1);
      int64_t output_bit_width_mask = (1l << output_bit_width) - 1;

      TF_ASSIGN_OR_RETURN(input,
                          BroadcastTo(Reshape(input, reshaped_input_shape),
                                      broadcasted_input_shape));
      input = BitcastConvertType(input, input_logical_type);
      TF_ASSIGN_OR_RETURN(Shape input_shape, b.GetShape(input));
      XlaOp iota = Iota(&b, input_shape, input_shape.dimensions_size() - 1);
      XlaOp iota_m = Mul(ScalarLike(input, output_bit_width), iota);
      input = And(ShiftRightLogical(input, iota_m),
                  ScalarLike(input, output_bit_width_mask));
      input = ConvertElementType(input, output_logical_type);
    } else if (input_bit_width < output_bit_width) {
      input = BitcastConvertType(input, input_logical_type);
      input = ConvertElementType(input, output_logical_type);

      // Shift bits and OR them together to reduce the inner dimension.
      XlaOp iota_m = Mul(
          ConstantR0WithType(&b, output_logical_type, input_bit_width),
          Iota(&b,
               ShapeUtil::ChangeElementType(from_shape, output_logical_type),
               from_shape.rank() - 1));
      input = ShiftLeft(input, iota_m);
      input = Reduce(input, Zero(&b, output_logical_type),
                     CreateScalarOrComputation(output_logical_type, &b),
                     {from_shape.rank() - 1});
    }

    BitcastConvertType(input, to_shape.element_type());

    TF_ASSIGN_OR_RETURN(XlaComputation xla_computation, b.Build());
    TF_ASSIGN_OR_RETURN(ProgramShape program_shape,
                        xla_computation.GetProgramShape());
    HloModuleConfig config(program_shape);
    TF_ASSIGN_OR_RETURN(auto new_module, HloModule::CreateFromProto(
                                             xla_computation.proto(), config));
    HloCloneContext context(module);
    computation =
        module->DeepCloneComputation(new_module->entry_computation(), &context);
  }

  return instruction->parent()->AddInstruction(HloInstruction::CreateCall(
      instruction->shape(), instruction->operands(), computation));
}

bool BitcastDtypesExpander::InstructionMatchesPattern(
    HloInstruction* instruction) {
  return instruction->opcode() == HloOpcode::kBitcastConvert &&
         primitive_util::BitWidth(instruction->shape().element_type()) !=
             primitive_util::BitWidth(
                 instruction->operand(0)->shape().element_type());
}

}  // namespace xla
