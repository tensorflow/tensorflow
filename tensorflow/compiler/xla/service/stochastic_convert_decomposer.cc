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

#include "tensorflow/compiler/xla/service/stochastic_convert_decomposer.h"

#include <cstdint>
#include <limits>

#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace xla {

Status DecomposeStochasticConvert(HloComputation* comp,
                                  HloInstruction* instruction) {
  CHECK(instruction->opcode() == HloOpcode::kStochasticConvert)
      << "requires a stochastic_convert instruction to decompose, but got: "
      << instruction->opcode();
  CHECK(instruction->operand_count() == 2)
      << "requires 2 operands for stochastic convert, but got: "
      << instruction->operand_count();
  HloInstruction* operand = instruction->mutable_operand(0);
  HloInstruction* random = instruction->mutable_operand(1);
  PrimitiveType from_type = operand->shape().element_type();
  PrimitiveType random_type = random->shape().element_type();
  PrimitiveType to_type = instruction->shape().element_type();
  TF_RETURN_IF_ERROR(ShapeInference::InferStochasticConvertShape(
                         operand->shape(), random->shape(), to_type)
                         .status());
  VLOG(1) << "Decomposing instruction: " << instruction->ToString();

  // For converting floats to integers, the fractional bits of the operands
  // are placed into an unsigned integer where the bit representing
  // 2^-1 is put in the most significant bit. This is then
  // compared (using an unsigned integer comparison) against the unsigned
  // random value. The fractional part will be rouneded up if the user-given
  // random value is less than the fractional bits, otherwise it will be
  // rounded down.
  if (primitive_util::IsSignedIntegralType(to_type)) {
    TF_ASSIGN_OR_RETURN(HloInstruction * operand_sign,
                        MakeUnaryHlo(HloOpcode::kSign, operand));
    TF_ASSIGN_OR_RETURN(HloInstruction * should_neg,
                        MakeCompareHlo(Comparison::Direction::kLt, operand_sign,
                                       MakeScalarLike(operand_sign, 0)));
    TF_ASSIGN_OR_RETURN(HloInstruction * operand_abs,
                        MakeUnaryHlo(HloOpcode::kAbs, operand));
    TF_ASSIGN_OR_RETURN(HloInstruction * truncated_fp,
                        MakeUnaryHlo(HloOpcode::kFloor, operand_abs));
    TF_ASSIGN_OR_RETURN(
        HloInstruction * fractional,
        MakeBinaryHlo(HloOpcode::kSubtract, operand_abs, truncated_fp));
    // Upcasts the operand to F32 as calculating fixed_fractional needs a
    // multiplier of 2^16  which can't be represented in F16(whose max
    // value is 2^16 - 2^5).
    if (from_type == F16) {
      fractional = MakeConvertToHlo(fractional, F32);
    }
    // Compares fractional values against unsigned random values by
    // normalizing random values into [0, 1): fractional vs. (random /
    // random_max). This equals to comparing (fractional * random_max) vs.
    // random.
    TF_ASSIGN_OR_RETURN(
        HloInstruction * fixed_fractional,
        MakeBinaryHlo(
            HloOpcode::kMultiply, fractional,
            MakeScalarLike(fractional, IPow<double>(2, primitive_util::BitWidth(
                                                           random_type)))));
    // Rounds the integer output up if the fractional pieces is larger than
    // the input random number.
    TF_ASSIGN_OR_RETURN(
        HloInstruction * should_round_up,
        MakeCompareHlo(Comparison::Direction::kLt, random,
                       MakeConvertToHlo(fixed_fractional, random_type)));
    HloInstruction* truncated_int = MakeConvertToHlo(truncated_fp, to_type);

    TF_ASSIGN_OR_RETURN(
        truncated_int,
        MakeSelectHlo(should_round_up,
                      MakeBinaryHlo(HloOpcode::kAdd, truncated_int,
                                    MakeScalarLike(truncated_int, 1))
                          .value(),
                      truncated_int));
    TF_ASSIGN_OR_RETURN(
        HloInstruction * result,
        MakeSelectHlo(should_neg,
                      MakeUnaryHlo(HloOpcode::kNegate, truncated_int).value(),
                      truncated_int));
    auto to_bits = primitive_util::BitWidth(to_type);
    // Deals with min values
    auto min = static_cast<int64_t>(
        (static_cast<uint64_t>(1) + ~static_cast<uint64_t>(1))
        << (to_bits - 1));
    TF_ASSIGN_OR_RETURN(HloInstruction * is_min,
                        MakeCompareHlo(Comparison::Direction::kLe, operand,
                                       MakeScalarLike(operand, min)));
    TF_ASSIGN_OR_RETURN(
        result, MakeSelectHlo(is_min, MakeScalarLike(result, min), result));
    // Deals with max values
    auto max =
        static_cast<int64_t>((static_cast<uint64_t>(1) << (to_bits - 1)) - 1);
    TF_ASSIGN_OR_RETURN(HloInstruction * is_max,
                        MakeCompareHlo(Comparison::Direction::kGe, operand,
                                       MakeScalarLike(operand, max)));
    TF_ASSIGN_OR_RETURN(
        result, MakeSelectHlo(is_max, MakeScalarLike(result, max), result));

    TF_RETURN_IF_ERROR(instruction->ReplaceAllUsesWith(result));
    TF_RETURN_IF_ERROR(comp->RemoveInstruction(instruction));
    return OkStatus();
  }

  // TODO(b/232442915): Add support for converting to floats.
  return InternalError("Unsupported stochastic convert: from %s to %s",
                       PrimitiveType_Name(from_type),
                       PrimitiveType_Name(to_type));
}

StatusOr<bool> StochasticConvertDecomposer::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      if (instruction->opcode() != HloOpcode::kStochasticConvert) {
        continue;
      }
      TF_RETURN_IF_ERROR(DecomposeStochasticConvert(computation, instruction));
      changed = true;
    }
  }
  return changed;
}

}  // namespace xla
