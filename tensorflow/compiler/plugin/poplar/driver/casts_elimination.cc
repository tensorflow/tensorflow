/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/casts_elimination.h"
#include "tensorflow/compiler/plugin/poplar/driver/matcher_predicates.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

#include <set>

namespace xla {
namespace poplarplugin {

static const std::vector<FusedGraphInfo> fuse_info = {
    {"reduction_no_convert", 1},
    {"reduction_no_convert", 1},
    {"convert_no_use", 0},
    {"convert_no_use", 0},
};

/*
 * Note about constructing these patterns.  Due to the behaviour of the fuser
 * there must be no backward references.  All nodes should appear after any
 * other nodes that refer to them.
 *
 */

static const std::vector<HloMatcherPattern> patterns = {
    // Remove convert to/from F32 before/after reduction, where initial value is
    // a constant
    {{HloOpcode::kConvert, true, 0, IsF32ToF16Convert, {1}},
     {HloOpcode::kReduce, true, 0, IsF32, {2, 3}},
     {HloOpcode::kConvert, true, 0, IsF16ToF32Convert, {4}},
     {HloOpcode::kConstant, true, 0, IsF32, {}},
     {HloOpcode::kParameter, false, 0, IsF16, {}}},

    // Remove convert to/from F32 before/after reduction, where initial value is
    // a convert from F16
    {{HloOpcode::kConvert, true, 0, IsF32ToF16Convert, {1}},
     {HloOpcode::kReduce, true, 0, IsF32, {2, 3}},
     {HloOpcode::kConvert, true, 0, IsF16ToF32Convert, {4}},
     {HloOpcode::kConvert, true, 0, IsF16ToF32Convert, {5}},
     {HloOpcode::kParameter, false, 0, IsF16, {}},
     {HloOpcode::kParameter, false, 1, IsF16, {}}},

    // Convert and then convert back F16 -> F32 -> F16
    {{HloOpcode::kConvert, true, 0, IsF32ToF16Convert, {1}},
     {HloOpcode::kConvert, true, 0, IsF16ToF32Convert, {2}},
     {HloOpcode::kParameter, false, 0, IsF16, {}}},

    // Convert and then convert back F32 -> F16 -> F32
    {{HloOpcode::kConvert, true, 0, IsF16ToF32Convert, {1}},
     {HloOpcode::kConvert, true, 0, IsF32ToF16Convert, {2}},
     {HloOpcode::kParameter, false, 0, IsF32, {}}},
};

CastsElimination::CastsElimination(struct CompilerAnnotations& annotations)
    : HloMatcher(patterns, annotations, false) {}

unsigned CastsElimination::ReplaceNodes() {
  unsigned int replacement_count = 0;

  // Handle all the reductions with a casts around them - remove all the casts
  const std::vector<unsigned> casts_around_reduction_patterns = {0, 1};
  for (const auto pattern_index : casts_around_reduction_patterns) {
    for (HloMatcherMatched& match : matches_[pattern_index]) {
      if (match.ok) {
        auto* convert_out = match.instructions[0];
        auto* reduction = match.instructions[1];
        auto* to_reduce_convert = match.instructions[2];
        auto* init_val = match.instructions[3];
        auto* value_in = match.instructions[4];

        // Create a new reduce_computation
        // Check the reduction op is elementwise binary and takes in two
        // parameters only. If that's not the case then we can't convert this
        // reduction.
        auto* reduce_computation = reduction->to_apply();
        auto* reduce_op = reduce_computation->root_instruction();
        if (!(reduce_op->IsElementwiseBinary() &&
              reduce_op->operand(0)->opcode() == HloOpcode::kParameter &&
              reduce_op->operand(1)->opcode() == HloOpcode::kParameter)) {
          continue;
        }
        // Build the new reduce_computation
        auto builder = HloComputation::Builder(reduce_computation->name());
        {
          auto* in0 = builder.AddInstruction(reduce_op->operand(0)->Clone());
          in0->mutable_shape()->set_element_type(F16);
          auto* in1 = builder.AddInstruction(reduce_op->operand(1)->Clone());
          in1->mutable_shape()->set_element_type(F16);
          const auto shape_op_fp16 =
              ShapeUtil::ChangeElementType(reduce_op->shape(), F16);
          builder.AddInstruction(
              reduce_op->CloneWithNewOperands(shape_op_fp16, {in0, in1}));
        }
        auto* new_reduce_computation =
            match.computation->parent()->AddEmbeddedComputation(
                builder.Build());

        // Get the initial value
        HloInstruction* new_init_val;
        if (init_val->opcode() == HloOpcode::kConstant) {
          // convert a constant from F32 to F16 and add it to the graph
          const auto shape_init_val_fp16 =
              ShapeUtil::ChangeElementType(init_val->shape(), F16);
          // std::unique_ptr<Literal> literal_f16;
          auto literal_f16 =
              init_val->literal().ConvertToShape(shape_init_val_fp16);
          // TF_ASSIGN_OR_RETURN(literal_f16,
          //                     );
          new_init_val =
              match.computation->AddInstruction(HloInstruction::CreateConstant(
                  std::move(literal_f16.ValueOrDie())));
        } else if (init_val->opcode() == HloOpcode::kConvert) {
          // init value is an output of a Convert from FP16 to FP32, so use the
          // argument to convert
          new_init_val = init_val->mutable_operand(0);
        } else {
          LOG(FATAL) << "Unsupported Op for Reduction init value";
        }

        // Create the new reduction
        const auto shape_reduction_fp16 =
            ShapeUtil::ChangeElementType(reduction->shape(), F16);
        auto* new_reduction =
            match.computation->AddInstruction(HloInstruction::CreateReduce(
                shape_reduction_fp16, value_in, new_init_val,
                reduction->dimensions(), new_reduce_computation));
        new_reduction->set_metadata(reduction->metadata());

        // Replace all uses with the new reduction
        OutlinedInfo outlined_info;
        outlined_info.removed_instructions.push_back(convert_out);
        TF_CHECK_OK(convert_out->ReplaceAllUsesWith(new_reduction));
        replacement_count += MarkReplacedInstructions(outlined_info);
      }
    }
  }

  // Handle all the unused casts
  const std::vector<unsigned> unsued_casts_patterns = {2, 3};
  for (const auto pattern_index : unsued_casts_patterns) {
    for (HloMatcherMatched& match : matches_[pattern_index]) {
      if (match.ok) {
        auto* convert_out = match.instructions[0];
        auto* convert_in = match.instructions[1];
        auto* val_in = match.instructions[2];

        // Replace all uses with the new reduction
        OutlinedInfo outlined_info;
        outlined_info.removed_instructions.push_back(convert_out);
        TF_CHECK_OK(convert_out->ReplaceAllUsesWith(val_in));
        replacement_count += MarkReplacedInstructions(outlined_info);
      }
    }
  }

  return replacement_count;
}

}  // namespace poplarplugin
}  // namespace xla
