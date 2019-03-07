/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/gpu_fusible.h"

#include <algorithm>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {
namespace gpu {

namespace {
void AppendParams(const HloInstruction& instr,
                  std::vector<HloInstruction*>* params) {
  if (instr.opcode() == HloOpcode::kFusion) {
    params->insert(std::end(*params), std::begin(instr.fused_parameters()),
                   std::end(instr.fused_parameters()));
  } else {
    for (HloInstruction* operand : instr.operands()) {
      params->push_back(operand);
    }
  }
}
}  // namespace

bool LayoutsAreReduceInputFusionFriendly(const HloInstruction& producer,
                                         const HloInstruction& reduce) {
  std::vector<HloInstruction*> params;
  AppendParams(producer, &params);
  AppendParams(reduce, &params);
  int64 max_rank = -1;
  const Layout* max_rank_layout;
  for (HloInstruction* param : params) {
    if (param->shape().IsArray() && param->shape().rank() > max_rank) {
      max_rank = param->shape().rank();
      max_rank_layout = &param->shape().layout();
    }
  }
  return absl::c_all_of(params, [&](HloInstruction* param) {
    return (!param->shape().IsArray()) || (param->shape().rank() < max_rank) ||
           (LayoutUtil::Equal(param->shape().layout(), *max_rank_layout));
  });
}

bool IsReduceInputFusion(const HloInstruction& instr) {
  if (instr.IsMultiOutputFusion()) {
    for (const HloInstruction* operand :
         instr.fused_expression_root()->operands()) {
      if (IsReductionToVector(*operand)) {
        CHECK(instr.fusion_kind() == HloInstruction::FusionKind::kInput)
            << " Multi-output fusion rooted at reduction-to-vector ops must be "
               "of kind kInput: "
            << instr.ToString();
        return true;
      }
    }
  } else if (instr.opcode() == HloOpcode::kFusion &&
             IsReductionToVector(*instr.fused_expression_root())) {
    CHECK(instr.fusion_kind() == HloInstruction::FusionKind::kInput)
        << " Fusion rooted at reduction-to-vector op must be of kind kInput: "
        << instr.ToString();
    return true;
  }
  return false;
}

bool IsInputFusibleReduction(const HloInstruction& instr) {
  return IsReduceInputFusion(instr) || IsReductionToVector(instr);
}

bool ShapesCompatibleForMultiOutputFusion(const HloInstruction& instr1,
                                          const HloInstruction& instr2) {
  // Returns the instructions that determines the emitter used for lowering,
  // sometimes referred to as "the real hero".
  auto get_real_hero =
      [&](const HloInstruction* instr) -> const HloInstruction* {
    if (instr->opcode() == HloOpcode::kFusion) {
      auto fused_expression_root = instr->fused_expression_root();
      if (instr->IsMultiOutputFusion()) {
        // If possible, we want to pick a reduction-to-vector operand of the
        // fusion root, because it has the most constraints.
        for (const auto* inst : fused_expression_root->operands()) {
          if (IsReductionToVector(*inst)) {
            return inst;
          }
        }
        return fused_expression_root->operands()[0];
      }
      return fused_expression_root;
    }
    return instr;
  };

  // Multi-output fusion kernels share a common parallel loop. The loop
  // dimenstions are determined by instruction shapes.
  auto get_loop_shape = [&](const HloInstruction* element_instr) {
    // Special-case reduction-to-vector ops: The loop dimensions are determined
    // by the shape of the first operand.
    if (IsReductionToVector(*element_instr)) {
      return element_instr->operand(0)->shape();
    }
    return element_instr->shape();
  };

  // All shapes of the root tuple of multi-output fusions should agree, i.e. all
  // root ops should have equal output shapes. An exception are
  // reduction-to-vector ops. Here the input shapes of the reduction (first
  // operand shape) and the reduction dimensions need to match.
  auto* instr_1 = get_real_hero(&instr1);
  auto* instr_2 = get_real_hero(&instr2);
  // TODO(tjoerg): Relax the shape constraint. The datatype does not matter.
  if (IsReductionToVector(*instr_1) && IsReductionToVector(*instr_2) &&
      (!ShapeUtil::Equal(instr_1->shape(), instr_2->shape()) ||
       instr_1->dimensions() != instr_2->dimensions())) {
    return false;
  }
  // The elementwise output shapes must be the same (including layout).
  // TODO(tjoerg): Further relax the constraint. The datatype does not matter.
  return ShapeUtil::EqualIgnoringFpPrecision(get_loop_shape(instr_1),
                                             get_loop_shape(instr_2));
}

bool IsInputFusibleScatter(const HloInstruction& instr) {
  if (instr.opcode() == HloOpcode::kScatter ||
      (instr.opcode() == HloOpcode::kFusion &&
       instr.fusion_kind() == HloInstruction::FusionKind::kInput &&
       instr.fused_expression_root()->opcode() == HloOpcode::kScatter)) {
    return true;
  }
  return false;
}

bool IsInputFusible(const HloInstruction& instr) {
  // Input fusion only handles non-elemental reduction and scatter operations.
  return IsInputFusibleReduction(instr) || IsInputFusibleScatter(instr);
}

bool IsLoopFusible(const HloInstruction& instr) {
  // Don't fuse get-tuple-element on GPU: We can, but it's slower than not
  // fusing.  We never generate kernels for unfused GTEs.  Instead, if an
  // unfused GTE is an input to a kernel (including a fusion kernel), we
  // compute the address of the GTE at the top of the kernel.  Often we know the
  // address of the GTE result statically, so we can do this without chasing any
  // pointers.
  return (instr.IsElementwise() && instr.operand_count() > 0) ||
         instr.opcode() == HloOpcode::kBitcast ||
         instr.opcode() == HloOpcode::kBroadcast ||
         instr.opcode() == HloOpcode::kConcatenate ||
         instr.opcode() == HloOpcode::kDynamicSlice ||
         instr.opcode() == HloOpcode::kDynamicUpdateSlice ||
         (instr.opcode() == HloOpcode::kFusion &&
          instr.fusion_kind() == HloInstruction::FusionKind::kLoop) ||
         instr.opcode() == HloOpcode::kGather ||
         instr.opcode() == HloOpcode::kIota ||
         instr.opcode() == HloOpcode::kPad ||
         (instr.opcode() == HloOpcode::kReduce &&
          !IsReductionToVector(instr)) ||
         instr.opcode() == HloOpcode::kReduceWindow ||
         instr.opcode() == HloOpcode::kReshape ||
         instr.opcode() == HloOpcode::kReverse ||
         instr.opcode() == HloOpcode::kSlice ||
         instr.opcode() == HloOpcode::kTranspose;
}

bool IsFusible(const HloInstruction& instr) {
  return IsInputFusible(instr) || IsLoopFusible(instr);
}

bool IsFusionEmitterInefficient(const HloInstruction* consumer,
                                const HloInstruction* producer) {
  if (consumer->opcode() != HloOpcode::kFusion) {
    return false;
  }
  // Collects for each instruction in the fusion node from which (indirect)
  // users newly created index values are passed. Roughly speaking, we reuse
  // index values if the shapes are equal when ignoring the element type (we may
  // reuse also if the shape change is a bitcast, but we don't consider that
  // here). By ignoring potential reuses our estimate whether the fusion emitter
  // is inefficient is a bit more conservative than necessary.
  absl::flat_hash_map<const HloInstruction*,
                      absl::flat_hash_set<const HloInstruction*>>
      indexing_users;
  // Stores the number of different index accesses for each instruction in the
  // fusion node. The fusion emitter caches access with the same index, so this
  // value indicates how many times a specific instruction will be emitted.
  absl::flat_hash_map<const HloInstruction*, int64> index_usage_count;

  auto postorder =
      consumer->fused_instructions_computation()->MakeInstructionPostOrder();
  std::reverse(postorder.begin(), postorder.end());
  for (const auto* instruction : postorder) {
    if (instruction->opcode() == HloOpcode::kParameter) {
      continue;
    }
    int64& total = index_usage_count[instruction];
    if (indexing_users[instruction].empty()) {
      total = 1;
    } else {
      total = 0;
      for (const auto* user : indexing_users[instruction]) {
        int64 weight = 1;
        // Concatenate is special: the index differs for each operand, so in the
        // worst case we have to deal with as many index values as the number of
        // operands of Concatenate. By considering the worst case, we are more
        // conservative than necessary regarding refusing to fuse.
        if (user->opcode() == HloOpcode::kConcatenate) {
          weight = user->operand_count();
        }
        total += index_usage_count[user] * weight;
      }
    }
    for (const auto* operand : instruction->operands()) {
      // For simplicity we assume that all shape and layout changing operations
      // invalidate index reuse.
      if (Shape::Equal().IgnoreElementType()(operand->shape(),
                                             instruction->shape())) {
        // If the index is reused, it means the operand gets index values from
        // the same set of (indirect) users as 'instruction' itself.
        indexing_users[operand].insert(indexing_users[instruction].begin(),
                                       indexing_users[instruction].end());
      } else {
        // If the index is not reused, it means 'instruction' computes a new
        // index derived from the index it gets.
        indexing_users[operand].insert(instruction);
      }
    }
  }
  // Also account for the 'producer' if it would be fused. Find the operand it
  // corresponds to.
  for (int64 operand_num = 0; operand_num < consumer->operand_count();
       ++operand_num) {
    if (consumer->operand(operand_num) == producer) {
      auto instruction = consumer->fused_parameter(operand_num);
      int64& total = index_usage_count[instruction];
      total = 0;
      for (const auto* user : indexing_users[instruction]) {
        total += index_usage_count[user];
      }
      break;
    }
  }
  int64 total = 0;
  for (const auto& entry : index_usage_count) {
    total += entry.second;
  }
  // Check that the code duplication has at most a factor of 8 (where 8 is an
  // arbitrary constant that seems to work).
  return total > 8 * index_usage_count.size();
}

}  // namespace gpu
}  // namespace xla
