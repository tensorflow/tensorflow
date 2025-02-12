/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/service/gpu/transforms/variadic_op_splitter.h"

#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

namespace {
// The parameter space on the GPU device is limited. We pick an arbitrary low
// constant here to try to prevent exceeding this parameter space. For a proper
// fix, we would have to take into account which parameters share a buffer, and
// how big these buffers are.
constexpr int32_t kMaxParameters = 128;

absl::StatusOr<bool> SplitConcatenate(HloInstruction* concat,
                                      HloComputation* comp) {
  auto operands = concat->operands();
  std::vector<HloInstruction*> operands_to_split(operands.begin(),
                                                 operands.end());
  while (operands_to_split.size() > 1) {
    std::vector<HloInstruction*> new_operands;
    absl::Span<HloInstruction*> operands_span(operands_to_split);
    for (int64_t offset = 0; offset < operands_to_split.size();
         offset += kMaxParameters) {
      // Check if there is a remainder of operands that does not completely fill
      // one "batch" of exactly 'kMaxParameters' operands. If there are only
      // less than 'kMaxParameters' operands left, then we still put them into a
      // concat together. Otherwise, we spare them for another round so that
      // they can be put together into a concat with some of the newly created
      // concats.
      if (offset > 0 && offset + kMaxParameters > operands_to_split.size()) {
        new_operands.insert(new_operands.end(),
                            operands_to_split.begin() + offset,
                            operands_to_split.end());
      } else {
        Shape new_shape = concat->shape();
        int64_t concat_dimension_size = 0;
        for (int64_t i = 0;
             i < kMaxParameters && offset + i < operands_to_split.size(); ++i) {
          concat_dimension_size +=
              operands_to_split[i + offset]->shape().dimensions(
                  concat->concatenate_dimension());
        }
        new_shape.set_dimensions(concat->concatenate_dimension(),
                                 concat_dimension_size);
        auto new_concat = comp->AddInstruction(concat->CloneWithNewOperands(
            new_shape, operands_span.subspan(offset, kMaxParameters)));
        new_operands.push_back(new_concat);
      }
    }
    operands_to_split = new_operands;
  }
  TF_RETURN_IF_ERROR(comp->ReplaceInstruction(concat, operands_to_split[0]));
  return true;
}

std::vector<HloInstruction*> GetRelevantVariadicOps(HloComputation* comp) {
  std::vector<HloInstruction*> ops;
  for (HloInstruction* instr : comp->instructions()) {
    if (HloPredicateIsOp<HloOpcode::kConcatenate>(instr) &&
        instr->operand_count() > kMaxParameters) {
      ops.push_back(instr);
    }
  }
  return ops;
}

}  // namespace

absl::StatusOr<bool> VariadicOpSplitter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* comp :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* op : GetRelevantVariadicOps(comp)) {
      // TODO(b/112613927): Handle also other ops than concatenate.
      TF_ASSIGN_OR_RETURN(bool result, SplitConcatenate(op, comp));
      changed |= result;
    }
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
