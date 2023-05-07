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

#include "tensorflow/compiler/xla/service/all_reduce_contiguous.h"

#include <vector>

#include "tensorflow/compiler/xla/hlo/ir/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instructions.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/hlo/utils/hlo_query.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"

namespace xla {
namespace {

Status ReplaceWithContiguousAllReduce(HloAllReduceInstruction* all_reduce) {
  TF_RET_CHECK(all_reduce);
  TF_RET_CHECK(!all_reduce->has_sharding());

  HloComputation& computation = *all_reduce->parent();  // never null
  PrimitiveType element_type = all_reduce->operand(0)->shape().element_type();

  // Bitcast operands to 1D so that they may be concatenated together.
  std::vector<HloInstruction*> flat_operands;
  flat_operands.reserve(all_reduce->operand_count());
  int64_t total_size = 0;
  for (HloInstruction* operand : all_reduce->operands()) {
    TF_RET_CHECK(operand->shape().IsArray());
    int64_t num_elements = ShapeUtil::ElementsIn(operand->shape());
    Shape flat_shape = ShapeUtil::MakeShape(element_type, {num_elements});
    flat_operands.push_back(computation.AddInstruction(
        HloInstruction::CreateBitcast(flat_shape, operand)));
    total_size += num_elements;
  }

  Shape concat_shape = ShapeUtil::MakeShape(element_type, {total_size});
  HloInstruction* concatenated =
      computation.AddInstruction(HloInstruction::CreateConcatenate(
          concat_shape, flat_operands, /*dimension=*/0));

  HloInstruction* new_all_reduce =
      computation.AddInstruction(HloInstruction::CreateAllReduce(
          concat_shape, {concatenated}, all_reduce->to_apply(),
          all_reduce->replica_groups(),
          /*constrain_layout=*/false, all_reduce->channel_id(),
          all_reduce->use_global_device_ids()));

  // Slice from all-reduce result and bitcast back to the original shapes.
  std::vector<HloInstruction*> outputs;
  outputs.reserve(all_reduce->operand_count());
  int64_t offset = 0;
  for (int64_t i = 0; i < all_reduce->operand_count(); ++i) {
    const Shape& flat_shape = flat_operands[i]->shape();
    int64_t end = offset + flat_shape.dimensions(0);
    HloInstruction* sliced = computation.AddInstruction(
        HloInstruction::CreateSlice(flat_shape, new_all_reduce,
                                    /*start_indices=*/{offset},
                                    /*limit_indices=*/{end},
                                    /*strides=*/{1}));
    outputs.push_back(computation.AddInstruction(HloInstruction::CreateBitcast(
        all_reduce->operand(i)->shape(), sliced)));
    offset = end;
  }
  // Replace original all-reduce with tuple of slices from new all-reduce.
  TF_RETURN_IF_ERROR(computation.ReplaceWithNewInstruction(
      all_reduce, HloInstruction::CreateTuple(outputs)));
  return OkStatus();
}
}  // namespace

StatusOr<bool> AllReduceContiguous::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  VLOG(1) << "Running AllReduceContiguous";

  if (hlo_query::ContainsLayoutConstrainedAllReduce(*module)) {
    VLOG(1)
        << "Skip AllReduceContiguous because the module contains all-reduce "
           "with constrained layouts";
    return false;
  }

  std::vector<HloAllReduceInstruction*> all_reduces;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kAllReduce &&
          instruction->operand_count() > 1) {
        all_reduces.push_back(Cast<HloAllReduceInstruction>(instruction));
      }
    }
  }

  for (HloAllReduceInstruction* all_reduce : all_reduces) {
    TF_RETURN_IF_ERROR(ReplaceWithContiguousAllReduce(all_reduce));
  }

  return !all_reduces.empty();
}

}  // namespace xla
