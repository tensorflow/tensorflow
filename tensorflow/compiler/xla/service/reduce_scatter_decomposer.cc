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

#include "tensorflow/compiler/xla/service/reduce_scatter_decomposer.h"

#include <sys/types.h>

#include <limits>

#include "tensorflow/compiler/xla/hlo/ir/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instructions.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/hlo/utils/hlo_query.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/collective_decomposer_utils.h"
#include "tensorflow/compiler/xla/service/collective_ops_utils.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {

StatusOr<bool> ReduceScatterDecomposer::Run(
    HloModule *module,
    const absl::flat_hash_set<absl::string_view> &execution_threads) {
  bool changed = false;
  int64_t next_channel_id = hlo_query::NextChannelId(*module);

  for (HloComputation *computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction *instruction :
         computation->MakeInstructionPostOrder()) {
      auto *rs = DynCast<HloReduceScatterInstruction>(instruction);
      if (!rs || !rs->shape().IsArray()) {
        continue;
      }

      std::optional<int64_t> channel_id;
      if (rs->channel_id()) {
        channel_id = next_channel_id++;
      }

      // Create an all-reduce
      HloInstruction *ar =
          computation->AddInstruction(HloInstruction::CreateAllReduce(
              rs->operand(0)->shape(), rs->operands(), rs->to_apply(),
              rs->replica_groups(), rs->constrain_layout(), channel_id,
              rs->use_global_device_ids()));
      // Create start indices for a dynamic slice to decompose the all-reduce
      // results.
      TF_ASSIGN_OR_RETURN(
          CollectiveOpGroupMode group_mode,
          GetCollectiveOpGroupMode(rs->channel_id().has_value(),
                                   rs->use_global_device_ids()));
      TF_ASSIGN_OR_RETURN(
          std::vector<HloInstruction *> start_indices,
          CreateStartIndicesForCollectiveDecomposition(
              group_mode, rs->replica_groups(), rs->shape(),
              rs->scatter_dimension(), computation, update_layout_));

      HloInstruction *ds =
          computation->AddInstruction(HloInstruction::CreateDynamicSlice(
              rs->shape(), ar, start_indices, rs->shape().dimensions()));

      TF_RETURN_IF_ERROR(rs->ReplaceAllUsesWith(ds));
      TF_RETURN_IF_ERROR(computation->RemoveInstruction(rs));
      changed = true;
    }
  }
  return changed;
}

}  // namespace xla
