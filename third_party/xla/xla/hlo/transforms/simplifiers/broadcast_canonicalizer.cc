/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/hlo/transforms/simplifiers/broadcast_canonicalizer.h"

#include <cstdint>
#include <iterator>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/hlo_creation_utils.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {

BroadcastCanonicalizer::BroadcastCanonicalizer() {}

absl::StatusOr<bool> BroadcastCanonicalizer::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  // Sort broadcast dims. Then insert a transpose on the broadcast to get the
  // original shape back.
  for (const auto& computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* hlo : computation->MakeInstructionPostOrder()) {
      if (hlo->opcode() != HloOpcode::kBroadcast) {
        continue;
      }
      if (absl::c_is_sorted(hlo->dimensions())) {
        continue;
      }
      std::vector<int64_t> new_dims(hlo->dimensions().begin(),
                                    hlo->dimensions().end());
      std::vector<int64_t> original_dims(hlo->dimensions().begin(),
                                         hlo->dimensions().end());

      std::vector<int64_t> new_broadcast_dims(hlo->shape().dimensions().begin(),
                                              hlo->shape().dimensions().end());
      absl::c_sort(new_dims);
      const int64_t rank = hlo->shape().rank();
      for (int i = 0; i < new_dims.size(); ++i) {
        new_broadcast_dims[new_dims[i]] =
            hlo->operand(0)->shape().dimensions(i);
      }

      auto new_broadcast = MakeBroadcastHlo(hlo->mutable_operand(0), new_dims,
                                            new_broadcast_dims);
      std::vector<int64_t> transpose_dims(rank);
      absl::c_iota(transpose_dims, 0);
      for (int i = 0; i < new_dims.size(); ++i) {
        transpose_dims[new_dims[i]] = new_dims[std::distance(
            original_dims.begin(), absl::c_find(original_dims, new_dims[i]))];
      }
      TF_RETURN_IF_ERROR(computation->ReplaceWithNewInstruction(
          hlo, HloInstruction::CreateTranspose(hlo->shape(), new_broadcast,
                                               transpose_dims)));
      changed = true;
    }
  }
  return changed;
}

}  // namespace xla
