/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/hlo/transforms/collectives/collective_permute_cse.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/hlo_reachability.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/tsl/platform/errors.h"

namespace xla {

static bool IsContained(const HloInstruction* large_slice,
                        const HloInstruction* small_slice) {
  for (int64_t i = 0; i < large_slice->slice_starts().size(); ++i) {
    if (large_slice->slice_strides()[i] != small_slice->slice_strides()[i]) {
      return false;
    }
    int64_t l_start = large_slice->slice_starts()[i];
    int64_t s_start = small_slice->slice_starts()[i];
    int64_t stride = large_slice->slice_strides()[i];
    if (s_start < l_start || (s_start - l_start) % stride != 0) {
      return false;
    }
    int64_t new_start = (s_start - l_start) / stride;
    int64_t n_l = large_slice->shape().dimensions(i);
    int64_t n_s = small_slice->shape().dimensions(i);
    if (new_start + n_s > n_l) {
      return false;
    }
  }
  return true;
}

static absl::Status RemoveControlDependencies(
    HloInstruction* small, HloInstruction* large,
    HloReachabilityMap* reachability) {
  CHECK(reachability != nullptr);
  std::vector<HloInstruction*> preds_to_remove;
  for (HloInstruction* pred : large->control_predecessors()) {
    if (pred == small || reachability->IsReachable(small, pred)) {
      preds_to_remove.push_back(pred);
    }
  }
  for (HloInstruction* pred : preds_to_remove) {
    TF_RETURN_IF_ERROR(pred->RemoveControlDependencyTo(large));
  }
  return absl::OkStatus();
}

absl::StatusOr<bool> CollectivePermuteCSE::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  for (HloComputation* computation : module->computations(execution_threads)) {
    if (computation->IsFusionComputation()) {
      continue;
    }

    std::unique_ptr<HloReachabilityMap> reachability;

    std::vector<HloInstruction*> permutes;
    for (HloInstruction* inst : computation->MakeInstructionPostOrder()) {
      if (inst->opcode() == HloOpcode::kCollectivePermute) {
        permutes.push_back(inst);
      }
    }

    if (permutes.size() < 2) {
      continue;
    }

    for (size_t i = 0; i < permutes.size(); ++i) {
      HloInstruction* a = permutes[i];
      if (!a || a->user_count() == 0) {
        continue;
      }
      auto a_pairs = a->source_target_pairs();
      absl::c_sort(a_pairs);
      for (size_t j = i + 1; j < permutes.size(); ++j) {
        HloInstruction* b = permutes[j];
        if (!b || b->user_count() == 0) {
          continue;
        }

        auto b_pairs = b->source_target_pairs();
        absl::c_sort(b_pairs);

        if (a_pairs != b_pairs) {
          continue;
        }

        HloInstruction* large = nullptr;
        HloInstruction* small = nullptr;
        bool is_slice_cse = false;

        if (a->operand(0) == b->operand(0)) {
          large = a;
          small = b;
        } else if (a->operand(0)->opcode() == HloOpcode::kSlice &&
                   a->operand(0)->operand(0) == b->operand(0)) {
          small = a;
          large = b;
          is_slice_cse = true;
        } else if (b->operand(0)->opcode() == HloOpcode::kSlice &&
                   b->operand(0)->operand(0) == a->operand(0)) {
          small = b;
          large = a;
          is_slice_cse = true;
        } else if (a->operand(0)->opcode() == HloOpcode::kSlice &&
                   b->operand(0)->opcode() == HloOpcode::kSlice &&
                   a->operand(0)->operand(0) == b->operand(0)->operand(0)) {
          if (IsContained(b->operand(0), a->operand(0))) {
            large = b;
            small = a;
            is_slice_cse = true;
          } else if (IsContained(a->operand(0), b->operand(0))) {
            large = a;
            small = b;
            is_slice_cse = true;
          }
        }

        if (large && small) {
          if (!reachability) {
            reachability = HloReachabilityMap::Build(computation);
          }
          CHECK(reachability != nullptr);
          TF_RETURN_IF_ERROR(
              RemoveControlDependencies(small, large, reachability.get()));

          HloInstruction* replacement = large;
          if (is_slice_cse) {
            const HloInstruction* small_slice = small->operand(0);
            if (large->operand(0)->opcode() == HloOpcode::kSlice) {
              const HloInstruction* large_slice = large->operand(0);
              std::vector<int64_t> new_starts, new_limits, new_strides;
              for (int64_t i = 0; i < small_slice->slice_starts().size(); ++i) {
                new_starts.push_back((small_slice->slice_starts()[i] -
                                      large_slice->slice_starts()[i]) /
                                     large_slice->slice_strides()[i]);
                new_limits.push_back(new_starts.back() +
                                     small_slice->shape().dimensions(i));
                new_strides.push_back(1);
              }
              replacement = computation->AddInstruction(
                  HloInstruction::CreateSlice(small->shape(), large, new_starts,
                                              new_limits, new_strides));
            } else {
              replacement = computation->AddInstruction(
                  small_slice->CloneWithNewOperands(small->shape(), {large}));
            }
          }

          // Force large to dominate small's location. By default, creating a
          // slice of large and replacing it means slice executes after large
          // and replaces small. Is it possible small was before large? Yes. If
          // small was before large, slice(large) will be computed where large
          // is, replacing the outputs that used small. We might affect memory
          // size if large is delayed. Replacement handles dependencies.
          TF_RETURN_IF_ERROR(small->ReplaceAllUsesWith(replacement));
          reachability->Replace(small, replacement);
          if (small->user_count() == 0) {
            TF_RETURN_IF_ERROR(computation->RemoveInstruction(small));
          }
          if (small == a) {
            permutes[i] = nullptr;
          }
          if (small == b) {
            permutes[j] = nullptr;
          }
          changed = true;
          if (small == a) {
            break;
          }
        }
      }
    }
  }
  return changed;
}

}  // namespace xla
