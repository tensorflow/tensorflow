/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/service/gpu/transforms/onehot_rewriter.h"

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "xla/tsl/platform/status_macros.h"

namespace xla {
namespace gpu {

namespace {

// Maps a dimension `output_dim` of `inst`'s output to the corresponding
// dimension in its 0-th operand for transparent ops.
std::optional<int64_t> GetOperandDimForOutputDim(const HloInstruction* inst,
                                                 int64_t output_dim) {
  if (inst->opcode() == HloOpcode::kConvert ||
      inst->opcode() == HloOpcode::kBitcastConvert) {
    return output_dim;
  }
  if (inst->opcode() == HloOpcode::kBroadcast) {
    for (int64_t i = 0; i < inst->dimensions().size(); ++i) {
      if (inst->dimensions(i) == output_dim) {
        return i;
      }
    }
    return std::nullopt;
  }
  if (inst->opcode() == HloOpcode::kReshape ||
      inst->opcode() == HloOpcode::kBitcast) {
    auto equality = ShapeUtil::InsertedOrDeleted1SizedDimensions(
        inst->operand(0)->shape(), inst->shape());
    if (!equality) {
      return std::nullopt;
    }

    for (int64_t inserted_dim : equality->inserted_dimensions) {
      if (inserted_dim == output_dim) {
        return std::nullopt;
      }
    }

    int64_t non_inserted_dims_before = 0;
    for (int64_t i = 0; i < output_dim; ++i) {
      if (!absl::c_linear_search(equality->inserted_dimensions, i)) {
        non_inserted_dims_before++;
      }
    }

    int64_t current_non_deleted_index = 0;
    for (int64_t i = 0; i < inst->operand(0)->shape().dimensions_size(); ++i) {
      if (!absl::c_linear_search(equality->deleted_dimensions, i)) {
        if (current_non_deleted_index == non_inserted_dims_before) {
          return i;
        }
        current_non_deleted_index++;
      }
    }
  }
  return std::nullopt;
}

// Traces `inst` through transparent ops.
// If `check_iota_dim` is provided, verifies if the root is an Iota at that dim.
// Returns the source instruction (Indices) or nullptr.
HloInstruction* TraceOneHotPattern(HloInstruction* inst,
                                   std::optional<int64_t> check_iota_dim) {
  HloInstruction* current = inst;
  std::optional<int64_t> current_dim = check_iota_dim;
  constexpr int64_t kMaxTraceDepth = 20;

  for (int64_t depth = 0; depth < kMaxTraceDepth; ++depth) {
    if (check_iota_dim.has_value()) {
      if (current->opcode() == HloOpcode::kIota) {
        return (Cast<HloIotaInstruction>(current)->iota_dimension() ==
                *current_dim)
                   ? current
                   : nullptr;
      }
      auto next_dim = GetOperandDimForOutputDim(current, *current_dim);
      if (!next_dim) {
        return nullptr;
      }
      current_dim = next_dim;
    } else {
      // Just tracing to source (Indices)
      if (current->opcode() != HloOpcode::kBroadcast &&
          current->opcode() != HloOpcode::kReshape &&
          current->opcode() != HloOpcode::kConvert &&
          current->opcode() != HloOpcode::kBitcast &&
          current->opcode() != HloOpcode::kBitcastConvert) {
        return current;
      }
    }
    current = current->mutable_operand(0);
  }
  return nullptr;
}

}  // namespace

absl::StatusOr<bool> OneHotGatherRewriter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instr : computation->MakeInstructionPostOrder()) {
      if (instr->opcode() != HloOpcode::kDot) {
        continue;
      }

      auto* dot = Cast<HloDotInstruction>(instr);
      const auto& dnums = dot->dot_dimension_numbers();

      if (dnums.lhs_batch_dimensions_size() > 0 ||
          dnums.rhs_batch_dimensions_size() > 0) {
        continue;
      }

      int64_t one_hot_idx = -1;
      HloInstruction* indices = nullptr;

      // Check both operands for the One-Hot pattern
      for (int i = 0; i < 2; ++i) {
        if (dnums.lhs_contracting_dimensions_size() != 1) {
          continue;
        }
        int64_t contract_dim = (i == 0) ? dnums.lhs_contracting_dimensions(0)
                                        : dnums.rhs_contracting_dimensions(0);

        HloInstruction* root = dot->mutable_operand(i);
        while (root->opcode() == HloOpcode::kConvert ||
               root->opcode() == HloOpcode::kBitcastConvert) {
          root = root->mutable_operand(0);
        }

        HloInstruction* compare = nullptr;
        HloInstruction* call_op = nullptr;

        if (root->opcode() == HloOpcode::kCompare) {
          compare = root;
        } else if (root->opcode() == HloOpcode::kCall) {
          call_op = root;
          if (call_op->to_apply()->root_instruction()->opcode() ==
              HloOpcode::kCompare) {
            compare = call_op->to_apply()->root_instruction();
          }
        }

        if (!compare ||
            compare->comparison_direction() != ComparisonDirection::kEq) {
          continue;
        }

        // Check operands of the compare: one must be Iota, one Indices
        HloInstruction* lhs = compare->mutable_operand(0);
        HloInstruction* rhs = compare->mutable_operand(1);

        bool lhs_iota = TraceOneHotPattern(lhs, contract_dim) != nullptr;
        bool rhs_iota = TraceOneHotPattern(rhs, contract_dim) != nullptr;

        HloInstruction* found = nullptr;
        if (lhs_iota && !rhs_iota) {
          found = TraceOneHotPattern(rhs, std::nullopt);
        } else if (rhs_iota && !lhs_iota) {
          found = TraceOneHotPattern(lhs, std::nullopt);
        }

        if (found) {
          if (call_op) {
            if (found->opcode() == HloOpcode::kParameter) {
              indices = call_op->mutable_operand(found->parameter_number());
            }
          } else {
            indices = found;
          }
          if (indices) {
            one_hot_idx = i;
            break;
          }
        }
      }

      if (one_hot_idx == -1) {
        continue;
      }

      HloInstruction* weights = dot->mutable_operand(1 - one_hot_idx);
      int64_t weights_contract_dim = (one_hot_idx == 0)
                                         ? dnums.rhs_contracting_dimensions(0)
                                         : dnums.lhs_contracting_dimensions(0);

      // Create Gather
      Shape indices_shape = indices->shape();
      std::vector<int64_t> new_dims(indices_shape.dimensions().begin(),
                                    indices_shape.dimensions().end());
      new_dims.push_back(1);
      Shape reshaped_shape =
          ShapeUtil::MakeShape(indices_shape.element_type(), new_dims);

      HloInstruction* reshaped_indices = nullptr;
      for (HloInstruction* user : indices->users()) {
        if (user->opcode() == HloOpcode::kReshape &&
            ShapeUtil::Equal(user->shape(), reshaped_shape)) {
          reshaped_indices = user;
          break;
        }
      }
      if (!reshaped_indices) {
        reshaped_indices = computation->AddInstruction(
            HloInstruction::CreateReshape(reshaped_shape, indices));
      }

      int64_t indices_rank = indices->shape().dimensions_size();
      int64_t weights_rank = weights->shape().dimensions_size();
      int64_t weights_nc_rank = weights_rank - 1;

      GatherDimensionNumbers gnums;
      gnums.mutable_offset_dims()->Reserve(weights_nc_rank);
      if (one_hot_idx == 0) {
        for (int i = 0; i < weights_nc_rank; ++i) {
          gnums.add_offset_dims(indices_rank + i);
        }
      } else {
        for (int i = 0; i < weights_nc_rank; ++i) {
          gnums.add_offset_dims(i);
        }
      }

      gnums.add_collapsed_slice_dims(weights_contract_dim);
      gnums.add_start_index_map(weights_contract_dim);
      gnums.set_index_vector_dim(indices_rank);

      std::vector<int64_t> slice_sizes;
      slice_sizes.reserve(weights_rank);
      for (int i = 0; i < weights_rank; ++i) {
        slice_sizes.push_back(
            i == weights_contract_dim ? 1 : weights->shape().dimensions(i));
      }

      HloInstruction* gather = computation->AddInstruction(
          HloInstruction::CreateGather(dot->shape(), weights, reshaped_indices,
                                       gnums, slice_sizes, false));

      RETURN_IF_ERROR(computation->ReplaceInstruction(dot, gather));
      changed = true;
    }
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
