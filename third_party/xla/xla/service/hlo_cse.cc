/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/hlo_cse.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/any_invocable.h"
#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/hlo_cse_constant_key.h"
#include "xla/service/hlo_domain_map.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/side_effect_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

// Find and combine identical constants. Constants are identical if they have
// the same type and value.
//
// While we're here, also combine identical iota instructions, since they need
// similar treatment.
template <bool kIsLayoutSensitive>
absl::StatusOr<bool> CombineConstants(
    HloComputation* computation,
    absl::AnyInvocable<bool(const HloInstruction*)> should_combine_constant) {
  // Populating the domain map is somewhat expensive -- only do it if there are
  // kDomain ops in the computation.  If there are no kDomain ops, the domain
  // map is trivial, every op gets mapped to the same domain.
  std::unique_ptr<HloDomainMap> domain_map;
  if (absl::c_any_of(computation->instructions(),
                     [&](const HloInstruction* instr) {
                       return instr->opcode() == HloOpcode::kDomain;
                     })) {
    TF_ASSIGN_OR_RETURN(domain_map, HloDomainMap::Create(computation, ""));
  }

  // Map from the literal hash of a constant or the shape hash of an iota all
  // equivalent instructions. This avoids extreme quadratic behavior with many
  // scalar constants.
  absl::flat_hash_map<CseConstantKey<kIsLayoutSensitive>,
                      HloConstantInstruction*>
      constants;
  int64_t combined = 0;
  auto inst_it = computation->instructions().begin();
  while (inst_it != computation->instructions().end()) {
    HloInstruction* instruction = *inst_it;

    // Advance list iterator before loop body because iterator may be
    // invalidated due to deletion.
    ++inst_it;

    if (should_combine_constant != nullptr &&
        !should_combine_constant(instruction)) {
      continue;
    }

    HloInstruction* match = nullptr;
    if (auto* constant_inst = DynCast<HloConstantInstruction>(instruction)) {
      auto [it, did_insert] = constants.insert(
          {CseConstantKey<kIsLayoutSensitive>{
               constant_inst->literal(), constant_inst->shape(),
               (domain_map != nullptr ? domain_map->GetDomainId(instruction)
                                      : 0)},
           constant_inst});
      if (!did_insert) {
        match = it->second;
      }
    }

    if (match != nullptr) {
      // Match found, replace this instruction with the one in the set.
      CHECK_OK(instruction->ReplaceAllUsesWith(match));
      CHECK_OK(computation->RemoveInstruction(instruction));
      ++combined;
    }
  }
  VLOG(4) << "Combined " << combined << " constants and iotas in "
          << computation->name() << " computation";
  return combined > 0;
}

// An instruction is considered to be equivalent to another only if they
// share the exact same set of operands.
struct CseKey {
  template <typename H>
  friend H AbslHashValue(H h, const CseKey& key) {
    auto instruction = key.hlo;
    h = instruction->shape().IsArray()
            ? H::combine(std::move(h), instruction->opcode(),
                         instruction->shape().dimensions())
            : H::combine(std::move(h), instruction->opcode());
    auto window_hash = [](H h, const Window& window) {
      const auto& window_dims = window.dimensions();
      for (const auto& window_dim : window_dims) {
        h = H::combine(std::move(h), window_dim.size(), window_dim.stride(),
                       window_dim.padding_low(), window_dim.padding_high(),
                       window_dim.window_dilation(), window_dim.base_dilation(),
                       window_dim.window_reversal());
      }
      return H::combine(std::move(h), window_dims.size());
    };

    auto result_accuracy_hash = [](H h, const ResultAccuracy& result_accuracy) {
      if (result_accuracy.has_tolerance()) {
        return H::combine(std::move(h), result_accuracy.tolerance().atol(),
                          result_accuracy.tolerance().rtol(),
                          result_accuracy.tolerance().ulps());
      }
      return H::combine(std::move(h), result_accuracy.mode());
    };
    h = result_accuracy_hash(std::move(h), instruction->result_accuracy());

    // Hash operands, ignoring operand order on commutative ops.
    if (HloOpcodeIsBinaryCommutative(instruction->opcode())) {
      CHECK_EQ(instruction->operand_count(), 2);
      auto id0 = instruction->operand(0)->unique_id();
      if (instruction->operand(0)->opcode() == HloOpcode::kIota) {
        id0 = 0;
      }
      auto id1 = instruction->operand(1)->unique_id();
      if (instruction->operand(1)->opcode() == HloOpcode::kIota) {
        id1 = 0;
      }
      if (id0 > id1) {
        std::swap(id0, id1);
      }
      h = H::combine(std::move(h), id0, id1);
    } else {
      for (auto operand : instruction->operands()) {
        if (operand->opcode() == HloOpcode::kIota) {
          continue;
        }
        h = H::combine(std::move(h), operand->unique_id());
      }
    }

    for (auto c : instruction->called_computations()) {
      h = H::combine(std::move(h), c->root_instruction()->opcode());
    }
    switch (instruction->opcode()) {
      case HloOpcode::kSlice:
        return H::combine(std::move(h), instruction->slice_starts(),
                          instruction->slice_strides());
      case HloOpcode::kPad: {
        const auto& padding_dims = instruction->padding_config().dimensions();
        for (const auto& padding_dim : padding_dims) {
          h = H::combine(std::move(h), padding_dim.edge_padding_low(),
                         padding_dim.edge_padding_high(),
                         padding_dim.interior_padding());
        }
        h = H::combine(std::move(h), padding_dims.size());
        return std::move(h);
      }
      case HloOpcode::kDot: {
        const auto& dot_dimension_numbers =
            instruction->dot_dimension_numbers();
        h = H::combine(
            std::move(h),
            absl::MakeSpan(dot_dimension_numbers.lhs_contracting_dimensions()),
            absl::MakeSpan(dot_dimension_numbers.rhs_contracting_dimensions()),
            absl::MakeSpan(dot_dimension_numbers.lhs_batch_dimensions()),
            absl::MakeSpan(dot_dimension_numbers.rhs_batch_dimensions()));
        return std::move(h);
      }
      case HloOpcode::kConvolution: {
        const auto& conv_dimension_numbers =
            instruction->convolution_dimension_numbers();
        h = H::combine(
            std::move(h), conv_dimension_numbers.input_batch_dimension(),
            conv_dimension_numbers.input_feature_dimension(),
            absl::MakeSpan(conv_dimension_numbers.input_spatial_dimensions()),
            conv_dimension_numbers.kernel_input_feature_dimension(),
            conv_dimension_numbers.kernel_output_feature_dimension(),
            absl::MakeSpan(conv_dimension_numbers.kernel_spatial_dimensions()),
            conv_dimension_numbers.output_batch_dimension(),
            conv_dimension_numbers.output_feature_dimension(),
            absl::MakeSpan(conv_dimension_numbers.output_spatial_dimensions()));
        return window_hash(std::move(h), instruction->window());
      }
      case HloOpcode::kReduceWindow:
        return window_hash(std::move(h), instruction->window());
      case HloOpcode::kConcatenate:
      case HloOpcode::kBroadcast:
      case HloOpcode::kTranspose:
      case HloOpcode::kReduce:
        return H::combine(std::move(h), instruction->dimensions());
      case HloOpcode::kGetTupleElement:
        return H::combine(std::move(h), instruction->tuple_index());
      case HloOpcode::kCompare:
        return H::combine(
            std::move(h),
            Cast<HloCompareInstruction>(instruction)->direction());
      default:
        return std::move(h);
    }
  }
  HloInstruction* hlo;
};

}  // namespace

/*static*/
bool HloCSE::ShouldEliminateInstruction(const HloInstruction* instruction) {
  // If the instruction has zero operands (constants, parameters, etc.) skip
  // over it.
  if (instruction->operand_count() == 0 &&
      instruction->opcode() != HloOpcode::kPartitionId &&
      instruction->opcode() != HloOpcode::kReplicaId) {
    return false;
  }

  const FrontendAttributes& frontend_attributes =
      instruction->frontend_attributes();
  if (frontend_attributes.IsInitialized()) {
    if (frontend_attributes.map().contains(kMustFuseAttr)) {
      return false;
    }
  }

  // Skip instructions which have side effects.
  if (instruction->HasSideEffect()) {
    return false;
  }

  return true;
}

absl::StatusOr<bool> HloCSE::RunOnComputation(HloComputation* computation) {
  if (should_eliminate_computation_ &&
      !should_eliminate_computation_(computation)) {
    return false;
  }

  TF_ASSIGN_OR_RETURN(
      bool changed,
      is_layout_sensitive_
          ? CombineConstants<true>(computation,
                                   std::move(should_combine_constant_))
          : CombineConstants<false>(computation,
                                    std::move(should_combine_constant_)));

  const auto eq_instructions = [&](const HloInstruction* a,
                                   const HloInstruction* b) {
    if (a == b) {
      return true;
    }
    if (a->opcode() != b->opcode() || a->opcode() != HloOpcode::kIota) {
      return false;
    }
    return a->dimensions(0) == b->dimensions(0) &&
           (is_layout_sensitive_
                ? ShapeUtil::Equal(a->shape(), b->shape())
                : ShapeUtil::Compatible(a->shape(), b->shape()));
  };
  const auto eq_computations = [](const HloComputation* lhs,
                                  const HloComputation* rhs) {
    return *lhs == *rhs;
  };

  auto cse_equal = [&](const CseKey& lhs, const CseKey& rhs) {
    return lhs.hlo->IdenticalIgnoringCommutativeOperandOrder(
        *rhs.hlo, eq_instructions, eq_computations, is_layout_sensitive_,
        /*sharding_sensitive=*/true);
  };

  // HLO instructions are grouped into equivalency classes by using the
  // cse_equal predicate defined above. This set holds a representative
  // instruction for each class.
  absl::flat_hash_set<CseKey, absl::Hash<CseKey>, decltype(cse_equal)>
      representatives(/*N=*/computation->instruction_count() + 1,
                      absl::Hash<CseKey>{}, cse_equal);
  for (auto instruction : computation->MakeInstructionPostOrder()) {
    if (should_eliminate_instruction_ != nullptr
            ? !should_eliminate_instruction_(instruction)
            : !ShouldEliminateInstruction(instruction)) {
      continue;
    }

    // Skip instructions that cannot be safely removed, regardless if they were
    // requested to be removed or not.
    if (!computation->IsSafelyRemovable(instruction,
                                        ignore_control_dependencies_)) {
      continue;
    }

    auto pair = representatives.insert(CseKey{instruction});
    if (!pair.second) {
      HloInstruction* equivalent_instruction = pair.first->hlo;
      TF_RETURN_IF_ERROR(
          instruction->ReplaceAllUsesWith(equivalent_instruction));
      TF_RETURN_IF_ERROR(computation->RemoveInstructionAndUnusedOperands(
          instruction, /*cleanup=*/std::nullopt, ignore_control_dependencies_));
      VLOG(4) << "Replaced " << instruction->name() << " with "
              << equivalent_instruction->name();
      changed = true;
      continue;
    }
    for (int64_t i = 0; i < instruction->operand_count(); ++i) {
      HloInstruction* a = instruction->mutable_operand(i);
      if (a->opcode() != HloOpcode::kIota) {
        continue;
      }
      for (int64_t j = i + 1; j < instruction->operand_count(); ++j) {
        HloInstruction* b = instruction->mutable_operand(j);
        if (a == b || !eq_instructions(a, b)) {
          continue;
        }
        TF_RETURN_IF_ERROR(instruction->ReplaceOperandWith(j, a));
        changed = true;
        if (b->IsDead()) {
          TF_RETURN_IF_ERROR(computation->RemoveInstruction(b));
        }
      }
    }
  }
  if (auto fusion = computation->FusionInstruction()) {
    if (fusion->IsMultiOutputFusion()) {
      // Attach users to the representative instruction, thus making the
      // duplicate fusion roots unused. HloDCE can then cleanup the unused
      // fusion roots.
      absl::flat_hash_map<const HloInstruction*, int64_t> root_to_unique_index;
      int64_t root_index = 0;
      HloInstruction* root = computation->root_instruction();
      for (const HloInstruction* hlo : root->operands()) {
        if (root_to_unique_index.find(hlo) == root_to_unique_index.end()) {
          root_to_unique_index[hlo] = root_to_unique_index[hlo] = root_index;
        }
        ++root_index;
      }
      if (root_to_unique_index.size() < root->operand_count()) {
        for (HloInstruction* user : fusion->users()) {
          if (user->opcode() == HloOpcode::kGetTupleElement) {
            const HloInstruction* fusion_root =
                root->operand(user->tuple_index());
            user->set_tuple_index(root_to_unique_index[fusion_root]);
          }
        }
      }
    }
  }
  return changed;
}

absl::StatusOr<bool> HloCSE::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  for (auto* computation : module->computations(execution_threads)) {
    TF_ASSIGN_OR_RETURN(bool computation_changed,
                        RunOnComputation(computation));
    changed |= computation_changed;
  }
  return changed;
}

}  // namespace xla
