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

#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/service/hlo_domain_map.h"
#include "xla/shape_util.h"
#include "tsl/platform/errors.h"

namespace xla {

namespace {

template <bool kIsLayoutSensitive>
struct ConstantKey {
  template <typename H>
  friend H AbslHashValue(H h, const ConstantKey& key) {
    h = H::combine(std::move(h), key.domain);
    return Literal::Hash<H, kIsLayoutSensitive, /*kByteLimit=*/64>(
        std::move(h), key.hlo->literal());
  }
  friend bool operator==(const ConstantKey& lhs, const ConstantKey& rhs) {
    return lhs.domain == rhs.domain &&
           (kIsLayoutSensitive ? Shape::Equal()
                               : Shape::Equal().IgnoreLayout())(
               lhs.hlo->shape(), rhs.hlo->shape()) &&
           lhs.hlo->literal().Equal(rhs.hlo->literal(), kIsLayoutSensitive);
  }
  HloConstantInstruction* hlo;
  int64_t domain;
};

// Find and combine identical constants. Constants are identical if they have
// the same type and value.
//
// While we're here, also combine identical iota instructions, since they need
// similar treatment.
template <bool kIsLayoutSensitive>
absl::StatusOr<bool> CombineConstants(HloComputation* computation,
                                      bool only_scalars) {
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
  absl::flat_hash_set<ConstantKey<kIsLayoutSensitive>> constants;
  int64_t combined = 0;
  auto inst_it = computation->instructions().begin();
  while (inst_it != computation->instructions().end()) {
    HloInstruction* instruction = *inst_it;

    // Advance list iterator before loop body because iterator may be
    // invalidated due to deletion.
    ++inst_it;

    if (only_scalars && !ShapeUtil::IsScalar(instruction->shape())) {
      continue;
    }

    HloInstruction* match = nullptr;
    if (auto* constant_inst = DynCast<HloConstantInstruction>(instruction)) {
      auto insert_result = constants.insert(ConstantKey<kIsLayoutSensitive>{
          constant_inst,
          (domain_map != nullptr ? domain_map->GetDomainId(instruction) : 0)});
      if (!insert_result.second) {
        match = insert_result.first->hlo;
      }
    }

    if (match != nullptr) {
      // Match found, replace this instruction with the one in the set.
      TF_CHECK_OK(instruction->ReplaceAllUsesWith(match));
      TF_CHECK_OK(computation->RemoveInstruction(instruction));
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
    h = H::combine(std::move(h), instruction->opcode(),
                   instruction->shape().dimensions());
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

absl::StatusOr<bool> HloCSE::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

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

  for (auto* computation : module->computations(execution_threads)) {
    if (only_fusion_computations_ && !computation->IsFusionComputation()) {
      continue;
    }

    TF_ASSIGN_OR_RETURN(
        bool combined,
        is_layout_sensitive_
            ? CombineConstants<true>(computation, only_scalars_)
            : CombineConstants<false>(computation, only_scalars_));
    changed |= combined;

    // HLO instructions are grouped into equivalency classes by using the
    // cse_equal predicate defined above. This set holds a representative
    // instruction for each class.
    absl::flat_hash_set<CseKey, absl::Hash<CseKey>, decltype(cse_equal)>
        representatives(/*N=*/computation->instruction_count() + 1,
                        absl::Hash<CseKey>{}, cse_equal);
    for (auto instruction : computation->MakeInstructionPostOrder()) {
      // If the instruction has zero operands (constants, parameters, etc.) skip
      // over it.
      if (instruction->operand_count() == 0 &&
          instruction->opcode() != HloOpcode::kPartitionId &&
          instruction->opcode() != HloOpcode::kReplicaId) {
        continue;
      }
      // Skip instructions which have side effects.
      if (instruction->HasSideEffect()) {
        continue;
      }

      if (only_scalars_ && !ShapeUtil::IsScalar(instruction->shape())) {
        continue;
      }

      auto pair = representatives.insert(CseKey{instruction});
      if (!pair.second) {
        HloInstruction* equivalent_instruction = pair.first->hlo;
        TF_RETURN_IF_ERROR(
            instruction->ReplaceAllUsesWith(equivalent_instruction));
        TF_RETURN_IF_ERROR(computation->RemoveInstructionAndUnusedOperands(
            instruction, /*cleanup=*/std::nullopt,
            ignore_control_dependencies_));
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
  }
  return changed;
}

}  // namespace xla
