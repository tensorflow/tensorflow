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

#include "xla/tests/test_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <utility>

#include "absl/status/status.h"
#include "xla/hlo/analysis/hlo_dataflow_analysis.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/service/hlo_verifier.h"
#include "xla/service/transfer_manager.h"
#include "xla/xla_data.pb.h"

namespace xla {

namespace {

enum class ConstantType { kUnknown, kZero, kOne };

// Return the constant type required by this computation, if known.
ConstantType GetInitValue(const HloComputation& computation) {
  // TODO(b/77635120): Add init values, for min, max, and their arg variants.
  const HloInstruction* const root = computation.root_instruction();
  if (computation.num_parameters() != 2 || root->operand_count() != 2 ||
      root->operand(0)->opcode() != HloOpcode::kParameter ||
      root->operand(1)->opcode() != HloOpcode::kParameter ||
      root->operand(0) == root->operand(1)) {
    return ConstantType::kUnknown;
  }

  switch (root->opcode()) {
    case HloOpcode::kAdd:
      return ConstantType::kZero;
    case HloOpcode::kMultiply:
      return ConstantType::kOne;
    default:
      return ConstantType::kUnknown;
  }
}

// Reduce, ReduceWindow, and SelectAndScatter ops may need a non-random
// initialization value.
bool NeedsInitValue(const HloUse& use) {
  const HloInstruction* const instruction = use.instruction;
  const HloOpcode opcode = instruction->opcode();
  const int64_t op_num = use.operand_number;
  return ((opcode == HloOpcode::kReduceWindow && op_num == 1) ||
          (opcode == HloOpcode::kSelectAndScatter && op_num == 2) ||
          (opcode == HloOpcode::kReduce &&
           op_num >= instruction->operand_count() / 2));
}

// Generate random values that are constrained to the input_shape minus the
// output_shape so as not to produce wrapping slices, for instance.
Literal MakeRandomIndex(int64_t index_bound, std::minstd_rand0* engine) {
  std::uniform_int_distribution<int32_t> generator(0, index_bound);
  return LiteralUtil::CreateR0<int32_t>(generator(*engine));
}

// Returns true if `dest' is reachable from `src' through data-formatting and
// custom call instructions within the same computation.
bool ReachableViaDataFormatting(const HloInstruction* src,
                                const HloInstruction* dest,
                                bool treat_gte_as_data_formatting) {
  if (src == dest) {
    return true;
  }
  switch (dest->opcode()) {
    case HloOpcode::kReshape:
    case HloOpcode::kTranspose:
    case HloOpcode::kCopy:
    case HloOpcode::kSlice:
      break;
    case HloOpcode::kCustomCall:
      if (dest->custom_call_target() == "AssumeGatherIndicesInBound") {
        break;
      }
      return false;
    // TODO(b/249417724): a workaround for tuple param.
    case HloOpcode::kGetTupleElement:
      if (treat_gte_as_data_formatting) {
        break;
      } else {
        return false;
      }
    default:
      return false;
  }
  for (const auto* operand : dest->operands()) {
    if (ReachableViaDataFormatting(src, operand,
                                   treat_gte_as_data_formatting)) {
      return true;
    }
  }
  return false;
}

// Use dataflow analysis on each parameter to see if there are uses that would
// be problematic when generating input data.  Returns the list of
// instructions that correspond to their uses.
//
// Should be paired with the CreateLiteralForConstrainedUses() function below.
std::vector<HloInstruction*> FindConstrainedUses(
    const HloDataflowAnalysis& dataflow, const HloInstruction& param,
    bool treat_gte_as_data_formatting) {
  std::vector<HloInstruction*> constrained_uses;
  for (const auto& pair : dataflow.GetInstructionValueSet(&param)) {
    const HloValue& value = dataflow.GetUniqueValueAt(&param, pair.first);
    for (const HloUse& use : value.GetUses()) {
      HloInstruction* instruction = use.instruction;
      const HloOpcode opcode = instruction->opcode();
      const int64_t op_num = use.operand_number;
      if ((opcode == HloOpcode::kDynamicSlice && op_num >= 1) ||
          (opcode == HloOpcode::kDynamicUpdateSlice && op_num >= 2)) {
        constrained_uses.push_back(instruction);
      } else if ((opcode == HloOpcode::kGather ||
                  opcode == HloOpcode::kScatter) &&
                 op_num == 1) {
        constrained_uses.push_back(instruction);
      } else if (opcode == HloOpcode::kFusion) {
        const HloInstruction* const to_analyze =
            instruction->fused_parameter(op_num);
        auto fused_uses = FindConstrainedUses(dataflow, *to_analyze,
                                              treat_gte_as_data_formatting);
        constrained_uses.insert(constrained_uses.end(), fused_uses.begin(),
                                fused_uses.end());
      } else if (NeedsInitValue(use)) {
        constrained_uses.push_back(instruction);
      } else if (opcode == HloOpcode::kConvert ||
                 opcode == HloOpcode::kReducePrecision) {
        auto converted_uses = FindConstrainedUses(dataflow, *instruction,
                                                  treat_gte_as_data_formatting);
        constrained_uses.insert(constrained_uses.end(), converted_uses.begin(),
                                converted_uses.end());
      } else if (opcode == HloOpcode::kSort &&
                 instruction->operand_count() >= 2 && op_num == 0) {
        // Operand 0 of sort is the array of keys used for key/value
        // (two-operand) kSort instructions. Since sort stability is not
        // guaranteed, constrain keys of key-value sort not to have
        // duplicates, since otherwise the value order may legitimately
        // differ.
        constrained_uses.push_back(instruction);
      }
    }
  }

  for (auto* instruction : param.parent()->instructions()) {
    const HloOpcode opcode = instruction->opcode();
    if (opcode == HloOpcode::kGather || opcode == HloOpcode::kScatter) {
      if (instruction->operand(1) == &param) {
        // Above already covers this case.
        continue;
      }
      if (ReachableViaDataFormatting(&param, instruction->operand(1),
                                     treat_gte_as_data_formatting)) {
        constrained_uses.push_back(instruction);
      }
    }
  }
  return constrained_uses;
}

// Given a parameter, generate a random Literal to use as input if there exist
// no constrained uses in the dataflow graph.  If such constraints exist,
// generate a constrained literal (either bounded in the case of indices, or
// zero in the case of init_values for reductions).
absl::StatusOr<Literal> CreateLiteralForConstrainedUses(
    const absl::Span<HloInstruction* const> constrained_uses,
    const HloInstruction& param, const Shape& param_shape,
    std::minstd_rand0* engine, bool use_large_range,
    std::optional<int64_t> max_bits_of_precision) {
  int64_t index_bound = INT64_MAX;
  bool no_duplicates = false;
  bool needs_constant = false;
  bool needs_sorted_indices = false;
  ConstantType constant_type = ConstantType::kUnknown;
  for (HloInstruction* use : constrained_uses) {
    switch (use->opcode()) {
      case HloOpcode::kDynamicSlice:
      case HloOpcode::kDynamicUpdateSlice: {
        const Shape& indexed_shape = use->operand(0)->shape();
        const Shape& slice_shape = use->opcode() == HloOpcode::kDynamicSlice
                                       ? use->shape()
                                       : use->operand(1)->shape();
        const int64_t first_index =
            Cast<HloDynamicIndexInstruction>(use)->first_index_operand_number();
        for (int64_t operand = first_index; operand < use->operand_count();
             ++operand) {
          if (use->operand(operand) == &param) {
            index_bound = std::min(
                index_bound,
                ShapeUtil::GetDimension(indexed_shape, operand - first_index) -
                    ShapeUtil::GetDimension(slice_shape,
                                            operand - first_index));
          }
        }
        break;
      }
      case HloOpcode::kGather:
      case HloOpcode::kScatter: {
        const Shape& operand_shape = use->operand(0)->shape();
        auto index_map = use->opcode() == HloOpcode::kGather
                             ? use->gather_dimension_numbers().start_index_map()
                             : use->scatter_dimension_numbers()
                                   .scatter_dims_to_operand_dims();
        for (const auto dim_in_operand : index_map) {
          index_bound = std::min(index_bound,
                                 operand_shape.dimensions(dim_in_operand) - 1);
        }
        if (use->opcode() == HloOpcode::kScatter) {
          needs_sorted_indices |=
              Cast<const HloScatterInstruction>(use)->indices_are_sorted();
        } else {
          needs_sorted_indices |=
              Cast<const HloGatherInstruction>(use)->indices_are_sorted();
        }
        break;
      }
      case HloOpcode::kReduce:
      case HloOpcode::kReduceWindow:
        needs_constant = true;
        constant_type = GetInitValue(*use->to_apply());
        break;

      case HloOpcode::kSelectAndScatter:
        needs_constant = true;
        constant_type = GetInitValue(*use->scatter());
        break;

      case HloOpcode::kSort:
        no_duplicates = true;
        break;

      default:
        return Unimplemented(
            "Constrained operand generation not implemented for %s.",
            use->ToString());
    }
  }
  int constraint_count = 0;
  constraint_count += no_duplicates ? 1 : 0;
  constraint_count += (index_bound != INT64_MAX) ? 1 : 0;
  constraint_count += needs_constant ? 1 : 0;
  if (constraint_count > 1) {
    return Unimplemented("Conflicting operand generation constraints.");
  }
  if (index_bound != INT64_MAX) {
    return MakeFakeLiteral(param_shape, engine,
                           std::pair<int64_t, int64_t>(0, index_bound),
                           needs_sorted_indices, no_duplicates, use_large_range,
                           max_bits_of_precision);
  } else if (needs_constant) {
    switch (constant_type) {
      case ConstantType::kZero:
        return LiteralUtil::Zero(param_shape.element_type());
      case ConstantType::kOne:
        return LiteralUtil::One(param_shape.element_type());
      case ConstantType::kUnknown:
        // We want the identity element for the computation, but we don't
        // really know what it is - so any value we generate will be just as
        // wrong.
        return MakeFakeLiteral(param_shape, engine, /*limit=*/std::nullopt,
                               /*is_sorted=*/needs_sorted_indices,
                               /*no_duplicates=*/false, use_large_range,
                               max_bits_of_precision);
    }
  } else {
    return MakeFakeLiteral(param_shape, engine, /*limit=*/std::nullopt,
                           /*is_sorted=*/needs_sorted_indices, no_duplicates,
                           use_large_range, max_bits_of_precision);
  }
}

// Given a module entry parameter, use the dataflow analysis to see if a
// special case literal must be created, or if we can generate fake data.
absl::StatusOr<Literal> MakeConstrainedArgument(
    const HloDataflowAnalysis& dataflow, const HloInstruction& param,
    const Shape& param_shape, std::minstd_rand0* engine, bool use_large_range,
    bool treat_gte_as_data_formatting,
    std::optional<int64_t> max_bits_of_precision) {
  const auto constrained_uses =
      FindConstrainedUses(dataflow, param, treat_gte_as_data_formatting);
  return CreateLiteralForConstrainedUses(constrained_uses, param, param_shape,
                                         engine, use_large_range,
                                         max_bits_of_precision);
}

}  // namespace

absl::StatusOr<std::vector<Literal>> MakeFakeArguments(
    const HloModule* module, bool pseudo_random, bool use_large_range,
    bool treat_gte_as_data_formatting,
    std::optional<int64_t> max_bits_of_precision, std::minstd_rand0* engine) {
  if (!pseudo_random) {
    return MakeFakeArguments(module, nullptr, use_large_range,
                             treat_gte_as_data_formatting,
                             max_bits_of_precision);
  }
  if (engine == nullptr) {
    auto new_engine =
        pseudo_random ? std::make_unique<std::minstd_rand0>() : nullptr;
    return MakeFakeArguments(module, new_engine.get(), use_large_range,
                             treat_gte_as_data_formatting,
                             max_bits_of_precision);
  }
  return MakeFakeArguments(module, engine, use_large_range,
                           treat_gte_as_data_formatting, max_bits_of_precision);
}

absl::StatusOr<std::vector<Literal>> MakeFakeArguments(
    const HloModule* module, std::minstd_rand0* engine, bool use_large_range,
    bool treat_gte_as_data_formatting,
    std::optional<int64_t> max_bits_of_precision) {
  TF_ASSIGN_OR_RETURN(auto dataflow, HloDataflowAnalysis::Run(*module));
  const auto params = module->entry_computation()->parameter_instructions();
  std::vector<Literal> arguments(params.size());
  for (int i = 0; i < params.size(); ++i) {
    const HloModuleConfig& module_config = module->config();
    const Shape& param_shape = (module_config.has_entry_computation_layout() &&
                                module_config.entry_computation_layout()
                                    .parameter_layout(i)
                                    .shape()
                                    .is_static())
                                   ? module_config.entry_computation_layout()
                                         .parameter_layout(i)
                                         .shape()
                                   : params[i]->shape();

    TF_ASSIGN_OR_RETURN(
        arguments[i],
        MakeConstrainedArgument(*dataflow, *params[i], param_shape, engine,
                                use_large_range, treat_gte_as_data_formatting,
                                max_bits_of_precision));
  }
  return std::move(arguments);
}

absl::Status VerifyHloModule(HloModule* const module, bool layout_sensitive,
                             bool allow_mixed_precision) {
  return HloVerifier(/*layout_sensitive=*/layout_sensitive,
                     /*allow_mixed_precision=*/allow_mixed_precision)
      .Run(module)
      .status();
}

std::unique_ptr<HloDotInstruction> CreateCanonicalDot(const Shape& shape,
                                                      HloInstruction* lhs,
                                                      HloInstruction* rhs) {
  CHECK_LE(lhs->shape().dimensions().size(), 2);
  CHECK_LE(rhs->shape().dimensions().size(), 2);
  PrecisionConfig precision_config;
  precision_config.mutable_operand_precision()->Resize(
      2, PrecisionConfig::DEFAULT);
  DotDimensionNumbers dot_dimension_numbers;
  dot_dimension_numbers.add_lhs_contracting_dimensions(
      lhs->shape().dimensions().size() > 1 ? 1 : 0);
  dot_dimension_numbers.add_rhs_contracting_dimensions(0);
  return std::make_unique<HloDotInstruction>(
      shape, lhs, rhs, dot_dimension_numbers, precision_config);
}

}  // namespace xla
