/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/gpu/triton_fusion_analysis.h"

#include <cstdint>
#include <optional>
#include <queue>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/gpu/cudnn_support_utils.h"
#include "xla/service/gpu/matmul_indexing_utils.h"
#include "xla/service/gpu/triton_tiling_propagation.h"
#include "xla/service/instruction_fusion.h"
#include "xla/status_macros.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

namespace {

using triton_fusion::DimOrdersAndReqs;
using triton_fusion::DimOrdersAndReqsOrError;
using triton_fusion::DotRequirements;
using triton_fusion::FusionContext;
using triton_fusion::GetPropagatedDimOrdersAndRequirements;
using triton_fusion::kNoSplitRequirement;
using triton_fusion::TransformDirection;

}  // namespace

namespace triton_fusion {

/*static*/ absl::StatusOr<FusionContext> FusionContext::FromDotOperand(
    const HloInstruction& dot, const int operand_number, const int split_k) {
  // There can be either none or one split-K batch dimension.
  const int num_split_k_batch_dims = split_k > 1;
  int split_k_dimension_index = kNoDimensionIndex;
  TF_ASSIGN_OR_RETURN(int contracting_dimension_index,
                      ContractingDimensionIndex(dot, operand_number));
  TF_ASSIGN_OR_RETURN(int non_contracting_dimension_index,
                      NonContractingDimensionIndex(dot, operand_number));
  if (split_k > 1) {
    split_k_dimension_index = contracting_dimension_index - 1;
  }
  int splittable_dimension_index = kNoDimensionIndex;
  // LHS non-contracting dimension can be split if non-splitK batch is absent.
  if (operand_number == 0 &&
      dot.dot_dimension_numbers().lhs_batch_dimensions_size() -
              num_split_k_batch_dims ==
          0) {
    splittable_dimension_index = non_contracting_dimension_index;
  }
  FusionContext context(DotProperties{non_contracting_dimension_index,
                                      splittable_dimension_index},
                        DotRequirements(kNoSplitRequirement));
  context.dim_orders_[dot.operand(operand_number)] =
      DimensionOrder::FromDotOperandOrOutput(*dot.operand(operand_number),
                                             split_k_dimension_index);
  return context;
}

/*static*/ FusionContext FusionContext::FromDotOutput(
    const HloInstruction& dot, const int split_k,
    DotRequirements requirements) {
  // Allow non-contracting dimension originating from LHS to split if
  // this dimension is split at the output at the same ratio as
  // at the input.
  int splittable_dimension_index = kNoDimensionIndex;
  if (requirements.splittable_dimension_major_part_size > 1) {
    // Split-K dimension is the first one in the output if present;
    // LHS non-contracting follows (batch is absent in this case).
    splittable_dimension_index = (split_k > 1) ? 1 : 0;
  }
  FusionContext context(DotProperties{/*noncontracting_dimension=*/-1,
                                      splittable_dimension_index},
                        std::move(requirements));
  context.dim_orders_[&dot] = DimensionOrder::FromDotOperandOrOutput(dot);
  return context;
}

bool FusionContext::CombineDimOrdersAndReqs(const DimOrdersAndReqs& update) {
  // First check that all updates to insert are compatible to avoid
  // incomplete merges.
  for (const auto& [key, value] : update.dim_orders) {
    auto it = dim_orders_.find(key);
    if (it != dim_orders_.cend() && !it->second.IsPhysicallyEquivalent(value)) {
      return false;
    }
  }

  DotRequirementsOrError requirements_or_error =
      CombineDotRequirements(requirements_, update.requirements);
  if (std::holds_alternative<FusionDecision>(requirements_or_error)) {
    return false;
  }

  requirements_ = std::move(std::get<DotRequirements>(requirements_or_error));
  dim_orders_.insert(update.dim_orders.begin(), update.dim_orders.end());
  return true;
}

// Returns the list of parameters that are inputs to the given instruction.
// Also updates the dimension orders and iteration specs for the parameters.
absl::Status FusionContext::PropagateDimensionOrdersToParameters(
    const HloInstruction& origin, ConstHloInstructionSet& parameters,
    ConstHloInstructionMap<TensorIterationSpec>& iter_specs) {
  absl::flat_hash_set<const HloInstruction*> visited;
  std::queue<const HloInstruction*> to_process;
  // Dimension orders describing outputs of corresponding instructions.
  visited.insert(&origin);
  to_process.push(&origin);
  while (!to_process.empty()) {
    const HloInstruction* hlo = to_process.front();
    to_process.pop();
    if (hlo->opcode() == HloOpcode::kParameter) {
      // One parameter corresponds to one iteration spec in the results of the
      // analysis. This describes well situations when a parameter has one or
      // more elementwise users - they share the same tiling. Situations when
      // one instruction is read differently by different users in the same
      // scope of the dot are currently prevented during the fusion.
      if (!parameters.insert(hlo).second) {
        return FailedPrecondition(
            "A parameter is read differently by different users. hlo: %s",
            hlo->ToString());
      }
      VLOG(5) << hlo->ToString();
    }
    DimOrdersAndReqsOrError result = GetPropagatedDimOrdersAndRequirements(
        *hlo, dim_orders_.at(hlo), TransformDirection::kOutputToInput,
        properties_);

    if (!std::holds_alternative<DimOrdersAndReqs>(result)) {
      return FailedPrecondition(
          "Can not propagate dim orders and requirements: %s",
          std::get<FusionDecision>(result).Explain());
    }

    if (!CombineDimOrdersAndReqs(std::get<DimOrdersAndReqs>(result))) {
      return FailedPrecondition("Can not combine dim orders and requirements.");
    }

    iter_specs[hlo] = dim_orders_.at(hlo).ToTensorIterationSpec();
    for (const HloInstruction* operand : hlo->operands()) {
      if (!visited.insert(operand).second) {
        continue;
      }
      if (operand->opcode() == HloOpcode::kDot) {
        // Encountering the dot itself happens during the processing of the
        // output fusion. The propagation should stop at it.
        continue;
      }
      to_process.push(operand);
    }
  }
  return absl::OkStatus();
}

}  // namespace triton_fusion

absl::StatusOr<TritonFusionAnalysis> TritonFusionAnalysis::Execute(
    const HloComputation& computation, const int split_k) {
  VLOG(5) << computation.ToString(HloPrintOptions::ShortParsable());
  TritonFusionAnalysis analysis;
  const HloInstruction* dot =
      hlo_query::GetFirstInstructionWithOpcode(computation, HloOpcode::kDot);
  TF_RET_CHECK(dot != nullptr);
  TF_RETURN_IF_ERROR(analysis.ExecuteForDotFusion(*dot, split_k));
  return analysis;
}

absl::StatusOr<TritonFusionAnalysis> TritonFusionAnalysis::Execute(
    const HloDotInstruction& dot, int split_k) {
  TritonFusionAnalysis analysis;
  TF_RETURN_IF_ERROR(analysis.ExecuteForDotFusion(dot, split_k));
  return analysis;
}

bool TritonFusionAnalysis::IsBatchDimMinorForInt4Parameter(
    const HloInstruction& dot, Scope scope) const {
  CHECK(scope == Scope::LHS || scope == Scope::RHS);
  const auto& dims = dot.dot_dimension_numbers();
  const auto& batch_dims = (scope == Scope::LHS) ? dims.lhs_batch_dimensions()
                                                 : dims.rhs_batch_dimensions();

  if (batch_dims.empty()) return true;

  int32_t batch_dim = batch_dims.Get(0);
  CHECK_EQ(batch_dims.size(), 1);
  const auto& params = parameters_.at(scope);
  for (const auto& param : params) {
    if (param->shape().element_type() != S4) continue;

    const auto* strides = IterSpec(scope, param, batch_dim);
    if (strides == nullptr) continue;
    // The hacky way to check if the batch dimension is minor.
    // It also covers the case when the batch dimension is second to last but
    // the minor dimension is equal to 1.
    if (strides->front().stride == 1) return false;
  }
  return true;
}

absl::Status TritonFusionAnalysis::ExecuteForDotFusion(
    const HloInstruction& dot, const int split_k) {
  DotRequirements lhs_requirements(kNoSplitRequirement);
  for (const Scope scope : {Scope::LHS, Scope::RHS, Scope::META}) {
    const int operand_number = static_cast<int>(scope);
    if (dot.operand_count() < operand_number + 1) {
      continue;  // Meta scope is optional.
    }
    TF_ASSIGN_OR_RETURN(auto context, FusionContext::FromDotOperand(
                                          dot, operand_number, split_k));
    TF_RETURN_IF_ERROR(context.PropagateDimensionOrdersToParameters(
        *dot.operand(operand_number), parameters_[scope], iter_specs_[scope]));
    if (scope == Scope::LHS) {
      lhs_requirements = context.requirements();
    }
  }

  // For now the RHS doesn't support splits, so it also doesn't impose any
  // requirements.
  auto context = FusionContext::FromDotOutput(dot, split_k, lhs_requirements);
  const HloInstruction* output = &dot;
  // Currently supported is one fusion output and one path from dot to it.
  // Propagate dimension order from dot to root.
  while (!output->IsRoot()) {
    if (output->user_count() != 1) {
      return absl::FailedPreconditionError(
          absl::StrCat("Expected one user for ", output->ToString()));
    }
    const HloInstruction* input = output;
    // Tuple with a custom call can be added at root to allocate a workspace
    // buffer. These do not need to participate in propagation of dimensions.
    if (IsWorkspaceAllocationRoot(*output->users()[0])) {
      break;
    }
    output = output->users()[0];
    DimOrdersAndReqsOrError result = GetPropagatedDimOrdersAndRequirements(
        *output, context.dim_orders().at(input),
        TransformDirection::kInputToOutput, context.dot_properties());
    if (std::holds_alternative<FusionDecision>(result)) {
      auto decision = std::get<FusionDecision>(result);
      return FailedPrecondition("Failed to propagate tiling with error: %s",
                                decision.Explain());
    }
    if (!context.CombineDimOrdersAndReqs(std::get<DimOrdersAndReqs>(result))) {
      return absl::InternalError(
          "Failed to combine dim orders and requirements.");
    }
  }

  bool spec_was_inserted =
      iter_specs_[Scope::OUTPUT]
          .insert(
              {output, context.dim_orders().at(output).ToTensorIterationSpec()})
          .second;
  if (!spec_was_inserted) {
    return absl::InternalError(
        "Failed to insert output spec for the output fusion.");
  }
  parameters_[Scope::OUTPUT] = {};
  if (output != &dot) {
    // Propagate back to parameters of the output fusion.
    TF_RETURN_IF_ERROR(context.PropagateDimensionOrdersToParameters(
        *output, parameters_[Scope::OUTPUT], iter_specs_[Scope::OUTPUT]));
  }
  return absl::OkStatus();
}

std::optional<TritonFusionAnalysis::Scope>
TritonFusionAnalysis::QueryInstructionScope(const HloInstruction& hlo) const {
  for (const Scope& scope : {Scope::LHS, Scope::RHS, Scope::OUTPUT}) {
    if (iter_specs_.at(scope).count(&hlo) > 0) {
      return scope;
    }
  }
  LOG(WARNING) << "No scope for hlo: " << hlo.ToString();
  return std::nullopt;
}

const TensorIterationSpec::DimIterationSpec* TritonFusionAnalysis::IterSpec(
    const TritonFusionAnalysis::Scope scope, const HloInstruction* hlo,
    const int dimension) const {
  auto hlo_spec = iter_specs_.at(scope).find(hlo);
  if (hlo_spec != iter_specs_.at(scope).cend()) {
    // The pointer returned here may also be nullptr.
    return hlo_spec->second.Find(dimension);
  }
  return nullptr;
}

namespace {

std::string IterationSpecByInstructionMapToString(
    const TritonFusionAnalysis::IterationSpecByInstructionMap& m) {
  return absl::StrCat("IterSpec{",
                      absl::StrJoin(m, ", ",
                                    [&](std::string* s, const auto& kv) {
                                      absl::StrAppend(s, kv.first->name(), ": ",
                                                      kv.second.ToString());
                                    }),
                      "}");
}

std::string ScopeToString(TritonFusionAnalysis::Scope s) {
  switch (s) {
    case TritonFusionAnalysis::Scope::LHS:
      return "LHS";
    case TritonFusionAnalysis::Scope::RHS:
      return "RHS";
    case TritonFusionAnalysis::Scope::META:
      return "META";
    case TritonFusionAnalysis::Scope::OUTPUT:
      return "OUTPUT";
  }
}

}  // namespace

std::string TritonFusionAnalysis::ToString() const {
  return absl::StrCat(
      "TritonFusionAnalysis{\n",
      absl::StrJoin(iter_specs_, ",\n",
                    [&](std::string* s, const auto& kv) {
                      absl::StrAppend(
                          s, ScopeToString(kv.first), ": ",
                          IterationSpecByInstructionMapToString(kv.second));
                    }),
      "\n}");
}

}  // namespace gpu
}  // namespace xla
