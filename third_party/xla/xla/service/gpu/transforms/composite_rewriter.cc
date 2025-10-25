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

#include "xla/service/gpu/transforms/composite_rewriter.h"

#include <cstdint>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

namespace {

absl::StatusOr<DotDimensionNumbers> ParseDimensionNumbers(
    absl::string_view composite_attributes) {
  mlir::MLIRContext context;
  mlir::Attribute attr = mlir::parseAttribute(composite_attributes, &context);
  mlir::DictionaryAttr dict_attrs = mlir::dyn_cast<mlir::DictionaryAttr>(attr);
  auto get_int = [&](absl::string_view key) -> absl::StatusOr<int32_t> {
    if (!dict_attrs.contains(key)) {
      return absl::InvalidArgumentError(absl::StrCat(key, " is not set"));
    }
    mlir::Attribute attr = dict_attrs.get(key);
    if (!mlir::isa<mlir::IntegerAttr>(attr)) {
      return absl::InvalidArgumentError(
          absl::StrCat(key, " is not an integer"));
    }
    return mlir::cast<mlir::IntegerAttr>(attr).getInt();
  };
  DotDimensionNumbers dot_dimension_numbers;
  TF_ASSIGN_OR_RETURN(int32_t lhs_contracting_dim_index,
                      get_int("lhs_contracting_dim_index"));
  dot_dimension_numbers.add_lhs_contracting_dimensions(
      lhs_contracting_dim_index);
  TF_ASSIGN_OR_RETURN(int32_t rhs_contracting_dim_index,
                      get_int("rhs_contracting_dim_index"));
  dot_dimension_numbers.add_rhs_contracting_dimensions(
      rhs_contracting_dim_index);

  if (dict_attrs.contains("lhs_batch_dim_index")) {
    TF_ASSIGN_OR_RETURN(int32_t lhs_batch_dim_index,
                        get_int("lhs_batch_dim_index"));
    dot_dimension_numbers.add_lhs_batch_dimensions(lhs_batch_dim_index);
  }
  if (dict_attrs.contains("rhs_batch_dim_index")) {
    TF_ASSIGN_OR_RETURN(int32_t rhs_batch_dim_index,
                        get_int("rhs_batch_dim_index"));
    dot_dimension_numbers.add_rhs_batch_dimensions(rhs_batch_dim_index);
  }
  if (dot_dimension_numbers.lhs_batch_dimensions_size() !=
      dot_dimension_numbers.rhs_batch_dimensions_size()) {
    return absl::InvalidArgumentError(
        "batch dimension should be specified for both lhs and rhs.");
  }
  return dot_dimension_numbers;
}

}  // namespace

absl::StatusOr<bool> CompositeRewriter::RewriteComputation(
    HloComputation* computation) {
  bool changed = false;
  for (HloInstruction* instruction : computation->MakeInstructionPostOrder()) {
    if (instruction->opcode() != HloOpcode::kCall) {
      continue;
    }
    auto call = Cast<HloCallInstruction>(instruction);
    if (!call->is_composite()) {
      continue;
    }
    if (!call->has_frontend_attributes()) {
      VLOG(3) << "No frontend attributes";
      continue;
    }
    auto frontend_attrs = call->frontend_attributes().map();
    auto key = "composite.name";
    if (!frontend_attrs.contains(key) ||
        frontend_attrs.at(key) != "xla.scaled_dot") {
      VLOG(3) << key << " is not xla.scaled_dot: " << frontend_attrs.at(key);
      continue;
    }
    if (!frontend_attrs.contains("composite.attributes")) {
      return absl::InvalidArgumentError(
          "composite.attributes is not set for xla.scaled_dot");
    }
    TF_ASSIGN_OR_RETURN(
        DotDimensionNumbers dot_dimension_numbers,
        ParseDimensionNumbers(frontend_attrs.at("composite.attributes")));
    PrecisionConfig precision{};
    precision.mutable_operand_precision()->Resize(2, PrecisionConfig::DEFAULT);
    auto* scaled_dot =
        computation->AddInstruction(HloInstruction::CreateScaledDot(
            call->shape(), call->mutable_operand(0), call->mutable_operand(1),
            call->mutable_operand(2), call->mutable_operand(3),
            dot_dimension_numbers, precision));
    TF_RETURN_IF_ERROR(call->ReplaceAllUsesWith(scaled_dot));
    TF_RETURN_IF_ERROR(computation->RemoveInstruction(call));
    changed = true;
  }
  return changed;
}

absl::StatusOr<bool> CompositeRewriter::RunImpl(
    HloModule* module, const absl::flat_hash_set<absl::string_view>&) {
  bool changed = false;
  for (HloComputation* computation : module->computations()) {
    TF_ASSIGN_OR_RETURN(bool result, RewriteComputation(computation));
    changed |= result;
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
