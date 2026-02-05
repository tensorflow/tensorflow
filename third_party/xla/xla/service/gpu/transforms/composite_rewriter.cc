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
#include "xla/literal.h"
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
  if (!dict_attrs.contains("dimension_numbers")) {
    return absl::InvalidArgumentError(
        "dimension_numbers are not set in composite attributes");
  }

  mlir::ArrayAttr dim_numbers =
      mlir::dyn_cast<mlir::ArrayAttr>(dict_attrs.get("dimension_numbers"));
  if (!dim_numbers || dim_numbers.size() != 2) {
    return absl::InvalidArgumentError(
        "dimension_numbers must be array of size 2");
  }

  mlir::ArrayAttr contracting = mlir::dyn_cast<mlir::ArrayAttr>(dim_numbers[0]);
  mlir::ArrayAttr batch = mlir::dyn_cast<mlir::ArrayAttr>(dim_numbers[1]);
  if (!contracting || contracting.size() != 2 || !batch || batch.size() != 2) {
    return absl::InvalidArgumentError(
        "invalid contracting or batch dimensions");
  }

  mlir::ArrayAttr lhs_contracting =
      mlir::dyn_cast<mlir::ArrayAttr>(contracting[0]);
  mlir::ArrayAttr rhs_contracting =
      mlir::dyn_cast<mlir::ArrayAttr>(contracting[1]);
  mlir::ArrayAttr lhs_batch = mlir::dyn_cast<mlir::ArrayAttr>(batch[0]);
  mlir::ArrayAttr rhs_batch = mlir::dyn_cast<mlir::ArrayAttr>(batch[1]);

  if (!lhs_contracting || !rhs_contracting || !lhs_batch || !rhs_batch) {
    return absl::InvalidArgumentError("Invalid dimension_numbers structure");
  }

  DotDimensionNumbers dnums;
  for (mlir::Attribute dim : lhs_contracting) {
    dnums.add_lhs_contracting_dimensions(
        mlir::cast<mlir::IntegerAttr>(dim).getInt());
  }
  for (mlir::Attribute dim : rhs_contracting) {
    dnums.add_rhs_contracting_dimensions(
        mlir::cast<mlir::IntegerAttr>(dim).getInt());
  }
  for (mlir::Attribute dim : lhs_batch) {
    dnums.add_lhs_batch_dimensions(mlir::cast<mlir::IntegerAttr>(dim).getInt());
  }
  for (mlir::Attribute dim : rhs_batch) {
    dnums.add_rhs_batch_dimensions(mlir::cast<mlir::IntegerAttr>(dim).getInt());
  }
  return dnums;
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

    if (dot_dimension_numbers.lhs_contracting_dimensions_size() != 1 ||
        dot_dimension_numbers.rhs_contracting_dimensions_size() != 1 ||
        dot_dimension_numbers.lhs_batch_dimensions_size() > 1 ||
        dot_dimension_numbers.rhs_batch_dimensions_size() > 1) {
      LOG(ERROR) << "Unsupported dimension numbers: "
                 << dot_dimension_numbers.DebugString();
      continue;
    }

    const HloInstruction* lhs = call->operand(0);
    const HloInstruction* rhs = call->operand(1);
    const HloInstruction* lhs_scale = call->operand(2);
    const HloInstruction* rhs_scale = call->operand(3);

    int64_t lhs_contracting_dim =
        dot_dimension_numbers.lhs_contracting_dimensions(0);
    int64_t rhs_contracting_dim =
        dot_dimension_numbers.rhs_contracting_dimensions(0);

    auto is_supported = [&](const HloInstruction* operand,
                            const HloInstruction* scale,
                            int64_t contracting_dim) {
      auto op_type = operand->shape().element_type();
      auto scale_type = scale->shape().element_type();
      if ((op_type == F8E4M3FN || op_type == F8E5M2 || op_type == F4E2M1FN) &&
          scale_type == F8E8M0FNU) {
        if (contracting_dim >= scale->shape().dimensions().size()) {
          return false;
        }
        int64_t operand_dim_size = operand->shape().dimensions(contracting_dim);
        int64_t scale_dim_size = scale->shape().dimensions(contracting_dim);

        if (scale_dim_size == 0 || operand_dim_size % scale_dim_size != 0) {
          return false;
        }
        int64_t scale_factor = operand_dim_size / scale_dim_size;
        return scale_factor % 32 == 0;
      }
      if (op_type == BF16 && scale_type == BF16) {
        if (scale->shape().dimensions().size() !=
            operand->shape().dimensions().size()) {
          return false;
        }
        for (int64_t dim : scale->shape().dimensions()) {
          if (dim != 1) {
            return false;
          }
        }
        if (scale->opcode() != HloOpcode::kConstant) {
          return false;
        }
        return scale->literal().IsAllFloat(1.0);
      }
      return false;
    };

    if (!is_supported(lhs, lhs_scale, lhs_contracting_dim) ||
        !is_supported(rhs, rhs_scale, rhs_contracting_dim)) {
      continue;
    }

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
