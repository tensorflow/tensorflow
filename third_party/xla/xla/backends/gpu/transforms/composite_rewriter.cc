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

#include "xla/backends/gpu/transforms/composite_rewriter.h"

#include <cstdint>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
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
#include "xla/literal.h"
#include "xla/service/decision.h"
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

Decision IsScaledDotComposite(const HloInstruction* instruction) {
  if (instruction->opcode() != HloOpcode::kCall) {
    return Decision::Forbid("instruction is not a call");
  }
  auto call = Cast<HloCallInstruction>(instruction);
  if (!call->is_composite()) {
    return Decision::Forbid(
        absl::StrCat(instruction->name(), " is not composite"));
  }
  if (!call->has_frontend_attributes()) {
    return Decision::Forbid("No frontend attributes");
  }
  const auto& frontend_attrs = call->frontend_attributes().map();
  auto it = frontend_attrs.find("composite.name");
  if (it == frontend_attrs.end() || it->second != "xla.scaled_dot") {
    return Decision::Forbid(
        it != frontend_attrs.end()
            ? absl::StrCat("composite.name is not xla.scaled_dot: ", it->second)
            : "composite.name attribute is missing");
  }
  return Decision::Allow();
}

Decision HasSupportedDimensionNumbers(
    const DotDimensionNumbers& dot_dimension_numbers) {
  if (dot_dimension_numbers.lhs_contracting_dimensions_size() != 1 ||
      dot_dimension_numbers.rhs_contracting_dimensions_size() != 1) {
    return Decision::Forbid(
        absl::StrCat("Expected 1 contracting dimension on LHS and RHS, got: ",
                     dot_dimension_numbers.DebugString()));
  }
  if (dot_dimension_numbers.lhs_batch_dimensions_size() > 1 ||
      dot_dimension_numbers.rhs_batch_dimensions_size() > 1) {
    return Decision::Forbid(
        absl::StrCat("Expected at most 1 batch dimension on LHS and RHS, got: ",
                     dot_dimension_numbers.DebugString()));
  }
  return Decision::Allow();
}

Decision IsValidBf16Scale(const HloInstruction* operand,
                          const HloInstruction* scale) {
  if (scale->shape().dimensions().size() !=
      operand->shape().dimensions().size()) {
    return Decision::Forbid("scale and operand rank mismatch for BF16");
  }
  for (int64_t dim : scale->shape().dimensions()) {
    if (dim != 1) {
      return Decision::Forbid("scale dim != 1 for BF16");
    }
  }
  if (scale->opcode() != HloOpcode::kConstant) {
    return Decision::Forbid("scale is not constant for BF16");
  }
  if (!scale->literal().IsAllFloat(1.0)) {
    return Decision::Forbid("scale is not 1.0 for BF16");
  }
  return Decision::Allow();
}

Decision IsSupportedBf16(const HloInstruction* lhs, const HloInstruction* rhs,
                         const HloInstruction* lhs_scale,
                         const HloInstruction* rhs_scale) {
  if (lhs_scale->shape().element_type() != BF16 ||
      rhs_scale->shape().element_type() != BF16) {
    return Decision::Forbid("BF16 operands require BF16 scales");
  }
  if (Decision d = IsValidBf16Scale(lhs, lhs_scale); !d) {
    return d;
  }
  return IsValidBf16Scale(rhs, rhs_scale);
}

Decision IsSupportedScaleAndOperand(const HloInstruction* operand,
                                    const HloInstruction* scale,
                                    int64_t contracting_dim) {
  PrimitiveType op_type = operand->shape().element_type();
  PrimitiveType scale_type = scale->shape().element_type();

  if (scale_type != F8E8M0FNU) {
    return Decision::Forbid(absl::StrCat("Unsupported scale type: ",
                                         PrimitiveType_Name(scale_type)));
  }

  if (contracting_dim >= scale->shape().dimensions().size()) {
    return Decision::Forbid("contracting_dim out of bounds for scale");
  }
  int64_t operand_dim_size = operand->shape().dimensions(contracting_dim);
  int64_t scale_dim_size = scale->shape().dimensions(contracting_dim);

  if (scale_dim_size == 0 || operand_dim_size % scale_dim_size != 0) {
    return Decision::Forbid("operand_dim_size not divisible by scale_dim_size");
  }
  int64_t scale_factor = operand_dim_size / scale_dim_size;

  if (op_type == F8E4M3FN || op_type == F8E5M2 || op_type == F4E2M1FN) {
    if (scale_factor % 16 != 0) {
      return Decision::Forbid(
          absl::StrCat("scale_factor % 16 != 0: ", scale_factor));
    }
    return Decision::Allow();
  }

  return Decision::Forbid(
      absl::StrCat("Unsupported operand type: ", PrimitiveType_Name(op_type)));
}

Decision IsSupportedScaledDot(
    const HloCallInstruction* call,
    const DotDimensionNumbers& dot_dimension_numbers) {
  if (call->operand_count() < 4) {
    return Decision::Forbid("Scaled dot call has fewer than 4 operands");
  }
  const HloInstruction* lhs = call->operand(0);
  const HloInstruction* rhs = call->operand(1);
  const HloInstruction* lhs_scale = call->operand(2);
  const HloInstruction* rhs_scale = call->operand(3);

  if (lhs->shape().element_type() == BF16 &&
      rhs->shape().element_type() == BF16) {
    return IsSupportedBf16(lhs, rhs, lhs_scale, rhs_scale);
  }

  int64_t lhs_contracting_dim =
      dot_dimension_numbers.lhs_contracting_dimensions(0);
  int64_t rhs_contracting_dim =
      dot_dimension_numbers.rhs_contracting_dimensions(0);

  if (Decision d =
          IsSupportedScaleAndOperand(lhs, lhs_scale, lhs_contracting_dim);
      !d) {
    return d;
  }
  return IsSupportedScaleAndOperand(rhs, rhs_scale, rhs_contracting_dim);
}

absl::Status RewriteScaledDot(
    HloComputation* computation, HloCallInstruction* call,
    const DotDimensionNumbers& dot_dimension_numbers) {
  PrecisionConfig precision{};
  precision.mutable_operand_precision()->Resize(2, PrecisionConfig::DEFAULT);
  auto* scaled_dot =
      computation->AddInstruction(HloInstruction::CreateScaledDot(
          call->shape(), call->mutable_operand(0), call->mutable_operand(1),
          call->mutable_operand(2), call->mutable_operand(3),
          dot_dimension_numbers, precision));
  ABSL_RETURN_IF_ERROR(call->ReplaceAllUsesWith(scaled_dot));
  ABSL_RETURN_IF_ERROR(computation->RemoveInstruction(call));
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<bool> CompositeRewriter::RewriteComputation(
    HloComputation* computation) {
  bool changed = false;
  for (HloInstruction* instruction : computation->MakeInstructionPostOrder()) {
    if (Decision d = IsScaledDotComposite(instruction); !d) {
      VLOG(3) << d.Explain();
      continue;
    }
    auto* call = Cast<HloCallInstruction>(instruction);
    const auto& frontend_attrs = call->frontend_attributes().map();
    auto it = frontend_attrs.find("composite.attributes");
    if (it == frontend_attrs.end()) {
      return absl::InvalidArgumentError(
          "composite.attributes is not set for xla.scaled_dot");
    }
    ABSL_ASSIGN_OR_RETURN(DotDimensionNumbers dot_dimension_numbers,
                     ParseDimensionNumbers(it->second));

    if (Decision d = HasSupportedDimensionNumbers(dot_dimension_numbers); !d) {
      LOG(ERROR) << d.Explain();
      continue;
    }

    if (Decision d = IsSupportedScaledDot(call, dot_dimension_numbers); !d) {
      VLOG(3) << "Scaled dot composite operands not supported: " << d.Explain();
      continue;
    }

    ABSL_RETURN_IF_ERROR(RewriteScaledDot(computation, call, dot_dimension_numbers));
    changed = true;
  }
  return changed;
}

absl::StatusOr<bool> CompositeRewriter::RunImpl(
    HloModule* module, const absl::flat_hash_set<absl::string_view>&) {
  bool changed = false;
  for (HloComputation* computation : module->computations()) {
    ABSL_ASSIGN_OR_RETURN(bool result, RewriteComputation(computation));
    changed |= result;
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
