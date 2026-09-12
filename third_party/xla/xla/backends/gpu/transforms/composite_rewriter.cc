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
    VLOG(3) << "Found call instruction: " << instruction->name();
    auto call = Cast<HloCallInstruction>(instruction);
    if (!call->is_composite()) {
      VLOG(3) << instruction->name() << " is not composite";
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
    ABSL_ASSIGN_OR_RETURN(
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

    auto is_supported = [&]() {
      auto lhs_type = lhs->shape().element_type();
      auto rhs_type = rhs->shape().element_type();
      auto lhs_scale_type = lhs_scale->shape().element_type();
      auto rhs_scale_type = rhs_scale->shape().element_type();

      if (lhs_type == BF16 && rhs_type == BF16) {
        if (lhs_scale_type != BF16 || rhs_scale_type != BF16) {
          VLOG(3) << "BF16 operands require BF16 scales";
          return false;
        }
        auto is_valid_bf16_scale = [](const HloInstruction* operand,
                                      const HloInstruction* scale) {
          if (scale->shape().dimensions().size() !=
              operand->shape().dimensions().size()) {
            VLOG(3) << "scale and operand rank mismatch for BF16";
            return false;
          }
          for (int64_t dim : scale->shape().dimensions()) {
            if (dim != 1) {
              VLOG(3) << "scale dim != 1 for BF16";
              return false;
            }
          }
          if (scale->opcode() != HloOpcode::kConstant) {
            VLOG(3) << "scale is not constant for BF16";
            return false;
          }
          bool supported = scale->literal().IsAllFloat(1.0);
          if (!supported) {
            VLOG(3) << "scale is not 1.0 for BF16";
          }
          return supported;
        };
        return is_valid_bf16_scale(lhs, lhs_scale) &&
               is_valid_bf16_scale(rhs, rhs_scale);
      }

      auto is_supported_operand_type = [](PrimitiveType type) {
        return type == F8E4M3FN || type == F8E5M2 || type == F4E2M1FN;
      };

      if (!is_supported_operand_type(lhs_type) ||
          !is_supported_operand_type(rhs_type)) {
        VLOG(3) << "Unsupported operand types: LHS="
                << PrimitiveType_Name(lhs_type)
                << ", RHS=" << PrimitiveType_Name(rhs_type);
        return false;
      }

      if (lhs_scale_type != rhs_scale_type) {
        VLOG(3) << "Scale type mismatch: LHS="
                << PrimitiveType_Name(lhs_scale_type)
                << ", RHS=" << PrimitiveType_Name(rhs_scale_type);
        return false;
      }

      if (lhs_scale_type != F8E8M0FNU && lhs_scale_type != F8E4M3FN) {
        VLOG(3) << "Unsupported scale type: "
                << PrimitiveType_Name(lhs_scale_type);
        return false;
      }

      if (lhs_contracting_dim >= lhs_scale->shape().dimensions().size() ||
          rhs_contracting_dim >= rhs_scale->shape().dimensions().size()) {
        VLOG(3) << "contracting_dim out of bounds for scale";
        return false;
      }

      int64_t lhs_operand_dim_size =
          lhs->shape().dimensions(lhs_contracting_dim);
      int64_t lhs_scale_dim_size =
          lhs_scale->shape().dimensions(lhs_contracting_dim);
      if (lhs_scale_dim_size == 0 ||
          lhs_operand_dim_size % lhs_scale_dim_size != 0) {
        VLOG(3) << "LHS operand dim not divisible by scale dim";
        return false;
      }
      int64_t lhs_scale_factor = lhs_operand_dim_size / lhs_scale_dim_size;

      int64_t rhs_operand_dim_size =
          rhs->shape().dimensions(rhs_contracting_dim);
      int64_t rhs_scale_dim_size =
          rhs_scale->shape().dimensions(rhs_contracting_dim);
      if (rhs_scale_dim_size == 0 ||
          rhs_operand_dim_size % rhs_scale_dim_size != 0) {
        VLOG(3) << "RHS operand dim not divisible by scale dim";
        return false;
      }
      int64_t rhs_scale_factor = rhs_operand_dim_size / rhs_scale_dim_size;

      if (lhs_scale_factor != rhs_scale_factor) {
        VLOG(3) << "Scale factor mismatch: LHS=" << lhs_scale_factor
                << ", RHS=" << rhs_scale_factor;
        return false;
      }

      int64_t scale_factor = lhs_scale_factor;
      if (lhs_scale_type == F8E8M0FNU) {
        if (scale_factor != 16 && scale_factor != 32) {
          VLOG(3) << "E8M0 scale_factor must be 16 or 32: " << scale_factor;
          return false;
        }
      } else if (lhs_scale_type == F8E4M3FN) {
        if (scale_factor != 16) {
          VLOG(3) << "E4M3 scale_factor must be 16: " << scale_factor;
          return false;
        }
      }

      return true;
    };

    if (!is_supported()) {
      VLOG(3) << "Scaled dot composite operands not supported";
      continue;
    }

    PrecisionConfig precision{};
    precision.mutable_operand_precision()->Resize(2, PrecisionConfig::DEFAULT);
    auto* scaled_dot =
        computation->AddInstruction(HloInstruction::CreateScaledDot(
            call->shape(), call->mutable_operand(0), call->mutable_operand(1),
            call->mutable_operand(2), call->mutable_operand(3),
            dot_dimension_numbers, precision));
    ABSL_RETURN_IF_ERROR(call->ReplaceAllUsesWith(scaled_dot));
    ABSL_RETURN_IF_ERROR(computation->RemoveInstruction(call));
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
