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

#include "xla/codegen/xtile/codegen/dot_algorithms.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/codegen/xtile/codegen/emitter_helpers.h"
#include "xla/codegen/xtile/ir/xtile_ops.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/translate/hlo_to_mhlo/attribute_importer.h"
#include "xla/service/algorithm_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace xtile {

namespace {

using ::mlir::ShapedType;
using ::mlir::Type;
using ::mlir::Value;

Type ElementType(Value v) { return mlir::getElementTypeOrSelf(v); }

mlir::stablehlo::Precision XlaPrecisionToStableHloPrecision(
    PrecisionConfig::Precision precision) {
  switch (precision) {
    case PrecisionConfig::DEFAULT:
      return mlir::stablehlo::Precision::DEFAULT;
    case PrecisionConfig::HIGH:
      return mlir::stablehlo::Precision::HIGH;
    case PrecisionConfig::HIGHEST:
      return mlir::stablehlo::Precision::HIGHEST;
    default:
      LOG(FATAL) << "Unsupported precision: " << precision;
  }
}

}  // namespace

namespace {

absl::StatusOr<Value> ScaledDot(mlir::ImplicitLocOpBuilder& b,
                                ScaledDotOperands& operands) {
  mlir::Type lhs_dot_elem_type = getElementTypeOrSelf(operands.lhs.getType());
  mlir::Type rhs_dot_elem_type = getElementTypeOrSelf(operands.rhs.getType());

  Value lhs_scale;
  if (lhs_dot_elem_type != b.getBF16Type()) {
    lhs_scale = Bitcast(b, operands.lhs_scale, b.getI8Type());
  }
  Value rhs_scale;
  if (rhs_dot_elem_type != b.getBF16Type()) {
    rhs_scale = Bitcast(b, operands.rhs_scale, b.getI8Type());
    rhs_scale = mlir::stablehlo::TransposeOp::create(
        b, rhs_scale, b.getDenseI64ArrayAttr({1, 0}));
  }

  // When operand type is subbyte size then it is packed along minor dim and for
  // RHS minor dim is not K.
  const auto& lhs_shaped_type =
      mlir::dyn_cast<ShapedType>(operands.lhs.getType());
  const bool rhs_k_pack = lhs_shaped_type.getElementType() !=
                          mlir::Float4E2M1FNType::get(b.getContext());
  auto dot_scaled_op = xtile::DotScaledOp::create(
      b, operands.accumulator.getType(), operands.lhs, operands.rhs, lhs_scale,
      rhs_scale, /*fastMath=*/true, /*lhs_k_pack=*/true, rhs_k_pack);

  auto add_result =
      mlir::isa<mlir::IntegerType>(
          dot_scaled_op.getResult().getType().getElementType())
          ? mlir::arith::AddIOp::create(b, operands.accumulator, dot_scaled_op)
          : mlir::arith::AddFOp::create(b, operands.accumulator, dot_scaled_op);
  return add_result->getResult(0);
}

namespace {

Value EmitStableHloDotAndAdd(mlir::ImplicitLocOpBuilder& b, Value lhs,
                             Value rhs, Value acc,
                             PrecisionSpec precision_spec) {
  auto lhs_type = mlir::cast<ShapedType>(lhs.getType());
  auto rhs_type = mlir::cast<ShapedType>(rhs.getType());

  CHECK(lhs_type.getRank() <= 2 && rhs_type.getRank() <= 2)
      << "Unsupported ranks. LHS rank: " << lhs_type.getRank()
      << " RHS rank: " << rhs_type.getRank();

  llvm::SmallVector<int64_t> array_attr{0};
  auto dot_dimension_numbers = mlir::stablehlo::DotDimensionNumbersAttr::get(
      b.getContext(), /*lhsBatchingDimensions=*/{},
      /*rhsBatchingDimensions=*/{},
      /*lhsContractingDimensions=*/
      {lhs_type.getRank() - 1},
      /*rhsContractingDimensions=*/
      {0});

  auto precision_config = mlir::stablehlo::PrecisionConfigAttr::get(
      b.getContext(), {precision_spec.lhs_operand_precision,
                       precision_spec.rhs_operand_precision});
  auto dot = mlir::stablehlo::DotGeneralOp::create(
      b, acc.getType(), lhs, rhs, dot_dimension_numbers,
      /*precision_config=*/precision_config,
      /*algorithm=*/
      stablehlo::ConvertDotAlgorithm(precision_spec.algorithm, &b));

  auto add_result =
      mlir::isa<mlir::IntegerType>(dot.getResult().getType().getElementType())
          ? mlir::arith::AddIOp::create(b, acc, dot)
          : mlir::arith::AddFOp::create(b, acc, dot);
  return add_result->getResult(0);
}

}  // namespace

absl::StatusOr<Type> GetAlgUnsetAccumulatorType(mlir::ImplicitLocOpBuilder& b,
                                                const HloDotInstruction& dot) {
  TF_ASSIGN_OR_RETURN(
      Type lhs_type,
      PrimitiveTypeToMlirType(b, dot.operand(0)->shape().element_type()));
  TF_ASSIGN_OR_RETURN(
      Type rhs_type,
      PrimitiveTypeToMlirType(b, dot.operand(1)->shape().element_type()));
  TF_ASSIGN_OR_RETURN(Type accumulator_type,
                      PrimitiveTypeToMlirType(b, dot.shape().element_type()));

  // The code below assumes that lhs and rhs have the same type. However
  // this may not always be the case with f8 matmuls, e.g. e4m3×e5m2 is
  // supported at the hardware level. NVIDIA GPUs currently only support f32
  // accumulators for such matmuls.
  if (lhs_type.isFloat(8) && rhs_type.isFloat(8)) {
    return b.getF32Type();
  }

  CHECK(lhs_type == rhs_type);

  // Currently allowing 8x8-bit ints -> i32.
  if (lhs_type == b.getIntegerType(8) && accumulator_type.isInteger(32)) {
    return b.getI32Type();
  }
  return (accumulator_type.isF64() && lhs_type.isF64()) ? b.getF64Type()
                                                        : b.getF32Type();
}

// Returns the `Type` that the dot operands should be casted to if there is a
// clear candidate. Raises an error if there are multiple allowed choices but
// the operands do not already conform to any of them. Returns `std::nullopt` if
// no casting is a priori needed.
absl::StatusOr<std::optional<Type>> GetForceOperandsType(
    mlir::ImplicitLocOpBuilder& b, const HloDotInstruction& dot,
    const DotOperands& dot_operands) {
  PrecisionConfig::Algorithm algorithm = dot.precision_config().algorithm();
  if (algorithm == PrecisionConfig::ALG_UNSET) {
    return std::nullopt;
  }

  TF_ASSIGN_OR_RETURN(
      std::vector<PrimitiveType> allowed_operands_primitive_types,
      algorithm_util::GetAllowedOperandsTypeForAlgorithm(algorithm));
  CHECK(!allowed_operands_primitive_types.empty());

  std::vector<Type> allowed_operands_types;
  allowed_operands_types.reserve(allowed_operands_primitive_types.size());
  for (PrimitiveType primitive_type : allowed_operands_primitive_types) {
    TF_ASSIGN_OR_RETURN(Type type, PrimitiveTypeToMlirType(b, primitive_type));
    allowed_operands_types.push_back(type);
  }

  Type lhs_type = ElementType(dot_operands.lhs);
  Type rhs_type = ElementType(dot_operands.rhs);
  if (allowed_operands_types.size() == 1) {
    // If there is a single allowed operand type, we force the operands to use
    // this type.
    return allowed_operands_types.front();
  }  // If there are several allowed operand types, we just check that the
  // operands have the same type, and that this type is one of the allowed
  // ones. Raise an error otherwise.
  if (lhs_type != rhs_type ||
      !absl::c_linear_search(allowed_operands_types, lhs_type)) {
    std::string allowed_operands_types_str = absl::StrJoin(
        allowed_operands_types, ", ", [&](std::string* out, Type type) {
          absl::StrAppend(out, MlirToString(type));
        });
    return absl::FailedPreconditionError(absl::StrCat(
        "Expected dot operands to both have the same type, and for this type "
        "to be one of the following types: ",
        allowed_operands_types_str, " but got ", MlirToString(lhs_type),
        " and ", MlirToString(rhs_type)));
  }

  return std::nullopt;
}

}  // namespace

absl::StatusOr<Type> GetDotAccumulatorType(mlir::ImplicitLocOpBuilder& b,
                                           const HloDotInstruction& dot) {
  const PrecisionConfig::Algorithm algorithm =
      dot.precision_config().algorithm();

  if (algorithm == PrecisionConfig::ALG_UNSET) {
    return GetAlgUnsetAccumulatorType(b, dot);
  }

  TF_ASSIGN_OR_RETURN(PrimitiveType accumulator_type,
                      algorithm_util::GetDotAccumulatorType(algorithm));
  return PrimitiveTypeToMlirType(b, accumulator_type);
}

absl::StatusOr<Value> EmitSingleTileDot(mlir::ImplicitLocOpBuilder& b,
                                        const HloDotInstruction& dot,
                                        DotOperands dot_operands) {
  PrecisionConfig::Algorithm algorithm = dot.precision_config().algorithm();
  PrecisionSpec precision_spec{
      algorithm,
      XlaPrecisionToStableHloPrecision(
          dot.precision_config().operand_precision(0)),
      XlaPrecisionToStableHloPrecision(
          dot.precision_config().operand_precision(1))};

  TF_ASSIGN_OR_RETURN(std::optional<Type> force_operands_type,
                      GetForceOperandsType(b, dot, dot_operands));

  TF_ASSIGN_OR_RETURN(Type force_accumulator_type,
                      GetDotAccumulatorType(b, dot));

  if (force_operands_type.has_value()) {
    if (ElementType(dot_operands.lhs) != *force_operands_type) {
      dot_operands.lhs = Cast(b, dot_operands.lhs, *force_operands_type);
    }

    if (ElementType(dot_operands.rhs) != *force_operands_type) {
      dot_operands.rhs = Cast(b, dot_operands.rhs, *force_operands_type);
    }
  }

  if (ElementType(dot_operands.accumulator) != force_accumulator_type) {
    dot_operands.accumulator =
        Cast(b, dot_operands.accumulator, force_accumulator_type);
  }

  Value result =
      EmitStableHloDotAndAdd(b, dot_operands.lhs, dot_operands.rhs,
                             dot_operands.accumulator, precision_spec);

  // TODO(b/393299275): once we've moved on from the legacy emitter, we should
  // make sure that this accumulator type is equal to the one derived here.
  Type outer_accumulator_type = ElementType(dot_operands.accumulator);
  if (ElementType(result) != outer_accumulator_type) {
    result = Cast(b, result, outer_accumulator_type);
  }

  return result;
}

absl::StatusOr<Value> EmitSingleTileScaledDot(
    mlir::ImplicitLocOpBuilder& b, const HloScaledDotInstruction& scaled_dot,
    ScaledDotOperands dot_operands) {
  return ScaledDot(b, dot_operands);
}

}  // namespace xtile
}  // namespace xla
