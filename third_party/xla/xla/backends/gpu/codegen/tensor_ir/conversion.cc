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

#include "xla/backends/gpu/codegen/tensor_ir/conversion.h"

#include <cstdint>
#include <numeric>
#include <optional>
#include <utility>

#include "tensor_ir/Dialect/TensorIR.h"
#include "tensor_ir/Dialect/TensorIRAttrs.h"
#include "absl/algorithm/container.h"
#include "absl/cleanup/cleanup.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

// NOLINTBEGIN(clang-diagnostic-pre-c++20-compat)

namespace xla::gpu::tensor_ir {
namespace {

namespace arith = ::mlir::arith;
namespace tir = ::mlir::nv_tensor_ir;

absl::StatusOr<mlir::Value> ConvertReductionInstruction(
    const HloInstruction& source, mlir::ValueRange operands,
    mlir::Block& target);

// Creates MLIR location from HLO instruction metadata.
mlir::Location GetLocationFromInstruction(const HloInstruction& source,
                                          mlir::MLIRContext* context) {
  const OpMetadata& metadata = source.metadata();

  llvm::SmallVector<mlir::Location> locations;
  if (!metadata.op_name().empty()) {
    locations.push_back(
        mlir::NameLoc::get(mlir::StringAttr::get(context, metadata.op_name())));
  }
  if (!metadata.source_file().empty()) {
    locations.push_back(mlir::FileLineColRange::get(
        context, metadata.source_file(), metadata.source_line(),
        metadata.source_column(), metadata.source_end_line(),
        metadata.source_end_column()));
  }

  if (locations.empty()) {
    return mlir::UnknownLoc::get(context);
  }
  if (locations.size() == 1) {
    return locations.front();
  }
  return mlir::FusedLoc::get(context, locations);
}

// Creates MLIR type from HLO primitive type.
absl::StatusOr<mlir::Type> GetElementType(PrimitiveType type,
                                          mlir::Builder& builder) {
  switch (type) {
    case PrimitiveType::PRED:
      return builder.getI1Type();
    case PrimitiveType::S8:
    case PrimitiveType::S16:
    case PrimitiveType::S32:
    case PrimitiveType::S64:
      return builder.getIntegerType(primitive_util::BitWidth(type),
                                    /*isSigned=*/true);
    case PrimitiveType::U8:
    case PrimitiveType::U16:
    case PrimitiveType::U32:
    case PrimitiveType::U64:
      return builder.getIntegerType(primitive_util::BitWidth(type),
                                    /*isSigned=*/false);
    case PrimitiveType::F16:
      return builder.getF16Type();
    case PrimitiveType::BF16:
      return builder.getBF16Type();
    case PrimitiveType::F32:
      return builder.getF32Type();
    case PrimitiveType::F64:
      return builder.getF64Type();
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported primitive type: ", type));
  }
}

// Converts a scalar literal to MLIR attribute.
absl::StatusOr<mlir::TypedAttr> GetScalarAttribute(
    const LiteralBase& literal, mlir::Builder& builder,
    bool use_signless_integer_type) {
  ABSL_ASSIGN_OR_RETURN(auto element_type,
                   GetElementType(literal.shape().element_type(), builder));
  if (element_type.isFloat()) {
    if (auto cst = literal.GetAsDouble({})) {
      return mlir::FloatAttr::get(element_type, *cst);
    }
  } else {
    if (auto cst = literal.GetIntegralAsS64({})) {
      if (use_signless_integer_type) {
        element_type =
            builder.getIntegerType(element_type.getIntOrFloatBitWidth());
      }
      return mlir::IntegerAttr::get(element_type, *cst);
    }
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Unsupported scalar literal: ", literal.ToString()));
}

// Creates MLIR tensor type from HLO shape.
absl::StatusOr<mlir::RankedTensorType> GetTensorType(const Shape& shape,
                                                     mlir::Builder& builder) {
  ABSL_ASSIGN_OR_RETURN(auto element_type,
                   GetElementType(shape.element_type(), builder));
  // Promote 0-D scalar shapes to 1-D tensors of size 1 for TensorIR.
  if (shape.dimensions().empty()) {
    return mlir::RankedTensorType::get({1}, element_type);
  }
  return mlir::RankedTensorType::get(shape.dimensions(), element_type);
}

// Builds a constant operation with given value and shape.
absl::StatusOr<mlir::Value> BuildConstant(const LiteralBase& literal,
                                          const Shape& shape,
                                          mlir::ImplicitLocOpBuilder& builder) {
  ABSL_ASSIGN_OR_RETURN(auto tensor_type, GetTensorType(shape, builder));
  auto dense_attr = primitive_util::ArrayTypeSwitch(
      [&](auto type) -> mlir::DenseElementsAttr {
        using NativeT = primitive_util::NativeTypeOf<type>;
        return mlir::DenseElementsAttr::get(tensor_type,
                                            literal.GetFirstElement<NativeT>());
      },
      literal.shape().element_type());
  return tir::ConstantOp::create(builder, dense_attr);
}

absl::StatusOr<mlir::Value> BuildFloatConstant(
    double value, const Shape& shape, mlir::ImplicitLocOpBuilder& builder) {
  return BuildConstant(LiteralUtil::CreateR0(shape.element_type(), value),
                       shape, builder);
}

// Builds a reshape operation from HLO bitcast instruction.
absl::StatusOr<mlir::Value> BuildBitcast(const HloInstruction& source,
                                         mlir::Value operand,
                                         mlir::ImplicitLocOpBuilder& builder) {
  mlir::Value result = operand;

  // Calculate the operand/result shapes with default layout.
  Shape operand_normal_shape =
      ShapeUtil::MakeShapeWithDescendingLayoutAndSamePhysicalLayout(
          source.operand(0)->shape());
  Shape result_normal_shape =
      ShapeUtil::MakeShapeWithDescendingLayoutAndSamePhysicalLayout(
          source.shape());

  // If the operand doesn't have a default layout, transpose it.
  // Example: [A,B,C]{2,0,1} -> [B,A,C]{2,1,0} uses permutation (1,0,2)
  if (source.operand(0)->shape() != operand_normal_shape) {
    auto permutation = llvm::to_vector(
        llvm::reverse(source.operand(0)->shape().layout().minor_to_major()));
    ABSL_ASSIGN_OR_RETURN(auto transpose_type,
                     GetTensorType(operand_normal_shape, builder));
    VLOG(3) << "Transposing operand: "
            << llvm_ir::DumpToString(result.getType()) << " to "
            << llvm_ir::DumpToString(transpose_type);
    result =
        tir::TransposeOp::create(builder, transpose_type, result, permutation);
  }

  // If the normalized shapes are not equal, create a reshape operation.
  if (operand_normal_shape != result_normal_shape) {
    ABSL_ASSIGN_OR_RETURN(auto reshape_type,
                     GetTensorType(result_normal_shape, builder));
    VLOG(3) << "Reshaping operand: " << llvm_ir::DumpToString(result.getType())
            << " to " << llvm_ir::DumpToString(reshape_type);
    result = tir::ReshapeOp::create(builder, reshape_type, result);
  }

  // If the result doesn't have a default layout, transpose it.
  // Example: [A,B,C]{2,1,0} -> [C,A,B]{0,2,1} uses permutation (2,0,1)
  if (source.shape() != result_normal_shape) {
    llvm::SmallVector<int64_t> permutation(source.shape().dimensions().size());
    for (auto [idx, pos] : llvm::enumerate(
             llvm::reverse(source.shape().layout().minor_to_major()))) {
      permutation[pos] = idx;
    }
    ABSL_ASSIGN_OR_RETURN(auto transpose_type,
                     GetTensorType(source.shape(), builder));
    VLOG(3) << "Transposing operand: "
            << llvm_ir::DumpToString(result.getType()) << " to "
            << llvm_ir::DumpToString(transpose_type);
    result =
        tir::TransposeOp::create(builder, transpose_type, result, permutation);
  }

  return result;
}

// Builds a broadcast operation from HLO broadcast instruction.
absl::StatusOr<mlir::Value> BuildBroadcast(
    const HloBroadcastInstruction& source, mlir::Value operand,
    mlir::ImplicitLocOpBuilder& builder) {
  auto dimensions = source.dimensions();

  // If the operand is a scalar constant, return shaped constant.
  if (const auto* constant = DynCast<HloConstantInstruction>(source.operand(0));
      constant != nullptr && constant->shape().dimensions().empty()) {
    VLOG(3) << "Broadcasting scalar constant: "
            << constant->literal().ToString() << " to " << source.shape();
    return BuildConstant(constant->literal(), source.shape(), builder);
  }

  // If the source dimensions are not sorted, a transpose is needed.
  // https://openxla.org/stablehlo/spec#broadcast_in_dim
  if (!absl::c_is_sorted(dimensions)) {
    // Get the transpose permutation by sorting the dimensions.
    llvm::SmallVector<int64_t> permutation(dimensions.size());
    std::iota(permutation.begin(), permutation.end(), 0);
    absl::c_sort(permutation, [&](int64_t a, int64_t b) {
      return dimensions[a] < dimensions[b];
    });

    // Create the transpose operation.
    llvm::SmallVector<int64_t> permuted_dims;
    permuted_dims.reserve(dimensions.size());
    for (int64_t i : permutation) {
      permuted_dims.push_back(source.operand(0)->shape().dimensions(i));
    }
    mlir::ShapedType transpose_type =
        llvm::cast<mlir::ShapedType>(operand.getType()).clone(permuted_dims);
    VLOG(3) << "Transposing operand: "
            << llvm_ir::DumpToString(operand.getType()) << " to "
            << llvm_ir::DumpToString(transpose_type);
    operand =
        tir::TransposeOp::create(builder, transpose_type, operand, permutation);
  }

  // Reshape operand, add unit dimensions for the broadcast.
  llvm::SmallVector<int64_t> pre_broadcast_dims;
  for (int i = 0, n = source.shape().dimensions().size(); i < n; ++i) {
    auto it = absl::c_find(dimensions, i);
    if (it != dimensions.end()) {
      pre_broadcast_dims.push_back(
          source.operand(0)->shape().dimensions(it - dimensions.begin()));
    } else {
      pre_broadcast_dims.push_back(1);
    }
  }
  mlir::ShapedType reshape_type =
      llvm::cast<mlir::ShapedType>(operand.getType()).clone(pre_broadcast_dims);
  VLOG(3) << "Reshaping operand: " << llvm_ir::DumpToString(operand.getType())
          << " to " << llvm_ir::DumpToString(reshape_type);
  operand = tir::ReshapeOp::create(builder, reshape_type, operand);

  // Create the broadcast operation.
  mlir::ShapedType broadcast_type =
      reshape_type.clone(source.shape().dimensions());
  VLOG(3) << "Broadcasting operand: "
          << llvm_ir::DumpToString(operand.getType()) << " to "
          << llvm_ir::DumpToString(broadcast_type);
  return tir::BroadcastOp::create(builder, broadcast_type, operand);
}

// Builds a dot operation from HLO dot instruction.
absl::StatusOr<mlir::Value> BuildDot(const HloDotInstruction& source,
                                     mlir::Value lhs, mlir::Value rhs,
                                     mlir::ImplicitLocOpBuilder& builder) {
  // Calculate LHS transpose permutation.
  int lhs_rank = source.operand(0)->shape().dimensions().size();
  const auto& lhs_batch = source.dot_dimension_numbers().lhs_batch_dimensions();
  const auto& lhs_contracting =
      source.dot_dimension_numbers().lhs_contracting_dimensions();
  auto lhs_non_contracting =
      GetNonContractingDims(lhs_rank, lhs_contracting, lhs_batch);

  llvm::SmallVector<int64_t> lhs_permutation(lhs_batch.begin(),
                                             lhs_batch.end());
  lhs_permutation.append(lhs_non_contracting.begin(),
                         lhs_non_contracting.end());
  lhs_permutation.append(lhs_contracting.begin(), lhs_contracting.end());

  // Transpose LHS, if needed.
  llvm::SmallVector<int64_t> lhs_permuted_dims;
  for (int i = 0; i < lhs_rank; ++i) {
    lhs_permuted_dims.push_back(
        source.operand(0)->shape().dimensions(lhs_permutation[i]));
  }

  if (!absl::c_is_sorted(lhs_permutation)) {
    mlir::ShapedType transpose_type =
        llvm::cast<mlir::ShapedType>(lhs.getType()).clone(lhs_permuted_dims);
    VLOG(3) << "Transposing LHS operand: "
            << llvm_ir::DumpToString(lhs.getType()) << " to "
            << llvm_ir::DumpToString(transpose_type);
    lhs =
        tir::TransposeOp::create(builder, transpose_type, lhs, lhs_permutation);
  }

  // Calculate LHS dot operand shape.
  int64_t batch_size =
      Product(absl::MakeSpan(lhs_permuted_dims).first(lhs_batch.size()));
  int64_t contracting_size =
      Product(absl::MakeSpan(lhs_permuted_dims).last(lhs_contracting.size()));
  int64_t lhs_non_contracting_size =
      Product(absl::MakeSpan(lhs_permuted_dims)
                  .subspan(lhs_batch.size(), lhs_non_contracting.size()));

  llvm::SmallVector<int64_t> lhs_expected_dims;
  if (!lhs_batch.empty()) {
    lhs_expected_dims.push_back(batch_size);
  }
  lhs_expected_dims.push_back(lhs_non_contracting_size);
  lhs_expected_dims.push_back(contracting_size);

  // Reshape LHS, if needed.
  auto lhs_reshape_type =
      llvm::cast<mlir::ShapedType>(lhs.getType()).clone(lhs_expected_dims);
  if (lhs.getType() != lhs_reshape_type) {
    VLOG(3) << "Reshaping LHS operand: " << llvm_ir::DumpToString(lhs.getType())
            << " to " << llvm_ir::DumpToString(lhs_reshape_type);
    lhs = tir::ReshapeOp::create(builder, lhs_reshape_type, lhs);
  }

  // Calculate RHS transpose permutation.
  int rhs_rank = source.operand(1)->shape().dimensions().size();
  const auto& rhs_batch = source.dot_dimension_numbers().rhs_batch_dimensions();
  const auto& rhs_contracting =
      source.dot_dimension_numbers().rhs_contracting_dimensions();
  auto rhs_non_contracting =
      GetNonContractingDims(rhs_rank, rhs_contracting, rhs_batch);

  llvm::SmallVector<int64_t> rhs_permutation(rhs_batch.begin(),
                                             rhs_batch.end());
  rhs_permutation.append(rhs_contracting.begin(), rhs_contracting.end());
  rhs_permutation.append(rhs_non_contracting.begin(),
                         rhs_non_contracting.end());

  // Transpose RHS, if needed.
  llvm::SmallVector<int64_t> rhs_permuted_dims;
  for (int i = 0; i < rhs_rank; ++i) {
    rhs_permuted_dims.push_back(
        source.operand(1)->shape().dimensions(rhs_permutation[i]));
  }

  if (!absl::c_is_sorted(rhs_permutation)) {
    mlir::ShapedType transpose_type =
        llvm::cast<mlir::ShapedType>(rhs.getType()).clone(rhs_permuted_dims);
    VLOG(3) << "Transposing RHS operand: "
            << llvm_ir::DumpToString(rhs.getType()) << " to "
            << llvm_ir::DumpToString(transpose_type);
    rhs =
        tir::TransposeOp::create(builder, transpose_type, rhs, rhs_permutation);
  }

  // Calculate RHS dot operand shape.
  int64_t rhs_non_contracting_size = Product(
      absl::MakeSpan(rhs_permuted_dims).last(rhs_non_contracting.size()));

  llvm::SmallVector<int64_t> rhs_expected_dims;
  if (!rhs_batch.empty()) {
    rhs_expected_dims.push_back(batch_size);
  }
  rhs_expected_dims.push_back(contracting_size);
  rhs_expected_dims.push_back(rhs_non_contracting_size);

  // Reshape RHS, if needed.
  auto rhs_reshape_type =
      llvm::cast<mlir::ShapedType>(rhs.getType()).clone(rhs_expected_dims);
  if (rhs.getType() != rhs_reshape_type) {
    VLOG(3) << "Reshaping RHS operand: " << llvm_ir::DumpToString(rhs.getType())
            << " to " << llvm_ir::DumpToString(rhs_reshape_type);
    rhs = tir::ReshapeOp::create(builder, rhs_reshape_type, rhs);
  }

  // Calculate matmul result shape.
  ABSL_ASSIGN_OR_RETURN(auto result_type, GetTensorType(source.shape(), builder));

  llvm::SmallVector<int64_t> matmul_shape;
  if (!lhs_batch.empty()) {
    matmul_shape.push_back(batch_size);
  }
  matmul_shape.push_back(lhs_non_contracting_size);
  matmul_shape.push_back(rhs_non_contracting_size);
  mlir::ShapedType matmul_type = result_type.clone(matmul_shape);

  // Create the matmul operation.
  VLOG(3) << "Creating matmul operation: " << source.name();
  auto matmul_op = tir::MatmulOp::create(builder, matmul_type, lhs, rhs);
  mlir::Value result = matmul_op.getResult();

  // Align the result to the source shape.
  if (result_type != matmul_type) {
    VLOG(3) << "Reshaping result: " << llvm_ir::DumpToString(matmul_type)
            << " to " << llvm_ir::DumpToString(result_type);
    result = tir::ReshapeOp::create(builder, result_type, result);
  }
  return result;
}

// Returns the shaped type for reduction output with unit dimensions for reduced
// axes.
mlir::ShapedType GetReduceOutputType(mlir::RankedTensorType result_type,
                                     const Shape& operand_shape,
                                     const Shape& result_shape,
                                     absl::Span<const int32_t> reduce_dims) {
  llvm::SmallVector<int64_t> output_shape;
  output_shape.reserve(operand_shape.dimensions().size());
  for (int i = 0, p = 0; i < operand_shape.dimensions().size(); ++i) {
    if (!absl::c_contains(reduce_dims, i)) {
      output_shape.push_back(result_shape.dimensions(p++));
    } else {
      output_shape.push_back(1);
    }
  }
  return result_type.clone(output_shape);
}

// Builds a reduction operation from HLO reduce instruction.
absl::StatusOr<mlir::Value> BuildReduce(const HloReduceInstruction& source,
                                        mlir::ValueRange operands,
                                        mlir::ImplicitLocOpBuilder& builder) {
  llvm::SmallVector<mlir::Attribute> initial_values;
  for (const HloInstruction* init_instr : source.init_values()) {
    const auto* constant = Cast<HloConstantInstruction>(init_instr);
    ABSL_ASSIGN_OR_RETURN(auto initial_value,
                     GetScalarAttribute(constant->literal(), builder,
                                        /*use_signless_integer_type=*/false));
    initial_values.push_back(initial_value);
  }

  llvm::SmallVector<int32_t> reduce_dims(source.dimensions().begin(),
                                         source.dimensions().end());
  ABSL_ASSIGN_OR_RETURN(auto result_type, GetTensorType(source.shape(), builder));
  mlir::ShapedType output_type = GetReduceOutputType(
      result_type, source.operand(0)->shape(), source.shape(), reduce_dims);

  VLOG(3) << "Creating reduction operation: " << source.name();
  auto reduce_op = tir::ReduceUDOp::create(
      builder, output_type, operands.take_front(source.input_count()),
      reduce_dims, builder.getArrayAttr(initial_values));

  mlir::Block* body = builder.createBlock(&reduce_op.getRegion());
  for (int i = 0; i < 2; ++i) {
    for (mlir::Attribute init : initial_values) {
      auto arg_type = llvm::cast<mlir::TypedAttr>(init).getType();
      if (arg_type.isInteger()) {
        arg_type = builder.getIntegerType(arg_type.getIntOrFloatBitWidth());
      }
      body->addArgument(arg_type, reduce_op.getLoc());
    }
  }

  const HloComputation* computation = source.to_apply();
  VLOG(3) << "Converting reduction HLO computation: " << computation->name();

  llvm::DenseMap<const HloInstruction*, mlir::Value> converted;
  for (const HloInstruction* instruction :
       computation->MakeInstructionPostOrder()) {
    llvm::SmallVector<mlir::Value> new_operands;
    for (const auto& operand : instruction->operands()) {
      new_operands.push_back(converted[operand]);
    }
    ABSL_ASSIGN_OR_RETURN(
        converted[instruction],
        ConvertReductionInstruction(*instruction, new_operands, *body));
  }

  mlir::Value result = converted[computation->root_instruction()];
  tir::YieldOp::create(builder, result);
  builder.setInsertionPointAfter(reduce_op);

  VLOG(3) << "Reshaping result: " << llvm_ir::DumpToString(output_type)
          << " to " << llvm_ir::DumpToString(result_type);
  return tir::ReshapeOp::create(builder, result_type, reduce_op.getResult(0));
}

// Returns the reduction type, if the body is a supported single-op reduction.
std::optional<tir::ReductionMode> GetReductionType(
    const HloReduceInstruction& reduce) {
  // Reduction must have exactly two parameters.
  const HloComputation* comp = reduce.to_apply();
  if (comp->num_parameters() != 2) {
    return std::nullopt;
  }
  const HloInstruction* param0 = comp->parameter_instruction(0);
  const HloInstruction* param1 = comp->parameter_instruction(1);

  // Reduction root must use the parameters as operands.
  const HloInstruction* root = comp->root_instruction();
  if (root->operand_count() != 2 || root->operand(0) != param0 ||
      root->operand(1) != param1) {
    return std::nullopt;
  }

  // Reduction initial value must be a constant.
  const auto* init_value =
      DynCast<HloConstantInstruction>(reduce.init_values().front());
  if (init_value == nullptr) {
    return std::nullopt;
  }

  if (root->opcode() == HloOpcode::kAdd && init_value->literal().IsAll(0)) {
    return tir::ReductionMode::add;
  }

  if (root->opcode() == HloOpcode::kMultiply &&
      init_value->literal().IsAll(1)) {
    return tir::ReductionMode::mul;
  }

  return std::nullopt;
}

// Builds a reduction operation for a reduction type.
absl::StatusOr<mlir::Value> BuildSimpleReduce(
    const HloReduceInstruction& source, tir::ReductionMode mode,
    mlir::Value operand, mlir::ImplicitLocOpBuilder& builder) {
  llvm::SmallVector<int32_t> reduce_dims(source.dimensions().begin(),
                                         source.dimensions().end());
  ABSL_ASSIGN_OR_RETURN(auto result_type, GetTensorType(source.shape(), builder));
  mlir::ShapedType output_type = GetReduceOutputType(
      result_type, source.operand(0)->shape(), source.shape(), reduce_dims);

  VLOG(3) << "Creating reduction operation: " << source.name();
  auto reduce_op =
      tir::ReduceOp::create(builder, output_type, operand, reduce_dims, mode);

  VLOG(3) << "Reshaping result: " << llvm_ir::DumpToString(output_type)
          << " to " << llvm_ir::DumpToString(result_type);
  return tir::ReshapeOp::create(builder, result_type, reduce_op.getResult());
}

// Maps HLO compare instruction to TensorIR comparator.
tir::Comparator GetComparator(const HloCompareInstruction& source) {
  bool is_float = primitive_util::IsFloatingPointType(
      source.operand(0)->shape().element_type());
  if (is_float) {
    switch (source.direction()) {
      case ComparisonDirection::kEq:
        return tir::Comparator::oeq;
      case ComparisonDirection::kNe:
        return tir::Comparator::une;
      case ComparisonDirection::kGt:
        return tir::Comparator::ogt;
      case ComparisonDirection::kGe:
        return tir::Comparator::oge;
      case ComparisonDirection::kLt:
        return tir::Comparator::olt;
      case ComparisonDirection::kLe:
        return tir::Comparator::ole;
    }
  } else {
    switch (source.direction()) {
      case ComparisonDirection::kEq:
        return tir::Comparator::eq;
      case ComparisonDirection::kNe:
        return tir::Comparator::neq;
      case ComparisonDirection::kGt:
        return tir::Comparator::gt;
      case ComparisonDirection::kGe:
        return tir::Comparator::ge;
      case ComparisonDirection::kLt:
        return tir::Comparator::lt;
      case ComparisonDirection::kLe:
        return tir::Comparator::le;
    }
  }
}

// Maps HLO compare instruction to `arith` comparator (floating-point).
arith::CmpFPredicate GetArithFloatComparator(
    const HloCompareInstruction& source) {
  switch (source.direction()) {
    case ComparisonDirection::kEq:
      return arith::CmpFPredicate::OEQ;
    case ComparisonDirection::kNe:
      return arith::CmpFPredicate::UNE;
    case ComparisonDirection::kGt:
      return arith::CmpFPredicate::OGT;
    case ComparisonDirection::kGe:
      return arith::CmpFPredicate::OGE;
    case ComparisonDirection::kLt:
      return arith::CmpFPredicate::OLT;
    case ComparisonDirection::kLe:
      return arith::CmpFPredicate::OLE;
  }
}

// Maps HLO compare instruction to `arith` comparator (integer).
arith::CmpIPredicate GetArithIntComparator(
    const HloCompareInstruction& source) {
  bool is_signed = primitive_util::IsSignedIntegralType(
      source.operand(0)->shape().element_type());
  switch (source.direction()) {
    case ComparisonDirection::kEq:
      return arith::CmpIPredicate::eq;
    case ComparisonDirection::kNe:
      return arith::CmpIPredicate::ne;
    case ComparisonDirection::kGt:
      return is_signed ? arith::CmpIPredicate::sgt : arith::CmpIPredicate::ugt;
    case ComparisonDirection::kGe:
      return is_signed ? arith::CmpIPredicate::sge : arith::CmpIPredicate::uge;
    case ComparisonDirection::kLt:
      return is_signed ? arith::CmpIPredicate::slt : arith::CmpIPredicate::ult;
    case ComparisonDirection::kLe:
      return is_signed ? arith::CmpIPredicate::sle : arith::CmpIPredicate::ule;
  }
}

// Template for creating `arith` operations.
template <typename Op, typename... Args>
mlir::Value CreateArithOp(Args&&... args) {
  return Op::create(std::forward<Args>(args)...).getResult();
}

// Creates strides attribute (helper).
mlir::NamedAttribute CreateStridesAttributeFromShape(mlir::MLIRContext* context,
                                                     const Shape& shape) {
  // Calculate strides from the shape and the layout.
  llvm::SmallVector<int64_t> strides(shape.dimensions().size());
  int64_t stride = 1;
  for (int dim : shape.layout().minor_to_major()) {
    strides[dim] = stride;
    stride *= shape.dimensions(dim);
  }

  // Create the attribute with strides formatted as a string.
  auto strides_fmt = absl::StrCat("(", absl::StrJoin(strides, ","), ")");
  auto strides_attr = mlir::StringAttr::get(context, strides_fmt);
  return {tir::TensorIRDialect::getStrideAttrName(), strides_attr};
}

mlir::DictionaryAttr CreateStridesDictionaryAttribute(
    mlir::MLIRContext* context, const Shape& shape) {
  if (LayoutUtil::IsMonotonicWithDim0Major(shape.layout())) {
    return mlir::DictionaryAttr::get(context);
  }
  auto attr = CreateStridesAttributeFromShape(context, shape);
  return mlir::DictionaryAttr::get(context, {attr});
}

// Converts HLO instruction `source` with already-converted `operands` values
// to TensorIR ops and insert them at the end of `target` block.
absl::StatusOr<mlir::Value> ConvertFusionInstruction(
    const HloInstruction& source, mlir::ValueRange operands,
    mlir::Block& target) {
  mlir::MLIRContext* context = target.getParent()->getContext();
  VLOG(3) << "Converting HLO instruction: " << source.ToString();

  mlir::Location location = GetLocationFromInstruction(source, context);
  mlir::ImplicitLocOpBuilder builder(location, context);
  builder.setInsertionPointToEnd(&target);

  switch (source.opcode()) {
    // Unary elementwise operations.
    case HloOpcode::kAbs:
      return tir::AbsOp::create(builder, operands);
    case HloOpcode::kCeil:
      return tir::CeilOp::create(builder, operands);
    case HloOpcode::kConvert: {
      ABSL_ASSIGN_OR_RETURN(auto convert_type,
                       GetTensorType(source.shape(), builder));
      return tir::ConvertOp::create(builder, convert_type, operands[0]);
    }
    case HloOpcode::kCos:
      return tir::CosOp::create(builder, operands);
    case HloOpcode::kErf:
      return tir::ErfOp::create(builder, operands);
    case HloOpcode::kExp:
      return tir::ExpOp::create(builder, operands);
    case HloOpcode::kExpm1: {
      // NOTE: Decomposing Expm1(x) to Exp(x) - 1.0 can result in a severe loss
      // of precision for values of x close to 0.
      ABSL_ASSIGN_OR_RETURN(auto one,
                       BuildFloatConstant(1.0, source.shape(), builder));
      return tir::SubOp::create(builder,
                                tir::ExpOp::create(builder, operands[0]), one);
    }
    case HloOpcode::kFloor:
      return tir::FloorOp::create(builder, operands);
    case HloOpcode::kLog:
      return tir::LogOp::create(builder, operands);
    case HloOpcode::kLog1p: {
      // NOTE: Decomposing Log1p(x) to Log(x + 1.0) can result in a severe loss
      // of precision for values of x close to 0.
      ABSL_ASSIGN_OR_RETURN(auto one,
                       BuildFloatConstant(1.0, source.shape(), builder));
      return tir::LogOp::create(builder,
                                tir::AddOp::create(builder, operands[0], one));
    }
    case HloOpcode::kNot:
      return tir::LogicalNotOp::create(builder, operands);
    case HloOpcode::kNegate:
      return tir::NegOp::create(builder, operands);
    case HloOpcode::kRsqrt:
      return tir::RsqrtOp::create(builder, operands);
    case HloOpcode::kSin:
      return tir::SinOp::create(builder, operands);
    case HloOpcode::kSqrt:
      return tir::SqrtOp::create(builder, operands);
    case HloOpcode::kTan:
      return tir::TanOp::create(builder, operands);
    case HloOpcode::kTanh:
      return tir::TanhFwdOp::create(builder, operands);

    // Binary elementwise operations.
    case HloOpcode::kAdd:
      return tir::AddOp::create(builder, operands);
    case HloOpcode::kAtan2:
      return tir::Atan2Op::create(builder, operands);
    case HloOpcode::kCompare: {
      tir::Comparator comparator =
          GetComparator(*Cast<HloCompareInstruction>(&source));
      return tir::CmpOp::create(builder, comparator, operands[0], operands[1]);
    }
    case HloOpcode::kDivide:
      return tir::DivOp::create(builder, operands);
    case HloOpcode::kMaximum:
      return tir::MaxOp::create(builder, operands);
    case HloOpcode::kMinimum:
      return tir::MinOp::create(builder, operands);
    case HloOpcode::kMultiply:
      return tir::MulOp::create(builder, operands);
    case HloOpcode::kPower:
      return tir::PowOp::create(builder, operands);
    case HloOpcode::kRemainder:
      return tir::RemOp::create(builder, operands);
    case HloOpcode::kSubtract:
      return tir::SubOp::create(builder, operands);
    case HloOpcode::kAnd:
      return tir::LogicalAndOp::create(builder, operands);
    case HloOpcode::kOr:
      return tir::LogicalOrOp::create(builder, operands);

    // Ternary elementwise operations.
    case HloOpcode::kSelect:
      return tir::BinarySelectOp::create(builder, operands);
    case HloOpcode::kClamp:
      return tir::MinOp::create(
          builder, operands[2],
          tir::MaxOp::create(builder, operands[0], operands[1]));

    // Layout modification operations.
    case HloOpcode::kReshape:
    case HloOpcode::kBitcast:
      return BuildBitcast(source, operands[0], builder);
    case HloOpcode::kBroadcast:
      return BuildBroadcast(*Cast<HloBroadcastInstruction>(&source),
                            operands[0], builder);
    case HloOpcode::kSlice: {
      ABSL_ASSIGN_OR_RETURN(auto slice_type, GetTensorType(source.shape(), builder));
      return tir::SliceOp::create(builder, slice_type, operands[0],
                                  source.slice_starts(), source.slice_limits(),
                                  source.slice_strides());
    }
    case HloOpcode::kTranspose: {
      auto transpose = Cast<HloTransposeInstruction>(&source);
      ABSL_ASSIGN_OR_RETURN(auto transpose_type,
                       GetTensorType(source.shape(), builder));
      return tir::TransposeOp::create(builder, transpose_type, operands[0],
                                      transpose->dimensions());
    }

    // Reduction operations.
    case HloOpcode::kDot:
      return BuildDot(*Cast<HloDotInstruction>(&source), operands[0],
                      operands[1], builder);
    case HloOpcode::kReduce: {
      auto reduce = Cast<HloReduceInstruction>(&source);
      auto type = GetReductionType(*reduce);
      return type.has_value()
                 ? BuildSimpleReduce(*reduce, *type, operands[0], builder)
                 : BuildReduce(*reduce, operands, builder);
    }

    // Miscellaneous operations.
    case HloOpcode::kParameter:
      return target.getArgument(source.parameter_number());
    case HloOpcode::kConstant:
      return BuildConstant(source.literal(), source.shape(), builder);
    case HloOpcode::kIota: {
      auto iota = Cast<HloIotaInstruction>(&source);
      ABSL_ASSIGN_OR_RETURN(auto iota_type, GetTensorType(source.shape(), builder));
      return tir::IotaOp::create(builder, iota_type, iota->iota_dimension(),
                                 /*dynamic_sizes=*/{});
    }
    case HloOpcode::kConcatenate: {
      ABSL_ASSIGN_OR_RETURN(auto concatenate_type,
                       GetTensorType(source.shape(), builder));
      return tir::ConcatenateOp::create(builder, concatenate_type, operands,
                                        source.concatenate_dimension());
    }

    default:
      return absl::UnimplementedError(absl::StrCat(
          "Unsupported instruction: ", HloOpcodeString(source.opcode())));
  }
}

// Converts HLO instruction `source` in the context of a reduction computation
// with already-converted `operands` values to TensorIR ops and insert them at
// the end of `target` block.
absl::StatusOr<mlir::Value> ConvertReductionInstruction(
    const HloInstruction& source, mlir::ValueRange operands,
    mlir::Block& target) {
  mlir::MLIRContext* context = target.getParent()->getContext();
  VLOG(3) << "Converting HLO instruction: " << source.ToString();

  mlir::Location location = GetLocationFromInstruction(source, context);
  mlir::ImplicitLocOpBuilder builder(location, context);
  builder.setInsertionPointToEnd(&target);

  // `arith` dialect (used in reduction body) has separate instructions for
  // floating-point and signed/unsigned integer types.
  const HloInstruction* type_source =
      source.operand_count() != 0 ? source.operand(0) : &source;
  PrimitiveType element_type = type_source->shape().element_type();
  bool is_float = primitive_util::IsFloatingPointType(element_type);
  bool is_signed = primitive_util::IsSignedIntegralType(element_type);

  switch (source.opcode()) {
    // Binary elementwise operations.
    case HloOpcode::kAdd:
      return is_float ? CreateArithOp<arith::AddFOp>(builder, operands)
                      : CreateArithOp<arith::AddIOp>(builder, operands);
    case HloOpcode::kCompare: {
      const auto& compare = *Cast<HloCompareInstruction>(&source);
      return is_float ? CreateArithOp<arith::CmpFOp>(
                            builder, GetArithFloatComparator(compare),
                            operands[0], operands[1])
                      : CreateArithOp<arith::CmpIOp>(
                            builder, GetArithIntComparator(compare),
                            operands[0], operands[1]);
    }
    case HloOpcode::kMaximum:
      return is_float    ? CreateArithOp<arith::MaximumFOp>(builder, operands)
             : is_signed ? CreateArithOp<arith::MaxSIOp>(builder, operands)
                         : CreateArithOp<arith::MaxUIOp>(builder, operands);
    case HloOpcode::kMinimum:
      return is_float    ? CreateArithOp<arith::MinimumFOp>(builder, operands)
             : is_signed ? CreateArithOp<arith::MinSIOp>(builder, operands)
                         : CreateArithOp<arith::MinUIOp>(builder, operands);
    case HloOpcode::kMultiply:
      return is_float ? CreateArithOp<arith::MulFOp>(builder, operands)
                      : CreateArithOp<arith::MulIOp>(builder, operands);
    case HloOpcode::kAnd:
      return CreateArithOp<arith::AndIOp>(builder, operands);
    case HloOpcode::kOr:
      return CreateArithOp<arith::OrIOp>(builder, operands);
    case HloOpcode::kXor:
      return CreateArithOp<arith::XOrIOp>(builder, operands);

    // Ternary elementwise operations.
    case HloOpcode::kSelect:
      return CreateArithOp<arith::SelectOp>(builder, operands);
    case HloOpcode::kClamp:
      return is_float    ? CreateArithOp<arith::MinimumFOp>(
                               builder, operands[2],
                               CreateArithOp<arith::MaximumFOp>(
                                   builder, operands[0], operands[1]))
             : is_signed ? CreateArithOp<arith::MinSIOp>(
                               builder, operands[2],
                               CreateArithOp<arith::MaxSIOp>(
                                   builder, operands[0], operands[1]))
                         : CreateArithOp<arith::MinUIOp>(
                               builder, operands[2],
                               CreateArithOp<arith::MaxUIOp>(
                                   builder, operands[0], operands[1]));

    // Miscellaneous operations.
    case HloOpcode::kParameter:
      return target.getArgument(source.parameter_number());
    case HloOpcode::kConstant: {
      const auto& constant = *Cast<HloConstantInstruction>(&source);
      ABSL_ASSIGN_OR_RETURN(auto value,
                       GetScalarAttribute(constant.literal(), builder,
                                          /*use_signless_integer_type=*/true));
      return arith::ConstantOp::create(builder, value);
    }

    default:
      return absl::UnimplementedError(absl::StrCat(
          "Unsupported instruction: ", HloOpcodeString(source.opcode())));
  }
}

}  // namespace

// Converts an HLO fusion computation into a TensorIR `GraphOp` appended
// to the body of `target`. If conversion fails, `target` is unmodified.
absl::StatusOr<mlir::nv_tensor_ir::GraphOp> ConvertFusionComputation(
    const HloComputation& source, mlir::ModuleOp target) {
  mlir::MLIRContext* context = target.getContext();
  VLOG(3) << "Converting HLO computation: " << source.name();

  mlir::ImplicitLocOpBuilder builder(mlir::UnknownLoc::get(context), context);
  builder.setInsertionPointToEnd(target.getBody());

  llvm::SmallVector<mlir::Type> input_types;
  llvm::SmallVector<mlir::Attribute> input_attrs;
  for (const auto& parameter : source.parameter_instructions()) {
    ABSL_ASSIGN_OR_RETURN(auto parameter_type,
                     GetTensorType(parameter->shape(), builder));
    input_types.push_back(parameter_type);
    input_attrs.push_back(
        CreateStridesDictionaryAttribute(context, parameter->shape()));
  }

  mlir::ArrayAttr arg_attrs;
  if (llvm::any_of(input_attrs, [](mlir::Attribute attr) {
        return !llvm::cast<mlir::DictionaryAttr>(attr).empty();
      })) {
    arg_attrs = mlir::ArrayAttr::get(context, input_attrs);
  }

  llvm::SmallVector<mlir::Type> output_types;
  ABSL_ASSIGN_OR_RETURN(auto output_type,
                   GetTensorType(source.root_instruction()->shape(), builder));
  output_types.push_back(output_type);

  mlir::ArrayAttr res_attrs;
  auto output_attr = CreateStridesDictionaryAttribute(
      context, source.root_instruction()->shape());
  if (!output_attr.empty()) {
    res_attrs = mlir::ArrayAttr::get(context, {output_attr});
  }

  auto function_type =
      mlir::FunctionType::get(context, input_types, output_types);
  VLOG(3) << "Function type: " << llvm_ir::DumpToString(function_type);

  auto graph_op = tir::GraphOp::create(builder, source.name(),
                                       /*sym_visibility=*/nullptr,
                                       function_type, arg_attrs, res_attrs);

  absl::Cleanup cleanup = [&] { graph_op.erase(); };

  mlir::Block* body = builder.createBlock(&graph_op.getRegion());
  for (mlir::Type type : input_types) {
    body->addArgument(type, target.getLoc());
  }

  llvm::DenseMap<const HloInstruction*, mlir::Value> converted;
  for (const HloInstruction* instruction : source.MakeInstructionPostOrder()) {
    llvm::SmallVector<mlir::Value> operands;
    operands.reserve(instruction->operand_count());
    for (const auto& operand : instruction->operands()) {
      operands.push_back(converted[operand]);
    }
    ABSL_ASSIGN_OR_RETURN(
        converted[instruction],
        ConvertFusionInstruction(*instruction, operands, *graph_op.getBody()));
  }

  mlir::Value result = converted[source.root_instruction()];
  tir::ResultsOp::create(builder, result);

  // Remove dead operations (scalar constants).
  for (mlir::Operation& op : llvm::make_early_inc_range(graph_op.getOps())) {
    if (mlir::isOpTriviallyDead(&op)) {
      op.erase();
    }
  }

  if (mlir::failed(mlir::verify(graph_op))) {
    return absl::InternalError(absl::StrCat(
        "Verification failed for TensorIR graph: ", source.name()));
  }

  std::move(cleanup).Cancel();
  return graph_op;
}

// Creates a new MLIR module containing the converted TensorIR graph for
// `source`.
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ConvertFusionComputation(
    const HloComputation& source, mlir::MLIRContext* context) {
  mlir::Location location = mlir::UnknownLoc::get(context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      llvm_ir::CreateMlirModuleOp(location);
  ABSL_RETURN_IF_ERROR(ConvertFusionComputation(source, *module).status());
  return module;
}

}  // namespace xla::gpu::tensor_ir

// NOLINTEND(clang-diagnostic-pre-c++20-compat)
