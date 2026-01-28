/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/codegen/xtile/codegen/emitter_helpers.h"

#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/codegen/emitters/elemental_hlo_to_mlir.h"
#include "xla/codegen/tiling/tiled_hlo_instruction.h"
#include "xla/codegen/xtile/ir/xtile_ops.h"
#include "xla/comparison_util.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/layout_util.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/mlir_hlo/mhlo/transforms/map_mhlo_to_scalar_op.h"
#include "xla/mlir_hlo/mhlo/transforms/transformation_helpers.h"
#include "xla/primitive_util.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/shape.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla::xtile {

using ::llvm::SmallVector;
using ::mlir::ArrayRef;
using ::mlir::ShapedType;
using ::mlir::Type;
using ::mlir::Value;
using ::mlir::ValueRange;

namespace ma = ::mlir::arith;
namespace mh = ::mlir::mhlo;
namespace mm = ::mlir::math;

namespace {
using TensorValue = mlir::TypedValue<mlir::RankedTensorType>;

// Emit a value as Index clamped to [lower, upper].
Value EmitClampedIndex(mlir::ImplicitLocOpBuilder& b, Value value,
                       int64_t lower, int64_t upper) {
  Value clamped_index =
      ma::MaxSIOp::create(b, value, CreateConst(b, value.getType(), lower));
  clamped_index = ma::MinSIOp::create(b, clamped_index,
                                      CreateConst(b, value.getType(), upper));
  return ma::IndexCastOp::create(b, b.getIndexType(), clamped_index);
}

absl::StatusOr<SmallVector<Value>> ComputeOffsetsForTile(
    mlir::ImplicitLocOpBuilder& b, Value pid, ValueRange runtime_values,
    const TiledHloInstruction& tiled_hlo) {
  TF_ASSIGN_OR_RETURN(IndexingMap tile_offsets_indexing,
                      tiled_hlo.tile_offsets_indexing());
  const std::vector<IndexingMap::Variable>& rt_vars =
      tile_offsets_indexing.GetRTVars();
  CHECK_EQ(rt_vars.size(), runtime_values.size())
      << absl::StrCat(tiled_hlo.ToString(), " has ", rt_vars.size(),
                      " runtime variables in tile_offsets_indexing but only ",
                      runtime_values.size(), " runtime values were provided");
  CHECK_EQ(tile_offsets_indexing.GetRangeVars().size(), 0)
      << "Range variables must be converted to dimensions. Instruction: "
      << tiled_hlo.ToString();
  // emitters::ApplyIndexing does not support symbols at the moment. As a
  // workaround we convert them to dimensions.
  IndexingMap dim_only_tiling =
      tile_offsets_indexing.ConvertSymbolsToDimensions();
  SmallVector<Value> dims;
  dims.reserve(1 /* pid */ + runtime_values.size());
  dims.push_back(pid);
  for (const auto& [rt_var, value] : llvm::zip(rt_vars, runtime_values)) {
    Value clamped_index =
        EmitClampedIndex(b, value, rt_var.bounds.lower, rt_var.bounds.upper);
    dims.push_back(Cast(b, clamped_index, pid.getType()));
  }
  return emitters::ApplyIndexing(dim_only_tiling, /*dims=*/dims,
                                 /*symbols=*/{}, b);
}

// Emit code corresponding to a fusion instruction somehow nested within the
// initial Triton fusion. This can happen when we carry around auxiliary
// computations, e.g. with reduces. Since we are emitting a single Triton
// fusion, we simply flatten the fusion inside the computation.
//
// TODO(b/331413981): get rid of this special handling once this is solved.
absl::StatusOr<TensorValue> EmitNestedFusion(
    mlir::ImplicitLocOpBuilder& b,
    const HloFusionInstruction& fusion_instruction,
    absl::flat_hash_map<const HloInstruction*, TensorValue>& values) {
  // TODO(b/331402498): revisit the order of scope once we completely
  // deprecate Triton fusion analysis.
  const HloComputation* fusion_computation =
      fusion_instruction.fused_instructions_computation();

  absl::flat_hash_map<const HloInstruction*, TensorValue> region_values;

  std::vector<const HloInstruction*> to_emit;
  for (const HloInstruction* instr :
       fusion_computation->MakeInstructionPostOrder()) {
    if (instr->opcode() == HloOpcode::kParameter) {
      int64_t parameter_number = instr->parameter_number();
      auto it = values.find(fusion_instruction.operand(parameter_number));
      TF_RET_CHECK(it != values.end());
      TF_RET_CHECK(region_values.insert({instr, it->second}).second);
    } else {
      to_emit.push_back(instr);
    }
  }

  TF_RET_CHECK(to_emit.back() == fusion_computation->root_instruction());

  return EmitScope(b, to_emit, region_values);
}

// Get a constant with all high bits of the same type as provided.
mlir::Value OnesLike(mlir::ImplicitLocOpBuilder& b, mlir::Type type) {
  mlir::Type element_type = mlir::getElementTypeOrSelf(type);
  CHECK(element_type.isInteger()) << "OnesLike only supports integer types.";

  int64_t width = element_type.getIntOrFloatBitWidth();
  mlir::APInt all_ones = mlir::APInt::getAllOnes(width);
  return mlir::createScalarOrSplatConstant(b, b.getLoc(), type, all_ones);
}

}  // namespace

SmallVector<int64_t> GetPaddedTileSizes(ArrayRef<int64_t> tile_sizes) {
  SmallVector<int64_t> result;
  result.reserve(tile_sizes.size());
  for (int64_t value : tile_sizes) {
    result.push_back(llvm::PowerOf2Ceil(value));
  }
  return result;
}

absl::StatusOr<Type> PrimitiveTypeToMlirType(mlir::ImplicitLocOpBuilder& b,

                                             PrimitiveType t) {
  switch (t) {
    case F64:
      return b.getF64Type();
    case F32:
      return b.getF32Type();
    case F16:
      return b.getF16Type();
    case BF16:
      return b.getBF16Type();
    case S64:
      return b.getI64Type();
    case U64:
      return b.getIntegerType(/*width=*/64, /*isSigned=*/false);
    case S32:
      return b.getI32Type();
    case U32:
      return b.getIntegerType(/*width=*/32, /*isSigned=*/false);
    case S16:
      return b.getI16Type();
    case U16:
      return b.getIntegerType(/*width=*/16, /*isSigned=*/false);
    case S8:
      return b.getI8Type();
    case U8:
      return b.getIntegerType(/*width=*/8, /*isSigned=*/false);
    case S4:
      return b.getI4Type();
    case U4:
      return b.getIntegerType(/*width=*/4, /*isSigned=*/false);
    case PRED:
      return b.getI1Type();
    case F8E5M2:
      return b.getType<mlir::Float8E5M2Type>();
    case F8E4M3FN:
      return b.getType<mlir::Float8E4M3FNType>();
    case F8E8M0FNU:
      return b.getType<mlir::Float8E8M0FNUType>();
    case F4E2M1FN:
      return b.getType<mlir::Float4E2M1FNType>();
    default:
      return absl::UnimplementedError(
          absl::StrCat("This type is not supported yet: ",
                       primitive_util::LowercasePrimitiveTypeName(t)));
  }
}

absl::StatusOr<PrimitiveType> GetPrimitiveType(Type t) {
  // NOLINTBEGIN(google-readability-braces-around-statements)
  if (t.isF64()) return F64;
  if (t.isF32()) return F32;
  if (t.isF16()) return F16;
  if (t.isBF16()) return BF16;
  if (t.isInteger(64)) return t.isSignedInteger() ? S64 : U64;
  if (t.isInteger(32)) return t.isSignedInteger() ? S32 : U32;
  if (t.isInteger(16)) return t.isSignedInteger() ? S16 : U16;
  if (t.isInteger(8)) return t.isSignedInteger() ? S8 : U8;
  if (t.isInteger(4)) return t.isSignedInteger() ? S4 : U4;
  if (t.isInteger(1)) return PRED;
  if (mlir::isa<mlir::Float8E5M2Type>(t)) return F8E5M2;
  if (mlir::isa<mlir::Float8E4M3FNType>(t)) return F8E4M3FN;
  if (mlir::isa<mlir::Float8E8M0FNUType>(t)) return F8E8M0FNU;
  // NOLINTEND(google-readability-braces-around-statements)
  return absl::UnimplementedError("Unsupported type in getPrimitiveType.\n");
}

Type StorageType(Type t) {
  if (auto i = mlir::dyn_cast<mlir::IntegerType>(t); i && i.getWidth() == 1) {
    return i.get(i.getContext(), 8, i.getSignedness());
  }
  return t;
}

bool IsFp8Type(Type t) {
  return llvm::isa<mlir::Float8E5M2Type, mlir::Float8E4M3FNType,
                   mlir::Float8E5M2FNUZType, mlir::Float8E4M3FNUZType,
                   mlir::Float8E4M3B11FNUZType>(t);
}

Value Cast(mlir::ImplicitLocOpBuilder& b, Value value, Type dst_element_ty) {
  auto src_ty = value.getType();
  auto dst_ty = dst_element_ty;
  if (auto src_shaped_ty = mlir::dyn_cast<ShapedType>(src_ty)) {
    dst_ty = src_shaped_ty.clone(src_shaped_ty.getShape(), dst_element_ty);
  }
  if (src_ty == dst_ty) {
    return value;
  }

  if (src_ty.isIndex() || dst_ty.isIndex()) {
    return ma::IndexCastOp::create(b, dst_ty, value);
  }

  return mlir::stablehlo::ConvertOp::create(b, dst_ty, value);
}

Value Subtract(mlir::ImplicitLocOpBuilder& b, ValueRange values) {
  return mlir::stablehlo::SubtractOp::create(b, values[0], values[1]);
}

Value Compare(mlir::ImplicitLocOpBuilder& b, ValueRange values,
              mlir::stablehlo::ComparisonDirection direction) {
  return mlir::stablehlo::CompareOp::create(b, values[0], values[1], direction);
}

Value Maximum(mlir::ImplicitLocOpBuilder& b, ValueRange values) {
  auto type = mlir::getElementTypeOrSelf(values[0]);

  if (type.isInteger(1)) {
    return ma::OrIOp::create(b, values);
  }

  return mlir::stablehlo::MaxOp::create(b, values);
}

Value Minimum(mlir::ImplicitLocOpBuilder& b, ValueRange values) {
  auto type = mlir::getElementTypeOrSelf(values[0]);

  if (type.isInteger(1)) {
    return ma::AndIOp::create(b, values);
  }

  return mlir::stablehlo::MinOp::create(b, values);
}

absl::StatusOr<Value> EmitElementwise(mlir::ImplicitLocOpBuilder& b,
                                      const HloInstruction& hlo,
                                      ValueRange inputs) {
  const bool is_integer =
      mlir::isa<mlir::IntegerType>(getElementTypeOrSelf(inputs[0].getType()));

  switch (hlo.opcode()) {
    case HloOpcode::kCopy:
      // Dimension transformations are taken care of separately.
      return inputs[0];
    case HloOpcode::kAbs:
      if (is_integer) {
        return mm::AbsIOp::create(b, inputs[0]);
      }
      return mm::AbsFOp::create(b, inputs[0]);
    case HloOpcode::kCeil:
      return mm::CeilOp::create(b, inputs[0]);
    case HloOpcode::kFloor:
      return mm::FloorOp::create(b, inputs[0]);
    case HloOpcode::kNot:
      return mlir::stablehlo::XorOp::create(b, inputs[0],
                                            OnesLike(b, inputs[0].getType()));
    case HloOpcode::kNegate:
      if (is_integer) {
        return Subtract(b, {ZerosLike(b, inputs[0]), inputs[0]});
      }
      return ma::NegFOp::create(b, inputs[0]);
    case HloOpcode::kConvert: {
      TF_ASSIGN_OR_RETURN(
          Type dst_ty, PrimitiveTypeToMlirType(b, hlo.shape().element_type()));
      return Cast(b, inputs[0], dst_ty);
    }
    case HloOpcode::kAdd:
      if (is_integer) {
        // XLA add semantics for predicates is equal to bitwise OR, while Arith
        // defines it differently. Replace add with or in this case.
        if (getElementTypeOrSelf(inputs[0]).isInteger(1)) {
          return ma::OrIOp::create(b, inputs[0], inputs[1]);
        }
      }
      return mlir::stablehlo::AddOp::create(b, inputs[0], inputs[1]);
    case HloOpcode::kSubtract:
      return Subtract(b, inputs);
    case HloOpcode::kMultiply:
      return mlir::stablehlo::MulOp::create(b, inputs[0], inputs[1]);
    case HloOpcode::kMaximum:
      return Maximum(b, inputs);
    case HloOpcode::kMinimum:
      return Minimum(b, inputs);
    case HloOpcode::kClamp:
      return Minimum(b, {Maximum(b, {inputs[0], inputs[1]}), inputs[2]});
    case HloOpcode::kAnd:
      return mlir::stablehlo::AndOp::create(b, inputs[0], inputs[1]);
    case HloOpcode::kOr:
      return mlir::stablehlo::OrOp::create(b, inputs[0], inputs[1]);
    case HloOpcode::kXor:
      return mlir::stablehlo::XorOp::create(b, inputs[0], inputs[1]);
    case HloOpcode::kDivide:
      return mlir::stablehlo::DivOp::create(b, inputs[0], inputs[1]);
    case HloOpcode::kCompare:
      return Compare(
          b, inputs,
          mlir::stablehlo::symbolizeComparisonDirection(
              ComparisonDirectionToString(hlo.comparison_direction()))
              .value());
    case HloOpcode::kSelect:
      return ma::SelectOp::create(
          b,
          Compare(b, {inputs[0], ZerosLike(b, inputs[0])},
                  mlir::stablehlo::ComparisonDirection::NE),
          inputs[1], inputs[2]);
    case HloOpcode::kReducePrecision:
      return mh::reducePrecision<mlir::tensor::BitcastOp>(
          b.getLoc(), inputs[0], hlo.exponent_bits(), hlo.mantissa_bits(), &b);
    case HloOpcode::kAcos:
      return mm::AcosOp::create(b, inputs[0]);
    case HloOpcode::kAcosh:
      return mm::AcoshOp::create(b, inputs[0]);
    case HloOpcode::kAsin:
      return mm::AsinOp::create(b, inputs[0]);
    case HloOpcode::kAsinh:
      return mm::AsinhOp::create(b, inputs[0]);
    case HloOpcode::kAtan2:
      return mm::Atan2Op::create(b, inputs[0], inputs[1]);
    case HloOpcode::kAtanh:
      return mm::AtanhOp::create(b, inputs[0]);
    case HloOpcode::kCos:
      return mm::CosOp::create(b, inputs[0]);
    case HloOpcode::kCosh:
      return mm::CoshOp::create(b, inputs[0]);
    case HloOpcode::kExp:
      return mm::ExpOp::create(b, inputs[0]);
    case HloOpcode::kErf:
      return mm::ErfOp::create(b, inputs[0]);
    case HloOpcode::kExpm1:
      return mm::ExpM1Op::create(b, inputs[0]);
    case HloOpcode::kLog:
      return mm::LogOp::create(b, inputs[0]);
    case HloOpcode::kLog1p:
      return mm::Log1pOp::create(b, inputs[0]);
    case HloOpcode::kPower:
      if (is_integer) {
        return mm::IPowIOp::create(b, inputs[0], inputs[1]);
      }
      return mm::PowFOp::create(b, inputs[0], inputs[1]);
    case HloOpcode::kRemainder:
      return mlir::stablehlo::RemOp::create(b, inputs[0], inputs[1]);
    case HloOpcode::kRsqrt:
      return mm::RsqrtOp::create(b, inputs[0]);
    case HloOpcode::kSin:
      return mm::SinOp::create(b, inputs[0]);
    case HloOpcode::kSinh:
      return mm::SinhOp::create(b, inputs[0]);
    case HloOpcode::kSqrt:
      return mm::SqrtOp::create(b, inputs[0]);
    case HloOpcode::kTan:
      return mm::TanOp::create(b, inputs[0]);
    case HloOpcode::kTanh:
      return mm::TanhOp::create(b, inputs[0]);
    case HloOpcode::kCbrt:
      return mm::CbrtOp::create(b, inputs[0]);
    case HloOpcode::kIsFinite:
      return mm::IsFiniteOp::create(b, inputs[0]);
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported elementwise operation ", hlo.ToString()));
  }
}

absl::StatusOr<mlir::TypedValue<mlir::RankedTensorType>> EmitConstant(
    mlir::ImplicitLocOpBuilder& b, const HloInstruction& constant) {
  TF_ASSIGN_OR_RETURN(
      Type ty, PrimitiveTypeToMlirType(b, constant.shape().element_type()));
  llvm::SmallVector<int64_t> shape{constant.shape().dimensions().begin(),
                                   constant.shape().dimensions().end()};

  if (constant.shape().AreAllLeavesIntegers()) {
    if (constant.shape().element_type() == U64) {
      return CreateConst(b, ty, ScalarConstantValue<uint64_t>(constant, U64),
                         shape);
    }
    return CreateConst(b, ty, ScalarConstantValue<int64_t>(constant, S64),
                       shape);
  }
  return CreateConst(b, ty, ScalarConstantValue<double>(constant, F64), shape);
}

Value Bitcast(mlir::ImplicitLocOpBuilder& b, Value value, Type type) {
  auto value_type = value.getType();
  value_type = mlir::dyn_cast<ShapedType>(value_type).clone(type);
  return mlir::arith::BitcastOp::create(b, value_type, value);
}

/*static */ absl::StatusOr<TileInfo> TileInfo::Construct(
    mlir::ImplicitLocOpBuilder& b, Value pid, ValueRange runtime_values,
    const TiledHloInstruction& tiled_hlo) {
  TF_ASSIGN_OR_RETURN(SmallVector<Value> offsets,
                      ComputeOffsetsForTile(b, pid, runtime_values, tiled_hlo));

  // Triton requires that all block dimensions are a power of 2.
  auto padded_tile_sizes = GetPaddedTileSizes(tiled_hlo.tile_sizes());
  SmallVector<int64_t> original_shape;
  original_shape.assign(tiled_hlo.hlo()->shape().dimensions().begin(),
                        tiled_hlo.hlo()->shape().dimensions().end());

  const Shape& shape = tiled_hlo.hlo()->shape();
  TF_ASSIGN_OR_RETURN(Type expected_element_type,
                      PrimitiveTypeToMlirType(b, shape.element_type()));
  auto storage_type = StorageType(expected_element_type);

  auto tile_strides = tiled_hlo.tile_strides();
  auto minor_to_major_layout = llvm::to_vector(LayoutUtil::MinorToMajor(shape));

  return TileInfo(offsets, tile_strides, original_shape, padded_tile_sizes,
                  minor_to_major_layout, storage_type);
}

TensorValue EmitParameterExtract(mlir::ImplicitLocOpBuilder& b,
                                 const TileInfo& tile_info, Value arg) {
  auto tensor_type = mlir::RankedTensorType::get(tile_info.padded_tile_sizes(),
                                                 tile_info.storage_type());

  return xla::xtile::ExtractTileOp::create(
      b, tensor_type, arg, tile_info.offsets(), tile_info.padded_tile_sizes(),
      tile_info.tile_strides());
}

absl::StatusOr<TensorValue> EmitScope(
    mlir::ImplicitLocOpBuilder& b,
    absl::Span<const HloInstruction* const> instructions,
    absl::flat_hash_map<const HloInstruction*, TensorValue>& values) {
  for (const HloInstruction* hlo : instructions) {
    TensorValue result;
    if (hlo->opcode() == HloOpcode::kConcatenate ||
        hlo->opcode() == HloOpcode::kDynamicSlice) {
      // Parameter loads and their concatenations are handled outside EmitScope.
      TF_RET_CHECK(values.contains(hlo)) << hlo->ToString();
      continue;
    }
    if (hlo->opcode() == HloOpcode::kParameter) {
      if (hlo->users()[0]->opcode() == HloOpcode::kConcatenate ||
          hlo->users()[0]->opcode() == HloOpcode::kDynamicSlice) {
        continue;
      }
      TF_RET_CHECK(values.contains(hlo)) << hlo->ToString();
      continue;
    }
    if (hlo->opcode() == HloOpcode::kBroadcast) {
      return absl::InvalidArgumentError(
          "Broadcast is not yet supported in EmitScope().");
    }
    if (hlo->opcode() == HloOpcode::kConstant) {
      TF_ASSIGN_OR_RETURN(result, EmitConstant(b, *hlo));
    } else if (HloInstruction::IsOpElementwise(hlo->opcode())) {
      std::vector<Value> operands;
      operands.reserve(hlo->operands().size());
      for (const HloInstruction* operand : hlo->operands()) {
        operands.push_back(values[operand]);
      }
      TF_ASSIGN_OR_RETURN(Value elementwise_result,
                          EmitElementwise(b, *hlo, operands));
      result = mlir::cast<TensorValue>(elementwise_result);
    } else if (hlo->opcode() == HloOpcode::kTuple) {
      TF_RET_CHECK(hlo->IsRoot()) << hlo->ToString();
    } else if (hlo->opcode() == HloOpcode::kBitcast ||
               hlo->opcode() == HloOpcode::kTranspose ||
               hlo->opcode() == HloOpcode::kSlice ||
               hlo->opcode() == HloOpcode::kReshape ||
               hlo->opcode() == HloOpcode::kPad) {
      // All these are currently supported only as operations on indices
      // which are pushed to loads and stores. No operations on tiles are
      // performed here.
      result = values[hlo->operand(0)];
    } else if (hlo->opcode() == HloOpcode::kFusion) {
      const auto* fusion_instruction = ::xla::Cast<HloFusionInstruction>(hlo);
      TF_ASSIGN_OR_RETURN(result,
                          EmitNestedFusion(b, *fusion_instruction, values));
    } else {
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported operation ", hlo->ToString()));
    }
    TF_RET_CHECK(values.insert({hlo, result}).second) << hlo->ToString();
    VLOG(8) << "Emitted " << hlo->ToString(HloPrintOptions::ShortParsable());
  }
  return values[instructions.back()];
}

TensorValue BroadcastInDims(mlir::ImplicitLocOpBuilder& b, TensorValue value,
                            ArrayRef<int64_t> output_shape,
                            ArrayRef<int64_t> dims) {
  CHECK(llvm::is_sorted(dims)) << "broadcast dims must be sorted";

  auto result_type = mlir::RankedTensorType::get(
      output_shape, value.getType().getElementType());

  return mlir::stablehlo::BroadcastInDimOp::create(b, result_type, value, dims);
}

TensorValue Splat(mlir::ImplicitLocOpBuilder& b, Value value,
                  ArrayRef<int64_t> output_shape) {
  auto tensor_value = mlir::dyn_cast<TensorValue>(value);
  if (!tensor_value) {
    tensor_value = mlir::tensor::FromElementsOp::create(
        b, mlir::RankedTensorType::get({}, value.getType()), value);
  }
  return BroadcastInDims(b, tensor_value, output_shape, /*dims=*/{});
}

Value UnsignedIntegerToSignlessInteger(mlir::OpBuilder& builder, Value value) {
  CHECK(getElementTypeOrSelf(value.getType()).isUnsignedInteger())
      << "Expected unsigned integer element type, got: "
      << ::xla::xtile::MlirToString(value.getType());
  Type signless_integer_type_type = mlir::IntegerType::get(
      builder.getContext(),
      getElementTypeOrSelf(value.getType()).getIntOrFloatBitWidth(),
      mlir::IntegerType::SignednessSemantics::Signless);
  if (auto shaped_type = mlir::dyn_cast<ShapedType>(value.getType())) {
    signless_integer_type_type =
        shaped_type.clone(shaped_type.getShape(), signless_integer_type_type);
  }
  return mlir::UnrealizedConversionCastOp::create(
             builder, value.getLoc(), signless_integer_type_type, value)
      .getResult(0);
}

}  // namespace xla::xtile
