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

#include "xla/backends/gpu/codegen/triton/fusion_emitter_legacy_matmul.h"

#include <array>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <queue>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "xla/backends/gpu/codegen/triton/dot_algorithms.h"
#include "xla/backends/gpu/codegen/triton/emitter_helpers.h"
#include "xla/codegen/emitter_loc_op_builder.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/mlir_hlo/mhlo/transforms/map_mhlo_to_scalar_op.h"
#include "xla/mlir_hlo/mhlo/transforms/transformation_helpers.h"
#include "xla/primitive_util.h"
#include "xla/service/algorithm_util.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/matmul_indexing_utils.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/model/tiled_hlo_computation.h"
#include "xla/service/gpu/triton_fusion_analysis.h"
#include "xla/service/gpu/triton_tiling_propagation.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/tma_metadata.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

namespace xla::gpu {

namespace ma = ::mlir::arith;
namespace mm = ::mlir::math;
namespace mt = ::mlir::triton;
namespace mh = ::mlir::mhlo;

using ::llvm::SmallVector;
using ::mlir::ArrayRef;
using ::mlir::ShapedType;
using ::mlir::Type;
using ::mlir::Value;
using ::mlir::ValueRange;

namespace {

const int kLhsIndex = 0;
const int kRhsIndex = 1;
const int kMetaIndex = 2;
const int kOutputIndex = 3;

// Returns the index of the operand in the fusion's parameter list. This
// might work better as a map from Scope->X, but it needs more refactoring.
size_t GetOperandIndex(TritonFusionAnalysis::Scope scope) {
  switch (scope) {
    case TritonFusionAnalysis::Scope::LHS:
      return kLhsIndex;
    case TritonFusionAnalysis::Scope::RHS:
      return kRhsIndex;
    case TritonFusionAnalysis::Scope::META:
      return kMetaIndex;
    case TritonFusionAnalysis::Scope::OUTPUT:
      return kOutputIndex;
  }
}

absl::StatusOr<Type> TritonType(EmitterLocOpBuilder& b, PrimitiveType t) {
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
    case S32:
      return b.getI32Type();
    case S16:
      return b.getI16Type();
    case PRED:
      return b.getI1Type();
    case S8:
      return b.getI8Type();
    case S4:
      return b.getI4Type();
    case F8E5M2:
      return b.getType<mlir::Float8E5M2Type>();
    case F8E4M3FN:
      return b.getType<mlir::Float8E4M3FNType>();
    default:
      return absl::UnimplementedError(
          absl::StrCat("This type is not supported yet: ",
                       primitive_util::LowercasePrimitiveTypeName(t)));
  }
}

Type StorageType(EmitterLocOpBuilder& b, Type t) {
  if (t.isInteger(/*width=*/1)) {
    return b.getI8Type();
  }
  return t;
}

// Create a scalar constant.
template <typename T>
ma::ConstantOp CreateConst(EmitterLocOpBuilder b, Type type, T value) {
  if (mlir::isa<mlir::IntegerType>(type)) {
    return b.create<ma::ConstantOp>(b.getIntegerAttr(type, value));
  }
  if (mlir::isa<mlir::FloatType>(type)) {
    return b.create<ma::ConstantOp>(
        b.getFloatAttr(type, static_cast<double>(value)));
  }
  LOG(FATAL) << "Constant type not supported: " << llvm_ir::DumpToString(type);
}

// Create a tensor constant.
template <typename T>
ma::ConstantOp CreateConst(EmitterLocOpBuilder b, Type type, T value,
                           llvm::ArrayRef<int64_t> shape) {
  auto tensor_type = mlir::RankedTensorType::get(shape, type);
  if (auto int_type = mlir::dyn_cast<mlir::IntegerType>(type)) {
    return b.create<ma::ConstantOp>(mlir::DenseElementsAttr::get(
        tensor_type,
        mlir::APInt(int_type.getIntOrFloatBitWidth(), value,
                    /*isSigned=*/std::is_signed_v<T>, /*implicitTrunc=*/true)));
  }
  if (auto float_type = mlir::dyn_cast<mlir::FloatType>(type)) {
    return b.create<ma::ConstantOp>(mlir::DenseElementsAttr::get(
        tensor_type, b.getFloatAttr(type, static_cast<double>(value))));
  }
  LOG(FATAL) << "Constant type not supported: " << llvm_ir::DumpToString(type);
}

Value ZerosLike(EmitterLocOpBuilder b, Value x) {
  if (auto src_shaped_ty = mlir::dyn_cast<ShapedType>(x.getType())) {
    Type src_ty = src_shaped_ty.getElementType();
    return CreateConst(b, src_ty, 0, src_shaped_ty.getShape());
  }
  return CreateConst(b, x.getType(), 0);
}

Value OnesLike(EmitterLocOpBuilder b, Value x) {
  if (auto src_shaped_ty = mlir::dyn_cast<ShapedType>(x.getType())) {
    Type src_ty = src_shaped_ty.getElementType();
    return CreateConst(b, src_ty, 1, src_shaped_ty.getShape());
  }
  return CreateConst(b, x.getType(), 1);
}

bool IsFp8Type(Type t) {
  return llvm::isa<mlir::Float8E5M2Type, mlir::Float8E4M3FNType,
                   mlir::Float8E5M2FNUZType, mlir::Float8E4M3FNUZType,
                   mlir::Float8E4M3B11FNUZType>(t);
}

Value Subtract(EmitterLocOpBuilder b, ValueRange values) {
  if (mlir::isa<mlir::IntegerType>(mlir::getElementTypeOrSelf(values[0]))) {
    return b.create<ma::SubIOp>(values[0], values[1]);
  } else {
    return b.create<ma::SubFOp>(values[0], values[1]);
  }
}

Value Compare(EmitterLocOpBuilder b, ValueRange values,
              mh::ComparisonDirection direction) {
  const Type type = mlir::getElementTypeOrSelf(values[0]);
  if (mlir::isa<mlir::IntegerType>(type)) {
    return b.create<ma::CmpIOp>(mh::impl::getCmpPredicate<ma::CmpIPredicate>(
                                    direction,
                                    /*isSigned=*/!type.isInteger(1))
                                    .value(),
                                values[0], values[1]);
  }
  return b.create<ma::CmpFOp>(
      mh::impl::getCmpPredicate<ma::CmpFPredicate>(direction,
                                                   /*isSigned=*/true)
          .value(),
      values[0], values[1]);
}

Value Maximum(EmitterLocOpBuilder b, const se::DeviceDescription& device_info,
              ValueRange values) {
  if (mlir::isa<mlir::FloatType>(mlir::getElementTypeOrSelf(values[0]))) {
    return b.create<ma::MaximumFOp>(values);
  }
  // logic: isNaN(lhs) || (!isNan(rhs) && lhs >= rhs) ? lhs : rhs
  // See also: IEEE Std 754-2008 5.11.
  //
  // This also works, but we wanted to make it similar to minimum.
  // logic: isNaN(lhs) || lhs >= rhs ? lhs : rhs
  Value lhs_is_nan =
      Compare(b, {values[0], values[0]}, mh::ComparisonDirection::NE);
  Value rhs_is_not_nan =
      Compare(b, {values[1], values[1]}, mh::ComparisonDirection::EQ);
  Value lhs_is_ge = Compare(b, values, mh::ComparisonDirection::GE);
  return b.create<ma::SelectOp>(
      b.create<ma::OrIOp>(lhs_is_nan,
                          b.create<ma::AndIOp>(rhs_is_not_nan, lhs_is_ge)),
      values[0], values[1]);
}

Value Minimum(EmitterLocOpBuilder b, const se::DeviceDescription& device_info,
              ValueRange values) {
  if (mlir::isa<mlir::FloatType>(mlir::getElementTypeOrSelf(values[0]))) {
    return b.create<ma::MinimumFOp>(values);
  }
  // logic: isNaN(lhs) || (!isNan(rhs) && lhs <= rhs) ? lhs : rhs
  // See also: IEEE Std 754-2008 5.11.
  //
  // This should also work, but the tests show that it doesn't work for
  // minimum(x, NaN):
  // logic: isNaN(lhs) || lhs <= rhs ? lhs : rhs
  Value lhs_is_nan =
      Compare(b, {values[0], values[0]}, mh::ComparisonDirection::NE);
  Value rhs_is_not_nan =
      Compare(b, {values[1], values[1]}, mh::ComparisonDirection::EQ);
  Value lhs_is_le = Compare(b, values, mh::ComparisonDirection::LE);
  return b.create<ma::SelectOp>(
      b.create<ma::OrIOp>(lhs_is_nan,
                          b.create<ma::AndIOp>(rhs_is_not_nan, lhs_is_le)),
      values[0], values[1]);
}

Value Splat(EmitterLocOpBuilder b, Value value, ArrayRef<int64_t> shape) {
  auto type = mlir::RankedTensorType::get(shape, value.getType());
  return b.create<mt::SplatOp>(type, value);
}

absl::StatusOr<Value> EmitElementwise(EmitterLocOpBuilder b,
                                      absl::string_view libdevice_path,
                                      const se::DeviceDescription& device_info,
                                      const HloInstruction& hlo,
                                      ValueRange inputs) {
  if (triton::IsSupportedElementwiseLibdeviceFunction(hlo)) {
    return triton::EmitElementwiseLibdeviceFunction(b, libdevice_path,
                                                    device_info, hlo, inputs);
  }
  const bool is_integer = mlir::isa<mlir::IntegerType>(
      mlir::getElementTypeOrSelf(inputs[0].getType()));

  switch (hlo.opcode()) {
    case HloOpcode::kCopy:
      // Dimension transformations are taken care of separately.
      return inputs[0];
    case HloOpcode::kAbs:
      if (is_integer) {
        return b.create<mm::AbsIOp>(inputs[0]);
      }
      return b.create<mm::AbsFOp>(inputs[0]);
    case HloOpcode::kCeil:
      return b.create<mm::CeilOp>(inputs[0]);
    case HloOpcode::kFloor:
      return b.create<mm::FloorOp>(inputs[0]);
    case HloOpcode::kNot:
      return b.create<ma::XOrIOp>(inputs[0], OnesLike(b, inputs[0]));
    case HloOpcode::kNegate:
      // NegFOp is not supported by Triton.
      return Subtract(b, {ZerosLike(b, inputs[0]), inputs[0]});
    case HloOpcode::kConvert: {
      TF_ASSIGN_OR_RETURN(Type dst_ty,
                          TritonType(b, hlo.shape().element_type()));
      return triton::Cast(b, inputs[0], dst_ty);
    }
    case HloOpcode::kAdd:
      if (is_integer) {
        return b.create<ma::AddIOp>(inputs[0], inputs[1]);
      }
      return b.create<ma::AddFOp>(inputs[0], inputs[1]);
    case HloOpcode::kSubtract:
      return Subtract(b, inputs);
    case HloOpcode::kMultiply:
      if (is_integer) {
        return b.create<ma::MulIOp>(inputs[0], inputs[1]);
      }
      return b.create<ma::MulFOp>(inputs[0], inputs[1]);
    case HloOpcode::kMaximum:
      return Maximum(b, device_info, inputs);
    case HloOpcode::kMinimum:
      return Minimum(b, device_info, inputs);
    case HloOpcode::kClamp:
      return Maximum(
          b, device_info,
          {Minimum(b, device_info, {inputs[1], inputs[2]}), inputs[0]});
    case HloOpcode::kAnd:
      return b.create<ma::AndIOp>(inputs[0], inputs[1]);
    case HloOpcode::kOr:
      return b.create<ma::OrIOp>(inputs[0], inputs[1]);
    case HloOpcode::kXor:
      return b.create<ma::XOrIOp>(inputs[0], inputs[1]);
    case HloOpcode::kDivide:
      if (is_integer) {
        // Unsigned not supported yet.
        return b.create<ma::DivSIOp>(inputs[0], inputs[1]);
      }
      return b.create<ma::DivFOp>(inputs[0], inputs[1]);
    case HloOpcode::kCompare:
      return Compare(
          b, inputs,
          mh::symbolizeComparisonDirection(
              ComparisonDirectionToString(hlo.comparison_direction()))
              .value());
    case HloOpcode::kSelect:
      return b.create<ma::SelectOp>(
          Compare(b, {inputs[0], ZerosLike(b, inputs[0])},
                  mh::ComparisonDirection::NE),
          inputs[1], inputs[2]);
    case HloOpcode::kReducePrecision:
      return mh::reducePrecision<mt::BitcastOp>(
          b.getLoc(), inputs[0], hlo.exponent_bits(), hlo.mantissa_bits(), &b);
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported elementwise operation ", hlo.ToString()));
  }
}

absl::StatusOr<Value> EmitConstant(EmitterLocOpBuilder b,
                                   const HloInstruction& constant) {
  CHECK_EQ(constant.opcode(), HloOpcode::kConstant);
  CHECK(ShapeUtil::IsEffectiveScalar(constant.shape()));

  TF_ASSIGN_OR_RETURN(Type ty, TritonType(b, constant.shape().element_type()));

  if (constant.shape().element_type() == U64) {
    TF_ASSIGN_OR_RETURN(Literal converted, constant.literal().Convert(U64));
    return CreateConst(b, ty, converted.GetFirstElement<uint64_t>());
  }

  if (constant.shape().AreAllLeavesIntegers()) {
    TF_ASSIGN_OR_RETURN(Literal converted, constant.literal().Convert(S64));
    return CreateConst(b, ty, converted.GetFirstElement<int64_t>());
  }

  TF_ASSIGN_OR_RETURN(Literal converted, constant.literal().Convert(F64));
  return CreateConst(b, ty, converted.GetFirstElement<double>());
}

using TensorValue = mlir::TypedValue<mlir::RankedTensorType>;

Value Broadcast(EmitterLocOpBuilder b, TensorValue value,
                ArrayRef<int64_t> shape) {
  return b.create<mt::BroadcastOp>(value.getType().clone(shape), value);
}

Value Range(EmitterLocOpBuilder b, int32_t limit) {
  auto type = mlir::RankedTensorType::get(limit, b.getI32Type());
  return b.create<mt::MakeRangeOp>(type, 0, limit);
}

Value AddPtr(EmitterLocOpBuilder b, Value ptr, Value offset) {
  return b.create<mt::AddPtrOp>(ptr.getType(), ptr, offset);
}

Value EmitParameterLoad(EmitterLocOpBuilder b, Value pointer,
                        ArrayRef<int32_t> boundary_checks) {
  if (mt::isTensorPointerType(pointer.getType())) {
    std::optional<mt::PaddingOption> padding;
    if (!boundary_checks.empty()) {
      padding = mt::PaddingOption::PAD_ZERO;
    }
    return b.create<mt::LoadOp>(pointer, boundary_checks, padding,
                                mt::CacheModifier::NONE,
                                mt::EvictionPolicy::NORMAL,
                                /*isVolatile=*/false);
  }

  // EmitTensorPointer will not create a MakeTensorPtrOp for scalars.
  return Splat(b,
               b.create<mt::LoadOp>(pointer, mt::CacheModifier::NONE,
                                    mt::EvictionPolicy::NORMAL,
                                    /*isVolatile=*/false),
               {});
}

// Grouped properties of tiled dimensions used to generate block pointers.
struct DimProperties {
  DimProperties(int64_t index, Value pid, int block_size, int split_value)
      : index(index),
        pid(pid),
        block_size(block_size),
        split_value(split_value) {}

  // Logical index of the dimension at the tiling-defining operation.
  int64_t index;
  // Block program ID corresponding to this dimension.
  Value pid;
  // Elements of the dimension to process per block program.
  int block_size;
  // Size of the major part of the dimension if it's split into two parts.
  int split_value;
};

struct Side {
  explicit Side(TritonFusionAnalysis::Scope scope,
                std::vector<DimProperties> tiled_dims = {},
                std::optional<int64_t> batch_dim_idx = std::nullopt)
      : scope(scope), tiled_dims(tiled_dims), batch_dim_idx(batch_dim_idx) {}
  TritonFusionAnalysis::Scope scope;
  std::vector<DimProperties> tiled_dims;
  std::optional<int64_t> batch_dim_idx;
};

absl::StatusOr<Value> EmitBroadcast(EmitterLocOpBuilder b,
                                    const TritonFusionAnalysis* analysis,
                                    const Side& side,
                                    const HloInstruction& broadcast,
                                    Value input) {
  TF_RET_CHECK(analysis != nullptr);
  std::vector<int64_t> out_shape;

  auto tensor_input = mlir::dyn_cast<TensorValue>(input);

  // The broadcasted dimension could be a non-trivial like broadcast + bitcast.
  // For example:
  // s8[2,256,128]broadcast(s8[2,128])
  // s8[512,128]bitcast(s8[2,256,128])
  // I.e. we broadcast the first dimension from 2 to 512.
  // When this is the case we don't need to expand the tile because it is
  // already 2d but we still need to broadcast it.
  bool non_trivial_broadcast = false;

  for (const DimProperties& dim : side.tiled_dims) {
    const TensorIterationSpec::DimIterationSpec* spec =
        analysis->IterSpec(side.scope, &broadcast, dim.index);
    if (spec != nullptr && spec->at(0).stride > 0) {
      out_shape.push_back(dim.block_size);
      non_trivial_broadcast |= spec->at(0).subfragments.size() != 1;
    }
  }

  if (!tensor_input) {
    // Input is scalar.
    return Splat(b, input, out_shape);
  }
  if (!non_trivial_broadcast &&
      tensor_input.getType().getRank() == out_shape.size()) {
    // No dimensions to broadcast.
    return input;
  }
  // Add broadcasted dimensions one by one.
  Value expanded_input = tensor_input;
  int dim_idx = 0;
  for (const DimProperties& dim : side.tiled_dims) {
    const auto* output_spec =
        analysis->IterSpec(side.scope, &broadcast, dim.index);
    if (output_spec != nullptr && output_spec->at(0).stride > 0) {
      const auto* input_spec =
          analysis->IterSpec(side.scope, broadcast.operand(0), dim.index);
      // A dimension is broadcasted if it's either absent in the input or
      // if its size is increased from the input to the output.
      if (tensor_input.getType().getRank() != out_shape.size()) {
        if (input_spec == nullptr ||
            output_spec->at(0).count > input_spec->at(0).count) {
          expanded_input = b.create<mt::ExpandDimsOp>(expanded_input, dim_idx);
        }
      }
      ++dim_idx;
    }
  }
  return Broadcast(b, mlir::cast<TensorValue>(expanded_input), out_shape);
}

// Emit sequence of instructions using compatible tiling ordered producers
// before consumers.
absl::StatusOr<Value> EmitScope(
    EmitterLocOpBuilder b, absl::string_view libdevice_path,
    const se::DeviceDescription& device_info,
    const TritonFusionAnalysis* analysis, const Side& side,
    absl::Span<const HloInstruction* const> instructions,
    absl::flat_hash_map<const HloInstruction*, Value>& values) {
  for (const HloInstruction* hlo : instructions) {
    Value result;
    if (hlo->opcode() == HloOpcode::kConvert &&
        hlo->operand(0)->shape().element_type() == S4) {
      Value unpacked;
      unpacked = triton::Cast(b, values[hlo->operand(0)], b.getI8Type());
      std::vector<Value> operands({unpacked});
      TF_ASSIGN_OR_RETURN(result, EmitElementwise(b, libdevice_path,
                                                  device_info, *hlo, operands));
    } else if (hlo->opcode() == HloOpcode::kConcatenate ||
               hlo->opcode() == HloOpcode::kDynamicSlice) {
      // Parameter loads and their concatenations are handled outside EmitScope.
      TF_RET_CHECK(values.contains(hlo)) << hlo->ToString();
      continue;
    } else if (hlo->opcode() == HloOpcode::kParameter) {
      if (hlo->users()[0]->opcode() == HloOpcode::kConcatenate ||
          hlo->users()[0]->opcode() == HloOpcode::kDynamicSlice) {
        continue;
      }
      TF_RET_CHECK(values.contains(hlo)) << hlo->ToString();
      continue;
    } else if (hlo->opcode() == HloOpcode::kConstant) {
      TF_ASSIGN_OR_RETURN(Value constant, EmitConstant(b, *hlo));
      // Splat makes it a tensor to avoid type mismatches.
      result = Splat(b, constant, {});
    } else if (hlo->opcode() == HloOpcode::kBroadcast) {
      TF_ASSIGN_OR_RETURN(result, EmitBroadcast(b, analysis, side, *hlo,
                                                values[hlo->operand(0)]));
    } else if (HloInstruction::IsOpElementwise(hlo->opcode())) {
      std::vector<Value> operands;
      operands.reserve(hlo->operands().size());
      for (const HloInstruction* operand : hlo->operands()) {
        operands.push_back(values[operand]);
      }
      TF_ASSIGN_OR_RETURN(result, EmitElementwise(b, libdevice_path,
                                                  device_info, *hlo, operands));
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
    } else {
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported operation ", hlo->ToString()));
    }
    TF_RET_CHECK(values.insert({hlo, result}).second) << hlo->ToString();
    VLOG(8) << "Emitted " << hlo->ToString(HloPrintOptions::ShortParsable());
  }
  return values[instructions.back()];
}

const TensorIterationSpec::DimIterationSpec* GetLhsNoncontractingSplitSpec(
    const TritonFusionAnalysis& analysis, int64_t lhs_noncontracting_dim_idx) {
  const TensorIterationSpec::DimIterationSpec* result = nullptr;
  for (const HloInstruction* lhs_param :
       analysis.ScopeParameters(TritonFusionAnalysis::Scope::LHS)) {
    const TensorIterationSpec::DimIterationSpec* spec =
        analysis.IterSpec(TritonFusionAnalysis::Scope::LHS, lhs_param,
                          lhs_noncontracting_dim_idx);
    if (spec != nullptr && spec->size() > 1) {
      CHECK_EQ(spec->size(), 2);
      if (result != nullptr) {
        CHECK_EQ(result->at(0).count, spec->at(0).count);
        CHECK_EQ(result->at(1).count, spec->at(1).count);
      }
      result = spec;
    }
  }
  return result;
}

// Structure for parameters relating to the MatMul shape and dimension indices.
//
// Variable naming: lhs [m, k] x rhs [k, n] -> out [m, n].
//
// The logical output dimensions are always ordered as:
//   split-K, batch, non-contracting LHS, non-contracting RHS,
// where split-K and batch are optional.
struct MatMulDims {
  static absl::StatusOr<MatMulDims> Create(
      const TritonGemmConfig& config, const HloDotInstruction& dot,
      const TritonFusionAnalysis& analysis);

  std::optional<int> out_split_k_dim_idx = std::nullopt;

  std::optional<int> lhs_batch_dim_idx = std::nullopt;
  std::optional<int> rhs_batch_dim_idx = std::nullopt;
  std::optional<int> out_batch_dim_idx = std::nullopt;

  // The LHS non-contracting can be split into two.
  std::optional<int64_t> lhs_noncontracting_split = std::nullopt;

  int lhs_contracting_dim_idx;
  int lhs_noncontracting_dim_idx;
  int rhs_contracting_dim_idx;
  int rhs_noncontracting_dim_idx;
  // The index of the LHS noncontracting dim in the output.
  int out_lhs_noncontracting_dim_idx;
  // The index of the RHS noncontracting dim in the output.
  int out_rhs_noncontracting_dim_idx;

  int64_t m;
  int64_t n;
  int64_t k;
  TritonGemmConfig config;

  std::string ToString() const {
    return absl::StrCat("MxNxK: ", m, "x", n, "x", k,
                        " contracting: lhs=", lhs_contracting_dim_idx,
                        " rhs=", rhs_contracting_dim_idx);
  }

 private:
  MatMulDims() = default;
};

// Structure for parameters relating to the MatMul launch grid.
struct MatMulLaunchConfig {
  explicit MatMulLaunchConfig(const TritonGemmConfig& config,
                              const HloDotInstruction& dot,
                              const MatMulDims& dims,
                              const se::DeviceDescription& device_info);

  int64_t grid_m;
  int64_t grid_n;
  LaunchDimensions launch_dims;
  mt::ProgramIDDim batch_program_id_dim;
  mt::ProgramIDDim noncontracting_program_id_dim;
};

ma::ConstantOp Cst(EmitterLocOpBuilder b, Type index_ty, int64_t v) {
  return CreateConst(b, index_ty, v);
}

AutotuneResult::TritonGemmKey DefaultTritonGemmKey() {
  AutotuneResult::TritonGemmKey triton_gemm_key;
  triton_gemm_key.set_block_m(64);
  triton_gemm_key.set_block_k(64);
  triton_gemm_key.set_block_n(64);
  triton_gemm_key.set_split_k(1);
  triton_gemm_key.set_num_stages(1);
  triton_gemm_key.set_num_warps(2);
  triton_gemm_key.set_num_ctas(1);
  return triton_gemm_key;
}

ma::ConstantOp Cst32(EmitterLocOpBuilder b, int32_t v) {
  return CreateConst(b, b.getI32Type(), v);
}

ma::ConstantOp Cst64(EmitterLocOpBuilder b, int64_t v) {
  return CreateConst(b, b.getI64Type(), v);
}

/*static*/ absl::StatusOr<MatMulDims> MatMulDims::Create(
    const TritonGemmConfig& config, const HloDotInstruction& dot,
    const TritonFusionAnalysis& analysis) {
  MatMulDims matmul_dims;
  matmul_dims.config = config;
  if (config.split_k > 1) {
    // split-k is always the first logical dimension.
    matmul_dims.out_split_k_dim_idx = 0;
  }

  int64_t num_split_k_dims = config.split_k > 1 ? 1 : 0;
  const auto& dims = dot.dot_dimension_numbers();
  matmul_dims.lhs_contracting_dim_idx = dims.lhs_contracting_dimensions(0);
  matmul_dims.lhs_noncontracting_dim_idx =
      GetNonContractingDims(dot.operand(0)->shape(),
                            dims.lhs_batch_dimensions(),
                            dims.lhs_contracting_dimensions())
          .value()[0];
  matmul_dims.rhs_contracting_dim_idx = dims.rhs_contracting_dimensions(0);
  matmul_dims.rhs_noncontracting_dim_idx =
      GetNonContractingDims(dot.operand(1)->shape(),
                            dims.rhs_batch_dimensions(),
                            dims.rhs_contracting_dimensions())
          .value()[0];

  if (dims.lhs_batch_dimensions_size() > num_split_k_dims) {
    matmul_dims.lhs_batch_dim_idx = *dims.lhs_batch_dimensions().rbegin();
    matmul_dims.rhs_batch_dim_idx = *dims.rhs_batch_dimensions().rbegin();
    // The batch dimension (if present) comes after the split-k dimension (if
    // present, otherwise it's the first dimension).
    matmul_dims.out_batch_dim_idx = num_split_k_dims;
  }

  // Logical output dimensions are always ordered as:
  //   split-K, batch, non-contracting LHS, non-contracting RHS,
  // where split-K and batch are optional.
  matmul_dims.out_rhs_noncontracting_dim_idx =
      dot.shape().dimensions_size() - 1;
  matmul_dims.out_lhs_noncontracting_dim_idx =
      dot.shape().dimensions_size() - 2;

  auto* root = dot.parent()->root_instruction();
  auto iter_spec =
      analysis.IterSpec(TritonFusionAnalysis::Scope::OUTPUT, root,
                        matmul_dims.out_rhs_noncontracting_dim_idx);
  TF_RET_CHECK(iter_spec != nullptr);
  matmul_dims.n = iter_spec->at(0).count;
  // Contracting dimension length.
  if (config.split_k > 1 &&
      dot.operand(1)->operand(0)->opcode() == HloOpcode::kPad) {
    // Unpadded LHS shape:  [..., k, ...]
    // Padded LHS shape:    [..., padded_k, ...]
    // Bitcasted LHS shape: [..., split_k, padded_k / split_k, ...]
    TF_RET_CHECK(dot.operand(1)->opcode() == HloOpcode::kBitcast);
    const Shape& unpadded_rhs_shape =
        dot.operand(1)->operand(0)->operand(0)->shape();
    matmul_dims.k =
        unpadded_rhs_shape.dimensions(dims.rhs_contracting_dimensions(0) - 1);
  } else {
    matmul_dims.k =
        dot.operand(1)->shape().dimensions(dims.rhs_contracting_dimensions(0)) *
        config.split_k;
  }

  auto* lhs_noncontracting_split_spec = GetLhsNoncontractingSplitSpec(
      analysis, matmul_dims.lhs_noncontracting_dim_idx);
  if (lhs_noncontracting_split_spec != nullptr) {
    // Just the fastest-varying part of it if the dimension is split.
    matmul_dims.m = lhs_noncontracting_split_spec->at(0).count;
    matmul_dims.lhs_noncontracting_split =
        lhs_noncontracting_split_spec->at(1).count;
  } else {
    matmul_dims.m = analysis
                        .IterSpec(TritonFusionAnalysis::Scope::OUTPUT, root,
                                  matmul_dims.out_lhs_noncontracting_dim_idx)
                        ->at(0)
                        .count;
  }

  // For now split non-contracting and batch are not supported
  // simultaneously because they are implemented via same mechanism.
  TF_RET_CHECK(!(matmul_dims.out_batch_dim_idx.has_value() &&
                 matmul_dims.lhs_noncontracting_split.has_value()));

  TF_RET_CHECK(matmul_dims.m >= 1);
  TF_RET_CHECK(matmul_dims.n >= 1);
  return std::move(matmul_dims);
}

MatMulLaunchConfig::MatMulLaunchConfig(const TritonGemmConfig& config,
                                       const HloDotInstruction& dot,
                                       const MatMulDims& dims,
                                       const se::DeviceDescription& device_info)
    : grid_m((dims.m + config.block_m - 1) / config.block_m),
      grid_n((dims.n + config.block_n - 1) / config.block_n) {
  int64_t batch_size = dims.lhs_noncontracting_split.value_or(
      dims.out_batch_dim_idx.has_value()
          ? dot.shape().dimensions(*dims.out_batch_dim_idx)
          : 1);
  // X block size is 32-bit, Y and Z are 16-bit. Use X for large dimensions.
  constexpr int64_t kBlockCountYZLimit = 65536;

  // In the imaginary situation where both batch size and grid_m * grid_n
  // are over 65535 we have to give up. Given the minimal m, n block sizes of 16
  // this requires at least 256 GB of output.
  CHECK_LT(batch_size * grid_m * grid_n,
           kBlockCountYZLimit * kBlockCountYZLimit);

  const bool large_batch = batch_size >= kBlockCountYZLimit;
  if (large_batch) {
    batch_program_id_dim = mt::ProgramIDDim::X;
    noncontracting_program_id_dim = mt::ProgramIDDim::Y;
    launch_dims = LaunchDimensions(
        se::BlockDim(batch_size, grid_m * grid_n, config.split_k),
        se::ThreadDim(config.num_warps * WarpSize(device_info), 1, 1));
  } else {
    batch_program_id_dim = mt::ProgramIDDim::Y;
    noncontracting_program_id_dim = mt::ProgramIDDim::X;
    launch_dims = LaunchDimensions(
        se::BlockDim(grid_m * grid_n, batch_size, config.split_k),
        se::ThreadDim(config.num_warps * WarpSize(device_info), 1, 1));
  }
}

absl::Status ValidateMatMulConfig(const TritonGemmConfig& config,
                                  const HloDotInstruction& dot) {
  TF_RET_CHECK(config.split_k >= 1);
  TF_RET_CHECK(config.block_m >= 16);
  TF_RET_CHECK(config.block_k >= 16);
  TF_RET_CHECK(config.block_n >= 16);

  const auto& dims = dot.dot_dimension_numbers();
  int num_batch_dims =
      dims.lhs_batch_dimensions_size() - (config.split_k > 1 ? 1 : 0);
  TF_RET_CHECK(num_batch_dims <= 1);
  if (config.split_k > 1) {
    // Split-K dimension has to be the first batch one and have an index
    // just before the contracting one.
    const int lhs_split_k_dim_idx = dims.lhs_contracting_dimensions(0) - 1;
    const int rhs_split_k_dim_idx = dims.rhs_contracting_dimensions(0) - 1;
    // Size of this dimension has to match the split_k value.
    TF_RET_CHECK(dims.lhs_batch_dimensions(0) == lhs_split_k_dim_idx);
    TF_RET_CHECK(dims.rhs_batch_dimensions(0) == rhs_split_k_dim_idx);
    TF_RET_CHECK(config.split_k ==
                 dot.operand(0)->shape().dimensions(lhs_split_k_dim_idx));
    TF_RET_CHECK(config.split_k ==
                 dot.operand(1)->shape().dimensions(rhs_split_k_dim_idx));
  }

  // Rely on dot decomposer: there is just one contracting and one
  // non-contracting dimension on each side + batch ones optionally.
  TF_RET_CHECK(dims.lhs_contracting_dimensions_size() == 1);
  TF_RET_CHECK(dims.rhs_contracting_dimensions_size() == 1);

  TF_RET_CHECK(dot.operand(0)->shape().dimensions_size() ==
               2 + (config.split_k > 1 ? 1 : 0) + num_batch_dims);
  return absl::OkStatus();
}

// if (index < limits[0]) {
//   return choices[0];
// } else if (index < limits[1]) {
//   return choices[1];
// } else if (...) {
// ...
// } else {
//   return choices.back();
// }
absl::StatusOr<Value> EmitMultiSelect(EmitterLocOpBuilder& b, Value index,
                                      ValueRange limits, ValueRange choices) {
  TF_RET_CHECK(choices.size() - 1 == limits.size());
  Value result = choices[0];
  for (int i = 0; i < choices.size() - 1; ++i) {
    result = b.create<ma::SelectOp>(
        b.create<ma::CmpIOp>(ma::CmpIPredicate::slt, index, limits[i]), result,
        choices[i + 1]);
  }
  return result;
}

absl::Status UncompilableMatmul(absl::string_view explanation) {
  absl::Status s = absl::CancelledError(explanation);
  s.SetPayload(kUncompilableFusion, absl::Cord(explanation));
  return s;
}

bool IsFp8Matmul(const HloDotInstruction* dot) {
  return primitive_util::IsF8Type(dot->operand(0)->shape().element_type()) &&
         primitive_util::IsF8Type(dot->operand(1)->shape().element_type());
}

class MatMulEmitterHelper {
 public:
  MatMulEmitterHelper(absl::string_view libdevice_path,
                      const se::DeviceDescription& device_info,
                      const HloDotInstruction* dot_instr, Type index_ty,
                      MatMulDims dims, const MatMulLaunchConfig& launch_config,
                      const TritonFusionAnalysis& analysis)
      : libdevice_path_(libdevice_path),
        device_info_(device_info),
        dot_instr_(dot_instr),
        index_ty_(index_ty),
        analysis_(analysis),
        dims_(dims),
        launch_config_(launch_config) {}

  std::vector<const HloInstruction*> EpiloguePostOrderTransitiveOperands(
      const HloInstruction* root) {
    // Collect all instructions of the dot's output scope.
    absl::flat_hash_set<const HloInstruction*> to_order;
    {
      std::queue<const HloInstruction*> to_add;
      if (root != dot_instr_) {
        to_add.push(root);
      }
      while (!to_add.empty()) {
        const HloInstruction* current = to_add.front();
        for (const HloInstruction* operand : current->operands()) {
          if (!to_order.contains(operand)) {
            if (operand != dot_instr_) {
              to_add.push(operand);
            }
          }
        }
        to_order.insert(current);
        to_add.pop();
      }
    }
    // Order them producers before consumers.
    std::vector<const HloInstruction*> to_emit;
    for (const HloInstruction* hlo :
         dot_instr_->parent()->MakeInstructionPostOrder()) {
      if (to_order.contains(hlo)) {
        to_emit.push_back(hlo);
      }
    }
    return to_emit;
  }

  Value MakeInput(EmitterLocOpBuilder b, const Side& side,
                  int64_t operand_index,
                  absl::flat_hash_map<const HloInstruction*, Value>& values) {
    return *EmitScope(
        b, libdevice_path_, device_info_, &analysis_, side,
        dot_instr_->parent()->MakeInstructionPostOrderFrom(
            const_cast<HloInstruction&>(*dot_instr_->operand(operand_index))),
        values);
  }

  int64_t GetNonContractingDimIdxForOperandScope(
      TritonFusionAnalysis::Scope scope) {
    if (scope == TritonFusionAnalysis::Scope::LHS) {
      return dims_.lhs_noncontracting_dim_idx;
    } else if (scope == TritonFusionAnalysis::Scope::RHS) {
      return dims_.rhs_noncontracting_dim_idx;
    } else {
      CHECK(false) << "This shouldn't be called for the other scopes.";
    }
  }

  bool IsNonTrivialTiledDimension(TritonFusionAnalysis::Scope scope,
                                  int64_t dim_index) {
    switch (scope) {
      case TritonFusionAnalysis::Scope::LHS:
        return (dim_index == dims_.lhs_noncontracting_dim_idx && dims_.m > 1) ||
               (dim_index == dims_.lhs_contracting_dim_idx && dims_.k > 1);
      case TritonFusionAnalysis::Scope::RHS:
        return (dim_index == dims_.rhs_noncontracting_dim_idx && dims_.n > 1) ||
               (dim_index == dims_.rhs_contracting_dim_idx && dims_.k > 1);
      case TritonFusionAnalysis::Scope::OUTPUT:
        return (dim_index == dims_.out_lhs_noncontracting_dim_idx &&
                dims_.m > 1) ||
               (dim_index == dims_.out_rhs_noncontracting_dim_idx &&
                dims_.n > 1);
      default:
        break;
    }
    return false;
  }

  bool NonTrivialTiledDimensionHasNoIterationAtParameter(
      TritonFusionAnalysis::Scope scope, const HloInstruction& hlo,
      int64_t dim_index) {
    const TensorIterationSpec::DimIterationSpec* spec =
        analysis_.IterSpec(scope, &hlo, dim_index);
    return spec == nullptr ||
           (IsNonTrivialTiledDimension(scope, dim_index) && spec->size() == 1 &&
            (spec->at(0).count <= 1 || spec->at(0).stride == 0));
  }

  // Returns the increments necessary to advance the pointer or offset of the
  // given side & hlo instruction.
  SmallVector<Value> EmitIncrements(EmitterLocOpBuilder b, const Side& side,
                                    const HloInstruction& hlo,
                                    int64_t contracting_dimension, Value ki,
                                    int64_t block_k) {
    SmallVector<Value> increments;
    for (const DimProperties& dim : side.tiled_dims) {
      if (NonTrivialTiledDimensionHasNoIterationAtParameter(side.scope, hlo,
                                                            dim.index)) {
        continue;
      }
      // Only the contracting dimensions are advanced.
      if (dim.index == contracting_dimension) {
        const TensorIterationSpec::DimIterationSpec* spec =
            analysis_.IterSpec(side.scope, &hlo, dim.index);
        int64_t broadcast = spec->at(0).broadcast_multiplier;
        if (broadcast > 1) {
          if (broadcast != block_k) {
            // If the broadcast multiplier is not equal to the block_k, we need
            // to compute the increment conditionally. It has to advance by 1
            // if the current block is the last one in the broadcasted fragment,
            // and by 0 otherwise. Advance by 1 is computed as:
            // ((ki + block_k) / broadcast) * broadcast == ki + block_k.
            Value one = Cst32(b, 1);
            Value zero = Cst32(b, 0);
            Value add = b.create<ma::AddIOp>(ki, Cst32(b, block_k));
            Value div = b.create<ma::DivSIOp>(add, Cst32(b, broadcast));
            Value mul = b.create<ma::MulIOp>(div, Cst32(b, broadcast));
            Value cond = b.create<ma::CmpIOp>(ma::CmpIPredicate::eq, mul, add);
            Value one_or_zero = b.create<ma::SelectOp>(cond, one, zero);
            increments.push_back(one_or_zero);
          } else {
            // If the broadcast multiplier is equal to the block_k, we can
            // advance by 1 unconditionally.
            increments.push_back(Cst32(b, 1));
          }
        } else {
          increments.push_back(Cst32(b, dim.block_size * dim.split_value));
        }
      } else {
        increments.push_back(Cst32(b, 0));
      }
    }
    return increments;
  }

  // Return the batch stride of the HLO passed as a parameter. If the
  // parameter HLO has no batch dimension, a zero stride is returned.
  // Also sets offset_batch and updates has_batch_offset as a side effect.
  absl::StatusOr<Value> GetBatchStride(EmitterLocOpBuilder b, const Side& side,
                                       const HloInstruction* hlo_param,
                                       int64_t& offset_batch,
                                       bool& has_batch_offset) {
    int64_t stride_batch = 0;
    if (side.scope != TritonFusionAnalysis::Scope::RHS &&
        dims_.lhs_noncontracting_split) {
      const TensorIterationSpec::DimIterationSpec* spec =
          analysis_.IterSpec(side.scope, hlo_param, side.tiled_dims[0].index);
      if (spec != nullptr) {
        if (spec->size() > 1) {
          // Support one specific kind of output transpose that splits the
          // dimension originating from the split LHS non-contracting one.
          stride_batch = spec->at(1).stride;
        } else {
          // Because the major part of the split is implemented using the
          // batch logic stride_batch is populated here as the stride of
          // the minor part times its size.
          stride_batch = spec->at(0).stride *
                         (spec->at(0).count / *dims_.lhs_noncontracting_split);
        }
        TF_RET_CHECK(stride_batch != 0);
      }
    } else if (side.batch_dim_idx.has_value()) {
      const TensorIterationSpec::DimIterationSpec* spec =
          analysis_.IterSpec(side.scope, hlo_param, *side.batch_dim_idx);
      if (spec != nullptr) {
        stride_batch = spec->at(0).stride;
        offset_batch = spec->at(0).slice_start;
        TF_RET_CHECK(stride_batch != 0);
      }
    }

    has_batch_offset |= stride_batch != 0;
    return Cst(b, index_ty_, stride_batch);
  }

  // bases: The base pointers of each argument.
  absl::StatusOr<Value> EmitTensorPointer(
      EmitterLocOpBuilder b, const HloInstruction* hlo, const Side& side,
      const ValueRange& bases, Value pid_k,
      std::vector<int32_t>& boundary_checks) {
    Value base;

    // Concatenations of parameters are handled during generation of block
    // pointers because of a limitation of implementation of block pointers
    // in the Triton compiler: block pointers are not supported inside
    // conditionals.
    // Therefore instead of directly using a conditional to emit a concatenation
    // and emitting its inputs inside the cases a single block pointer is
    // emitted for all inputs, but all its properties (base, strides etc) get
    // generated conditionally on the position of the current thread block
    // within the concatenated dimension.
    ConcatParams concat_params;

    if (hlo->opcode() == HloOpcode::kConcatenate) {
      TF_ASSIGN_OR_RETURN(
          std::tie(concat_params, base),
          CalculateConcatParamsAndUpdateBase(b, hlo, side, bases));
    } else {
      concat_params.dim_idx = -1;
      base = bases[0];
    }

    // Parameters of MakeTensorPtrOp to be generated by this function.
    TensorParams tensor_params;
    for (const DimProperties& dim : side.tiled_dims) {
      TF_RETURN_IF_ERROR(AddDimToTensorParams(b, hlo, side, dim, concat_params,
                                              bases, boundary_checks,
                                              tensor_params));
    }

    TF_ASSIGN_OR_RETURN(base, UpddateBaseForBatchAndConcatCases(
                                  b, hlo, concat_params, side, base));

    if (dims_.out_split_k_dim_idx.has_value()) {
      TF_ASSIGN_OR_RETURN(base,
                          UpdateBaseForSplitKCase(b, hlo, side, base, pid_k));
    }

    if (tensor_params.block_dims.empty()) {
      // Load of a scalar.
      return base;
    }
    auto tensor_ptr = mlir::cast<Value>(
        b.create<mt::MakeTensorPtrOp>(
             base, tensor_params.bounds, tensor_params.strides,
             tensor_params.tensor_offsets, tensor_params.block_dims,
             tensor_params.dim_order)
            .getResult());
    tensor_ptr = b.create<mt::AdvanceOp>(tensor_ptr.getType(), tensor_ptr,
                                         tensor_params.block_offsets);
    return tensor_ptr;
  }

 private:
  struct ConcatParams {
    // Index of concatenated dimension if present, -1 otherwise.
    int dim_idx;
    // Offsets along the concatenated dimension at which operands change.
    std::vector<Value> boundaries;
    // Block index along the concatenated dimension * block size.
    Value dim_pid_offset;
  };

  struct TensorParams {
    std::vector<Value> bounds;
    std::vector<Value> strides;

    // Offsets from tensor origin, same for all thread blocks.
    std::vector<Value> tensor_offsets;
    std::vector<int32_t> block_dims;
    std::vector<int32_t> dim_order;

    // Offsets for a given thread block, typically pid * block size.
    // Used in a one-off AdvanceOp applied to the generated MakeTensorPtrOp.
    std::vector<Value> block_offsets;
  };

  absl::Status AddDimToTensorParams(EmitterLocOpBuilder b,
                                    const HloInstruction* hlo, const Side& side,
                                    const DimProperties& properties,
                                    const ConcatParams& concat_params,
                                    const ValueRange& bases,
                                    std::vector<int32_t>& boundary_checks,
                                    TensorParams& tensor_params) {
    if (NonTrivialTiledDimensionHasNoIterationAtParameter(side.scope, *hlo,
                                                          properties.index)) {
      // If a non-trivial tiled dimension has only one element at
      // the parameter, it's being broadcasted. Skip it in the tensor
      // pointer to prevent it from being padded to the tile size on load
      // instead of being broadcasted.
      return absl::OkStatus();
    }
    Value pid_offset;
    if (properties.pid == nullptr) {
      pid_offset = Cst32(b, 0);
    } else {
      pid_offset =
          b.create<ma::MulIOp>(properties.pid, Cst32(b, properties.block_size));
    }
    std::vector<const HloInstruction*> inputs;
    if (hlo->opcode() == HloOpcode::kConcatenate) {
      inputs.insert(inputs.end(), hlo->operands().cbegin(),
                    hlo->operands().cend());
    } else {
      inputs = {hlo};
    }
    std::vector<const TensorIterationSpec::DimIterationSpec*> specs;
    std::vector<Value> input_strides;
    std::vector<Value> input_offsets;
    std::vector<Value> input_bounds;
    specs.reserve(inputs.size());
    input_strides.reserve(inputs.size());
    input_offsets.reserve(inputs.size());
    input_bounds.reserve(inputs.size());
    for (const HloInstruction* input : inputs) {
      auto* spec = analysis_.IterSpec(side.scope, input, properties.index);
      specs.push_back(spec);
      auto dim_spec = spec->at(0);
      input_strides.push_back(Cst64(b, dim_spec.stride));
      input_offsets.push_back(
          b.create<ma::AddIOp>(pid_offset, Cst32(b, dim_spec.slice_start)));
      input_bounds.push_back(Cst64(b, dim_spec.count));
    }
    {
      TF_ASSIGN_OR_RETURN(
          Value select_input_strides,
          EmitMultiSelect(b, concat_params.dim_pid_offset,
                          concat_params.boundaries, input_strides));
      tensor_params.strides.push_back(select_input_strides);
    }

    if (properties.index == concat_params.dim_idx) {
      TF_RETURN_IF_ERROR(AddConcatDimToTensorParams(
          b, hlo, side, properties, concat_params, pid_offset, specs.front(),
          input_offsets, input_bounds, tensor_params));
    } else if (hlo->opcode() == HloOpcode::kDynamicSlice &&
               (side.scope == TritonFusionAnalysis::Scope::LHS ||
                side.scope == TritonFusionAnalysis::Scope::RHS) &&
               properties.index ==
                   GetNonContractingDimIdxForOperandScope(side.scope)) {
      TF_RETURN_IF_ERROR(
          AddDynamicSliceDimToTensorParams(b, hlo, side, properties, pid_offset,
                                           specs.back(), bases, tensor_params));
    } else {
      TF_RETURN_IF_ERROR(AddStandaloneDimToTensorParams(
          b, side, properties, pid_offset, specs.front(), bases, tensor_params,
          boundary_checks));
    }
    if (specs.back()->at(0).broadcast_multiplier > 1) {
      if (specs.back()->at(0).broadcast_multiplier % properties.block_size) {
        return UncompilableMatmul(
            absl::StrCat("Broadcast multiplier is not a multiple of the block "
                         "size. block_size: ",
                         properties.block_size, " vs broadcast_multiplier: ",
                         specs.back()->at(0).broadcast_multiplier));
      }
      if (properties.split_value > 1) {
        return UncompilableMatmul(
            "Broadcasted dimension is split, which is not supported yet.");
      }
      tensor_params.block_dims.push_back(1);
    } else {
      tensor_params.block_dims.push_back(properties.block_size);
    }
    tensor_params.dim_order.emplace(tensor_params.dim_order.begin(),
                                    tensor_params.dim_order.size());
    return absl::OkStatus();
  }

  absl::Status AddStandaloneDimToTensorParams(
      EmitterLocOpBuilder b, const Side& side, const DimProperties& properties,
      Value pid_offset, const TensorIterationSpec::DimIterationSpec* spec,
      const ValueRange& bases, TensorParams& tensor_params,
      std::vector<int32_t>& boundary_checks) {
    tensor_params.tensor_offsets.push_back(Cst32(b, spec->at(0).slice_start));
    tensor_params.block_offsets.push_back(pid_offset);
    int64_t dim_bound = spec->at(0).count;
    if (side.scope == TritonFusionAnalysis::Scope::OUTPUT &&
        properties.index == dims_.out_lhs_noncontracting_dim_idx &&
        spec->size() == 1 && dims_.lhs_noncontracting_split.has_value()) {
      // Dimension of the output produced by the non-contracting LHS one
      // is logically split, major part is addressed using pid_batch.
      dim_bound /= *dims_.lhs_noncontracting_split;
    }
    tensor_params.bounds.push_back(Cst64(b, dim_bound));
    if (dim_bound % (properties.block_size * properties.split_value) != 0) {
      boundary_checks.push_back(tensor_params.bounds.size() - 1);
    }
    return absl::OkStatus();
  }

  absl::Status AddDynamicSliceDimToTensorParams(
      EmitterLocOpBuilder b, const HloInstruction* hlo, const Side& side,
      const DimProperties& properties, const Value pid_offset,
      const TensorIterationSpec::DimIterationSpec* spec,
      const ValueRange& bases, TensorParams& tensor_params) {
    // Here we compute the offset of where we should read the slice from.
    // TODO(b/323255699): Add support for slices of the contracting dim.
    // Dynamic slices are guaranteed to only be offset along the majormost
    // dimension.

    // The only fragment of the non-contracting dim of the dot's input in
    // the current scope:
    TF_RET_CHECK(spec->size() == 1);
    const TensorIterationSpec::IterationSpecFragment only_fragment_of_nc_dim =
        spec->at(0);
    // The majormost dim index in the dynamic slice's output.
    const int majormost_dim = hlo->shape().layout().minor_to_major().back();

    // dynamic slice operands are (input, start_index0, start_index1, ...)
    // so the start index corresponding to the ith dimension is bases[i+1].
    Value majormost_dim_start_index_ptr_val = bases[majormost_dim + 1];
    Value majormost_dim_start_index_val = b.create<mt::LoadOp>(
        majormost_dim_start_index_ptr_val, mt::CacheModifier::NONE,
        mt::EvictionPolicy::NORMAL,
        /*isVolatile=*/false);
    int64_t majormost_dim_start_index_upper_limit =
        hlo->operand(0)->shape().dimensions(majormost_dim) -
        hlo->dynamic_slice_sizes().at(majormost_dim);
    // We don't want to cast S64 indices to S32, because that could result
    // in an incorrect value.
    if (majormost_dim_start_index_val.getType().isInteger() &&
        majormost_dim_start_index_val.getType().getIntOrFloatBitWidth() == 64) {
      return UncompilableMatmul(
          "64 bit dynamic-slice indices are not supported yet.");
    }
    majormost_dim_start_index_val =
        triton::Cast(b, majormost_dim_start_index_val, b.getI32Type());
    majormost_dim_start_index_val =
        b.create<ma::MaxSIOp>(majormost_dim_start_index_val, Cst32(b, 0));
    majormost_dim_start_index_val =
        b.create<ma::MinSIOp>(majormost_dim_start_index_val,
                              Cst32(b, majormost_dim_start_index_upper_limit));

    // How many "rows" (non-contracting dim values) are there in a slice of
    // size 1?
    int64_t rows_per_majormost_dim = 1;
    for (int i = 0; i < hlo->shape().dimensions().size() - 1; ++i) {
      rows_per_majormost_dim *= hlo->shape().dimensions_minor(i);
    }
    rows_per_majormost_dim =
        rows_per_majormost_dim / only_fragment_of_nc_dim.stride;
    Value rows_per_majormost_dim_val = Cst32(b, rows_per_majormost_dim);

    Value tensor_offset_val_i32 = b.create<ma::MulIOp>(
        majormost_dim_start_index_val, rows_per_majormost_dim_val);
    tensor_params.tensor_offsets.push_back(tensor_offset_val_i32);

    // tt.make_tensor_ptr expects an i64 for shape and size, but expects
    // i32 for offsets. We extend the offset to calculate the upper bound.
    Value tensor_offset_val_i64 =
        b.create<ma::ExtSIOp>(b.getI64Type(), tensor_offset_val_i32);
    Value sliced_count_val = Cst64(b, only_fragment_of_nc_dim.sliced_count);
    Value upper_bound_val =
        b.create<ma::AddIOp>(tensor_offset_val_i64, sliced_count_val);
    tensor_params.bounds.push_back(upper_bound_val);

    tensor_params.block_offsets.push_back(pid_offset);
    return absl::OkStatus();
  }

  absl::Status AddConcatDimToTensorParams(
      EmitterLocOpBuilder b, const HloInstruction* hlo, const Side& side,
      const DimProperties& properties, const ConcatParams& concat_params,
      const Value pid_offset, const TensorIterationSpec::DimIterationSpec* spec,
      const std::vector<Value>& input_offsets, std::vector<Value>& input_bounds,
      TensorParams& tensor_params) {
    TF_ASSIGN_OR_RETURN(Value select_input_offsets,
                        EmitMultiSelect(b, pid_offset, concat_params.boundaries,
                                        input_offsets));
    tensor_params.block_offsets.push_back(select_input_offsets);

    TF_ASSIGN_OR_RETURN(
        Value select_input_bounds,
        EmitMultiSelect(b, pid_offset, concat_params.boundaries, input_bounds));
    tensor_params.bounds.push_back(select_input_bounds);
    tensor_params.tensor_offsets.push_back(Cst32(b, spec->at(0).slice_start));
    return absl::OkStatus();
  }

  absl::StatusOr<Value> UpddateBaseForBatchAndConcatCases(
      EmitterLocOpBuilder b, const HloInstruction* hlo,
      const ConcatParams& concat_params, const Side& side, Value base) {
    int64_t offset_batch = 0;
    bool has_batch_offset = false;
    Value batch_stride;

    if (hlo->opcode() == HloOpcode::kConcatenate) {
      std::vector<Value> batch_strides;
      batch_strides.reserve(hlo->operands().size());
      for (const HloInstruction* operand : hlo->operands()) {
        TF_ASSIGN_OR_RETURN(
            Value op_stride,
            GetBatchStride(b, side, operand, offset_batch, has_batch_offset));
        batch_strides.push_back(op_stride);
      }
      TF_ASSIGN_OR_RETURN(
          batch_stride,
          EmitMultiSelect(b, concat_params.dim_pid_offset,
                          concat_params.boundaries, batch_strides));
    } else {
      TF_ASSIGN_OR_RETURN(
          batch_stride,
          GetBatchStride(b, side, hlo, offset_batch, has_batch_offset));
    }

    // Avoid generating logic to compute batch offset if unnecessary.
    if (has_batch_offset) {
      Value pid_batch =
          b.create<mt::GetProgramIdOp>(launch_config_.batch_program_id_dim);

      Value pid_offset_batch = b.create<ma::MulIOp>(
          b.create<ma::AddIOp>(Cst(b, index_ty_, offset_batch),
                               ConvertScalar(b, pid_batch)),
          batch_stride);

      base = AddPtr(b, base, pid_offset_batch);
    }
    return base;
  }

  absl::StatusOr<Value> UpdateBaseForSplitKCase(EmitterLocOpBuilder b,
                                                const HloInstruction* hlo,
                                                const Side& side, Value base,
                                                Value pid_k) {
    const TensorIterationSpec::DimIterationSpec* spec = analysis_.IterSpec(
        TritonFusionAnalysis::Scope::OUTPUT, hlo, *dims_.out_split_k_dim_idx);
    if (spec != nullptr && spec->at(0).count > 1) {
      TF_RET_CHECK(pid_k != nullptr);
      base =
          AddPtr(b, base,
                 b.create<ma::MulIOp>(ConvertScalar(b, pid_k),
                                      Cst(b, index_ty_, spec->at(0).stride)));
    }
    return base;
  }

  absl::StatusOr<std::pair<ConcatParams, Value>>
  CalculateConcatParamsAndUpdateBase(EmitterLocOpBuilder b,
                                     const HloInstruction* hlo,
                                     const Side& side,
                                     const ValueRange& bases) {
    ConcatParams concat_params;
    // For now only non-contracting dimension can be concatenated.
    concat_params.dim_idx = (side.scope == TritonFusionAnalysis::Scope::LHS)
                                ? dims_.lhs_noncontracting_dim_idx
                                : dims_.rhs_noncontracting_dim_idx;
    const DimProperties& properties = [&] {
      for (const DimProperties& dim : side.tiled_dims) {
        if (dim.index == concat_params.dim_idx) {
          return dim;
        }
      }
      LOG(FATAL) << "Missing dimension.";
    }();
    TF_RET_CHECK(bases.size() == hlo->operand_count());

    concat_params.boundaries.reserve(hlo->operand_count() - 1);
    for (int i = 0; i < hlo->operand_count() - 1; ++i) {
      const TensorIterationSpec::IterationSpecFragment& fragment =
          analysis_
              .IterSpec(side.scope, hlo->operand(i), concat_params.dim_idx)
              ->at(0);
      if (fragment.sliced_count % properties.block_size != 0) {
        return UncompilableMatmul(
            "Operand is not divisible by the block size.");
      }
      concat_params.boundaries.push_back(
          Cst32(b, -fragment.slice_start + fragment.sliced_count));
    }

    concat_params.dim_pid_offset =
        b.create<ma::MulIOp>(properties.pid, Cst32(b, properties.block_size));
    TF_ASSIGN_OR_RETURN(Value base,
                        EmitMultiSelect(b, concat_params.dim_pid_offset,
                                        concat_params.boundaries, bases));
    return std::make_pair(concat_params, base);
  }

  // Extend int32 indexes to int64, if necessary.
  Value ConvertScalar(EmitterLocOpBuilder b, Value value) {
    if (index_ty_.getIntOrFloatBitWidth() == 64) {
      return b.create<ma::ExtSIOp>(index_ty_, value);
    }
    return value;
  }

  absl::string_view libdevice_path_;
  const se::DeviceDescription& device_info_;
  const HloDotInstruction* dot_instr_;
  Type index_ty_;
  TritonFusionAnalysis analysis_;
  MatMulDims dims_;
  MatMulLaunchConfig launch_config_;
};

absl::StatusOr<SmallVector<Value>> GetArguments(mlir::FunctionOpInterface fn,
                                                const HloInstruction& input) {
  if (input.opcode() == HloOpcode::kParameter) {
    return {{fn.getArgument(input.parameter_number())}};
  } else if (input.opcode() == HloOpcode::kConcatenate ||
             input.opcode() == HloOpcode::kDynamicSlice) {
    // As defined in GemmFusion, all inputs of concatenate and dynamic slice are
    // parameters.
    SmallVector<Value> result;
    for (const HloInstruction* operand : input.operands()) {
      TF_RET_CHECK(operand->opcode() == HloOpcode::kParameter);
      result.push_back(fn.getArgument(operand->parameter_number()));
    }
    return result;
  }
  LOG(FATAL) << "Unexpected opcode: " << input.opcode();
}

// Concatenations can currently only be applied directly to parameters;
// all concatenated parameters share the same block pointer. This function
// returns all inputs of a kernel: concatenations of parameters and standalone
// parameters.
ConstHloInstructionSet ScopeInputs(const TritonFusionAnalysis& analysis,
                                   const TritonFusionAnalysis::Scope scope) {
  ConstHloInstructionSet result;
  for (const HloInstruction* parameter : analysis.ScopeParameters(scope)) {
    if (absl::c_any_of(parameter->users(), [](const HloInstruction* user) {
          return user->opcode() == HloOpcode::kConcatenate ||
                 user->opcode() == HloOpcode::kDynamicSlice;
        })) {
      // Concatenation is always the only user of its parameters by
      // construction.
      CHECK_EQ(parameter->users().size(), 1);
      for (const HloInstruction* operand : parameter->users()[0]->operands()) {
        // All operands of a concatenation have to be computation parameters.
        CHECK_EQ(operand->opcode(), HloOpcode::kParameter);
      }
      result.insert(parameter->users()[0]);
    } else {
      result.insert(parameter);
    }
  }
  return result;
}


// This is a heuristic that serves as a proxy for register usage and code size.
//
// We have noticed that tilings with very long LLVM IR code are both slow to
// compile and slow to run. This can be for example due to register spills. So
// we should skip these tilings to save time. But it's better to skip them
// before the LLVM IR is generated. To do that, we came up with a formula that
// strongly correlates with the LLVM IR size. The formula is the size of the two
// input and the output thread block tiles divided by the number of warps. We
// read https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/ as a
// reference, and found the formula by trial and error.
//
// To regenerate the limit, we have to run an exhaustive search on all tilings
// for a few different HLOs, printing the runtimes and the heuristic values.
//
// From that, we can find a limit, such that all tilings within alpha *
// optimal_runtime have a heuristic value less than or equal to the limit.
//
// In our measurements, all tilings which were within 1.13 * optimal_runtime had
// a complexity_heuristic_value <= kComplexityHeuristicLimit.
//
// See go/tiling-heuristic for more details.
absl::Status CheckGemmTilingComplexityHeuristic(
    const TritonGemmConfig& config) {
  constexpr int64_t kComplexityHeuristicLimit = 9000;
  int64_t complexity_heuristic_value =
      (config.block_m * config.block_n +
       (config.block_m + config.block_n) * config.block_k) /
      config.num_warps;
  VLOG(2) << "Complexity heuristic: " << complexity_heuristic_value;
  if (complexity_heuristic_value > kComplexityHeuristicLimit) {
    return ResourceExhausted("Tiling complexity heuristic exceeded: %d > %d",
                             complexity_heuristic_value,
                             kComplexityHeuristicLimit);
  }
  return absl::OkStatus();
}

class Scopes {
 public:
  Scopes(EmitterLocOpBuilder& b, const HloInstruction* dot_instr,
         const TritonFusionAnalysis& analysis, const MatMulDims& dims,
         const TritonGemmConfig& config, const MatMulLaunchConfig launch_config)
      : lhs_(TritonFusionAnalysis::Scope::LHS),
        rhs_(TritonFusionAnalysis::Scope::RHS),
        out_(TritonFusionAnalysis::Scope::OUTPUT) {
    constexpr int group_m = 8;
    const int64_t width = group_m * launch_config.grid_n;

    auto pid_nc = b.create<mt::GetProgramIdOp>(
        launch_config.noncontracting_program_id_dim);
    pid_k_ = (config.split_k > 1)
                 ? b.create<mt::GetProgramIdOp>(mt::ProgramIDDim::Z)
                 : Value{};

    auto group_id = b.create<ma::DivSIOp>(pid_nc, Cst32(b, width));
    ma::ConstantOp group_m_op = Cst32(b, group_m);
    auto first_pid_m = b.create<ma::MulIOp>(group_id, group_m_op);
    auto sub0 =
        b.create<ma::SubIOp>(Cst32(b, launch_config.grid_m), first_pid_m);
    auto group_size = b.create<ma::SelectOp>(
        b.create<ma::CmpIOp>(ma::CmpIPredicate::slt, sub0, group_m_op), sub0,
        group_m_op);

    pid_m_ = b.create<ma::AddIOp>(first_pid_m,
                                  b.create<ma::RemSIOp>(pid_nc, group_size));

    pid_n_ = b.create<ma::DivSIOp>(
        b.create<ma::RemSIOp>(pid_nc, Cst32(b, width)), group_size);

    int lhs_non_contracting_block_size = config.block_m;
    int lhs_contracting_block_size = config.block_k;
    lhs_.tiled_dims = {
        DimProperties(dims.lhs_noncontracting_dim_idx, pid_m_,
                      lhs_non_contracting_block_size,
                      /*split_value=*/1),
        DimProperties(dims.lhs_contracting_dim_idx, pid_k_,
                      lhs_contracting_block_size, config.split_k)};
    lhs_.batch_dim_idx = dims.lhs_batch_dim_idx;

    int rhs_contracting_block_size = config.block_k;
    int rhs_non_contracting_block_size = config.block_n;
    rhs_.tiled_dims = {
        DimProperties(dims.rhs_contracting_dim_idx, pid_k_,
                      rhs_contracting_block_size, config.split_k),
        DimProperties(dims.rhs_noncontracting_dim_idx, pid_n_,
                      rhs_non_contracting_block_size,
                      /*split_value=*/1)};
    rhs_.batch_dim_idx = dims.rhs_batch_dim_idx;

    out_.tiled_dims = {DimProperties(dims.out_lhs_noncontracting_dim_idx,
                                     pid_m_, config.block_m,
                                     /*split_value=*/1),
                       DimProperties(dims.out_rhs_noncontracting_dim_idx,
                                     pid_n_, config.block_n,
                                     /*split_value=*/1)};
    out_.batch_dim_idx = dims.out_batch_dim_idx;
  }

  std::vector<const Side*> input_scopes() const {
    if (meta_.has_value()) {
      return {&lhs_, &rhs_, &meta_.value()};
    }
    return {&lhs_, &rhs_};
  }
  const Side& lhs() const { return lhs_; }
  const Side& rhs() const { return rhs_; }
  const Side& out() const { return out_; }
  const std::optional<Side>& meta() const { return meta_; }
  const Value& pid_m() const { return pid_m_; }
  const Value& pid_k() const { return pid_k_; }
  const Value& pid_n() const { return pid_n_; }

 private:
  Side lhs_;
  Side rhs_;
  Side out_;
  std::optional<Side> meta_;

  Value pid_m_;
  Value pid_k_;
  Value pid_n_;
};

// This class represents a loadable input that needs to be partially loaded on
// each iteration of a ForOp. It keeps track of which iterable argument indices
// correspond to this input & all information necessary to load & modify
// iteration arguments for the next iteration.
class IterableInput {
 public:
  IterableInput(size_t iter_arg_index, size_t operand_index,
                int contracting_dimension, Type type, Type storage_type,
                const HloInstruction* hlo_instr, const Side* side,
                std::vector<int32_t> boundary_checks, int64_t block_k)
      : iter_arg_index_(iter_arg_index),
        operand_index_(operand_index),
        contracting_dimension_(contracting_dimension),
        type_(type),
        storage_type_(storage_type),
        block_k_(block_k),
        hlo_instr_(hlo_instr),
        side_(side),
        boundary_checks_(boundary_checks) {};

  static absl::StatusOr<IterableInput> CreateIterableInput(
      size_t iter_arg_index, EmitterLocOpBuilder& b, const MatMulDims& dims,
      const Side* side, const HloInstruction* hlo_instr, int64_t block_k) {
    Type input_ty;
    if (side->scope == TritonFusionAnalysis::Scope::META) {
      input_ty = b.getI16Type();
    } else {
      TF_ASSIGN_OR_RETURN(input_ty,
                          TritonType(b, hlo_instr->shape().element_type()));
    }
    int contracting_dimension =
        (side->scope == TritonFusionAnalysis::Scope::RHS)
            ? dims.rhs_contracting_dim_idx
            : dims.lhs_contracting_dim_idx;

    return IterableInput(iter_arg_index, GetOperandIndex(side->scope),
                         contracting_dimension, input_ty,
                         StorageType(b, input_ty), hlo_instr, side,
                         /*boundary_checks=*/{}, block_k);
  }

  Value EmitAdvance(EmitterLocOpBuilder b, MatMulEmitterHelper& emitter,
                    Value ki, ValueRange iter_args) const {
    SmallVector<Value> increments = emitter.EmitIncrements(
        b, *side_, *hlo_instr_, contracting_dimension_, ki, block_k_);

    if (increments.empty()) {
      return iter_args[iter_arg_index_];
    }

    const Value& iter_arg = iter_args[iter_arg_index_];
    return b.create<mt::AdvanceOp>(iter_arg.getType(), iter_arg, increments);
  }

  Value EmitLoad(EmitterLocOpBuilder b, ValueRange args) const {
    Value param_value = EmitParameterLoad(b, args.front(), boundary_checks_);
    if (type_ != storage_type_) {
      // For example cast i8 to i1.
      param_value = triton::Cast(b, param_value, type_);
    }
    return param_value;
  }

  // Index of the iter_arg of the ForOp associated with this input.
  size_t iter_arg_index_;
  // Index used to differentiate it between LHS, RHS, and META inputs.
  size_t operand_index_;
  // Index of the contracting dimension in the input.
  int contracting_dimension_;
  // Type of the input.
  Type type_;
  // Storage type of the input (in case it is different & needs to be casted).
  Type storage_type_;
  // Step size of the contracting dimension.
  int64_t block_k_;

  // Necessary for some operations at the moment. Maybe could be refactored out.
  const HloInstruction* hlo_instr_;
  const Side* side_;

  // This is currently set afterwards, but needs to be associated with the input
  // during iteration.
  std::vector<int32_t> boundary_checks_;
};

enum MaskExpandDimension { kMajor = 0, kMinor = 1 };

Value EmitMaskOnInput(EmitterLocOpBuilder& b,
                      MaskExpandDimension expand_along_dimension, Value input,
                      int dim_k_denom, Value k, int64_t dims_k, int64_t block_k,
                      Value pid_k, int64_t other_dim_block_size) {
  int block_k_size = block_k / dim_k_denom;
  auto dim_k_elements_to_keep =
      b.create<ma::SubIOp>(Cst32(b, dims_k / dim_k_denom), k);
  auto is_last_tile_cond = b.create<ma::CmpIOp>(
      ma::CmpIPredicate::slt, dim_k_elements_to_keep, Cst32(b, block_k_size));
  auto input_type = mlir::cast<mlir::RankedTensorType>(input.getType());
  auto input_element_type = input_type.getElementType();

  // If the input is a scalar, we need to expand it to a 2D tensor.
  // Otherwise, keep the input type.
  auto expanded_input_type = [&](Value input) {
    if (input_type.getRank() != 0) return input_type;
    // expand along the major dimension.
    if (expand_along_dimension == kMajor) {
      return mlir::RankedTensorType::get(
          ArrayRef<int64_t>{other_dim_block_size, block_k_size},
          input_element_type);
    }
    // expand along the minor dimension.
    return mlir::RankedTensorType::get(
        ArrayRef<int64_t>{block_k_size, other_dim_block_size},
        input_element_type);
  }(input);

  auto expanded_input = input;
  // If the input is a scalar, we need to expand it to a 2D tensor.
  if (input_type.getRank() == 0) {
    expanded_input = b.create<mt::ExpandDimsOp>(expanded_input, 0);
    expanded_input = b.create<mt::ExpandDimsOp>(expanded_input, 0);
    expanded_input =
        b.create<mt::BroadcastOp>(expanded_input_type, expanded_input);
  }

  auto if_op = b.create<mlir::scf::IfOp>(
      is_last_tile_cond, /*thenBranch=*/
      [&, &parent_builder = b](mlir::OpBuilder& builder, mlir::Location loc) {
        EmitterLocOpBuilder b(loc, builder, parent_builder.annotate_loc());
        // Make a range vector from 0 to block_k.
        auto range_from_0_to_k = Range(b, block_k_size);
        if (pid_k != nullptr) {
          range_from_0_to_k = b.create<ma::AddIOp>(
              range_from_0_to_k,
              Splat(b, b.create<ma::MulIOp>(pid_k, Cst32(b, block_k_size)),
                    block_k_size));
        }
        // Make it a 2D matrix.
        TensorValue range_from_0_to_k_2d = mlir::cast<TensorValue>(
            b.create<mt::ExpandDimsOp>(range_from_0_to_k,
                                       expand_along_dimension)
                .getResult());
        // Make 2d vector of dim_k_elements_to_keep.
        auto dim_k_elements_to_keep_2d =
            Splat(b, dim_k_elements_to_keep,
                  range_from_0_to_k_2d.getType().getShape());
        // The mask is true for elements in range_from_0_to_k_2d that are less
        // than dim_k_elements_to_keep.
        auto elements_mask_vector =
            b.create<ma::CmpIOp>(ma::CmpIPredicate::slt, range_from_0_to_k_2d,
                                 dim_k_elements_to_keep_2d);

        Value elements_mask_matrix = b.create<mt::BroadcastOp>(
            expanded_input_type.clone(b.getI1Type()), elements_mask_vector);

        // Zeros to use instead of the masked elements.
        auto zeros = CreateConst(b, input_element_type, 0,
                                 expanded_input_type.getShape());
        auto result =
            b.create<ma::SelectOp>(elements_mask_matrix, expanded_input, zeros);
        b.create<mlir::scf::YieldOp>(mlir::ValueRange(result));
      },
      /*elseBranch=*/
      [&, &parent_builder = b](mlir::OpBuilder& builder, mlir::Location loc) {
        // We don't need to mask anything but we need to expand the input.
        // Otherwise Triton complains.
        EmitterLocOpBuilder b(loc, builder, parent_builder.annotate_loc());
        b.create<mlir::scf::YieldOp>(mlir::ValueRange(expanded_input));
      });
  return if_op.getResult(0);
}

absl::StatusOr<TritonGemmConfig> GetTritonGemmConfig(
    const HloFusionInstruction* fusion) {
  auto backend_config =
      fusion->backend_config<GpuBackendConfig>()->fusion_backend_config();

  if (!backend_config.has_triton_gemm_config()) {
    // TODO(bchetioui): consolidate default parameters. At the moment, these
    // may be constructed in two distinct places.
    LOG(WARNING) << "Using fallback triton GEMM config for op "
                 << fusion->name();
    *backend_config.mutable_triton_gemm_config() = DefaultTritonGemmKey();
  }

  return TritonGemmConfig::FromProto(backend_config.triton_gemm_config());
}

Type GetIndexType(EmitterLocOpBuilder& b, const HloDotInstruction& dot_instr,
                  const TritonGemmConfig& config) {
  // Use 32-bit indexing if addressing any of the inputs or the output (which
  // could grow if split_k is set) does not cross the INT_MAX boundary.
  // Otherwise, fall back to 64-bit indexing, which is slower.
  bool use_64bit_indexing =
      ShapeUtil::ElementsIn(dot_instr.operand(0)->shape()) > INT_MAX ||
      ShapeUtil::ElementsIn(dot_instr.operand(1)->shape()) > INT_MAX ||
      ShapeUtil::ElementsIn(dot_instr.shape()) * config.split_k > INT_MAX;
  return b.getIntegerType(use_64bit_indexing ? 64 : 32);
}

absl::Status EmitForLoopBody(EmitterLocOpBuilder& b,
                             MatMulEmitterHelper& emitter, const Scopes& scopes,
                             const HloDotInstruction* dot_instr,
                             const MatMulDims& dims,
                             const llvm::SmallVector<IterableInput>& inputs,
                             Value ki, ValueRange iter_args) {
  SmallVector<Value> args_for_yield;
  std::array<absl::flat_hash_map<const HloInstruction*, Value>, 3> values;

  // Load tiles of all parameters of LHS and RHS scopes and advance pointers.
  for (const IterableInput& input : inputs) {
    Value param_value = input.EmitLoad(b, iter_args[input.iter_arg_index_]);
    CHECK(values[input.operand_index_]
              .insert({input.hlo_instr_, param_value})
              .second);
    args_for_yield.push_back(input.EmitAdvance(b, emitter, ki, iter_args));
  }

  // Emit all operations of LHS and RHS scopes.
  Value dot_lhs =
      emitter.MakeInput(b, scopes.lhs(), kLhsIndex, values[kLhsIndex]);
  Value dot_rhs =
      emitter.MakeInput(b, scopes.rhs(), kRhsIndex, values[kRhsIndex]);

  // Operation in the fusion before the dot can alter the elements of the
  // tiles that were zero masked during loads. These have to be zeroed here
  // again just before the dot so that they do not affect the output.
  // Only the K dimension needs masking here because unnecessary elements in
  // the other two get discarded by the masked store at the end.
  const bool need_masking =
      dims.k % (dims.config.block_k * dims.config.split_k) > 0;
  if (need_masking) {
    dot_lhs = EmitMaskOnInput(b, MaskExpandDimension::kMajor, dot_lhs, 1, ki,
                              dims.k, dims.config.block_k, scopes.pid_k(),
                              dims.config.block_m);
    dot_rhs = EmitMaskOnInput(b, MaskExpandDimension::kMinor, dot_rhs, 1, ki,
                              dims.k, dims.config.block_k, scopes.pid_k(),
                              dims.config.block_n);
    // Masking the metadata is not necessary, as the inputs are masked
    // (i.e. zeroed out), so the padded metadata can hold any values.
  }

  TF_ASSIGN_OR_RETURN(
      Value acc_next,
      triton::EmitSingleTileDot(
          b, *dot_instr,
          triton::DotOperands{dot_lhs, dot_rhs, iter_args.back()}));
  args_for_yield.push_back(acc_next);
  b.create<mlir::scf::YieldOp>(args_for_yield);

  return absl::OkStatus();
}

}  // namespace

// Use tiling and execution parameters from 'config'. BlockLevelParameters are
// ignored.
// Variable naming: lhs [m, k] x rhs [k, n] -> out [m, n].
absl::StatusOr<std::optional<stream_executor::gpu::TmaMetadata>> EmitMatMul(
    EmitterLocOpBuilder& b, absl::string_view libdevice_path,
    const se::DeviceDescription& device_info,
    const HloFusionInstruction* fusion, mlir::FunctionOpInterface fn,
    const BlockLevelParameters&) {
  // TODO b/315957220: Populate tma_metadata.
  stream_executor::gpu::TmaMetadata tma_metadata;

  TF_ASSIGN_OR_RETURN(TritonGemmConfig config, GetTritonGemmConfig(fusion));
  TF_ASSIGN_OR_RETURN(auto analysis,
                      TritonFusionAnalysis::Execute(
                          *fusion->called_computation(), config.split_k));

  TF_RETURN_IF_ERROR(CheckGemmTilingComplexityHeuristic(config));

  const HloComputation* computation = fusion->fused_instructions_computation();
  const HloInstruction* instr =
      hlo_query::GetFirstInstructionWithOpcode(*computation, HloOpcode::kDot);
  const HloDotInstruction* dot_instr = DynCast<HloDotInstruction>(instr);

  Type index_ty = GetIndexType(b, *dot_instr, config);

  const HloInstruction* root = dot_instr->parent()->root_instruction();
  TF_RET_CHECK(!root->shape().IsTuple());

  TF_RETURN_IF_ERROR(ValidateMatMulConfig(config, *dot_instr));
  const int split_k = config.split_k;
  const int block_m = config.block_m;
  const int block_k = config.block_k;
  const int block_n = config.block_n;

  TF_ASSIGN_OR_RETURN(const MatMulDims dims,
                      MatMulDims::Create(config, *dot_instr, analysis));
  const MatMulLaunchConfig launch_config(config, *dot_instr, dims, device_info);
  VLOG(6) << analysis.ToString();

  MatMulEmitterHelper emitter(libdevice_path, device_info, dot_instr, index_ty,
                              dims, launch_config, analysis);

  TF_ASSIGN_OR_RETURN(mlir::Type acc_ty,
                      triton::GetDotAccumulatorType(b, *dot_instr));

  ma::ConstantOp accumulator_init =
      CreateConst(b, acc_ty, 0, {block_m, block_n});

  // Calculate the sizes of the lhs, rhs, meta, and output sides.
  Scopes scopes(b, dot_instr, analysis, dims, config, launch_config);

  llvm::SmallVector<IterableInput> inputs;

  // Pointers to inputs of LHS scope, then RHS, then the accumulator
  // that change with every loop iteration and are passed between them.
  SmallVector<Value> iter_args;
  int64_t step_k = block_k * split_k;
  for (const Side* side : scopes.input_scopes()) {
    for (const HloInstruction* input_hlo : ScopeInputs(analysis, side->scope)) {
      TF_ASSIGN_OR_RETURN(SmallVector<Value> arguments,
                          GetArguments(fn, *input_hlo));
      TF_ASSIGN_OR_RETURN(
          IterableInput iter_input,
          IterableInput::CreateIterableInput(iter_args.size(), b, dims, side,
                                             input_hlo, step_k));
      TF_ASSIGN_OR_RETURN(Value tensor_ptr,
                          emitter.EmitTensorPointer(
                              b, input_hlo, *side, arguments, scopes.pid_k(),
                              iter_input.boundary_checks_));
      inputs.push_back(iter_input);
      iter_args.push_back(tensor_ptr);
    }
  }

  auto body_builder_callback = [&](mlir::OpBuilder&, mlir::Location, Value ki,
                                   ValueRange iter_args) -> void {
    CHECK_OK(EmitForLoopBody(b, emitter, scopes, dot_instr, dims, inputs, ki,
                             iter_args));
  };

  iter_args.push_back(accumulator_init);
  Value acc_final = b.create<mlir::scf::ForOp>(
                         /*lowerBound*/ Cst32(b, 0),
                         /*upperBound*/ Cst32(b, dims.k),
                         /*step*/ Cst32(b, step_k),
                         /*iterArgs*/ iter_args, body_builder_callback)
                        .getResult(iter_args.size() - 1);
  absl::flat_hash_map<const HloInstruction*, Value> values_out;
  TF_ASSIGN_OR_RETURN(Type acc_final_ty,
                      TritonType(b, dot_instr->shape().element_type()));
  values_out[dot_instr] = triton::Cast(b, acc_final, acc_final_ty);

  // Emit the output scope.
  if (std::vector<const HloInstruction*> to_emit =
          emitter.EpiloguePostOrderTransitiveOperands(root);
      !to_emit.empty()) {
    for (const HloInstruction* input :
         ScopeInputs(analysis, TritonFusionAnalysis::Scope::OUTPUT)) {
      std::vector<int32_t> boundary_checks;
      TF_ASSIGN_OR_RETURN(SmallVector<Value> arguments,
                          GetArguments(fn, *input));
      TF_ASSIGN_OR_RETURN(
          Value tensor_pointer,
          emitter.EmitTensorPointer(b, input, scopes.out(), arguments,
                                    scopes.pid_k(), boundary_checks));
      TF_RET_CHECK(values_out
                       .insert({input, EmitParameterLoad(b, tensor_pointer,
                                                         boundary_checks)})
                       .second);
    }
    TF_RETURN_IF_ERROR(EmitScope(b, libdevice_path, device_info, &analysis,
                                 scopes.out(), to_emit, values_out)
                           .status());
  }

  // Emit tensor store operations for all outputs.
  for (int i = 0;
       i < fn.getNumArguments() - dot_instr->parent()->num_parameters(); ++i) {
    const HloInstruction* producer =
        root->shape().IsTuple() ? root->operand(i) : root;
    std::vector<int32_t> boundary_checks;
    TF_ASSIGN_OR_RETURN(
        Value tensor_pointer,
        emitter.EmitTensorPointer(
            b, producer, scopes.out(),
            {fn.getArgument(i + dot_instr->parent()->num_parameters())},
            scopes.pid_k(), boundary_checks));
    b.create<mt::StoreOp>(tensor_pointer, values_out[producer], boundary_checks,
                          mt::CacheModifier::NONE, mt::EvictionPolicy::NORMAL);
  }
  return tma_metadata;
}

absl::StatusOr<LaunchDimensions> GetMatMulLaunchDimensions(
    const TritonFusionAnalysis& analysis, const HloFusionAdaptor& fusion,
    const TritonGemmConfig& config, const se::DeviceDescription& device_info) {
  auto dot = HloBfsFindIf(fusion.GetRoots(), fusion, [](auto node) {
    return node.opcode() == HloOpcode::kDot;
  });
  TF_RET_CHECK(dot != std::nullopt);
  const auto& dot_instr =
      *static_cast<const HloDotInstruction*>(&dot->instruction());
  TF_ASSIGN_OR_RETURN(MatMulDims dims,
                      MatMulDims::Create(config, dot_instr, analysis));
  MatMulLaunchConfig launch_config(config, dot_instr, dims, device_info);
  return launch_config.launch_dims;
}

}  // namespace xla::gpu
