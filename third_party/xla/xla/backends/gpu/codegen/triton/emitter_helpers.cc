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

#include "xla/backends/gpu/codegen/triton/emitter_helpers.h"

#include <cstdint>
#include <variant>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/TargetParser/Triple.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "xla/codegen/emitter_loc_op_builder.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/mlir_hlo/mhlo/transforms/map_mhlo_to_scalar_op.h"
#include "xla/mlir_hlo/mhlo/transforms/transformation_helpers.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/target_util.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace xla::gpu::triton {

using ::llvm::SmallVector;
using ::mlir::ArrayRef;
using ::mlir::ShapedType;
using ::mlir::Type;
using ::mlir::Value;
using ::mlir::ValueRange;

namespace ma = ::mlir::arith;
namespace mh = ::mlir::mhlo;
namespace mm = ::mlir::math;
namespace mt = ::mlir::triton;

ScalarOrTensor::ScalarOrTensor(mlir::Value value) : value_(value) {
  CHECK(IsScalar() || UnwrapTensor().getType().getRank() > 0)
      << "0D tensors are not supported by Triton";
}

SmallVector<int64_t> GetPaddedTileSizes(ArrayRef<int64_t> tile_sizes) {
  SmallVector<int64_t> result;
  result.reserve(tile_sizes.size());
  for (int64_t value : tile_sizes) {
    result.push_back(llvm::PowerOf2Ceil(value));
  }
  return result;
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
    case S8:
      return b.getI8Type();
    case S4:
      return b.getI4Type();
    case PRED:
      return b.getI1Type();
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

absl::StatusOr<PrimitiveType> GetPrimitiveType(Type t) {
  if (t.isF64()) return F64;
  if (t.isF32()) return F32;
  if (t.isF16()) return F16;
  if (t.isBF16()) return BF16;
  if (t.isInteger(64)) return S64;
  if (t.isInteger(32)) return S32;
  if (t.isInteger(16)) return S16;
  if (t.isInteger(8)) return S8;
  if (t.isInteger(4)) return S4;
  if (t.isInteger(1)) return PRED;
  if (mlir::isa<mlir::Float8E5M2Type>(t)) return F8E5M2;
  if (mlir::isa<mlir::Float8E4M3FNType>(t)) return F8E4M3FN;
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

Value Cast(EmitterLocOpBuilder& b, Value value, Type dst_element_ty) {
  Type src_ty = value.getType();
  Type src_element_ty = src_ty;
  Type fp16_ty = b.getF16Type();
  Type fp32_ty = b.getF32Type();
  Type dst_ty = dst_element_ty;
  if (auto src_shaped_ty = mlir::dyn_cast<ShapedType>(src_ty)) {
    src_element_ty = src_shaped_ty.getElementType();
    dst_ty = src_shaped_ty.clone(src_shaped_ty.getShape(), dst_element_ty);
    fp16_ty = src_shaped_ty.clone(src_shaped_ty.getShape(), b.getF16Type());
    fp32_ty = src_shaped_ty.clone(src_shaped_ty.getShape(), b.getF32Type());
  }
  if (src_ty == dst_ty) {
    return value;
  }

  // All operations on bf16 are done through f32.
  if (src_element_ty.isBF16()) {
    return Cast(b, b.create<ma::ExtFOp>(fp32_ty, value), dst_element_ty);
  }
  if (dst_element_ty.isBF16()) {
    // S8 -> BF16 is directly supported and doesn't need to go through f32.
    if (!src_element_ty.isInteger(8)) {
      return b.create<ma::TruncFOp>(dst_ty, Cast(b, value, b.getF32Type()));
    }
  }

  // float => float
  auto src_fp_element_ty = mlir::dyn_cast<mlir::FloatType>(src_element_ty);
  auto dst_fp_element_ty = mlir::dyn_cast<mlir::FloatType>(dst_element_ty);
  if (src_fp_element_ty && dst_fp_element_ty) {
    // F8 <-> FP16, BF16, FP32, FP64 need to be handled via Triton's tt.fp_to_fp
    // because LLVM doesn't support casts from/to FP8.
    // TODO(b/266862493): Add end-to-end test once FP8 support lands in XLA as
    // we can't test the code below without patching the feature.
    if (IsFp8Type(src_element_ty) && !IsFp8Type(dst_element_ty)) {
      return b.create<mt::FpToFpOp>(dst_ty, value);
    }
    if (IsFp8Type(dst_element_ty) && !IsFp8Type(src_element_ty)) {
      return b.create<mt::FpToFpOp>(
          dst_ty, value,
          mt::RoundingModeAttr::get(b.getContext(), mt::RoundingMode::RTNE));
    }
    if (IsFp8Type(src_element_ty) && IsFp8Type(dst_element_ty)) {
      // FP8 <-> FP8 conversion needs to go through FP16
      auto fp16_value = b.create<mt::FpToFpOp>(fp16_ty, value);
      return b.create<mt::FpToFpOp>(
          dst_ty, fp16_value,
          mt::RoundingModeAttr::get(b.getContext(), mt::RoundingMode::RTNE));
    }

    if (src_fp_element_ty.getFPMantissaWidth() >
        dst_fp_element_ty.getFPMantissaWidth()) {
      return b.create<ma::TruncFOp>(dst_ty, value);
    } else {
      return b.create<ma::ExtFOp>(dst_ty, value);
    }
  }
  // int => int
  if (mlir::isa<mlir::IntegerType>(src_element_ty) &&
      mlir::isa<mlir::IntegerType>(dst_element_ty)) {
    if (src_element_ty.getIntOrFloatBitWidth() <
        dst_element_ty.getIntOrFloatBitWidth()) {
      if (src_element_ty.isInteger(1)) {
        return b.create<ma::ExtUIOp>(dst_ty, value);
      }
      return b.create<ma::ExtSIOp>(dst_ty, value);
    }
    return b.create<ma::TruncIOp>(dst_ty, value);
  }
  // int => float
  if (mlir::isa<mlir::IntegerType>(src_element_ty) && dst_fp_element_ty) {
    // TODO(b/266862493): Support unsigned integer types.
    if (src_element_ty.isInteger(1)) {
      return b.create<ma::UIToFPOp>(dst_ty, value);
    }
    return b.create<ma::SIToFPOp>(dst_ty, value);
  }
  // float => int
  if (src_fp_element_ty && mlir::isa<mlir::IntegerType>(dst_element_ty)) {
    if (dst_element_ty.isInteger(1)) {
      return b.create<ma::CmpFOp>(ma::CmpFPredicate::UNE, value,
                                  ZerosLike(b, value));
    }
    // TODO(b/266862493): Support unsigned integer types.
    // The current logic handles signed integer types only. Additional handling
    // is needed for unsigned integer types.
    auto cst_int = [&](int64_t x) {
      if (auto src_shaped_ty = mlir::dyn_cast<ShapedType>(src_ty)) {
        return CreateConst(b, dst_element_ty, x, src_shaped_ty.getShape())
            .UnwrapUnsafe();
      } else {
        return CreateConst(b, dst_element_ty, x).UnwrapUnsafe();
      }
    };
    auto cst_float = [&](int64_t x) {
      if (auto src_shaped_ty = mlir::dyn_cast<ShapedType>(src_ty)) {
        return CreateConst(b, src_fp_element_ty, x, src_shaped_ty.getShape())
            .UnwrapUnsafe();
      } else {
        return CreateConst(b, src_fp_element_ty, x).UnwrapUnsafe();
      }
    };
    auto fptosi = b.create<ma::FPToSIOp>(dst_ty, value);
    int64_t min = llvm::minIntN(dst_element_ty.getIntOrFloatBitWidth());
    int64_t max = llvm::maxIntN(dst_element_ty.getIntOrFloatBitWidth());

    // value <= static_cast<float>(INT_MIN) ? INT_MIN : ...
    auto clamped = b.create<ma::SelectOp>(
        b.create<ma::CmpFOp>(ma::CmpFPredicate::OLE, value, cst_float(min)),
        cst_int(min), fptosi);
    // value >= static_cast<float>(INT_MAX) ? INT_MAX : ...
    clamped = b.create<ma::SelectOp>(
        b.create<ma::CmpFOp>(ma::CmpFPredicate::OGE, value, cst_float(max)),
        cst_int(max), clamped);
    // isnan(value) ? 0 : ...
    return b.create<ma::SelectOp>(
        b.create<ma::CmpFOp>(ma::CmpFPredicate::UNO, value, value), cst_int(0),
        clamped);
  }

  LOG(FATAL) << "Type conversion not supported: "
             << llvm_ir::DumpToString(src_element_ty) << " -> "
             << llvm_ir::DumpToString(dst_element_ty);
}

Value Subtract(EmitterLocOpBuilder& b, ValueRange values) {
  if (mlir::isa<mlir::IntegerType>(mlir::getElementTypeOrSelf(values[0]))) {
    return b.create<ma::SubIOp>(values[0], values[1]);
  } else {
    return b.create<ma::SubFOp>(values[0], values[1]);
  }
}

Value Compare(EmitterLocOpBuilder& b, ValueRange values,
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

Value Maximum(EmitterLocOpBuilder& b, const se::DeviceDescription& device_info,
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

Value Minimum(EmitterLocOpBuilder& b, const se::DeviceDescription& device_info,
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

ScalarOrTensor Splat(EmitterLocOpBuilder& b, ScalarOrTensor value,
                     ArrayRef<int64_t> shape) {
  CHECK(!shape.empty());
  auto type = mlir::RankedTensorType::get(shape, value.getType());
  return ScalarOrTensor(b.create<mt::SplatOp>(type, value.UnwrapUnsafe()));
}

bool IsSupportedElementwiseLibdeviceFunction(const HloInstruction& hlo) {
  auto dev_fn_id = GetTargetDeviceFunctionID(hlo.opcode());
  if (!dev_fn_id.has_value()) {
    return false;
  }
  PrimitiveType output_type = hlo.shape().element_type();
  return output_type == PrimitiveType::BF16 ||
         output_type == PrimitiveType::F16 ||
         output_type == PrimitiveType::F32 || output_type == PrimitiveType::F64;
}

absl::StatusOr<Value> EmitElementwiseLibdeviceFunction(
    EmitterLocOpBuilder& b, absl::string_view libdevice_path,
    const se::DeviceDescription& device_info, const HloInstruction& hlo,
    ValueRange inputs) {
  auto dev_fn_id = GetTargetDeviceFunctionID(hlo.opcode());
  if (!dev_fn_id.has_value()) {
    return absl::InvalidArgumentError(
        absl::StrCat("No libdevice function for operation ", hlo.ToString()));
  }
  PrimitiveType output_type = hlo.shape().element_type();
  if (output_type != PrimitiveType::BF16 && output_type != PrimitiveType::F16 &&
      output_type != PrimitiveType::F32 && output_type != PrimitiveType::F64) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported elementwise operation ", hlo.ToString()));
  }
  llvm::Triple triple("nvptx64-unknown-unknown");
  if (std::holds_alternative<se::RocmComputeCapability>(
          device_info.gpu_compute_capability())) {
    triple.setTriple("amdgcn-unknown-unknown");
  }
  llvm::SmallVector<Value, 2> casted_inputs;
  if (output_type == PrimitiveType::BF16 || output_type == PrimitiveType::F16) {
    // Upcast the inputs to F32.
    for (int64_t i = 0; i < inputs.size(); ++i) {
      casted_inputs.push_back(Cast(b, inputs[i], b.getF32Type()));
    }
  } else {
    casted_inputs.assign(inputs.begin(), inputs.end());
  }
  Value res = b.create<mt::ExternElementwiseOp>(
      casted_inputs[0].getType(), casted_inputs, "libdevice", libdevice_path,
      ObtainDeviceFunctionName(dev_fn_id.value(), output_type, triple),
      /*pure=*/true);
  if (output_type == PrimitiveType::BF16 || output_type == PrimitiveType::F16) {
    // Downcast back to the original output type.
    TF_ASSIGN_OR_RETURN(auto dst_ty, TritonType(b, output_type));
    res = Cast(b, res, dst_ty);
  }
  return res;
}

absl::StatusOr<Value> EmitElementwise(EmitterLocOpBuilder& b,
                                      absl::string_view libdevice_path,
                                      const se::DeviceDescription& device_info,
                                      const HloInstruction& hlo,
                                      ValueRange inputs) {
  if (IsSupportedElementwiseLibdeviceFunction(hlo)) {
    return EmitElementwiseLibdeviceFunction(b, libdevice_path, device_info, hlo,
                                            inputs);
  }
  const bool is_integer =
      mlir::isa<mlir::IntegerType>(getElementTypeOrSelf(inputs[0].getType()));

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
      return Cast(b, inputs[0], dst_ty);
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

absl::StatusOr<ScalarOrTensor> EmitConstant(EmitterLocOpBuilder& b,
                                            const HloInstruction& constant) {
  TF_ASSIGN_OR_RETURN(Type ty, TritonType(b, constant.shape().element_type()));
  llvm::SmallVector<int64_t> shape{constant.shape().dimensions().begin(),
                                   constant.shape().dimensions().end()};

  if (constant.shape().AreAllLeavesIntegers()) {
    if (constant.shape().element_type() == U64) {
      return CreateConst(b, ty, ScalarConstantValue<uint64_t>(constant, U64),
                         shape);
    } else {
      return CreateConst(b, ty, ScalarConstantValue<int64_t>(constant, S64),
                         shape);
    }
  }
  return CreateConst(b, ty, ScalarConstantValue<double>(constant, F64), shape);
}

}  // namespace xla::gpu::triton
