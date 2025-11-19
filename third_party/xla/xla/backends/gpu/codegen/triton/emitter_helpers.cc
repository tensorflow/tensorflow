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
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/TargetParser/Triple.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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
#include "xla/backends/gpu/codegen/triton/ir/triton_xla_ops.h"
#include "xla/backends/gpu/codegen/triton/tma_utils.h"
#include "xla/codegen/emitter_loc_op_builder.h"
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
#include "xla/service/gpu/target_util.h"
#include "xla/service/gpu/triton_fusion_analysis.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/shape.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/tma_metadata.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

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

namespace {
using TensorValue = mlir::TypedValue<mlir::RankedTensorType>;

// Emit a value as Index clamped to [lower, upper].
Value EmitClampedIndex(EmitterLocOpBuilder b, Value value, int64_t lower,
                       int64_t upper) {
  Value clamped_index =
      b.create<ma::MaxSIOp>(value, CreateConst(b, value.getType(), lower));
  clamped_index = b.create<ma::MinSIOp>(clamped_index,
                                        CreateConst(b, value.getType(), upper));
  return b.create<ma::IndexCastOp>(b.getIndexType(), clamped_index);
}

absl::StatusOr<SmallVector<Value>> ComputeOffsetsForTile(
    EmitterLocOpBuilder b, Value pid, ValueRange runtime_values,
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
    dims.push_back(triton::Cast(b, clamped_index, pid.getType()));
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
    EmitterLocOpBuilder b, const HloFusionInstruction& fusion_instruction,
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

  return EmitScope(b, /*analysis=*/nullptr, to_emit, region_values);
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
  if (t.isInteger(64)) return S64;
  if (t.isInteger(32)) return S32;
  if (t.isInteger(16)) return S16;
  if (t.isInteger(8)) return S8;
  if (t.isInteger(4)) return S4;
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

  if (src_ty.isIndex() || dst_ty.isIndex()) {
    return b.create<ma::IndexCastOp>(dst_ty, value);
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
    if (IsFp8Type(src_element_ty) && IsFp8Type(dst_element_ty)) {
      // FP8 <-> FP8 conversion needs to go through FP16
      auto fp16_value = b.create<ma::ExtFOp>(fp16_ty, value);
      return b.create<ma::TruncFOp>(dst_ty, fp16_value);
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
    // int => bool is always value != 0.
    if (dst_element_ty.isInteger(1)) {
      return b.create<ma::CmpIOp>(ma::CmpIPredicate::ne, value,
                                  ZerosLike(b, value));
    }
    return b.create<ma::TruncIOp>(dst_ty, value);
  }
  // int => float
  if (mlir::isa<mlir::IntegerType>(src_element_ty) && dst_fp_element_ty) {
    // The current logic handles signed integer types only.
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
    // The current logic handles signed integer types only. Additional handling
    // is needed for unsigned integer types.
    auto cst_int = [&](int64_t x) -> Value {
      if (auto src_shaped_ty = mlir::dyn_cast<ShapedType>(src_ty)) {
        return CreateConst(b, dst_element_ty, x, src_shaped_ty.getShape());
      } else {
        return CreateConst(b, dst_element_ty, x);
      }
    };
    auto cst_float = [&](int64_t x) -> Value {
      if (auto src_shaped_ty = mlir::dyn_cast<ShapedType>(src_ty)) {
        return CreateConst(b, src_fp_element_ty, x, src_shaped_ty.getShape());
      } else {
        return CreateConst(b, src_fp_element_ty, x);
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

Value Maximum(EmitterLocOpBuilder& b, ValueRange values) {
  auto type = mlir::getElementTypeOrSelf(values[0]);
  if (mlir::isa<mlir::FloatType>(type)) {
    return b.create<ma::MaximumFOp>(values);
  }

  if (type.isInteger(1)) {
    return b.create<ma::OrIOp>(values);
  }

  return b.create<ma::MaxSIOp>(values);
}

Value Minimum(EmitterLocOpBuilder& b, ValueRange values) {
  auto type = mlir::getElementTypeOrSelf(values[0]);
  if (mlir::isa<mlir::FloatType>(type)) {
    return b.create<ma::MinimumFOp>(values);
  }

  if (type.isInteger(1)) {
    return b.create<ma::AndIOp>(values);
  }

  return b.create<ma::MinSIOp>(values);
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

// TODO(willfroom): Remove this (and associated functions) once the legacy
// matmul is removed.
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
  if (device_info.gpu_compute_capability().IsRocm()) {
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
        // XLA add semantics for predicates is equal to bitwise OR, while Arith
        // defines it differently. Replace add with or in this case.
        if (getElementTypeOrSelf(inputs[0]).isInteger(1)) {
          return b.create<ma::OrIOp>(inputs[0], inputs[1]);
        }
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
      return Maximum(b, inputs);
    case HloOpcode::kMinimum:
      return Minimum(b, inputs);
    case HloOpcode::kClamp:
      return Minimum(b, {Maximum(b, {inputs[0], inputs[1]}), inputs[2]});
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
      return mh::reducePrecision<mlir::tensor::BitcastOp>(
          b.getLoc(), inputs[0], hlo.exponent_bits(), hlo.mantissa_bits(), &b);
    case HloOpcode::kAcos:
      return b.create<mm::AcosOp>(inputs[0]);
    case HloOpcode::kAcosh:
      return b.create<mm::AcoshOp>(inputs[0]);
    case HloOpcode::kAsin:
      return b.create<mm::AsinOp>(inputs[0]);
    case HloOpcode::kAsinh:
      return b.create<mm::AsinhOp>(inputs[0]);
    case HloOpcode::kAtan2:
      return b.create<mm::Atan2Op>(inputs[0], inputs[1]);
    case HloOpcode::kAtanh:
      return b.create<mm::AtanhOp>(inputs[0]);
    case HloOpcode::kCos:
      return b.create<mm::CosOp>(inputs[0]);
    case HloOpcode::kCosh:
      return b.create<mm::CoshOp>(inputs[0]);
    case HloOpcode::kExp:
      return b.create<mm::ExpOp>(inputs[0]);
    case HloOpcode::kErf:
      return b.create<mm::ErfOp>(inputs[0]);
    case HloOpcode::kExpm1:
      return b.create<mm::ExpM1Op>(inputs[0]);
    case HloOpcode::kLog:
      return b.create<mm::LogOp>(inputs[0]);
    case HloOpcode::kLog1p:
      return b.create<mm::Log1pOp>(inputs[0]);
    case HloOpcode::kPower:
      return b.create<mm::PowFOp>(inputs[0], inputs[1]);
    case HloOpcode::kRemainder:
      return b.create<ma::RemFOp>(inputs[0], inputs[1]);
    case HloOpcode::kRsqrt:
      return b.create<mm::RsqrtOp>(inputs[0]);
    case HloOpcode::kSin:
      return b.create<mm::SinOp>(inputs[0]);
    case HloOpcode::kSinh:
      return b.create<mm::SinhOp>(inputs[0]);
    case HloOpcode::kSqrt:
      return b.create<mm::SqrtOp>(inputs[0]);
    case HloOpcode::kTan:
      return b.create<mm::TanOp>(inputs[0]);
    case HloOpcode::kTanh:
      return b.create<mm::TanhOp>(inputs[0]);
    case HloOpcode::kCbrt:
      return b.create<mm::CbrtOp>(inputs[0]);
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported elementwise operation ", hlo.ToString()));
  }
}

absl::StatusOr<mlir::TypedValue<mlir::RankedTensorType>> EmitConstant(
    EmitterLocOpBuilder& b, const HloInstruction& constant) {
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

Value Bitcast(EmitterLocOpBuilder& b, Value value, Type type) {
  auto value_type = value.getType();
  value_type = mlir::dyn_cast<ShapedType>(value_type).clone(type);
  return b.create<mlir::arith::BitcastOp>(value_type, value);
}

std::vector<llvm::Metadata*> ExtractNvvmAnnotations(
    llvm::Module* ll_triton_module) {
  std::vector<llvm::Metadata*> captured_nvvm_annotations;
  llvm::NamedMDNode* nvvm_annotations =
      ll_triton_module->getNamedMetadata("nvvm.annotations");
  if (nvvm_annotations) {
    for (llvm::MDNode* operand : nvvm_annotations->operands()) {
      captured_nvvm_annotations.push_back(operand);
    }
    ll_triton_module->eraseNamedMetadata(nvvm_annotations);
  }
  return captured_nvvm_annotations;
}

absl::StatusOr<stream_executor::ThreadDim> ExtractThreadDims(
    mlir::ModuleOp triton_module, mlir::LLVM::LLVMFuncOp func_op) {
  // Extract the launch information from the Triton module.
  auto threads_per_warp_attr =
      triton_module->getAttrOfType<mlir::IntegerAttr>("ttg.threads-per-warp");
  if (!threads_per_warp_attr) {
    return absl::InternalError("ttg.threads-per-warp attribute not found.");
  }
  auto num_warps_attr =
      triton_module->getAttrOfType<mlir::IntegerAttr>("ttg.num-warps");
  if (!num_warps_attr) {
    return absl::InternalError("ttg.num-warps attribute not found.");
  }
  auto total_num_warps_attr =
      triton_module->getAttrOfType<mlir::IntegerAttr>("ttg.total-num-warps");
  if (!total_num_warps_attr) {
    return absl::InternalError("ttg.total-num-warps attribute not found.");
  }
  auto reqntid_attr =
      func_op->getAttrOfType<mlir::DenseI32ArrayAttr>("nvvm.reqntid");
  if (!reqntid_attr) {
    return absl::InternalError("nvvm.reqntid attribute not found.");
  }
  auto reqntids = reqntid_attr.asArrayRef();
  if (reqntids.empty()) {
    return absl::InternalError("nvvm.reqntid attribute is empty.");
  }
  if (reqntids.size() > 3) {
    return absl::InternalError(
        "nvvm.reqntid attribute has more than 3 dimensions.");
  }

  // Validate the launch information.
  if (num_warps_attr.getInt() != total_num_warps_attr.getInt()) {
    VLOG(6)
        << "num_warps and total_num_warps are different! This can happen if "
           "Triton compilation decides to use a different number of warps than "
           "configured. e.g. auto warp specialization can do that.";
  }
  int64_t expected_total_threads = xla::Product<int32_t>(reqntids);
  int64_t actual_total_threads =
      total_num_warps_attr.getInt() * threads_per_warp_attr.getInt();
  if (actual_total_threads != expected_total_threads) {
    return absl::InternalError(absl::StrCat(
        "Expected total threads as per reqntid attribute to be ",
        expected_total_threads, " but got ", actual_total_threads,
        " as per ttg.total-num-warps and tt.threads-per-warp attributes."));
  }

  stream_executor::ThreadDim thread_dims(reqntids[0],
                                         reqntids.size() > 1 ? reqntids[1] : 1,
                                         reqntids.size() > 2 ? reqntids[2] : 1);
  return thread_dims;
}

absl::StatusOr<stream_executor::gpu::TmaMetadata> ExtractTmaMetadata(
    mlir::LLVM::LLVMFuncOp func_op) {
  stream_executor::gpu::TmaMetadata tma_metadata;
  for (auto [idx, arg] : llvm::enumerate(func_op.getArguments())) {
    if (auto attr =
            func_op.getArgAttrOfType<mlir::triton::xla::TmaDescriptorAttr>(
                idx, "tt.tma_descriptor")) {
      TF_ASSIGN_OR_RETURN(
          auto tma_desc,
          CreateTmaDescriptor(attr.getGlobalShape(), attr.getTileShape(),
                              attr.getTileStrides(), attr.getLayout(),
                              attr.getElementByteSize(),
                              attr.getSwizzleMode().getValue()));
      tma_metadata.arg_index_to_tma_info.insert({idx, tma_desc});
    }
  }
  return tma_metadata;
}

mt::PointerType GetGlobalPointerType(mlir::Type element_type) {
  return mlir::cast<mt::PointerType>(mt::getPointerTypeToElement(element_type));
}

/*static */ absl::StatusOr<TileInfo> TileInfo::Construct(
    EmitterLocOpBuilder b, Value pid, ValueRange runtime_values,
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
                      TritonType(b, shape.element_type()));
  auto storage_type = StorageType(expected_element_type);

  auto tile_strides = tiled_hlo.tile_strides();
  auto minor_to_major_layout = llvm::to_vector(LayoutUtil::MinorToMajor(shape));

  return TileInfo(offsets, tile_strides, original_shape, padded_tile_sizes,
                  minor_to_major_layout, storage_type);
}

TensorValue EmitParameterExtract(EmitterLocOpBuilder b,
                                 const TileInfo& tile_info, Value arg) {
  auto tensor_type = mlir::RankedTensorType::get(tile_info.padded_tile_sizes(),
                                                 tile_info.storage_type());

  return b.create<xla::xtile::ExtractTileOp>(
      tensor_type, arg, tile_info.offsets(), tile_info.padded_tile_sizes(),
      tile_info.tile_strides());
}

absl::StatusOr<TensorValue> EmitScope(
    EmitterLocOpBuilder b, const TritonFusionAnalysis* analysis,
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

}  // namespace xla::gpu::triton
