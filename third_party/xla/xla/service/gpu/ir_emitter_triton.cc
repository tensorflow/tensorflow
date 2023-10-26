/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/ir_emitter_triton.h"

#include <climits>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <system_error>  // NOLINT(build/c++11): required to interface with LLVM
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"  // from @llvm-project
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"  // from @llvm-project
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"  // from @llvm-project
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"  // from @llvm-project
#include "mlir/Dialect/Math/IR/Math.h"  // from @llvm-project
#include "mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/ExecutionEngine/OptUtils.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
#include "mlir/IR/Verifier.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"  // from @llvm-project
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"  // from @llvm-project
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"  // from @llvm-project
#include "mlir/Target/LLVMIR/Export.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "xla/autotuning.pb.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/literal.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/mlir_hlo/mhlo/transforms/map_mhlo_to_scalar_op.h"
#include "xla/primitive_util.h"
#include "xla/service/dump.h"
#include "xla/service/gpu/gemm_rewriter_triton.h"
#include "xla/service/gpu/hlo_traversal.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/llvm_gpu_backend/gpu_backend_lib.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/target_util.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/shape_util.h"
#include "xla/status.h"
#include "xla/status_macros.h"
#include "xla/statusor.h"
#include "xla/stream_executor/device_description.h"
#include "xla/translate/hlo_to_mhlo/hlo_function_importer.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/path.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"
#include "triton/Conversion/NVGPUToLLVM/NVGPUToLLVMPass.h"
#include "triton/Conversion/TritonGPUToLLVM/TritonGPUToLLVMPass.h"
#include "triton/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

namespace xla {
namespace gpu {

namespace ma = ::mlir::arith;
namespace mm = ::mlir::math;
namespace ml = ::mlir::LLVM;
namespace mn = ::mlir::NVVM;
namespace mt = ::mlir::triton;

using ::llvm::SmallVector;
using mlir::ArrayRef;
using mlir::ImplicitLocOpBuilder;
using ::mlir::Type;
using ::mlir::Value;
using mlir::ValueRange;

namespace {

// XLA -> Triton type conversions.
Type TritonType(mlir::OpBuilder b, PrimitiveType t) {
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
    default:
      LOG(FATAL) << "This type is not supported yet: "
                 << primitive_util::LowercasePrimitiveTypeName(t);
  }
}

Type StorageType(mlir::OpBuilder b, Type t) {
  if (t.isInteger(1)) {
    return b.getI8Type();
  }
  return t;
}

// Get the value of the scalar constant's literal in a C++ type.
template <typename T>
T ScalarConstantValue(const HloInstruction& instr, PrimitiveType dst_type) {
  CHECK(hlo_query::IsScalarConstant(&instr));
  StatusOr<Literal> converted = instr.literal().Convert(dst_type);
  TF_CHECK_OK(converted.status());
  return converted.value().GetFirstElement<T>();
}

// Create a scalar constant.
template <typename T>
ma::ConstantOp CreateConst(ImplicitLocOpBuilder b, Type type, T value) {
  if (type.isa<mlir::IntegerType>()) {
    return b.create<ma::ConstantOp>(b.getIntegerAttr(type, value));
  }
  if (type.isa<mlir::FloatType>()) {
    return b.create<ma::ConstantOp>(
        b.getFloatAttr(type, static_cast<double>(value)));
  }
  LOG(FATAL) << "Constant type not supported: " << llvm_ir::DumpToString(type);
}

// Create a tensor constant.
template <typename T>
ma::ConstantOp CreateConst(ImplicitLocOpBuilder& b, Type type, T value,
                           ArrayRef<int64_t> shape) {
  auto tensor_type = mlir::RankedTensorType::get(shape, type);
  if (auto int_type = type.dyn_cast<mlir::IntegerType>()) {
    return b.create<ma::ConstantOp>(mlir::DenseElementsAttr::get(
        tensor_type, mlir::APInt(int_type.getIntOrFloatBitWidth(), value)));
  }
  if (auto float_type = type.dyn_cast<mlir::FloatType>()) {
    return b.create<ma::ConstantOp>(mlir::DenseElementsAttr::get(
        tensor_type, b.getFloatAttr(type, static_cast<double>(value))));
  }
  LOG(FATAL) << "Constant type not supported: " << llvm_ir::DumpToString(type);
}

Value ZerosLike(ImplicitLocOpBuilder& b, Value x) {
  if (auto src_shaped_ty = x.getType().dyn_cast<mlir::ShapedType>()) {
    Type src_ty = src_shaped_ty.getElementType();
    return CreateConst(b, src_ty, 0, src_shaped_ty.getShape());
  }
  return CreateConst(b, x.getType(), 0);
}

Value OnesLike(ImplicitLocOpBuilder& b, Value x) {
  if (auto src_shaped_ty = x.getType().dyn_cast<mlir::ShapedType>()) {
    Type src_ty = src_shaped_ty.getElementType();
    return CreateConst(b, src_ty, 1, src_shaped_ty.getShape());
  }
  return CreateConst(b, x.getType(), 1);
}

// Triton type conversions.
Value Cast(ImplicitLocOpBuilder& b, Value value, Type dst_element_ty) {
  Type src_ty = value.getType();
  Type src_element_ty = src_ty;
  Type fp32_ty = b.getF32Type();
  Type dst_ty = dst_element_ty;
  if (auto src_shaped_ty = src_ty.dyn_cast<mlir::ShapedType>()) {
    src_element_ty = src_shaped_ty.getElementType();
    dst_ty = src_shaped_ty.clone(src_shaped_ty.getShape(), dst_element_ty);
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
    return b.create<ma::TruncFOp>(dst_ty, Cast(b, value, b.getF32Type()));
  }

  // float => float
  auto src_fp_element_ty = src_element_ty.dyn_cast<mlir::FloatType>();
  auto dst_fp_element_ty = dst_element_ty.dyn_cast<mlir::FloatType>();
  if (src_fp_element_ty && dst_fp_element_ty) {
    if (src_fp_element_ty.getFPMantissaWidth() >
        dst_fp_element_ty.getFPMantissaWidth()) {
      return b.create<ma::TruncFOp>(dst_ty, value);
    } else {
      return b.create<ma::ExtFOp>(dst_ty, value);
    }
  }
  // int => int
  if (src_element_ty.isa<mlir::IntegerType>() &&
      dst_element_ty.isa<mlir::IntegerType>()) {
    if (src_element_ty.getIntOrFloatBitWidth() <
        dst_element_ty.getIntOrFloatBitWidth()) {
      return b.create<ma::ExtSIOp>(dst_ty, value);
    }
    return b.create<ma::TruncIOp>(dst_ty, value);
  }
  // int => float
  if (src_element_ty.isa<mlir::IntegerType>() && dst_fp_element_ty) {
    // TODO(b/266862493): Support unsigned integer types.
    if (src_element_ty.isInteger(1)) {
      return b.create<ma::UIToFPOp>(dst_ty, value);
    }
    return b.create<ma::SIToFPOp>(dst_ty, value);
  }
  // float => int
  if (src_fp_element_ty && dst_element_ty.isa<mlir::IntegerType>()) {
    // TODO(b/266862493): Support unsigned integer types.
    if (dst_element_ty.isInteger(1)) {
      return b.create<ma::CmpFOp>(ma::CmpFPredicate::UNE, value,
                                  ZerosLike(b, value));
    }
    return b.create<ma::FPToSIOp>(dst_ty, value);
  }

  LOG(FATAL) << "Type conversion not supported: "
             << llvm_ir::DumpToString(src_element_ty) << " -> "
             << llvm_ir::DumpToString(dst_element_ty);
}

Value Subtract(ImplicitLocOpBuilder& b, ValueRange values) {
  if (mlir::getElementTypeOrSelf(values[0]).isa<mlir::IntegerType>()) {
    return b.create<ma::SubIOp>(values[0], values[1]);
  } else {
    return b.create<ma::SubFOp>(values[0], values[1]);
  }
}

Value Compare(ImplicitLocOpBuilder& b, ValueRange values,
              mlir::mhlo::ComparisonDirection direction) {
  if (mlir::getElementTypeOrSelf(values[0]).isa<mlir::IntegerType>()) {
    return b.create<ma::CmpIOp>(
        mlir::mhlo::impl::getCmpPredicate<ma::CmpIPredicate>(direction,
                                                             /*isSigned=*/true)
            .value(),
        values[0], values[1]);
  }
  return b.create<ma::CmpFOp>(
      mlir::mhlo::impl::getCmpPredicate<ma::CmpFPredicate>(direction,
                                                           /*isSigned=*/true)
          .value(),
      values[0], values[1]);
}

Value Maximum(ImplicitLocOpBuilder& b, ValueRange values) {
  // ma::MaximumFOp seems to think that max(NaN, x) = x, so we don't use that.
  //
  // logic: isNaN(lhs) || (!isNan(rhs) && lhs > rhs) ? lhs : rhs
  // See also: IEEE Std 754-2008 5.11.
  //
  // This also works, but we wanted to make it similar to minimum.
  // logic: isNaN(lhs) || lhs > rhs ? lhs : rhs
  Value lhs_is_nan =
      Compare(b, {values[0], values[0]}, mlir::mhlo::ComparisonDirection::NE);
  Value rhs_is_not_nan =
      Compare(b, {values[1], values[1]}, mlir::mhlo::ComparisonDirection::EQ);
  Value lhs_is_greater =
      Compare(b, values, mlir::mhlo::ComparisonDirection::GT);
  return b.create<ma::SelectOp>(
      b.create<ma::OrIOp>(lhs_is_nan,
                          b.create<ma::AndIOp>(rhs_is_not_nan, lhs_is_greater)),
      values[0], values[1]);
}

Value Minimum(ImplicitLocOpBuilder& b, ValueRange values) {
  // ma::MinimumFOp seems to think that min(NaN, x) = x, so we don't use that.
  //
  // logic: isNaN(lhs) || (!isNan(rhs) && lhs < rhs) ? lhs : rhs
  // See also: IEEE Std 754-2008 5.11.
  //
  // This should also work, but the tests show that it doesn't work for
  // minimum(x, NaN):
  // logic: isNaN(lhs) || lhs < rhs ? lhs : rhs
  Value lhs_is_nan =
      Compare(b, {values[0], values[0]}, mlir::mhlo::ComparisonDirection::NE);
  Value rhs_is_not_nan =
      Compare(b, {values[1], values[1]}, mlir::mhlo::ComparisonDirection::EQ);
  Value lhs_is_less = Compare(b, values, mlir::mhlo::ComparisonDirection::LT);
  return b.create<ma::SelectOp>(
      b.create<ma::OrIOp>(lhs_is_nan,
                          b.create<ma::AndIOp>(rhs_is_not_nan, lhs_is_less)),
      values[0], values[1]);
}

// TODO(b/269489810): Contribute nicer builders to Triton, so we don't need to
// define these utilities.
Value Splat(ImplicitLocOpBuilder& b, Value value, ArrayRef<int64_t> shape) {
  auto type = mlir::RankedTensorType::get(shape, value.getType());
  return b.create<mt::SplatOp>(type, value);
}

using TensorValue = mlir::TypedValue<mlir::RankedTensorType>;

Value Broadcast(ImplicitLocOpBuilder& b, TensorValue value,
                ArrayRef<int64_t> shape) {
  return b.create<mt::BroadcastOp>(value.getType().clone(shape), value);
}

Value Range(ImplicitLocOpBuilder& b, int32_t limit) {
  auto type = mlir::RankedTensorType::get(limit, b.getI32Type());
  return b.create<mt::MakeRangeOp>(type, 0, limit);
}

Value AddPtr(ImplicitLocOpBuilder& b, Value ptr, Value offset) {
  return b.create<mt::AddPtrOp>(ptr.getType(), ptr, offset);
}

Value EmitElementwise(ImplicitLocOpBuilder& b, absl::string_view libdevice_path,
                      const HloInstruction& hlo, ValueRange inputs) {
  if (mlir::getElementTypeOrSelf(inputs[0]).isF32() ||
      mlir::getElementTypeOrSelf(inputs[0]).isF64()) {
    auto dev_fn_id = GetTargetDeviceFunctionID(hlo.opcode());
    if (dev_fn_id.ok()) {
      return b.create<mt::ExternElementwiseOp>(
          inputs[0].getType(), inputs, "libdevice", libdevice_path,
          ObtainDeviceFunctionName(dev_fn_id.value(),
                                   hlo.shape().element_type(),
                                   llvm::Triple("nvptx64-unknown-unknown")),
          /*pure=*/true);
    }
  }
  const bool is_integer =
      mlir::getElementTypeOrSelf(inputs[0]).isa<mlir::IntegerType>();

  switch (hlo.opcode()) {
    case HloOpcode::kCopy:
      // Dimension transformations are taken care of separately.
      return inputs[0];
    case HloOpcode::kAbs:
      if (is_integer) {
        return b.create<mm::AbsIOp>(inputs[0]);
      }
      return b.create<mm::AbsFOp>(inputs[0]);
    case HloOpcode::kNot:
      return b.create<ma::XOrIOp>(inputs[0], OnesLike(b, inputs[0]));
    case HloOpcode::kNegate:
      // NegFOp is not supported by Triton.
      return Subtract(b, {ZerosLike(b, inputs[0]), inputs[0]});
    case HloOpcode::kConvert:
      return Cast(b, inputs[0], TritonType(b, hlo.shape().element_type()));
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
      return Maximum(b, inputs);
    case HloOpcode::kMinimum:
      return Minimum(b, inputs);
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
          mlir::mhlo::symbolizeComparisonDirection(
              ComparisonDirectionToString(hlo.comparison_direction()))
              .value());
    case HloOpcode::kSelect:
      return b.create<ma::SelectOp>(
          Compare(b, {inputs[0], ZerosLike(b, inputs[0])},
                  mlir::mhlo::ComparisonDirection::NE),
          inputs[1], inputs[2]);
    default:
      LOG(FATAL) << "Unsupported operation " << hlo.ToString();
  }
}

Value EmitParameterLoad(ImplicitLocOpBuilder& b, Value pointer,
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
  return Splat(b,
               b.create<mt::LoadOp>(pointer, mt::CacheModifier::NONE,
                                    mt::EvictionPolicy::NORMAL,
                                    /*isVolatile=*/false),
               {});
}

Value EmitConstant(ImplicitLocOpBuilder& b, const HloInstruction& constant) {
  Type ty = TritonType(b, constant.shape().element_type());
  if (constant.shape().IsInteger()) {
    if (constant.shape().element_type() == U64) {
      return CreateConst(b, ty, ScalarConstantValue<uint64_t>(constant, U64));
    } else {
      return CreateConst(b, ty, ScalarConstantValue<int64_t>(constant, S64));
    }
  }
  return CreateConst(b, ty, ScalarConstantValue<double>(constant, F64));
}

struct DimProperties {
  DimProperties(int64_t index, Value offset, int block_size, int split_value)
      : index(index),
        offset(offset),
        block_size(block_size),
        split_value(split_value) {}

  int64_t index;
  Value offset;
  int block_size;
  int split_value;
};

Value EmitBroadcast(ImplicitLocOpBuilder& b,
                    const TritonFusionAnalysis* analysis,
                    TritonFusionAnalysis::Scope scope,
                    absl::Span<const DimProperties> tiled_dimensions,
                    const HloInstruction& broadcast, Value input) {
  CHECK(analysis != nullptr);
  std::vector<int64_t> out_shape;
  for (const DimProperties& dim : tiled_dimensions) {
    const TensorIterationSpec::DimIterationSpec* spec =
        analysis->IterSpec(scope, &broadcast, dim.index);
    if (spec != nullptr && spec->at(0).stride > 0) {
      out_shape.push_back(dim.block_size);
    }
  }
  auto tensor_input = input.dyn_cast<TensorValue>();
  if (!tensor_input) {
    // Input is scalar.
    return Splat(b, input, out_shape);
  }
  if (tensor_input.getType().getRank() == out_shape.size()) {
    // No dimensions to broadcast.
    return input;
  }
  // Add broadcasted dimensions one by one.
  Value expanded_input = tensor_input;
  int dim_idx = 0;
  for (const DimProperties& dim : tiled_dimensions) {
    if (analysis->IterSpec(scope, &broadcast, dim.index) != nullptr &&
        analysis->IterSpec(scope, &broadcast, dim.index)->at(0).stride > 0) {
      if (analysis->IterSpec(scope, broadcast.operand(0), dim.index) ==
          nullptr) {
        // Broadcasted dimension.
        expanded_input = b.create<mt::ExpandDimsOp>(expanded_input, dim_idx);
      }
      ++dim_idx;
    }
  }
  return Broadcast(b, expanded_input.cast<TensorValue>(), out_shape);
}

StatusOr<Value> EmitScope(
    ImplicitLocOpBuilder& b, absl::string_view libdevice_path,
    const TritonFusionAnalysis* analysis, TritonFusionAnalysis::Scope scope,
    absl::Span<const DimProperties> tiled_dimensions,
    absl::Span<const HloInstruction* const> instructions,
    absl::flat_hash_map<const HloInstruction*, Value>& values);

StatusOr<Value> EmitReduce(ImplicitLocOpBuilder& b,
                           absl::string_view libdevice_path,
                           const HloInstruction& hlo_reduce, Value input) {
  llvm::ArrayRef<int64_t> input_shape =
      input.cast<TensorValue>().getType().getShape();

  // At the moment, we should only emit a full reduction over the last axis of
  // a single input.
  CHECK_EQ(hlo_reduce.operand_count(), 2);
  CHECK_EQ(hlo_reduce.dimensions().size(), 1);
  CHECK_EQ(hlo_reduce.dimensions(0), hlo_reduce.operand(0)->shape().rank() - 1);
  const int block_row = input_shape.back();
  const int row_len = hlo_reduce.operand(0)->shape().dimensions_minor(0);
  CHECK_GE(block_row, row_len);

  const HloInstruction* operand = hlo_reduce.operand(1);
  Value neutral;

  // We assume that the reduction value was input as a constant, or in the case
  // of a data type affected by float normalization, a convert of a constant.
  if (operand->opcode() == HloOpcode::kConvert) {
    CHECK_EQ(operand->operand(0)->opcode(), HloOpcode::kConstant);
    CHECK_EQ(operand->operand(0)->shape().element_type(), BF16);
    PrimitiveType dest_ty = operand->shape().element_type();
    CHECK_EQ(dest_ty, F32);
    neutral = EmitConstant(b, *operand->operand(0));
    neutral = Cast(b, neutral, TritonType(b, dest_ty));
  } else {
    CHECK_EQ(operand->opcode(), HloOpcode::kConstant);
    neutral = EmitConstant(b, *operand);
  }

  // Since every shape is padded to a power of 2 in Triton, the input tile may
  // be padded with arbitrary values. These values could affect the result of
  // the reduction, so we need to mask them away. Luckily, we have a monoid
  // structure (element_type, hlo_reduce.to_apply(), hlo_reduce.operand(1))---
  // up to floating-point inaccuracies. Masking the input using
  // hlo_reduce.operand(1) is thus always the right choice to ensure that the
  // reduction is computed correctly, since it is the neutral value with regards
  // to the reducer.
  if (block_row != row_len) {
    Value mask = b.create<ma::CmpIOp>(
        ma::CmpIPredicate::slt, Range(b, block_row),
        Splat(b, CreateConst(b, b.getI32Type(), row_len), block_row));
    input = b.create<ma::SelectOp>(mask, input, Splat(b, neutral, input_shape));
  }

  // Triton actually only performs reductions on float32 inputs, and we must
  // thus upcast/downcast our input if its data type is different.
  Value casted_input = Cast(b, input, b.getF32Type());

  mt::ReduceOp reduction = b.create<mt::ReduceOp>(
      SmallVector<Value>({casted_input}), (int)input_shape.size() - 1);
  {
    mlir::Location loc = b.getLoc();
    mlir::Block* reducer =
        b.createBlock(&reduction->getRegion(0), {},
                      {b.getF32Type(), b.getF32Type()}, {loc, loc});

    HloComputation* reduction_computation = hlo_reduce.to_apply();

    std::vector<const HloInstruction*> to_emit;
    absl::flat_hash_map<const HloInstruction*, Value> region_values;
    for (const HloInstruction* instr :
         reduction_computation->MakeInstructionPostOrder()) {
      if (instr->opcode() == HloOpcode::kParameter) {
        int parameter_number = instr->parameter_number();
        CHECK_LT(parameter_number, 2);
        CHECK(region_values
                  .insert({instr, reducer->getArgument(parameter_number)})
                  .second);
      } else {
        to_emit.push_back(instr);
      }
    }

    CHECK(!to_emit.empty());

    b.setInsertionPointToStart(reducer);
    TF_ASSIGN_OR_RETURN(Value result,
                        EmitScope(b, libdevice_path, /*analysis=*/nullptr,
                                  TritonFusionAnalysis::Scope::OUTPUT, {},
                                  to_emit, region_values));
    b.create<mt::ReduceReturnOp>(SmallVector<Value>({result}));
    b.setInsertionPointAfter(reduction);
  }

  Value result = reduction.getResult().front();

  // We want to return a tensor of float32, but the ReturnReduceOp produces an
  // f32 constant when reducing a single dim. To convert to a tensor we splat
  // the result.
  if (!reduction.getResult().front().dyn_cast<TensorValue>()) {
    result = Splat(b, result, {});
  }

  return Cast(b, result, TritonType(b, hlo_reduce.shape().element_type()));
}

// Emit sequence of instructions using compatible tiling ordered producers
// before consumers.
StatusOr<Value> EmitScope(
    ImplicitLocOpBuilder& b, absl::string_view libdevice_path,
    const TritonFusionAnalysis* analysis, TritonFusionAnalysis::Scope scope,
    absl::Span<const DimProperties> tiled_dimensions,
    absl::Span<const HloInstruction* const> instructions,
    absl::flat_hash_map<const HloInstruction*, Value>& values) {
  for (const HloInstruction* hlo : instructions) {
    Value result;
    if (hlo->opcode() == HloOpcode::kParameter) {
      // Parameter loads are handled outside EmitScope.
      TF_RET_CHECK(values.contains(hlo)) << hlo->ToString();
      continue;
    } else if (hlo->opcode() == HloOpcode::kConstant) {
      // Splat makes it a tensor to avoid type mismatches.
      result = Splat(b, EmitConstant(b, *hlo), {});
    } else if (hlo->opcode() == HloOpcode::kBroadcast) {
      result = EmitBroadcast(b, analysis, scope, tiled_dimensions, *hlo,
                             values[hlo->operand(0)]);
    } else if (hlo->opcode() == HloOpcode::kReduce) {
      TF_ASSIGN_OR_RETURN(
          result, EmitReduce(b, libdevice_path, *hlo, values[hlo->operand(0)]));
    } else if (hlo->IsElementwise()) {
      std::vector<Value> operands;
      operands.reserve(hlo->operands().size());
      for (const HloInstruction* operand : hlo->operands()) {
        operands.push_back(values[operand]);
      }
      result = EmitElementwise(b, libdevice_path, *hlo, operands);
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
      LOG(FATAL) << hlo->ToString();
    }
    TF_RET_CHECK(values.insert({hlo, result}).second) << hlo->ToString();
    VLOG(8) << "Emitted " << hlo->ToString(HloPrintOptions::ShortParsable());
  }
  return values[instructions.back()];
}

void CreateTritonPipeline(mlir::OpPassManager& pm,
                          const se::CudaComputeCapability& cc, int num_warps,
                          int num_stages) {
  const int ccAsInt = cc.major * 10 + cc.minor;
  // Based on optimize_ttir() in
  // @triton//:python/triton/compiler/compiler.py
  pm.addPass(mt::createRewriteTensorPointerPass());
  pm.addPass(mlir::createInlinerPass());
  pm.addPass(mt::createCombineOpsPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mt::createReorderBroadcastPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createLoopInvariantCodeMotionPass());
  pm.addPass(mlir::createSymbolDCEPass());
  // Based on ttir_to_ttgir() in
  // @triton//:python/triton/compiler/compiler.py
  pm.addPass(mt::createConvertTritonToTritonGPUPass(num_warps));
  // Based on optimize_ttgir() in
  // @triton//:python/triton/compiler/compiler.py
  pm.addPass(mlir::createTritonGPUCoalescePass());
  pm.addPass(mlir::createTritonNvidiaGPUPlanCTAPass());
  pm.addPass(mlir::createTritonGPURemoveLayoutConversionsPass());
  pm.addPass(mlir::createTritonGPUAccelerateMatmulPass(ccAsInt));
  pm.addPass(mlir::createTritonGPURemoveLayoutConversionsPass());
  pm.addPass(mlir::createTritonGPUOptimizeDotOperandsPass());
  pm.addPass(mlir::createTritonGPUPipelinePass(num_stages, num_warps));
  pm.addPass(mlir::createTritonNvidiaGPUMaterializeLoadStorePass());
  pm.addPass(mlir::createTritonGPUPrefetchPass());
  pm.addPass(mlir::createTritonGPUOptimizeDotOperandsPass());
  pm.addPass(mlir::createTritonGPURemoveLayoutConversionsPass());
  pm.addPass(mlir::createTritonGPUDecomposeConversionsPass());
  pm.addPass(mlir::createTritonNvidiaGPUWSFixupMissingAttrs());
  pm.addPass(mlir::createTritonGPUReorderInstructionsPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(mlir::createTritonNvidiaGPUWSFixupMissingAttrs());
  // Based on translateTritonGPUToLLVMIR() in
  // @triton//:lib/Target/LLVMIR/LLVMIRTranslation.cpp
  pm.addPass(mlir::createConvertSCFToCFPass());
  pm.addPass(mlir::createConvertIndexToLLVMPass());
  pm.addPass(
      mt::createConvertTritonGPUToLLVMPass(ccAsInt, mt::Default, nullptr));
  pm.addPass(mt::createConvertNVGPUToLLVMPass());
  pm.addPass(mlir::createArithToLLVMConversionPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createSymbolDCEPass());
  // Note: translateTritonGPUToLLVMIR adds line info with LLVMDIScopePass.
}

// Extract additional attributes from an LLVM function that are not passed
// to the builder directly.
SmallVector<mlir::NamedAttribute> GetExtraAttrs(ml::LLVMFuncOp func) {
  llvm::StringSet<> registered_attr_names{
      func.getSymNameAttrName().getValue(),
      func.getFunctionTypeAttrName().getValue(),
      func.getLinkageAttrName().getValue(),
      func.getDsoLocalAttrName().getValue(),
      func.getCConvAttrName().getValue(),
      func.getArgAttrsAttrName().getValue(),
      func.getFunctionEntryCountAttrName().getValue()};
  return llvm::to_vector(
      llvm::make_filter_range(func->getAttrs(), [&](mlir::NamedAttribute attr) {
        return !registered_attr_names.contains(attr.getName().getValue());
      }));
}

// Strip address spaces from function parameters.
void StripParameterAddressSpaces(mlir::RewriterBase& rewriter,
                                 ml::LLVMFuncOp func) {
  // Figure out what the new signature should be.
  ml::LLVMFunctionType func_ty = func.getFunctionType();
  SmallVector<Type> generic_func_params(
      llvm::map_range(func_ty.getParams(), [](Type type) -> Type {
        auto ptr_ty = type.dyn_cast<ml::LLVMPointerType>();
        if (!ptr_ty) return type;
        if (ptr_ty.getAddressSpace() != mn::kGlobalMemorySpace) return type;
        return ml::LLVMPointerType::get(ptr_ty.getElementType());
      }));
  ml::LLVMFunctionType generic_func_ty =
      func_ty.clone(generic_func_params, func_ty.getReturnTypes());

  // Create a function with the new signature.
  SmallVector<mlir::DictionaryAttr> arg_attrs(llvm::map_range(
      func.getArgAttrsAttr().getValue(),
      [](mlir::Attribute attr) { return attr.cast<mlir::DictionaryAttr>(); }));
  auto generic_func = rewriter.create<ml::LLVMFuncOp>(
      func.getLoc(), func.getSymName(), generic_func_ty, func.getLinkage(),
      func.getDsoLocal(), func.getCConv(), /*comdat=*/nullptr,
      GetExtraAttrs(func), arg_attrs, func.getFunctionEntryCount());

  // Convert generic address spaces back to original ones within the function
  // body.
  mlir::Block* entry = generic_func.addEntryBlock();
  rewriter.setInsertionPointToEnd(entry);
  SmallVector<Value> converted_args;
  for (auto [arg, type] :
       llvm::zip(generic_func.getArguments(), func_ty.getParams())) {
    Value converted = arg;
    if (arg.getType() != type) {
      converted = rewriter.create<ml::AddrSpaceCastOp>(arg.getLoc(), type, arg);
    }
    converted_args.push_back(converted);
  }

  // Move the rest of function body from the original function.
  rewriter.cloneRegionBefore(func.getBody(), generic_func.getBody(),
                             generic_func.getBody().end());
  rewriter.eraseOp(func);
  rewriter.mergeBlocks(entry->getNextNode(), entry, converted_args);
}

// Rewrite signatures of kernel functions to use generic data pointers and
// cast them to global ones within the kernel.
struct GeneralizeKernelSignaturePass
    : mlir::PassWrapper<GeneralizeKernelSignaturePass, mlir::OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GeneralizeKernelSignaturePass);
  void runOnOperation() override {
    mlir::IRRewriter rewriter(&getContext());
    getOperation()->walk([&](ml::LLVMFuncOp func) {
      if (!func->hasAttr(mn::NVVMDialect::getKernelFuncAttrName())) {
        return;
      }
      rewriter.setInsertionPointAfter(func);
      StripParameterAddressSpaces(rewriter, func);
    });
  }
};

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
  MatMulDims(const AutotuneResult::TritonGemmKey& config,
             const HloDotInstruction& dot,
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
};

// Structure for parameters relating to the MatMul launch grid.
struct MatMulLaunchConfig {
  explicit MatMulLaunchConfig(const AutotuneResult::TritonGemmKey& config,
                              const HloDotInstruction& dot,
                              const MatMulDims& dims);

  int64_t grid_m;
  int64_t grid_n;
  LaunchDimensions launch_dims;
  mt::ProgramIDDim batch_program_id_dim;
  mt::ProgramIDDim noncontracting_program_id_dim;
};

MatMulDims::MatMulDims(const AutotuneResult::TritonGemmKey& config,
                       const HloDotInstruction& dot,
                       const TritonFusionAnalysis& analysis) {
  if (config.split_k() > 1) {
    // split-k is always the first logical dimension.
    out_split_k_dim_idx = 0;
  }

  int64_t num_split_k_dims = config.split_k() > 1 ? 1 : 0;
  const auto& dims = dot.dot_dimension_numbers();
  lhs_contracting_dim_idx = dims.lhs_contracting_dimensions(0);
  lhs_noncontracting_dim_idx =
      GetNonContractingDims(dot.operand(0)->shape(),
                            dims.lhs_batch_dimensions(),
                            dims.lhs_contracting_dimensions())
          .value()[0];
  rhs_contracting_dim_idx = dims.rhs_contracting_dimensions(0);
  rhs_noncontracting_dim_idx =
      GetNonContractingDims(dot.operand(1)->shape(),
                            dims.rhs_batch_dimensions(),
                            dims.rhs_contracting_dimensions())
          .value()[0];

  if (dims.lhs_batch_dimensions_size() > num_split_k_dims) {
    lhs_batch_dim_idx = *dims.lhs_batch_dimensions().rbegin();
    rhs_batch_dim_idx = *dims.rhs_batch_dimensions().rbegin();
    // The batch dimension (if present) comes after the split-k dimension (if
    // present, otherwise it's the first dimension).
    out_batch_dim_idx = num_split_k_dims;
  }

  // Logical output dimensions are always ordered as:
  //   split-K, batch, non-contracting LHS, non-contracting RHS,
  // where split-K and batch are optional.
  out_rhs_noncontracting_dim_idx = dot.shape().rank() - 1;
  out_lhs_noncontracting_dim_idx = dot.shape().rank() - 2;

  auto* root = dot.parent()->root_instruction();
  n = analysis
          .IterSpec(TritonFusionAnalysis::Scope::OUTPUT, root,
                    out_rhs_noncontracting_dim_idx)
          ->at(0)
          .count;
  // Contracting dimension length.
  if (config.split_k() > 1 &&
      dot.operand(0)->operand(0)->opcode() == HloOpcode::kPad) {
    // Unpadded LHS shape:  [..., k, ...]
    // Padded LHS shape:    [..., padded_k, ...]
    // Bitcasted LHS shape: [..., split_k, padded_k / split_k, ...]
    CHECK_EQ(dot.operand(0)->opcode(), HloOpcode::kBitcast);
    const Shape& unpadded_lhs_shape =
        dot.operand(0)->operand(0)->operand(0)->shape();
    k = unpadded_lhs_shape.dimensions(dims.lhs_contracting_dimensions(0) - 1);
  } else {
    k = dot.operand(0)->shape().dimensions(dims.lhs_contracting_dimensions(0)) *
        config.split_k();
  }

  auto* lhs_noncontracting_split_spec =
      GetLhsNoncontractingSplitSpec(analysis, lhs_noncontracting_dim_idx);
  if (lhs_noncontracting_split_spec != nullptr) {
    // Just the fastest-varying part of it if the dimension is split.
    m = lhs_noncontracting_split_spec->at(0).count;
    lhs_noncontracting_split = lhs_noncontracting_split_spec->at(1).count;
  } else {
    m = analysis
            .IterSpec(TritonFusionAnalysis::Scope::OUTPUT, root,
                      out_lhs_noncontracting_dim_idx)
            ->at(0)
            .count;
  }

  // For now split non-contracting and batch are not supported
  // simultaneously because they are implemented via same mechanism.
  CHECK(
      !(out_batch_dim_idx.has_value() && lhs_noncontracting_split.has_value()));

  CHECK_GE(m, 1);
  CHECK_GE(n, 1);
}

MatMulLaunchConfig::MatMulLaunchConfig(
    const AutotuneResult::TritonGemmKey& config, const HloDotInstruction& dot,
    const MatMulDims& dims)
    : grid_m((dims.m + config.block_m() - 1) / config.block_m()),
      grid_n((dims.n + config.block_n() - 1) / config.block_n()) {
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
    launch_dims = {{batch_size, grid_m * grid_n, config.split_k()},
                   {config.num_warps() * WarpSize(), 1, 1}};
  } else {
    batch_program_id_dim = mt::ProgramIDDim::Y;
    noncontracting_program_id_dim = mt::ProgramIDDim::X;
    launch_dims =
        LaunchDimensions{{grid_m * grid_n, batch_size, config.split_k()},
                         {config.num_warps() * WarpSize(), 1, 1}};
  }
}

void ValidateMatMulConfig(const AutotuneResult::TritonGemmKey& config,
                          const HloDotInstruction& dot) {
  CHECK_GE(config.split_k(), 1);
  CHECK_GE(config.block_m(), 16);
  CHECK_GE(config.block_k(), 16);
  CHECK_GE(config.block_n(), 16);

  const auto& dims = dot.dot_dimension_numbers();
  int num_batch_dims =
      dims.lhs_batch_dimensions_size() - (config.split_k() > 1 ? 1 : 0);
  CHECK_LE(num_batch_dims, 1);
  if (config.split_k() > 1) {
    // Split-K dimension has to be the first batch one and have an index
    // just before the contracting one.
    const int lhs_split_k_dim_idx = dims.lhs_contracting_dimensions(0) - 1;
    const int rhs_split_k_dim_idx = dims.rhs_contracting_dimensions(0) - 1;
    // Size of this dimension has to match the split_k value.
    CHECK_EQ(dims.lhs_batch_dimensions(0), lhs_split_k_dim_idx);
    CHECK_EQ(dims.rhs_batch_dimensions(0), rhs_split_k_dim_idx);
    CHECK_EQ(config.split_k(),
             dot.operand(0)->shape().dimensions(lhs_split_k_dim_idx));
    CHECK_EQ(config.split_k(),
             dot.operand(1)->shape().dimensions(rhs_split_k_dim_idx));
  }

  // Rely on dot decomposer: there is just one contracting and one
  // non-contracting dimension on each side + batch ones optionally.
  CHECK_EQ(dims.lhs_contracting_dimensions_size(), 1);
  CHECK_EQ(dims.rhs_contracting_dimensions_size(), 1);

  CHECK_EQ(dot.operand(0)->shape().rank(),
           2 + (config.split_k() > 1 ? 1 : 0) + num_batch_dims);
}

struct Side {
  TritonFusionAnalysis::Scope scope;
  std::vector<DimProperties> tiled_dims;
  std::optional<int64_t> batch_dim_idx;
};

class MatMulEmitterHelper {
 public:
  MatMulEmitterHelper(absl::string_view libdevice_path,
                      const HloDotInstruction* dot_instr,
                      ImplicitLocOpBuilder& b, Type index_ty, MatMulDims dims,
                      const MatMulLaunchConfig& launch_config,
                      const TritonFusionAnalysis& analysis)
      : b_(b),
        libdevice_path_(libdevice_path),
        dot_instr_(dot_instr),
        index_ty_(index_ty),
        analysis_(analysis),
        dims_(dims),
        launch_config_(launch_config) {}

  // TODO(b/266862493): Accumulator can be integer too.
  // Otherwise only f64 x f64 -> f64 uses f64 accumulator.
  mlir::FloatType GetDotAccumulatorType() {
    Type dot_output_ty = TritonType(b_, dot_instr_->shape().element_type());
    // Data type of dot() immediate inputs.
    Type dot_input_ty = [&] {
      const Type lhs_ty =
          TritonType(b_, dot_instr_->operand(0)->shape().element_type());
      const Type rhs_ty =
          TritonType(b_, dot_instr_->operand(1)->shape().element_type());
      CHECK(lhs_ty == rhs_ty);
      return lhs_ty;
    }();
    // TODO(b/266862493): Accumulator can be integer too.
    // Otherwise only f64 x f64 -> f64 uses f64 accumulator.
    return (dot_output_ty.isF64() && dot_input_ty.isF64()) ? b_.getF64Type()
                                                           : b_.getF32Type();
  }

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
        CHECK(to_order.insert(current).second);
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

  Value MakeInput(Side& side, int64_t operand_index,
                  absl::flat_hash_map<const HloInstruction*, Value>& values) {
    return *EmitScope(
        b_, libdevice_path_, &analysis_, side.scope, side.tiled_dims,
        dot_instr_->parent()->MakeInstructionPostOrderFrom(
            const_cast<HloInstruction&>(*dot_instr_->operand(operand_index))),
        values);
  }

  Value EmitTensorPointer(const HloInstruction* hlo, const Side& side,
                          Value base, Value pid_k,
                          std::vector<int32_t>& boundary_checks) {
    auto pid_batch =
        b_.create<mt::GetProgramIdOp>(launch_config_.batch_program_id_dim);

    std::vector<Value> bounds;
    std::vector<Value> strides;
    // Offsets from tensor origin, same for all thread blocks.
    std::vector<Value> tensor_offsets;
    // Offsets for a given thread block, typically pid * block size.
    std::vector<Value> block_offsets;
    std::vector<int32_t> block_dims;
    std::vector<int32_t> dim_order;

    auto add_dim = [&](const DimProperties& properties) {
      const TensorIterationSpec::DimIterationSpec* spec =
          analysis_.IterSpec(side.scope, hlo, properties.index);
      if (spec == nullptr) {
        return;
      }
      const int64_t stride = spec->at(0).stride;
      int64_t count = spec->at(0).count;
      if (side.scope == TritonFusionAnalysis::Scope::OUTPUT &&
          properties.index == dims_.out_lhs_noncontracting_dim_idx &&
          spec->size() == 1 && dims_.lhs_noncontracting_split.has_value()) {
        // Dimension of the output produced by the non-contracting LHS one
        // is logically split, major part is addressed using pid_batch.
        count /= *dims_.lhs_noncontracting_split;
      }
      if (count % (properties.block_size * properties.split_value) != 0) {
        boundary_checks.push_back(bounds.size());
      }
      bounds.push_back(Cst64(count));
      strides.push_back(Cst64(stride));
      block_offsets.push_back(properties.offset);
      tensor_offsets.push_back(Cst32(spec->at(0).slice_start));
      block_dims.push_back(properties.block_size);
      dim_order.emplace(dim_order.begin(), dim_order.size());
    };
    for (const DimProperties& dim : side.tiled_dims) {
      add_dim(dim);
    }

    int64_t stride_batch = 0;
    int64_t offset_batch = 0;
    if (side.scope != TritonFusionAnalysis::Scope::RHS &&
        dims_.lhs_noncontracting_split) {
      const TensorIterationSpec::DimIterationSpec* spec =
          analysis_.IterSpec(side.scope, hlo, side.tiled_dims[0].index);
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
        CHECK_NE(stride_batch, 0);
      }
    } else if (side.batch_dim_idx.has_value()) {
      const TensorIterationSpec::DimIterationSpec* spec =
          analysis_.IterSpec(side.scope, hlo, *side.batch_dim_idx);
      if (spec != nullptr) {
        stride_batch = spec->at(0).stride;
        offset_batch = spec->at(0).slice_start;
        CHECK_NE(stride_batch, 0);
      }
    }
    if (stride_batch != 0) {
      Value pid_offset_batch = b_.create<ma::MulIOp>(
          b_.create<ma::AddIOp>(Cst(offset_batch), ConvertScalar(pid_batch)),
          Cst(stride_batch));
      base = AddPtr(b_, base, pid_offset_batch);
    }

    if (dims_.out_split_k_dim_idx.has_value()) {
      const TensorIterationSpec::DimIterationSpec* spec = analysis_.IterSpec(
          TritonFusionAnalysis::Scope::OUTPUT, hlo, *dims_.out_split_k_dim_idx);
      if (spec != nullptr) {
        int64_t stride_split_k = spec->at(0).stride;
        Value offset_split_k =
            b_.create<ma::MulIOp>(ConvertScalar(pid_k), Cst(stride_split_k));
        base = AddPtr(b_, base, offset_split_k);
      }
    }

    if (block_dims.empty()) {
      return base;
    }
    auto tensor_ptr =
        b_.create<mt::MakeTensorPtrOp>(base, bounds, strides, tensor_offsets,
                                       block_dims, dim_order)
            .getResult()
            .cast<Value>();
    tensor_ptr = b_.create<mt::AdvanceOp>(tensor_ptr.getType(), tensor_ptr,
                                          block_offsets);
    return tensor_ptr;
  }

 private:
  // Extend int32 indexes to int64, if necessary.
  Value ConvertScalar(Value value) {
    if (index_ty_.getIntOrFloatBitWidth() == 64) {
      return b_.create<ma::ExtSIOp>(index_ty_, value);
    }
    return value;
  }

  Value Cst(int64_t v) { return CreateConst(b_, index_ty_, v); }
  Value Cst32(int64_t v) { return CreateConst(b_, i32_ty_, v); }
  Value Cst64(int64_t v) { return CreateConst(b_, i64_ty_, v); }

  ImplicitLocOpBuilder& b_;
  absl::string_view libdevice_path_;
  const HloDotInstruction* dot_instr_;
  Type index_ty_;
  TritonFusionAnalysis analysis_;
  MatMulDims dims_;
  MatMulLaunchConfig launch_config_;
  Type i32_ty_ = b_.getI32Type();
  Type i64_ty_ = b_.getI64Type();
};

}  // namespace

LaunchDimensions GetMatMulLaunchDimensions(
    const TritonFusionAnalysis& analysis,
    absl::Span<const HloInstruction* const> roots,
    const FusionBoundaryFn& fusion_boundary,
    const AutotuneResult::TritonGemmKey& config) {
  const auto* dot = static_cast<const HloDotInstruction*>(
      HloFindIf(roots, fusion_boundary, [](const HloInstruction& node) {
        return node.opcode() == HloOpcode::kDot;
      }));
  CHECK_NE(dot, nullptr);
  const MatMulDims dims(config, *dot, analysis);
  const MatMulLaunchConfig launch_config(config, *dot, dims);
  return launch_config.launch_dims;
}

// Variable naming: lhs [m, k] x rhs [k, n] -> out [m, n].
Status EmitMatMul(mlir::OpBuilder builder, absl::string_view libdevice_path,
                  const TritonFusionAnalysis& analysis,
                  const HloComputation* computation, mlir::triton::FuncOp fn,
                  const AutotuneResult::TritonGemmKey& config,
                  int shmem_budget) {
  const HloDotInstruction* dot_instr = DynCast<HloDotInstruction>(
      hlo_query::GetFirstInstructionWithOpcode(*computation, HloOpcode::kDot));
  // Use 32-bit indexing if addressing any of the inputs or the output (which
  // could grow if split_k is set) does not cross the INT_MAX boundary.
  // Otherwise, fall back to 64-bit indexing, which is slower.
  bool use_64bit_indexing =
      ShapeUtil::ElementsIn(dot_instr->operand(0)->shape()) > INT_MAX ||
      ShapeUtil::ElementsIn(dot_instr->operand(1)->shape()) > INT_MAX ||
      ShapeUtil::ElementsIn(dot_instr->shape()) * config.split_k() > INT_MAX;
  Type index_ty = builder.getIntegerType(use_64bit_indexing ? 64 : 32);

  const HloInstruction* root = dot_instr->parent()->root_instruction();
  CHECK(!root->shape().IsTuple());

  // We'll be creating a lot of instructions from a single dot, use an
  // implicit loc builder so we don't have to pass around the location all the
  // time.
  auto loc = mlir::NameLoc::get(builder.getStringAttr(dot_instr->name()));
  ImplicitLocOpBuilder b(loc, builder);
  Type i32_ty = b.getI32Type();

  ValidateMatMulConfig(config, *dot_instr);
  const int split_k = config.split_k();
  const int block_m = config.block_m();
  const int block_k = config.block_k();
  const int block_n = config.block_n();

  const MatMulDims dims(config, *dot_instr, analysis);
  const MatMulLaunchConfig launch_config(config, *dot_instr, dims);
  VLOG(6) << analysis.ToString();

  MatMulEmitterHelper emitter(libdevice_path, dot_instr, b, index_ty, dims,
                              launch_config, analysis);

  constexpr int group_m = 8;
  const int64_t width = group_m * launch_config.grid_n;

  auto c32 = [&](int64_t v) { return CreateConst(b, b.getI32Type(), v); };

  auto pid_nc =
      b.create<mt::GetProgramIdOp>(launch_config.noncontracting_program_id_dim);
  auto pid_k = b.create<mt::GetProgramIdOp>(mt::ProgramIDDim::Z);

  auto group_id = b.create<ma::DivSIOp>(pid_nc, c32(width));
  ma::ConstantOp group_m_op = c32(group_m);
  auto first_pid_m = b.create<ma::MulIOp>(group_id, group_m_op);
  auto sub0 = b.create<ma::SubIOp>(c32(launch_config.grid_m), first_pid_m);
  auto group_size = b.create<ma::SelectOp>(
      b.create<ma::CmpIOp>(ma::CmpIPredicate::slt, sub0, group_m_op), sub0,
      group_m_op);

  auto pid_m = b.create<ma::AddIOp>(first_pid_m,
                                    b.create<ma::RemSIOp>(pid_nc, group_size));
  auto pid_m_offset = b.create<ma::MulIOp>(pid_m, c32(block_m));

  auto pid_n = b.create<ma::DivSIOp>(b.create<ma::RemSIOp>(pid_nc, c32(width)),
                                     group_size);
  auto pid_n_offset = b.create<ma::MulIOp>(pid_n, c32(block_n));

  auto pid_k_offset = b.create<ma::MulIOp>(pid_k, c32(block_k));

  mlir::FloatType acc_ty = emitter.GetDotAccumulatorType();

  ma::ConstantOp accumulator_init =
      CreateConst(b, acc_ty, 0, {block_m, block_n});

  // Parameters are passed to the loop in non-trivial order, these maps help
  // finding them and their attributes.
  absl::flat_hash_map<int, const HloInstruction*> iter_args_to_parameters;
  absl::flat_hash_map<int, std::vector<int32_t>> iter_args_to_boundary_checks;

  Side lhs{TritonFusionAnalysis::Scope::LHS,
           /*tiled_dims=*/
           {DimProperties(dims.lhs_noncontracting_dim_idx, pid_m_offset,
                          block_m, /*split_value=*/1),
            DimProperties(dims.lhs_contracting_dim_idx, pid_k_offset, block_k,
                          split_k)},
           dims.lhs_batch_dim_idx};
  Side rhs{TritonFusionAnalysis::Scope::RHS,
           /*tiled_dims=*/
           {DimProperties(dims.rhs_contracting_dim_idx, pid_k_offset, block_k,
                          split_k),
            DimProperties(dims.rhs_noncontracting_dim_idx, pid_n_offset,
                          block_n, /*split_value=*/1)},
           dims.rhs_batch_dim_idx};
  Side out{TritonFusionAnalysis::Scope::OUTPUT,
           /*tiled_dims=*/
           {DimProperties(dims.out_lhs_noncontracting_dim_idx, pid_m_offset,
                          block_m, /*split_value=*/1),
            DimProperties(dims.out_rhs_noncontracting_dim_idx, pid_n_offset,
                          block_n, /*split_value=*/1)},
           dims.out_batch_dim_idx};

  auto body_builder = [&](mlir::OpBuilder&, mlir::Location, Value ki,
                          ValueRange iter_args) {
    SmallVector<Value> iter_args_next;
    iter_args_next.reserve(iter_args.size());
    absl::flat_hash_map<const HloInstruction*, Value> values_lhs;
    absl::flat_hash_map<const HloInstruction*, Value> values_rhs;
    // Load tiles of all parameters of LHS and RHS scopes and advance pointers.
    for (int i = 0; i < iter_args.size() - 1; ++i) {
      const bool is_lhs =
          i < analysis.ScopeParameters(TritonFusionAnalysis::Scope::LHS).size();
      Side& side = is_lhs ? lhs : rhs;
      auto& values = is_lhs ? values_lhs : values_rhs;
      CHECK(values
                .insert({iter_args_to_parameters[i],
                         EmitParameterLoad(b, iter_args[i],
                                           iter_args_to_boundary_checks[i])})
                .second);
      SmallVector<Value> increments;
      for (const DimProperties& dim : side.tiled_dims) {
        const TensorIterationSpec::DimIterationSpec* spec = analysis.IterSpec(
            side.scope, iter_args_to_parameters[i], dim.index);
        if (spec == nullptr || spec->at(0).stride == 0) {
          continue;
        }
        // Only the contracting dimensions are advanced.
        if (dim.index == (is_lhs ? dims.lhs_contracting_dim_idx
                                 : dims.rhs_contracting_dim_idx)) {
          increments.push_back(c32(dim.block_size * split_k));
        } else {
          increments.push_back(c32(0));
        }
      }
      if (increments.empty()) {
        iter_args_next.push_back(iter_args[i]);
      } else {
        iter_args_next.push_back(b.create<mt::AdvanceOp>(
            iter_args[i].getType(), iter_args[i], increments));
      }
    }

    // Emit all operations of LHS and RHS scopes.
    Value dot_input_lhs = emitter.MakeInput(lhs, 0, values_lhs);
    Value dot_input_rhs = emitter.MakeInput(rhs, 1, values_rhs);

    // Operation in the fusion before the dot can alter the elements of the
    // tiles that were zero masked during loads. These have to be zeroed here
    // again just before the dot so that they do not affect the output.
    // Only the K dimension needs masking here because unnecessary elements in
    // the other two get discarded by the masked store at the end.
    const bool need_masking = dims.k % (block_k * split_k) > 0;
    if (need_masking) {
      auto elements_in_tile =
          b.create<ma::SubIOp>(CreateConst(b, i32_ty, dims.k), ki);
      auto range_k = b.create<ma::AddIOp>(
          Splat(b, b.create<ma::MulIOp>(pid_k, CreateConst(b, i32_ty, block_k)),
                block_k),
          Range(b, block_k));
      auto apply_mask = [&](int64_t dim, Value input) {
        auto ty = input.getType().cast<mlir::RankedTensorType>();
        TensorValue range_expanded = b.create<mt::ExpandDimsOp>(range_k, dim)
                                         .getResult()
                                         .cast<TensorValue>();
        Value mask = b.create<mt::BroadcastOp>(
            ty.clone(b.getI1Type()),
            b.create<ma::CmpIOp>(ma::CmpIPredicate::slt, range_expanded,
                                 Splat(b, elements_in_tile,
                                       range_expanded.getType().getShape())));
        return b.create<ma::SelectOp>(mask, input, ZerosLike(b, input));
      };
      dot_input_lhs = apply_mask(0, dot_input_lhs);
      dot_input_rhs = apply_mask(1, dot_input_rhs);
    }

    const bool allow_tf32 =
        absl::c_none_of(dot_instr->precision_config().operand_precision(),
                        [](const int precision) {
                          return precision != PrecisionConfig::DEFAULT;
                        });

    // Execute matrix multiplication of input tiles and pass the accumulator.
    // TODO(manany): Should be looked into once we enable Hopper workloads.
    // maxNumImpreciseAcc flag was introduced for Hopper to accumulate in a
    // lower precision than the output type. The change was introduced here:
    // https://github.com/openai/triton/commit/31b0c521427109a8eda609b58d756c380b21599a
    Value accumulator_next = b.create<mt::DotOp>(dot_input_lhs, dot_input_rhs,
                                                 iter_args.back(), allow_tf32,
                                                 /*maxNumImpreciseAcc=*/0);
    iter_args_next.push_back(accumulator_next);

    b.create<mlir::scf::YieldOp>(iter_args_next);
  };

  // Pointers to parameters of LHS scope, then RHS, then the accumulator
  // that change with every loop iteration and are passed between them.
  // LHS and RHS can use same HLO computation parameters, but because they use
  // different pointers they have to be stored separately for each scope.
  SmallVector<Value> iter_args;
  iter_args.reserve(
      analysis.ScopeParameters(TritonFusionAnalysis::Scope::LHS).size() +
      analysis.ScopeParameters(TritonFusionAnalysis::Scope::RHS).size() + 1);

  for (const Side& side : {lhs, rhs}) {
    for (const HloInstruction* param : analysis.ScopeParameters(side.scope)) {
      CHECK(iter_args_to_parameters.insert({iter_args.size(), param}).second);
      iter_args.push_back(emitter.EmitTensorPointer(
          param, side, fn.getArgument(param->parameter_number()), pid_k,
          iter_args_to_boundary_checks[iter_args.size()]));
    }
  }

  iter_args.push_back(accumulator_init);
  Value acc_final = b.create<mlir::scf::ForOp>(
                         /*lowerBound=*/c32(0),
                         /*upperBound=*/c32(dims.k),
                         /*step=*/c32(block_k * split_k),
                         /*iterArgs=*/iter_args, body_builder)
                        .getResult(iter_args.size() - 1);
  absl::flat_hash_map<const HloInstruction*, Value> values_out;
  values_out[dot_instr] =
      Cast(b, acc_final, TritonType(b, dot_instr->shape().element_type()));

  // Emit the output scope.
  if (std::vector<const HloInstruction*> to_emit =
          emitter.EpiloguePostOrderTransitiveOperands(root);
      !to_emit.empty()) {
    for (const HloInstruction* parameter :
         analysis.ScopeParameters(TritonFusionAnalysis::Scope::OUTPUT)) {
      std::vector<int32_t> boundary_checks;
      Value tensor_pointer = emitter.EmitTensorPointer(
          parameter, out, fn.getArgument(parameter->parameter_number()), pid_k,
          boundary_checks);
      CHECK(values_out
                .insert({parameter,
                         EmitParameterLoad(b, tensor_pointer, boundary_checks)})
                .second);
    }
    TF_RETURN_IF_ERROR(EmitScope(b, libdevice_path, &analysis,
                                 TritonFusionAnalysis::Scope::OUTPUT,
                                 out.tiled_dims, to_emit, values_out)
                           .status());
  }

  // Emit tensor store operations for all outputs.
  for (int i = 0;
       i < fn.getNumArguments() - dot_instr->parent()->num_parameters(); ++i) {
    const HloInstruction* producer =
        root->shape().IsTuple() ? root->operand(i) : root;
    std::vector<int32_t> boundary_checks;
    Value tensor_pointer = emitter.EmitTensorPointer(
        producer, out,
        fn.getArgument(i + dot_instr->parent()->num_parameters()), pid_k,
        boundary_checks);
    b.create<mt::StoreOp>(tensor_pointer, values_out[producer], boundary_checks,
                          mt::CacheModifier::NONE, mt::EvictionPolicy::NORMAL);
  }
  return OkStatus();
}

LaunchDimensions GetSoftMaxLaunchDimensions(
    absl::Span<const HloInstruction* const> roots,
    const FusionBoundaryFn& fusion_boundary,
    const AutotuneResult::TritonGemmKey& config) {
  const HloInstruction* reduce =
      HloFindIf(roots, fusion_boundary, [](const HloInstruction& node) {
        return node.opcode() == HloOpcode::kReduce;
      });
  CHECK_NE(reduce, nullptr);
  const Shape& reduce_input_shape = reduce->operand(0)->shape();
  int num_rows = 1;
  for (int minor_axis = 1; minor_axis < reduce_input_shape.rank();
       ++minor_axis) {
    num_rows *= reduce_input_shape.dimensions_minor(minor_axis);
  }

  return {{num_rows, 1, 1}, {config.num_warps() * WarpSize(), 1, 1}};
}

Status EmitSoftMax(mlir::OpBuilder builder, absl::string_view libdevice_path,
                   const TritonFusionAnalysis& analysis,
                   const HloComputation* computation, mlir::triton::FuncOp fn,
                   const AutotuneResult::TritonGemmKey& config, int) {
  const HloInstruction* root = computation->root_instruction();
  auto loc = mlir::NameLoc::get(builder.getStringAttr(root->name()));
  ImplicitLocOpBuilder b(loc, builder);

  // Assumptions we make about the matcher:
  //   * matches Softmax "diamonds" on the last axis, along with any number of
  //     elementwise operations/bitcasts on any edge
  //   * within a given fusion, every argument to a Softmax diamond has the same
  //     shape
  //   * every reduction is on the last axis
  //   * the last axis of every reduction parameter has the same length
  //   * reductions only reduce a single operand
  //   * all the shapes have canonical layout (logical layout = physical layout)
  //   * the computation has a single input and a single output

  // TODO(bchetioui): allow doing several rows per block (e.g. for when rows
  // are smaller than the minimum transaction size)

  const HloInstruction* reduce = hlo_query::GetFirstInstructionWithOpcode(
      *computation, HloOpcode::kReduce);

  CHECK_NE(reduce, nullptr);

  Shape reduce_input_shape = reduce->operand(0)->shape();

  CHECK_EQ(reduce->opcode(), HloOpcode::kReduce);
  CHECK_EQ(reduce->dimensions().size(), 1);
  CHECK_EQ(reduce->dimensions()[0], reduce_input_shape.rank() - 1);

  int row_len = reduce_input_shape.dimensions_minor(0);
  int block_row = 1;

  // block_row must be a power of two.
  while (block_row < row_len) {
    block_row *= 2;
  }

  Value row_index = b.create<ma::ExtSIOp>(
      b.getI64Type(), b.create<mt::GetProgramIdOp>(mt::ProgramIDDim::X));
  Value row_stride = CreateConst(b, b.getI32Type(), row_len);

  absl::flat_hash_map<const HloInstruction*, Value> values_out;
  auto make_tensor_pointer = [&](Value base) {
    Value offset = b.create<ma::MulIOp>(
        row_index, b.create<ma::ExtSIOp>(b.getI64Type(), row_stride));
    return b.create<mt::MakeTensorPtrOp>(
        /*base=*/AddPtr(b, base, offset),
        /*shape=*/ValueRange{CreateConst(b, b.getI64Type(), row_len)},
        /*strides=*/ValueRange{CreateConst(b, b.getI64Type(), 1)},
        /*offsets=*/ValueRange{CreateConst(b, b.getI32Type(), 0)},
        /*tensorShape=*/std::vector<int32_t>{block_row},
        /*order=*/std::vector<int32_t>{0});
  };

  std::vector<int32_t> boundary_checks;
  if (block_row != row_len) {
    boundary_checks.push_back(0);
  }
  values_out[computation->parameter_instruction(0)] = EmitParameterLoad(
      b, make_tensor_pointer(fn.getArgument(0)), boundary_checks);
  // Dimension 0 is the reduced one by construction and it's the only one
  // present in the tile shapes.
  std::vector<DimProperties> tiled_dims = {
      DimProperties(0, row_index, block_row, /*split_value=*/1)};
  TF_ASSIGN_OR_RETURN(
      Value result,
      EmitScope(b, libdevice_path, &analysis,
                TritonFusionAnalysis::Scope::OUTPUT, tiled_dims,
                computation->MakeInstructionPostOrder(), values_out));

  b.create<mt::StoreOp>(make_tensor_pointer(fn.getArgument(1)), result,
                        std::vector<int32_t>{0}, mt::CacheModifier::NONE,
                        mt::EvictionPolicy::NORMAL);
  return OkStatus();
}

// Simplified copy of translateLLVMToLLVMIR which in addition takes
// path to libdevice directly as an argument.
StatusOr<std::unique_ptr<llvm::Module>> TranslateLLVMToLLVMIR(
    llvm::LLVMContext* llvmContext, mlir::ModuleOp module,
    absl::string_view libdevice_path) {
  mlir::DialectRegistry registry;
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerNVVMDialectTranslation(registry);
  module->getContext()->appendDialectRegistry(registry);

  std::unique_ptr<llvm::Module> llvmModule =
      mlir::translateModuleToLLVMIR(module, *llvmContext);
  if (!llvmModule) {
    return InternalError("Failed to emit LLVM IR.");
  }

  // Link external libraries before performing optimizations.
  TF_RETURN_IF_ERROR(nvptx::LinkLibdeviceIfNecessary(
      llvmModule.get(), std::string(libdevice_path)));

  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/3, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);

  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << err;
    return InternalError("Failed to optimize LLVM IR.");
  }

  return llvmModule;
}

namespace {

std::string GetLibdevicePath(const HloComputation* hlo_computation) {
  return nvptx::LibDevicePath(hlo_computation->parent()
                                  ->config()
                                  .debug_options()
                                  .xla_gpu_cuda_data_dir());
}

}  // namespace

StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> CreateTritonModule(
    const TritonFusionAnalysis& analysis, absl::string_view fn_name,
    const HloComputation* hlo_computation,
    const se::DeviceDescription& device_info,
    const AutotuneResult::TritonGemmKey& config, TritonIrEmitter ir_emitter,
    mlir::MLIRContext& mlir_context) {
  mlir_context.loadDialect<mt::TritonDialect>();
  mlir::OpBuilder b(&mlir_context);
  auto loc = mlir::NameLoc::get(b.getStringAttr(hlo_computation->name()));
  mlir::OwningOpRef<mlir::ModuleOp> triton_module =
      llvm_ir::CreateMlirModuleOp(loc);
  b.setInsertionPointToEnd(triton_module->getBody());

  // Build Triton kernel.
  SmallVector<Type> fn_arg_types;
  for (HloInstruction* p : hlo_computation->parameter_instructions()) {
    fn_arg_types.push_back(mt::PointerType::get(
        StorageType(b, TritonType(b, p->shape().element_type())),
        mn::kGlobalMemorySpace));
  }

  for (const ShapeUtil::IndexedShape& s :
       ShapeUtil::GetLeafShapes(hlo_computation->root_instruction()->shape())) {
    fn_arg_types.push_back(mt::PointerType::get(
        StorageType(b, TritonType(b, s.shape.element_type())),
        mn::kGlobalMemorySpace));
  }

  auto fn = b.create<mt::FuncOp>(loc, fn_name,
                                 b.getFunctionType(fn_arg_types, std::nullopt));
  for (int i = 0; i < fn.getNumArguments(); ++i) {
    fn.setArgAttr(i, "tt.divisibility", b.getIntegerAttr(b.getI32Type(), 16));
  }
  fn.addEntryBlock();
  b.setInsertionPointToStart(&fn.front());

  TF_RETURN_IF_ERROR(ir_emitter(b, GetLibdevicePath(hlo_computation), analysis,
                                hlo_computation, fn, config,
                                device_info.shared_memory_per_block_optin()));

  b.create<mt::ReturnOp>(loc);

  VLOG(6) << llvm_ir::DumpToString(*triton_module);
  if (DumpingEnabledForHloModule(*hlo_computation->parent())) {
    DumpToFileInDirOrStdout(*hlo_computation->parent(), "triton_ir", "ttir",
                            llvm_ir::DumpToString(*triton_module));
  }

  CHECK(mlir::succeeded(mlir::verify(*triton_module)));
  return std::move(triton_module);
}

StatusOr<TritonWrapperResult> TritonWrapper(
    const TritonFusionAnalysis& analysis, absl::string_view fn_name,
    const HloComputation* hlo_computation, absl::string_view fusion_kind,
    const se::CudaComputeCapability& cc,
    const se::DeviceDescription& device_info,
    const AutotuneResult::TritonGemmKey& config, llvm::Module* llvm_module,
    TritonIrEmitter ir_emitter, mlir::MLIRContext& mlir_context) {
  if (fusion_kind == kTritonGemmFusionKind) {
    // This is a heuristic that serves as a proxy for register usage and code
    // size.
    //
    // We have noticed that tilings with very long LLVM IR code are both slow to
    // compile and slow to run. This can be for example due to register spills.
    // So we should skip these tilings to save time. But it's better to skip
    // them before the LLVM IR is generated. To do that, we came up with a
    // formula that strongly correlates with the LLVM IR size. The formula is
    // the size of the two input and the output thread block tiles divided by
    // the number of warps. We read
    // https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/ as a
    // reference, and found the formula by trial and error.
    //
    // To regenerate the limit, we have to run an exhaustive search on all
    // tilings for a few different HLOs, printing the runtimes and the heuristic
    // values.
    // From that, we can find a limit, such that all tilings within alpha *
    // optimal_runtime have a heuristic value less than or equal to the limit.
    //
    // In our measurements, all tilings which were within 1.13 * optimal_runtime
    // had a complexity_heuristic_value <= kComplexityHeuristicLimit.
    //
    // See go/tiling-heuristic for more details.
    constexpr int64_t kComplexityHeuristicLimit = 9000;
    int64_t complexity_heuristic_value =
        (config.block_m() * config.block_n() +
         (config.block_m() + config.block_n()) * config.block_k()) /
        config.num_warps();
    VLOG(2) << "Complexity heuristic: " << complexity_heuristic_value;
    if (complexity_heuristic_value > kComplexityHeuristicLimit) {
      return ResourceExhausted("Tiling complexity heuristic exceeded: %d > %d",
                               complexity_heuristic_value,
                               kComplexityHeuristicLimit);
    }
  }

  TF_ASSIGN_OR_RETURN(
      auto triton_module,
      CreateTritonModule(analysis, fn_name, hlo_computation, device_info,
                         config, ir_emitter, mlir_context));

  VLOG(3) << hlo_computation->ToString(HloPrintOptions::ShortParsable());
  VLOG(2) << config.ShortDebugString();

  // Compile Triton kernel to LLVM.
  std::optional<llvm::raw_fd_ostream> log_stream;
  const HloModule* hlo_module = hlo_computation->parent();

  bool should_verify =
      (hlo_module->config().debug_options().xla_gpu_llvm_verification_level() >=
       1);
#ifndef NDEBUG
  should_verify = true;
#endif

  mlir::PassManager pm(&mlir_context);
  pm.enableVerifier(should_verify);

  if (hlo_module->config().debug_options().xla_gpu_dump_llvmir()) {
    const std::string basename =
        absl::StrCat(absl::string_view(tsl::io::Basename(hlo_module->name())),
                     ".triton-passes.log");
    std::string outputs_dir;
    if (!tsl::io::GetTestUndeclaredOutputsDir(&outputs_dir)) {
      outputs_dir = hlo_module->config().debug_options().xla_dump_to();
    }
    if (!outputs_dir.empty()) {
      std::string path = tsl::io::JoinPath(outputs_dir, basename);
      std::error_code err;
      log_stream.emplace(path, err, llvm::sys::fs::OF_None);
      if (err) {
        log_stream.reset();
      }
      pm.getContext()->disableMultithreading();
      auto print_always = [](mlir::Pass*, mlir::Operation*) { return true; };
      pm.enableIRPrinting(/*shouldPrintBeforePass=*/print_always,
                          /*shouldPrintAfterPass=*/print_always,
                          /*printModuleScope=*/true,
                          /*printAfterOnlyOnChange=*/false,
                          /*printAfterOnlyOnFailure=*/true, *log_stream,
                          /*opPrintingFlags=*/{});
    } else {
      LOG(ERROR) << "--xla_gpu_dump_llvmir is set, but neither the environment "
                 << "variable TEST_UNDECLARED_OUTPUTS_DIR nor the flag "
                 << "--xla_dump_to is set, so the llvm dumps are disabled.";
    }
  }

  CreateTritonPipeline(pm, cc, config.num_warps(), config.num_stages());
  if (log_stream.has_value()) {
    pm.printAsTextualPipeline(log_stream.value());
    log_stream->write("\n\n", 2);
  }
  // Triton generates pointers to the global address space, while XLA needs a
  // kernel signature with pointers to the generic address space.
  pm.addPass(std::make_unique<GeneralizeKernelSignaturePass>());
  // llvm::Linker::linkModules() segfaults if we don't strip locations.
  pm.addPass(mlir::createStripDebugInfoPass());

  bool succeeded = mlir::succeeded(pm.run(*triton_module));

  if (log_stream.has_value()) {
    log_stream->flush();
  }

  if (!succeeded) {
    return InternalError("Failed to compile Triton kernel.");
  }

  const int shared_mem_bytes =
      (*triton_module)
          ->getAttrOfType<mlir::IntegerAttr>("triton_gpu.shared")
          .getInt();
  VLOG(2) << "Shared memory usage: " << shared_mem_bytes << " B";
  if (shared_mem_bytes > device_info.shared_memory_per_block_optin()) {
    return ResourceExhausted("Shared memory size limit exceeded.");
  }

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<llvm::Module> ll_triton_module,
      TranslateLLVMToLLVMIR(&llvm_module->getContext(), *triton_module,
                            GetLibdevicePath(hlo_computation)));
  VLogModule(5, *ll_triton_module);
  if (should_verify) {
    VerifyModule(*ll_triton_module);
  }

  // Integrate LLVM matmul kernel into XLA's LLVM module.
  ll_triton_module->eraseNamedMDNode(
      ll_triton_module->getNamedMetadata("nvvm.annotations"));
  ll_triton_module->setDataLayout(llvm_module->getDataLayout());
  ll_triton_module->setTargetTriple(llvm_module->getTargetTriple());
  // Use override flag because libdevice functions can be present in both.
  CHECK(!llvm::Linker::linkModules(*llvm_module, std::move(ll_triton_module),
                                   llvm::Linker::Flags::OverrideFromSrc));
  VLogModule(5, *llvm_module);
  if (should_verify) {
    VerifyModule(*llvm_module);
  }

  return {{shared_mem_bytes}};
}

}  // namespace gpu
}  // namespace xla
