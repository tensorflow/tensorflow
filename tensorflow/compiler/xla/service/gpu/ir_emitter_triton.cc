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

#include "tensorflow/compiler/xla/service/gpu/ir_emitter_triton.h"

#include <climits>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <system_error>  // NOLINT(build/c++11): required to interface with LLVM
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
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
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
#include "mlir/IR/Verifier.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"  // from @llvm-project
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"  // from @llvm-project
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"  // from @llvm-project
#include "mlir/Target/LLVMIR/Export.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/xla/comparison_util.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instructions.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/hlo/utils/hlo_query.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/transforms/map_mhlo_to_scalar_op.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/gpu/gemm_rewriter_triton.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_device_info.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/gpu/launch_dimensions.h"
#include "tensorflow/compiler/xla/service/gpu/llvm_gpu_backend/gpu_backend_lib.h"
#include "tensorflow/compiler/xla/service/gpu/matmul_utils.h"
#include "tensorflow/compiler/xla/service/gpu/target_util.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/translate/hlo_to_mhlo/hlo_function_importer.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/path.h"
#include "tensorflow/tsl/platform/statusor.h"
#include "tensorflow/tsl/platform/tensor_float_32_utils.h"
#include "triton/Conversion/TritonGPUToLLVM/TritonGPUToLLVMPass.h"
#include "triton/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

namespace xla {
namespace gpu {

namespace ma = ::mlir::arith;
namespace mm = ::mlir::math;
namespace ml = ::mlir::LLVM;
namespace mn = ::mlir::NVVM;
namespace mt = ::mlir::triton;

using ::llvm::SmallVector;
using ::mlir::Type;
using ::mlir::Value;

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
      // Treat PRED as S8.
    case S8:
      return b.getI8Type();
    default:
      LOG(FATAL) << "This type is not supported yet: "
                 << primitive_util::LowercasePrimitiveTypeName(t);
  }
}

// Triton type conversions.
Value Cast(mlir::ImplicitLocOpBuilder& b, Value value, Type dst_element_ty) {
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

  // Float <=> float
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
    return b.create<ma::FPToSIOp>(dst_ty, value);
  }

  LOG(FATAL) << "Type conversion not supported: "
             << llvm_ir::DumpToString(src_element_ty) << " -> "
             << llvm_ir::DumpToString(dst_element_ty);
}

Type ElementType(Value v) {
  Type src_ty = v.getType();
  if (auto src_shaped_ty = src_ty.dyn_cast<mlir::ShapedType>()) {
    return src_shaped_ty.getElementType();
  }
  return src_ty;
}

// Get the value of the scalar constant's literal in a C++ type.
template <typename T>
T ScalarConstantValue(const HloInstruction& instr) {
  CHECK(hlo_query::IsScalarConstant(&instr));
  PrimitiveType dst_type;
  if constexpr (std::is_integral_v<T>) {
    if constexpr (std::numeric_limits<T>::is_signed) {
      dst_type = S64;
    } else {
      dst_type = U64;
    }
  } else {
    dst_type = F64;
  }
  StatusOr<Literal> converted = instr.literal().Convert(dst_type);
  TF_CHECK_OK(converted.status());
  if constexpr (std::is_integral_v<T>) {
    if constexpr (std::numeric_limits<T>::is_signed) {
      return converted.value().GetFirstElement<int64_t>();
    } else {
      return converted.value().GetFirstElement<uint64_t>();
    }
  } else {
    return converted.value().GetFirstElement<double>();
  }
}

// Create a scalar constant.
template <typename T>
ma::ConstantOp CreateConst(mlir::ImplicitLocOpBuilder b, Type type, T value) {
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
ma::ConstantOp CreateConst(mlir::ImplicitLocOpBuilder& b, Type type, T value,
                           mlir::ArrayRef<int64_t> shape) {
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

Value Subtract(mlir::ImplicitLocOpBuilder& b, mlir::ValueRange values) {
  if (ElementType(values[0]).isa<mlir::IntegerType>()) {
    return b.create<ma::SubIOp>(values[0], values[1]);
  } else {
    return b.create<ma::SubFOp>(values[0], values[1]);
  }
}

Value Compare(mlir::ImplicitLocOpBuilder& b, mlir::ValueRange values,
              ComparisonDirection direction) {
  if (ElementType(values[0]).isa<mlir::IntegerType>()) {
    return b.create<ma::CmpIOp>(
        mlir::mhlo::impl::getCmpPredicate<ma::CmpIPredicate>(
            mlir::mhlo::symbolizeComparisonDirection(
                ComparisonDirectionToString(direction))
                .value(),
            /*isSigned=*/true)
            .value(),
        values[0], values[1]);
  }
  return b.create<ma::CmpFOp>(
      mlir::mhlo::impl::getCmpPredicate<ma::CmpFPredicate>(
          mlir::mhlo::symbolizeComparisonDirection(
              ComparisonDirectionToString(direction))
              .value(),
          /*isSigned=*/true)
          .value(),
      values[0], values[1]);
}

Value Maximum(mlir::ImplicitLocOpBuilder& b, mlir::ValueRange values) {
  auto cmp = Compare(b, values, ComparisonDirection::kGt);
  return b.create<ma::SelectOp>(cmp, values[0], values[1]);
}

Value Minimum(mlir::ImplicitLocOpBuilder& b, mlir::ValueRange values) {
  auto cmp = Compare(b, values, ComparisonDirection::kLt);
  return b.create<ma::SelectOp>(cmp, values[0], values[1]);
}

Value ZerosLike(mlir::ImplicitLocOpBuilder& b, Value x) {
  if (auto src_shaped_ty = x.getType().dyn_cast<mlir::ShapedType>()) {
    Type src_ty = src_shaped_ty.getElementType();
    return CreateConst(b, src_ty, 0, src_shaped_ty.getShape());
  }
  return CreateConst(b, x.getType(), 0);
}

Value OnesLike(mlir::ImplicitLocOpBuilder& b, Value x) {
  if (auto src_shaped_ty = x.getType().dyn_cast<mlir::ShapedType>()) {
    Type src_ty = src_shaped_ty.getElementType();
    return CreateConst(b, src_ty, 1, src_shaped_ty.getShape());
  }
  return CreateConst(b, x.getType(), 1);
}

// TODO(b/269489810): Contribute nicer builders to Triton, so we don't need to
// define these utilities.
Value Splat(mlir::ImplicitLocOpBuilder& b, Value value,
            mlir::ArrayRef<int64_t> shape) {
  auto type = mlir::RankedTensorType::get(shape, value.getType());
  return b.create<mt::SplatOp>(type, value);
}

using TensorValue = mlir::TypedValue<mlir::RankedTensorType>;

Value Broadcast(mlir::ImplicitLocOpBuilder& b, TensorValue value,
                mlir::ArrayRef<int64_t> shape) {
  auto type =
      mlir::RankedTensorType::get(shape, value.getType().getElementType());
  return b.create<mt::BroadcastOp>(type, value);
}

Value Range(mlir::ImplicitLocOpBuilder& b, int32_t limit) {
  auto type = mlir::RankedTensorType::get(limit, b.getI32Type());
  return b.create<mt::MakeRangeOp>(type, 0, limit);
}

Value AddPtr(mlir::ImplicitLocOpBuilder& b, Value ptr, Value offset) {
  return b.create<mt::AddPtrOp>(ptr.getType(), ptr, offset);
}

Value EmitElementwise(mlir::ImplicitLocOpBuilder& b,
                      absl::string_view libdevice_path,
                      const HloInstruction& hlo, mlir::ValueRange inputs) {
  if (ElementType(inputs[0]).isF32() || ElementType(inputs[0]).isF64()) {
    auto dev_fn_id = GetTargetDeviceFunctionID(hlo.opcode());
    if (dev_fn_id.ok()) {
      return b.create<mt::PureExternElementwiseOp>(
          inputs[0].getType(), inputs, "libdevice", libdevice_path,
          ObtainDeviceFunctionName(dev_fn_id.value(),
                                   hlo.shape().element_type(),
                                   llvm::Triple("nvptx64-unknown-unknown")));
    }
  }
  const bool is_integer = ElementType(inputs[0]).isa<mlir::IntegerType>();

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
      return Compare(b, inputs, hlo.comparison_direction());
    case HloOpcode::kSelect:
      return b.create<ma::SelectOp>(
          Compare(b, {inputs[0], ZerosLike(b, inputs[0])},
                  ComparisonDirection::kNe),
          inputs[1], inputs[2]);
    default:
      LOG(FATAL) << "Unsupported operation " << hlo.ToString();
  }
}

Value EmitParameterLoad(mlir::ImplicitLocOpBuilder& b, Value tensor_pointer,
                        mlir::ArrayRef<int32_t> boundary_checks) {
  std::optional<mt::PaddingOption> padding;
  if (!boundary_checks.empty()) {
    padding = mt::PaddingOption::PAD_ZERO;
  }
  return b.create<mt::LoadOp>(tensor_pointer, boundary_checks, padding,
                              mt::CacheModifier::NONE,
                              mt::EvictionPolicy::NORMAL,
                              /*isVolatile=*/false);
}

Value EmitConstant(mlir::ImplicitLocOpBuilder& b,
                   const HloInstruction& constant) {
  Type ty = TritonType(b, constant.shape().element_type());
  if (constant.shape().IsInteger()) {
    if (constant.shape().element_type() == U64) {
      return CreateConst(b, ty, ScalarConstantValue<uint64_t>(constant));
    } else {
      return CreateConst(b, ty, ScalarConstantValue<int64_t>(constant));
    }
  }
  return CreateConst(b, ty, ScalarConstantValue<double>(constant));
}

Value EmitBroadcast(mlir::ImplicitLocOpBuilder& b,
                    const HloInstruction& broadcast, Value input,
                    mlir::ArrayRef<int64_t> tile_shape) {
  if (!input.dyn_cast<TensorValue>()) {
    return Splat(b, input, tile_shape);
  }
  // The only other kind of broadcast that can happen currently is a
  // broadcast into the split-K batch dimension which requires
  // no action here.
  return input;
}

StatusOr<Value> EmitScope(
    mlir::ImplicitLocOpBuilder& b, absl::string_view libdevice_path,
    absl::Span<const HloInstruction* const> instructions,
    absl::flat_hash_map<const HloInstruction*, Value>& values,
    mlir::ArrayRef<int64_t> tile_shape, Value tile_mask);

StatusOr<Value> EmitReduce(mlir::ImplicitLocOpBuilder& b,
                           const HloInstruction& hlo_reduce,
                           absl::string_view libdevice_path, Value input,
                           Value tile_mask) {
  llvm::ArrayRef<int64_t> input_shape =
      input.cast<TensorValue>().getType().getShape();

  // At the moment, we should only emit a full reduction over the last axis of
  // a single input.
  CHECK_EQ(hlo_reduce.operand_count(), 2);
  CHECK_EQ(hlo_reduce.dimensions().size(), 1);
  CHECK_EQ(hlo_reduce.dimensions(0), hlo_reduce.operand(0)->shape().rank() - 1);
  CHECK_GE(input_shape.back(),
           hlo_reduce.operand(0)->shape().dimensions().back());
  // We assume here that the reduction value was input as a constant, and/or has
  // been constant-folded.
  CHECK_EQ(hlo_reduce.operand(1)->opcode(), HloOpcode::kConstant);

  // Since every shape is padded to a power of 2 in Triton, the input tile may
  // be padded with arbitrary values. These values could affect the result of
  // the reduction, so we need to mask them away. Luckily, we have a monoid
  // structure (element_type, hlo_reduce.to_apply(), hlo_reduce.operand(1))---
  // up to floating-point inaccuracies. Masking the input using
  // hlo_reduce.operand(1) is thus always the right choice to ensure that the
  // reduction is computed correctly, since it is the neutral value with regards
  // to the reducer.
  Value neutral = EmitConstant(b, *hlo_reduce.operand(1));
  if (tile_mask) {
    input = b.create<ma::SelectOp>(tile_mask, input,
                                   Splat(b, neutral, input_shape));
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
                        EmitScope(b, libdevice_path, to_emit, region_values,
                                  /*tile_shape=*/{}, /*tile_mask=*/{}));
    b.create<mt::ReduceReturnOp>(SmallVector<Value>({result}));
    b.setInsertionPointAfter(reduction);
  }

  return Cast(b, reduction.getResult().front(),
              TritonType(b, hlo_reduce.shape().element_type()));
}

// Emit sequence of instructions using compatible tiling ordered producers
// before consumers.
StatusOr<Value> EmitScope(
    mlir::ImplicitLocOpBuilder& b, absl::string_view libdevice_path,
    absl::Span<const HloInstruction* const> instructions,
    absl::flat_hash_map<const HloInstruction*, Value>& values,
    mlir::ArrayRef<int64_t> tile_shape, Value tile_mask) {
  for (const HloInstruction* hlo : instructions) {
    Value result;
    if (hlo->opcode() == HloOpcode::kParameter) {
      // Parameter loads are handled outside EmitScope.
      TF_RET_CHECK(values.contains(hlo)) << hlo->ToString();
      continue;
    } else if (hlo->opcode() == HloOpcode::kConstant) {
      result = EmitConstant(b, *hlo);
    } else if (hlo->opcode() == HloOpcode::kBroadcast) {
      result = EmitBroadcast(b, *hlo, values[hlo->operand(0)], tile_shape);
    } else if (hlo->opcode() == HloOpcode::kReduce) {
      TF_ASSIGN_OR_RETURN(
          result, EmitReduce(b, *hlo, libdevice_path, values[hlo->operand(0)],
                             tile_mask));
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
               hlo->opcode() == HloOpcode::kReshape) {
      result = values[hlo->operand(0)];
    } else {
      LOG(FATAL) << hlo->ToString();
    }
    TF_RET_CHECK(values.insert({hlo, result}).second) << hlo->ToString();
    VLOG(8) << "Emitted " << hlo->ToString();
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
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createLoopInvariantCodeMotionPass());
  pm.addPass(mlir::createSymbolDCEPass());
  // Based on ttir_to_ttgir() in
  // @triton//:python/triton/compiler/compiler.py
  pm.addPass(mt::createConvertTritonToTritonGPUPass(num_warps));
  // Based on optimize_ttgir() in
  // @triton//:python/triton/compiler/compiler.py
  pm.addPass(mlir::createTritonGPUCoalescePass());
  pm.addPass(mlir::createTritonGPURemoveLayoutConversionsPass());
  pm.addPass(mlir::createTritonGPUAccelerateMatmulPass(ccAsInt));
  pm.addPass(mlir::createTritonGPURemoveLayoutConversionsPass());
  pm.addPass(mlir::createTritonGPUOptimizeDotOperandsPass());
  pm.addPass(mlir::createTritonGPUPipelinePass(num_stages));
  pm.addPass(mlir::createTritonGPUPrefetchPass());
  pm.addPass(mlir::createTritonGPUOptimizeDotOperandsPass());
  pm.addPass(mlir::createTritonGPURemoveLayoutConversionsPass());
  pm.addPass(mlir::createTritonGPUDecomposeConversionsPass());
  pm.addPass(mlir::createTritonGPUReorderInstructionsPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createSymbolDCEPass());
  // Based on translateTritonGPUToLLVMIR() in
  // @triton//:lib/Target/LLVMIR/LLVMIRTranslation.cpp
  pm.addPass(mlir::createConvertSCFToCFPass());
  pm.addPass(mlir::createConvertIndexToLLVMPass());
  pm.addPass(mt::createConvertTritonGPUToLLVMPass(ccAsInt));
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

// Variable naming: lhs [m, k] x rhs [k, n] -> out [m, n].
// TODO(b/270937368): Split this up into smaller functions.
template <typename IndexT>
StatusOr<LaunchDimensions> MatMulImpl(
    mlir::OpBuilder builder, absl::string_view libdevice_path,
    const HloDotInstruction* dot_instr, mlir::triton::FuncOp fn,
    const AutotuneResult::TritonGemmKey& config, int shmem_budget) {
  const HloInstruction* root = dot_instr->parent()->root_instruction();
  CHECK(!root->shape().IsTuple());

  // We'll be creating a lot of instructions from a single dot, use an
  // implicit loc builder so we don't have to pass around the location all the
  // time.
  auto loc = mlir::NameLoc::get(builder.getStringAttr(dot_instr->name()));
  mlir::ImplicitLocOpBuilder b(loc, builder);
  Type i32_ty = b.getI32Type();
  Type i64_ty = b.getI64Type();
  Type int_ty;
  if constexpr (std::is_same_v<IndexT, int64_t>) {
    int_ty = i64_ty;
  } else {
    int_ty = i32_ty;
  }
  const DotDimensionNumbers& dims = dot_instr->dot_dimension_numbers();
  const DotFusionAnalysis analysis(dot_instr->parent(), config.split_k());

  // Rely on dot decomposer: there is just one contracting and one
  // non-contracting dimension on each side + batch ones optionally.
  CHECK_EQ(dims.lhs_contracting_dimensions_size(), 1);
  CHECK_EQ(dims.rhs_contracting_dimensions_size(), 1);

  const bool have_split_k = config.split_k() > 1;
  if (have_split_k) {
    // Split-K dimension has to be the first batch one and have an index
    // just before the contracting one.
    // Size of this dimension has to match the split_k value.
    CHECK_EQ(dims.lhs_batch_dimensions(0),
             dims.lhs_contracting_dimensions(0) - 1);
    CHECK_EQ(dims.rhs_batch_dimensions(0),
             dims.rhs_contracting_dimensions(0) - 1);
    CHECK_EQ(config.split_k(), dot_instr->operand(0)->shape().dimensions(
                                   dims.lhs_contracting_dimensions(0) - 1));
    CHECK_EQ(config.split_k(), dot_instr->operand(1)->shape().dimensions(
                                   dims.rhs_contracting_dimensions(0) - 1));
  }

  CHECK_LE(dims.lhs_batch_dimensions_size(), 1 + have_split_k);
  const bool have_batch = dims.lhs_batch_dimensions_size() - have_split_k;
  CHECK_EQ(dot_instr->operand(0)->shape().rank(),
           2 + have_split_k + have_batch);
  const int lhs_noncontracting_dim_idx =
      GetNonContractingDims(dot_instr->operand(0)->shape(),
                            dims.lhs_batch_dimensions(),
                            dims.lhs_contracting_dimensions())
          .value()[0];
  const int rhs_noncontracting_dim_idx =
      GetNonContractingDims(dot_instr->operand(1)->shape(),
                            dims.rhs_batch_dimensions(),
                            dims.rhs_contracting_dimensions())
          .value()[0];

  // Logical output dimensions are always ordered as:
  //   split-K, batch, non-contracting LHS, non-contracting RHS,
  // where split-K and batch are optional.
  const int rhs_nc_out_idx = dot_instr->shape().rank() - 1;
  const int lhs_nc_out_idx = dot_instr->shape().rank() - 2;
  const int split_k_out_idx = have_split_k ? 0 : -1;
  const int batch_out_idx = have_batch ? (have_split_k ? 1 : 0) : -1;

  // LHS non-contracting dimension length.
  // LHS non-contracting can be split, so this holds its full size unlike the
  // m_minor.
  int m =
      analysis.IterSpec(DotFusionAnalysis::Scope::OUTPUT, root, lhs_nc_out_idx)
          ->at(0)
          .count;

  // Contracting dimension length.
  const int k = dot_instr->operand(0)->shape().dimensions(
                    dims.lhs_contracting_dimensions(0)) *
                config.split_k();

  // For now all parameters of one scope (dot LHS, RHS) are required to have the
  // same physical layout = use the same indices in tiles. This is enforced by
  // construction in the Triton GEMM rewriter.

  // LHS non-contracting can be split into two.
  bool lhs_nc_split = false;
  // Either batch size or upper part of the length of a split nc dimension.
  int batch_size = 1;
  IndexT stride_lhs_batch = 0;
  IndexT stride_rhs_batch = 0;
  if (!analysis.ScopeParameters(DotFusionAnalysis::Scope::LHS).empty()) {
    const HloInstruction* lhs_param0 =
        *analysis.ScopeParameters(DotFusionAnalysis::Scope::LHS).begin();
    const TensorIterationSpec::DimIterationSpec* lhs_nc_iter_spec =
        analysis.IterSpec(DotFusionAnalysis::Scope::LHS, lhs_param0,
                          lhs_noncontracting_dim_idx);
    lhs_nc_split = lhs_nc_iter_spec->size() > 1;
    // For now split non-contracting and batch are not supported simultaneously
    // because they are implemented via same mechanism.
    CHECK_LE(have_batch + lhs_nc_split, 1);
    if (lhs_nc_split) {
      batch_size = lhs_nc_iter_spec->at(1).count;
      CHECK_GE(batch_size, 1);
      stride_lhs_batch = lhs_nc_iter_spec->at(1).stride;
      CHECK_GE(stride_lhs_batch, 1);
    } else if (have_batch) {
      const int64_t lhs_batch_dim_idx =
          *(dims.lhs_batch_dimensions().cend() - 1);
      batch_size = analysis
                       .IterSpec(DotFusionAnalysis::Scope::LHS, lhs_param0,
                                 lhs_batch_dim_idx)
                       ->at(0)
                       .count;
      CHECK_GE(batch_size, 1);
      stride_lhs_batch = analysis
                             .IterSpec(DotFusionAnalysis::Scope::LHS,
                                       lhs_param0, lhs_batch_dim_idx)
                             ->at(0)
                             .stride;
      CHECK_GE(stride_lhs_batch, 1);
    }

    CHECK_EQ(lhs_nc_iter_spec->size(), 1 + lhs_nc_split);
    CHECK_EQ(analysis
                 .IterSpec(DotFusionAnalysis::Scope::LHS, lhs_param0,
                           dims.lhs_contracting_dimensions(0))
                 ->size(),
             1);
    // Just the fastest-varying part of it if the dimension is split.
    m = lhs_nc_iter_spec->at(0).count;
  }

  CHECK_GE(m, 1);

  if (!analysis.ScopeParameters(DotFusionAnalysis::Scope::RHS).empty()) {
    const HloInstruction* rhs_param0 =
        *analysis.ScopeParameters(DotFusionAnalysis::Scope::RHS).begin();
    // Splitting of RHS non-contracting is not supported yet.
    CHECK_EQ(analysis
                 .IterSpec(DotFusionAnalysis::Scope::RHS, rhs_param0,
                           rhs_noncontracting_dim_idx)
                 ->size(),
             1);
    if (have_batch) {
      const int64_t rhs_batch_dim_idx =
          *(dims.rhs_batch_dimensions().cend() - 1);
      stride_rhs_batch = analysis
                             .IterSpec(DotFusionAnalysis::Scope::RHS,
                                       rhs_param0, rhs_batch_dim_idx)
                             ->at(0)
                             .stride;
      CHECK_GE(stride_rhs_batch, 1);
    }
  }

  constexpr int group_m = 8;

  const int n =
      analysis.IterSpec(DotFusionAnalysis::Scope::OUTPUT, root, rhs_nc_out_idx)
          ->at(0)
          .count;
  CHECK_GE(n, 1);

  const int block_m = config.block_m();
  const int block_k = config.block_k();
  const int block_n = config.block_n();

  CHECK_GE(block_m, 16);
  CHECK_GE(block_k, 16);
  CHECK_GE(block_n, 16);

  const int grid_m = ceil(1.0 * m / block_m);
  const int grid_n = ceil(1.0 * n / block_n);
  const int width = group_m * grid_n;

  Type dot_output_ty = TritonType(b, dot_instr->shape().element_type());

  // Data type of dot() immediate inputs.
  Type dot_input_ty = b.getF32Type();
  {
    const Type lhs_ty =
        TritonType(b, dot_instr->operand(0)->shape().element_type());
    const Type rhs_ty =
        TritonType(b, dot_instr->operand(1)->shape().element_type());
    CHECK(lhs_ty == rhs_ty);
    dot_input_ty = lhs_ty;
  }
  // TODO(b/266862493): Accumulator can be integer too.
  // Otherwise only f64 x f64 -> f64 uses f64 accumulator.
  mlir::FloatType acc_ty = (dot_output_ty.isF64() && dot_input_ty.isF64())
                               ? b.getF64Type()
                               : b.getF32Type();

  // X block size is 32-bit, Y and Z are 16-bit. Use X for large dimensions.
  constexpr int64_t kBlockCountYZLimit = 65536;
  const bool large_batch = batch_size >= kBlockCountYZLimit;
  auto pid_batch = b.create<mt::GetProgramIdOp>(
      large_batch ? mt::ProgramIDDim::X : mt::ProgramIDDim::Y);
  auto pid_nc = b.create<mt::GetProgramIdOp>(large_batch ? mt::ProgramIDDim::Y
                                                         : mt::ProgramIDDim::X);
  auto pid_k = b.create<mt::GetProgramIdOp>(mt::ProgramIDDim::Z);

  // In the imaginary situation where both batch size and grid_m * grid_n
  // are over 65535 we have to give up. Given the minimal m, n block sizes of 16
  // this requires at least 256 GB of output.
  CHECK_LT(batch_size * grid_m * grid_n,
           kBlockCountYZLimit * kBlockCountYZLimit);

  const LaunchDimensions launch_dimensions{
      {large_batch ? batch_size : grid_m * grid_n,
       large_batch ? grid_m * grid_n : batch_size, config.split_k()},
      {config.num_warps() * WarpSize(), 1, 1}};

  auto group_id = b.create<ma::DivSIOp>(pid_nc, CreateConst(b, i32_ty, width));
  ma::ConstantOp group_m_op = CreateConst(b, i32_ty, group_m);
  auto first_pid_m = b.create<ma::MulIOp>(group_id, group_m_op);
  auto sub0 = b.create<ma::SubIOp>(CreateConst(b, i32_ty, grid_m), first_pid_m);
  auto group_size = b.create<ma::SelectOp>(
      b.create<ma::CmpIOp>(ma::CmpIPredicate::slt, sub0, group_m_op), sub0,
      group_m_op);

  // Extend int32 indexes to int64, if necessary.
  auto convert_scalar = [&](Value value) -> Value {
    if constexpr (std::is_same_v<IndexT, int64_t>) {
      return b.create<ma::ExtSIOp>(int_ty, value);
    }
    return value;
  };

  auto pid_m = b.create<ma::AddIOp>(first_pid_m,
                                    b.create<ma::RemSIOp>(pid_nc, group_size));
  auto pid_m_offset =
      b.create<ma::MulIOp>(pid_m, CreateConst(b, i32_ty, block_m));

  auto pid_n = b.create<ma::DivSIOp>(
      b.create<ma::RemSIOp>(pid_nc, CreateConst(b, i32_ty, width)), group_size);
  auto pid_n_offset =
      b.create<ma::MulIOp>(pid_n, CreateConst(b, i32_ty, block_n));

  auto pid_k_offset =
      b.create<ma::MulIOp>(pid_k, CreateConst(b, i32_ty, block_k));

  ma::ConstantOp accumulator_init =
      CreateConst(b, acc_ty, 0, {block_m, block_n});

  // Numbers of dimensions of tensor pointers that need masking on loads or
  // stores.
  std::vector<int32_t> boundary_checks_lhs;
  std::vector<int32_t> boundary_checks_rhs;
  std::vector<int32_t> boundary_checks_out;
  if (m % block_m != 0) {
    boundary_checks_lhs.push_back(0);
    boundary_checks_out.push_back(0);
  }
  if (k % (block_k * config.split_k()) != 0) {
    boundary_checks_lhs.push_back(1);
    boundary_checks_rhs.push_back(0);
  }
  if (n % block_n != 0) {
    boundary_checks_rhs.push_back(1);
    boundary_checks_out.push_back(1);
  }

  // Parameters are passed to the loop in non-trivial order, this map helps
  // finding them.
  absl::flat_hash_map<int, const HloInstruction*> iter_args_to_parameters;

  auto body_builder = [&](mlir::OpBuilder&, mlir::Location, Value ki,
                          mlir::ValueRange iter_args) {
    SmallVector<Value> iter_args_next;
    iter_args_next.reserve(iter_args.size());
    absl::flat_hash_map<const HloInstruction*, Value> values_lhs;
    absl::flat_hash_map<const HloInstruction*, Value> values_rhs;
    // Load tiles of all parameters of LHS and RHS scopes and advance pointers.
    for (int i = 0; i < iter_args.size() - 1; ++i) {
      const bool is_lhs =
          i < analysis.ScopeParameters(DotFusionAnalysis::Scope::LHS).size();
      const int increment_dim0 = block_k * config.split_k() * (is_lhs ? 0 : 1);
      const int increment_dim1 = block_k * config.split_k() * (is_lhs ? 1 : 0);
      absl::flat_hash_map<const HloInstruction*, Value>& values =
          is_lhs ? values_lhs : values_rhs;
      CHECK(values
                .insert({iter_args_to_parameters[i],
                         EmitParameterLoad(b, iter_args[i],
                                           is_lhs ? boundary_checks_lhs
                                                  : boundary_checks_rhs)})
                .second);
      iter_args_next.push_back(b.create<mt::AdvanceOp>(
          iter_args[i].getType(), iter_args[i],
          mlir::ValueRange{CreateConst(b, i32_ty, increment_dim0),
                           CreateConst(b, i32_ty, increment_dim1)}));
    }

    // TODO(b/269726484): Peel the loop instead of inserting a masked load in
    // every iteration, even the ones that do not need it.
    const bool need_masking = k % (block_k * config.split_k()) > 0;
    Value lhs_mask;
    Value rhs_mask;
    if (need_masking) {
      auto elements_in_tile =
          b.create<ma::SubIOp>(CreateConst(b, i32_ty, k), ki);
      auto range_k = b.create<ma::AddIOp>(
          Splat(b, b.create<ma::MulIOp>(pid_k, CreateConst(b, i32_ty, block_k)),
                block_k),
          Range(b, block_k));
      lhs_mask = Broadcast(
          b,
          b.create<ma::CmpIOp>(ma::CmpIPredicate::slt,
                               b.create<mt::ExpandDimsOp>(range_k, 0),
                               Splat(b, elements_in_tile, {1, block_k}))
              .getResult()
              .template cast<TensorValue>(),
          {block_m, block_k});
      rhs_mask = Broadcast(
          b,
          b.create<ma::CmpIOp>(ma::CmpIPredicate::slt,
                               b.create<mt::ExpandDimsOp>(range_k, 1),
                               Splat(b, elements_in_tile, {block_k, 1}))
              .getResult()
              .template cast<TensorValue>(),
          {block_k, block_n});
    }

    // Emit all operations of LHS and RHS scopes.
    TF_ASSIGN_OR_RETURN(
        Value dot_input_lhs,
        EmitScope(b, libdevice_path,
                  dot_instr->parent()->MakeInstructionPostOrderFrom(
                      const_cast<HloInstruction&>(*dot_instr->operand(0))),
                  values_lhs, {block_m, block_k}, lhs_mask));
    TF_ASSIGN_OR_RETURN(
        Value dot_input_rhs,
        EmitScope(b, libdevice_path,
                  dot_instr->parent()->MakeInstructionPostOrderFrom(
                      const_cast<HloInstruction&>(*dot_instr->operand(1))),
                  values_rhs, {block_k, block_n}, rhs_mask));

    // Operation in the fusion before the dot can alter the elements of the
    // tiles that were zero masked during loads. These have to be zeroed here
    // again just before the dot so that they do not affect the output.
    // Only the K dimension needs masking here because unnecessary elements in
    // the other two get discarded by the masked store at the end.
    if (need_masking) {
      dot_input_lhs = b.create<ma::SelectOp>(lhs_mask, dot_input_lhs,
                                             ZerosLike(b, dot_input_lhs));
      dot_input_rhs = b.create<ma::SelectOp>(rhs_mask, dot_input_rhs,
                                             ZerosLike(b, dot_input_rhs));
    }

    // Execute matrix multiplication of input tiles and pass the accumulator.
    Value accumulator_next = b.create<mt::DotOp>(
        dot_input_lhs, dot_input_rhs, iter_args.back(),
        /*allowTF32=*/tsl::tensor_float_32_execution_enabled());
    iter_args_next.push_back(accumulator_next);

    b.create<mlir::scf::YieldOp>(iter_args_next);
    return OkStatus();
  };

  // Pointers to parameters of LHS scope, then RHS, then the accumulator
  // that change with every loop iteration and are passed between them.
  // LHS and RHS can use same HLO computation parameters, but because they use
  // different pointers they have to be stored separately for each scope.
  SmallVector<Value> iter_args;
  iter_args.reserve(
      analysis.ScopeParameters(DotFusionAnalysis::Scope::LHS).size() +
      analysis.ScopeParameters(DotFusionAnalysis::Scope::RHS).size() + 1);

  Value lhs_offset_batch = b.create<ma::MulIOp>(
      convert_scalar(pid_batch), CreateConst(b, int_ty, stride_lhs_batch));
  for (const HloInstruction* parameter :
       analysis.ScopeParameters(DotFusionAnalysis::Scope::LHS)) {
    Value base = fn.getArgument(parameter->parameter_number());
    const int64_t stride_lhs_m =
        analysis
            .IterSpec(DotFusionAnalysis::Scope::LHS, parameter,
                      lhs_noncontracting_dim_idx)
            ->at(0)
            .stride;
    const int64_t stride_lhs_k =
        analysis
            .IterSpec(DotFusionAnalysis::Scope::LHS, parameter,
                      dims.lhs_contracting_dimensions(0))
            ->at(0)
            .stride;
    Value ptrs = b.create<mt::MakeTensorPtrOp>(
        /*base=*/AddPtr(b, base, lhs_offset_batch),
        /*shape=*/
        mlir::ValueRange{CreateConst(b, i64_ty, m), CreateConst(b, i64_ty, k)},
        /*strides=*/
        mlir::ValueRange{CreateConst(b, i64_ty, stride_lhs_m),
                         CreateConst(b, i64_ty, stride_lhs_k)},
        /*offsets=*/mlir::ValueRange{pid_m_offset, pid_k_offset},
        /*tensorShape=*/std::vector<int32_t>{block_m, block_k},
        /*order=*/std::vector<int32_t>{1, 0});
    CHECK(iter_args_to_parameters.insert({iter_args.size(), parameter}).second)
        << parameter->ToString();
    iter_args.push_back(ptrs);
  }

  Value rhs_offset_batch = b.create<ma::MulIOp>(
      convert_scalar(pid_batch), CreateConst(b, int_ty, stride_rhs_batch));
  for (const HloInstruction* parameter :
       analysis.ScopeParameters(DotFusionAnalysis::Scope::RHS)) {
    Value base = fn.getArgument(parameter->parameter_number());
    const IndexT stride_rhs_k =
        analysis
            .IterSpec(DotFusionAnalysis::Scope::RHS, parameter,
                      dims.rhs_contracting_dimensions(0))
            ->at(0)
            .stride;
    const IndexT stride_rhs_n =
        analysis
            .IterSpec(DotFusionAnalysis::Scope::RHS, parameter,
                      rhs_noncontracting_dim_idx)
            ->at(0)
            .stride;
    Value ptrs = b.create<mt::MakeTensorPtrOp>(
        /*base=*/AddPtr(b, base, rhs_offset_batch),
        /*shape=*/
        mlir::ValueRange{CreateConst(b, i64_ty, k), CreateConst(b, i64_ty, n)},
        /*strides=*/
        mlir::ValueRange{CreateConst(b, i64_ty, stride_rhs_k),
                         CreateConst(b, i64_ty, stride_rhs_n)},
        /*offsets=*/mlir::ValueRange{pid_k_offset, pid_n_offset},
        /*tensorShape=*/std::vector<int32_t>{block_k, block_n},
        /*order=*/std::vector<int32_t>{1, 0});
    CHECK(iter_args_to_parameters.insert({iter_args.size(), parameter}).second)
        << parameter->ToString();
    iter_args.push_back(ptrs);
  }

  iter_args.push_back(accumulator_init);
  Value acc_final =
      b.create<mlir::scf::ForOp>(
           /*lowerBound=*/b.create<ma::ConstantIntOp>(0, /*width=*/32),
           /*upperBound=*/b.create<ma::ConstantIntOp>(k, /*width=*/32),
           /*step=*/
           b.create<ma::ConstantIntOp>(block_k * config.split_k(),
                                       /*width=*/32),
           /*iterArgs=*/iter_args, body_builder)
          .getResult(iter_args.size() - 1);
  absl::flat_hash_map<const HloInstruction*, Value> values_out;
  values_out[dot_instr] =
      Cast(b, acc_final, TritonType(b, dot_instr->shape().element_type()));

  // Generate tensor pointer for a parameter load or output store within the
  // dot's output scope.
  auto output_scope_tensor_pointer = [&](const HloInstruction* hlo, Value base,
                                         bool add_split_k_offset) {
    const IndexT stride_m =
        analysis
            .IterSpec(DotFusionAnalysis::Scope::OUTPUT, hlo, lhs_nc_out_idx)
            ->at(0)
            .stride;
    {
      IndexT stride_batch = 0;
      if (have_batch) {
        stride_batch =
            analysis
                .IterSpec(DotFusionAnalysis::Scope::OUTPUT, hlo, batch_out_idx)
                ->at(0)
                .stride;
        CHECK_GE(stride_batch, 1);
      }
      {
        const TensorIterationSpec::DimIterationSpec* spec = analysis.IterSpec(
            DotFusionAnalysis::Scope::OUTPUT, hlo, lhs_nc_out_idx);
        if (spec->size() > 1) {
          CHECK_EQ(spec->size(), 2);
          // Support one specific kind of output transpose that splits the
          // dimension originating from the split LHS non-contracting one.
          CHECK(!have_batch);
          CHECK(lhs_nc_split);
          CHECK_EQ(spec->at(1).count, batch_size);
          stride_batch = spec->at(1).stride;
        } else if (lhs_nc_split) {
          // Dimension of the output produced by the non-contracting LHS one
          // is physically contiguous though the producing LHS one is split.
          // Because the major part of the split is implemented using the batch
          // logic stride_out_batch is populated here as the stride of the minor
          // part times its size.
          stride_batch = stride_m * m;
        }
      }
      Value offset_batch = b.create<ma::MulIOp>(
          convert_scalar(pid_batch), CreateConst(b, int_ty, stride_batch));
      base = AddPtr(b, base, offset_batch);
    }
    if (add_split_k_offset) {
      IndexT stride_split_k = 0;
      if (have_split_k) {
        stride_split_k = analysis
                             .IterSpec(DotFusionAnalysis::Scope::OUTPUT, hlo,
                                       split_k_out_idx)
                             ->at(0)
                             .stride;
        CHECK_GE(stride_split_k, 1);
      }
      Value offset_split_k = b.create<ma::MulIOp>(
          convert_scalar(pid_k), CreateConst(b, int_ty, stride_split_k));
      base = AddPtr(b, base, offset_split_k);
    }
    const IndexT stride_n =
        analysis
            .IterSpec(DotFusionAnalysis::Scope::OUTPUT, hlo, rhs_nc_out_idx)
            ->at(0)
            .stride;
    return b.create<mt::MakeTensorPtrOp>(
        /*base=*/base,
        /*shape=*/
        mlir::ValueRange{CreateConst(b, i64_ty, m), CreateConst(b, i64_ty, n)},
        /*strides=*/
        mlir::ValueRange{CreateConst(b, i64_ty, stride_m),
                         CreateConst(b, i64_ty, stride_n)},
        /*offsets=*/mlir::ValueRange{pid_m_offset, pid_n_offset},
        /*tensorShape=*/std::vector<int32_t>{block_m, block_n},
        /*order=*/std::vector<int32_t>{1, 0});
  };

  // Collect all instructions of the dot's output scope.
  absl::flat_hash_set<const HloInstruction*> to_order;
  {
    std::queue<const HloInstruction*> to_add;
    if (root != dot_instr) {
      to_add.push(root);
    }
    while (!to_add.empty()) {
      const HloInstruction* current = to_add.front();
      for (const HloInstruction* operand : current->operands()) {
        if (!to_order.contains(operand)) {
          if (operand != dot_instr) {
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
       dot_instr->parent()->MakeInstructionPostOrder()) {
    if (to_order.contains(hlo)) {
      to_emit.push_back(hlo);
    }
  }
  // Emit the output scope.
  if (!to_emit.empty()) {
    for (const HloInstruction* parameter :
         analysis.ScopeParameters(DotFusionAnalysis::Scope::OUTPUT)) {
      Value tensor_pointer = output_scope_tensor_pointer(
          parameter, fn.getArgument(parameter->parameter_number()),
          /*add_split_k_offset=*/false);
      CHECK(values_out
                .insert({parameter, EmitParameterLoad(b, tensor_pointer,
                                                      boundary_checks_out)})
                .second);
    }
    TF_RETURN_IF_ERROR(EmitScope(b, libdevice_path, to_emit, values_out,
                                 {block_m, block_n}, /*tile_mask=*/{})
                           .status());
  }

  // Emit tensor store operations for all outputs.
  for (int i = 0;
       i < fn.getNumArguments() - dot_instr->parent()->num_parameters(); ++i) {
    const HloInstruction* producer =
        root->shape().IsTuple() ? root->operand(i) : root;
    Value tensor_pointer = output_scope_tensor_pointer(
        producer, fn.getArgument(i + dot_instr->parent()->num_parameters()),
        /*add_split_k_offset=*/true);
    b.create<mt::StoreOp>(tensor_pointer, values_out[producer],
                          boundary_checks_out, mt::CacheModifier::NONE,
                          mt::EvictionPolicy::NORMAL);
  }
  return launch_dimensions;
}

}  // namespace

StatusOr<LaunchDimensions> MatMul(mlir::OpBuilder builder,
                                  absl::string_view libdevice_path,
                                  const HloComputation* computation,
                                  mlir::triton::FuncOp fn,
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
  if (use_64bit_indexing) {
    return MatMulImpl<int64_t>(builder, libdevice_path, dot_instr, fn, config,
                               shmem_budget);
  } else {
    return MatMulImpl<int32_t>(builder, libdevice_path, dot_instr, fn, config,
                               shmem_budget);
  }
}

StatusOr<LaunchDimensions> SoftMax(mlir::OpBuilder builder,
                                   absl::string_view libdevice_path,
                                   const HloComputation* computation,
                                   mlir::triton::FuncOp fn,
                                   const AutotuneResult::TritonGemmKey& config,
                                   int) {
  const HloInstruction* root = computation->root_instruction();
  auto loc = mlir::NameLoc::get(builder.getStringAttr(root->name()));
  mlir::ImplicitLocOpBuilder b(loc, builder);

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

  int num_rows = 1;
  for (int minor_axis = 1; minor_axis < reduce_input_shape.rank(); ++minor_axis)
    num_rows *= reduce_input_shape.dimensions_minor(minor_axis);

  Value row_index = b.create<mt::GetProgramIdOp>(mt::ProgramIDDim::X);
  Value row_stride = CreateConst(b, b.getI32Type(), row_len);

  absl::flat_hash_map<const HloInstruction*, Value> values_out;
  auto make_tensor_pointer = [&](Value base) {
    Value offset = b.create<ma::MulIOp>(row_index, row_stride);
    return b.create<mt::MakeTensorPtrOp>(
        /*base=*/AddPtr(b, base, offset),
        /*shape=*/mlir::ValueRange{CreateConst(b, b.getI64Type(), row_len)},
        /*strides=*/mlir::ValueRange{CreateConst(b, b.getI64Type(), 1)},
        /*offsets=*/mlir::ValueRange{CreateConst(b, b.getI32Type(), 0)},
        /*tensorShape=*/std::vector<int32_t>{block_row},
        /*order=*/std::vector<int32_t>{0});
  };

  std::vector<int32_t> boundary_checks;
  if (block_row != row_len) {
    boundary_checks.push_back(0);
  }
  values_out[computation->parameter_instruction(0)] = EmitParameterLoad(
      b, make_tensor_pointer(fn.getArgument(0)), boundary_checks);
  Value mask = b.create<ma::CmpIOp>(ma::CmpIPredicate::slt, Range(b, block_row),
                                    Splat(b, row_stride, block_row));
  TF_ASSIGN_OR_RETURN(
      Value result,
      EmitScope(b, libdevice_path, computation->MakeInstructionPostOrder(),
                values_out, {block_row}, mask));

  b.create<mt::StoreOp>(make_tensor_pointer(fn.getArgument(1)), result,
                        std::vector<int32_t>{0}, mt::CacheModifier::NONE,
                        mt::EvictionPolicy::NORMAL);

  const LaunchDimensions launch_dimensions{
      {num_rows, 1, 1}, {config.num_warps() * WarpSize(), 1, 1}};

  return launch_dimensions;
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

StatusOr<LaunchDimensions> TritonWrapper(
    absl::string_view fn_name, const HloComputation* hlo_computation,
    absl::string_view fusion_kind, const se::CudaComputeCapability& cc,
    const GpuDeviceInfo& device_info,
    const AutotuneResult::TritonGemmKey& config, llvm::Module* llvm_module,
    LaunchDimensionsGenerator generator, mlir::MLIRContext& mlir_context) {
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

  mlir_context.loadDialect<mt::TritonDialect>();
  mlir::OpBuilder b(&mlir_context);
  auto loc = mlir::NameLoc::get(b.getStringAttr(hlo_computation->name()));
  auto triton_module = mlir::ModuleOp::create(loc);
  b.setInsertionPointToEnd(triton_module.getBody());

  VLOG(3) << hlo_computation->ToString();
  VLOG(2) << config.ShortDebugString();

  // Build Triton kernel.
  SmallVector<Type> fn_arg_types;
  for (HloInstruction* p : hlo_computation->parameter_instructions()) {
    fn_arg_types.push_back(mt::PointerType::get(
        TritonType(b, p->shape().element_type()), mn::kGlobalMemorySpace));
  }

  for (const ShapeUtil::IndexedShape& s :
       ShapeUtil::GetLeafShapes(hlo_computation->root_instruction()->shape())) {
    fn_arg_types.push_back(mt::PointerType::get(
        TritonType(b, s.shape.element_type()), mn::kGlobalMemorySpace));
  }

  auto fn = b.create<mt::FuncOp>(loc, fn_name,
                                 b.getFunctionType(fn_arg_types, std::nullopt));
  for (int i = 0; i < fn.getNumArguments(); ++i) {
    fn.setArgAttr(i, "tt.divisibility", b.getIntegerAttr(b.getI32Type(), 16));
  }
  fn.addEntryBlock();
  b.setInsertionPointToStart(&fn.front());

  const std::string libdevice_path =
      nvptx::LibDevicePath(hlo_computation->parent()
                               ->config()
                               .debug_options()
                               .xla_gpu_cuda_data_dir());

  TF_ASSIGN_OR_RETURN(LaunchDimensions launch_dimensions,
                      generator(b, libdevice_path, hlo_computation, fn, config,
                                device_info.shared_memory_per_block_optin));

  b.create<mt::ReturnOp>(loc);
  VLOG(6) << llvm_ir::DumpToString(triton_module);
  CHECK(mlir::succeeded(mlir::verify(triton_module)));

  // Compile Triton kernel to LLVM.
  mlir::PassManager pm(&mlir_context);

  std::optional<llvm::raw_fd_ostream> log_stream;
  const HloModule* hlo_module = hlo_computation->parent();
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
  // Triton generates pointers to the global address space, while XLA needs a
  // kernel signature with pointers to the generic address space.
  pm.addPass(std::make_unique<GeneralizeKernelSignaturePass>());
  // llvm::Linker::linkModules() segfaults if we don't strip locations.
  pm.addPass(mlir::createStripDebugInfoPass());

  bool succeeded = mlir::succeeded(pm.run(triton_module));

  if (log_stream.has_value()) {
    log_stream->flush();
  }

  if (!succeeded) {
    return InternalError("Failed to compile Triton kernel.");
  }

  const int shared_mem_bytes =
      triton_module->getAttrOfType<mlir::IntegerAttr>("triton_gpu.shared")
          .getInt();
  VLOG(2) << "Shared memory usage: " << shared_mem_bytes << " B";
  if (shared_mem_bytes > device_info.shared_memory_per_block_optin) {
    return ResourceExhausted("Shared memory size limit exceeded.");
  }
  launch_dimensions.SetSharedMemBytes(shared_mem_bytes);

  TF_ASSIGN_OR_RETURN(std::unique_ptr<llvm::Module> ll_triton_module,
                      TranslateLLVMToLLVMIR(&llvm_module->getContext(),
                                            triton_module, libdevice_path));
  LogAndVerify(ll_triton_module.get());

  // Integrate LLVM matmul kernel into XLA's LLVM module.
  ll_triton_module->eraseNamedMDNode(
      ll_triton_module->getNamedMetadata("nvvm.annotations"));
  ll_triton_module->setDataLayout(llvm_module->getDataLayout());
  // Use override flag because libdevice functions can be present in both.
  CHECK(!llvm::Linker::linkModules(*llvm_module, std::move(ll_triton_module),
                                   llvm::Linker::Flags::OverrideFromSrc));
  LogAndVerify(llvm_module);

  return launch_dimensions;
}

}  // namespace gpu
}  // namespace xla
