/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/gpu/elemental_ir_emitter.h"

#include <cstdint>
#include <string>
#include <vector>

// IWYU pragma: no_include "llvm/IR/Attributes.gen.inc"
// IWYU pragma: no_include "llvm/IR/Intrinsics.gen.inc"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/ModRef.h"
#include "llvm/TargetParser/Triple.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout.h"
#include "xla/service/elemental_ir_emitter.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/ir_emitter_nested.h"
#include "xla/service/gpu/target_util.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/llvm_ir/math_ops.h"
#include "xla/stream_executor/device_description.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

GpuElementalIrEmitter::GpuElementalIrEmitter(
    IrEmitterContext& ir_emitter_context, llvm::IRBuilder<>* b)
    : ElementalIrEmitter(ir_emitter_context.llvm_module(), b),
      ir_emitter_context_(ir_emitter_context) {}

absl::StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitDeviceMathCall(
    TargetDeviceFunctionID funcid, absl::Span<llvm::Value* const> operands,
    absl::Span<const PrimitiveType> input_types, PrimitiveType output_type,
    absl::string_view name) {
  // Device functions don't have f16 math functions, so we convert the operands
  // to f32 before calling the function and then convert the result back to f16.
  bool cast_result_to_fp16 = false;
  std::vector<llvm::Value*> converted_operands(operands.begin(),
                                               operands.end());
  std::vector<PrimitiveType> converted_input_types(input_types.begin(),
                                                   input_types.end());
  switch (output_type) {
    case F16:
      cast_result_to_fp16 = true;
      for (int64_t i = 0; i < operands.size(); ++i) {
        if (input_types[i] == F16) {
          converted_operands[i] =
              FPCast(converted_operands[i], b()->getFloatTy());
          converted_input_types[i] = F32;
        }
      }
      output_type = F32;
      [[fallthrough]];
    case F32:
      break;
    case F64:
      break;
    default:
      return Unimplemented("Bad type for device math call: %s",
                           PrimitiveType_Name(output_type));
  }
  const std::string& munged_callee = ObtainDeviceFunctionName(
      funcid, output_type,
      llvm::Triple(b()->GetInsertBlock()->getModule()->getTargetTriple()));
  llvm::Value* result = EmitMathCall(munged_callee, converted_operands,
                                     converted_input_types, output_type, name)
                            .value();
  if (cast_result_to_fp16) {
    result = FPCast(result, b()->getHalfTy());
  }
  return result;
}

absl::StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitMathCall(
    const std::string& callee_name, absl::Span<llvm::Value* const> operands,
    absl::Span<const PrimitiveType> input_types, PrimitiveType output_type,
    absl::string_view name) {
  // Binary math functions transform are of type [T] -> T.
  for (PrimitiveType input_type : input_types) {
    if (output_type != input_type) {
      return Unimplemented("Input type != output type: %s != %s",
                           PrimitiveType_Name(input_type),
                           PrimitiveType_Name(output_type));
    }
  }

  return EmitDeviceFunctionCall(callee_name, operands, input_types, output_type,
                                llvm::AttrBuilder(b()->getContext())
                                    .addMemoryAttr(llvm::MemoryEffects::none())
                                    .addAttribute(llvm::Attribute::NoUnwind),
                                b(), name);
}

llvm_ir::IrArray::Index GpuElementalIrEmitter::GetSourceIndexOfBitcast(
    const llvm_ir::IrArray::Index& index, const HloInstruction* hlo) {
  Shape shape = hlo->shape();
  Shape operand_shape = hlo->operand(0)->shape();

  // Decode the layout of the shape from the Protobufs attached to
  // backend_config_.
  auto gpu_config = hlo->backend_config<GpuBackendConfig>();
  CHECK_OK(gpu_config);

  const BitcastBackendConfig& bitcast_config =
      gpu_config.value().bitcast_backend_config();

  // If there is no layout in the protobuf, do not override it.
  if (!bitcast_config.result_layout().minor_to_major().empty()) {
    *shape.mutable_layout() =
        xla::Layout::CreateFromProto(bitcast_config.result_layout());
  }
  if (!bitcast_config.source_layout().minor_to_major().empty()) {
    *operand_shape.mutable_layout() =
        xla::Layout::CreateFromProto(bitcast_config.source_layout());
  }
  return index.SourceIndexOfBitcast(shape, operand_shape, b());
}

absl::StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitFloatBinaryOp(
    const HloInstruction* op, llvm::Value* lhs_value, llvm::Value* rhs_value) {
  PrimitiveType lhs_input_type = op->operand(0)->shape().element_type();
  PrimitiveType rhs_input_type = op->operand(1)->shape().element_type();
  PrimitiveType output_type = op->shape().element_type();
  HloOpcode opcode = op->opcode();

  if (ir_emitter_context_.debug_options().xla_gpu_enable_fast_min_max() &&
      (opcode == HloOpcode::kMaximum || opcode == HloOpcode::kMinimum)) {
    return llvm_ir::EmitCallToIntrinsic(
        opcode == HloOpcode::kMaximum ? llvm::Intrinsic::maxnum
                                      : llvm::Intrinsic::minnum,
        {lhs_value, rhs_value}, {lhs_value->getType()}, b());
  }

  // sm_80 and up has min.NaN and max.NaN instructions.
  if (output_type == F32 &&
      ir_emitter_context_.cuda_compute_capability().IsAtLeast(
          se::CudaComputeCapability::AMPERE) &&
      (opcode == HloOpcode::kMaximum || opcode == HloOpcode::kMinimum)) {
    return llvm_ir::EmitCallToIntrinsic(
        opcode == HloOpcode::kMaximum ? llvm::Intrinsic::maximum
                                      : llvm::Intrinsic::minimum,
        {lhs_value, rhs_value}, {lhs_value->getType()}, b());
  }

  switch (op->opcode()) {
    case HloOpcode::kRemainder: {
      return EmitDeviceMathCall(TargetDeviceFunctionID::kFmod,
                                {lhs_value, rhs_value},
                                {lhs_input_type, rhs_input_type}, output_type);
    }
    case HloOpcode::kPower: {
      return EmitPowerOp(op, lhs_value, rhs_value);
    }
    default:
      return ElementalIrEmitter::EmitFloatBinaryOp(op, lhs_value, rhs_value);
  }
}

absl::StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitPowerOp(
    const HloInstruction* op, llvm::Value* lhs_value, llvm::Value* rhs_value) {
  CHECK_EQ(op->opcode(), HloOpcode::kPower);
  PrimitiveType lhs_input_type = op->operand(0)->shape().element_type();
  PrimitiveType rhs_input_type = op->operand(1)->shape().element_type();
  PrimitiveType output_type = op->shape().element_type();
  return EmitDeviceMathCall(TargetDeviceFunctionID::kPow,
                            {lhs_value, rhs_value},
                            {lhs_input_type, rhs_input_type}, output_type);
}

absl::StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitLog(
    PrimitiveType prim_type, llvm::Value* value) {
  return EmitDeviceMathCall(TargetDeviceFunctionID::kLog, {value}, {prim_type},
                            prim_type);
}

absl::StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitLog1p(
    PrimitiveType prim_type, llvm::Value* value) {
  return EmitDeviceMathCall(TargetDeviceFunctionID::kLog1p, {value},
                            {prim_type}, prim_type);
}

absl::StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitSin(
    PrimitiveType prim_type, llvm::Value* value) {
  return EmitDeviceMathCall(TargetDeviceFunctionID::kSin, {value}, {prim_type},
                            prim_type);
}

absl::StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitCos(
    PrimitiveType prim_type, llvm::Value* value) {
  return EmitDeviceMathCall(TargetDeviceFunctionID::kCos, {value}, {prim_type},
                            prim_type);
}

absl::StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitTan(
    PrimitiveType prim_type, llvm::Value* value) {
  return EmitDeviceMathCall(TargetDeviceFunctionID::kTan, {value}, {prim_type},
                            prim_type);
}

absl::StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitExp(
    PrimitiveType prim_type, llvm::Value* value, absl::string_view /*name*/) {
  return EmitDeviceMathCall(TargetDeviceFunctionID::kExp, {value}, {prim_type},
                            prim_type);
}

absl::StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitExpm1(
    PrimitiveType prim_type, llvm::Value* value) {
  return EmitDeviceMathCall(TargetDeviceFunctionID::kExpm1, {value},
                            {prim_type}, prim_type);
}

absl::StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitPow(
    PrimitiveType prim_type, llvm::Value* lhs, llvm::Value* rhs,
    absl::string_view name) {
  return EmitDeviceMathCall(TargetDeviceFunctionID::kPow, {lhs, rhs},
                            {prim_type, prim_type}, prim_type, name);
}

absl::StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitSqrt(
    PrimitiveType prim_type, llvm::Value* value) {
  return EmitDeviceMathCall(TargetDeviceFunctionID::kSqrt, {value}, {prim_type},
                            prim_type);
}

absl::StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitRsqrt(
    PrimitiveType prim_type, llvm::Value* value) {
  return EmitDeviceMathCall(TargetDeviceFunctionID::kRsqrt, {value},
                            {prim_type}, prim_type);
}

absl::StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitAtan2(
    PrimitiveType prim_type, llvm::Value* lhs, llvm::Value* rhs,
    absl::string_view name) {
  return EmitDeviceMathCall(TargetDeviceFunctionID::kAtan2, {lhs, rhs},
                            {prim_type, prim_type}, prim_type, name);
}

absl::StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitTanh(
    PrimitiveType prim_type, llvm::Value* value) {
  // When F64 is being requested, assume performance is less important and use
  // the more numerically precise tanh function.
  if (prim_type == F64) {
    return EmitDeviceMathCall(TargetDeviceFunctionID::kTanh, {value},
                              {prim_type}, prim_type);
  }

  // Emit a fast approximation of tanh instead of calling __nv_tanh.
  // __nv_tanh is particularly bad because it contains branches, thus
  // preventing LLVM's load-store vectorizer from working its magic across a
  // function which contains tanh calls.
  //
  // This routine isn't numerically precise, but it's good enough for ML.

  // Upcast F16 to F32 if necessary.
  llvm::Type* type = prim_type == F16 ? b()->getFloatTy() : value->getType();
  llvm::Value* input = FPCast(value, type);

  // If |value| >= kMaxValue, tanh() is set to -1.0 or 1.0.
  constexpr double kMaxValue = 20.0;
  auto max_value = llvm::ConstantFP::get(type, kMaxValue);
  llvm::Value* abs_value =
      llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::fabs, {input}, {type}, b());

  llvm::Value* fast_tanh = llvm_ir::EmitFastTanh(b(), input);
  auto one = llvm::ConstantFP::get(type, 1.0);
  auto one_with_sign = llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::copysign,
                                                    {one, input}, {type}, b());
  return FPCast(Select(FCmpULT(abs_value, max_value), fast_tanh, one_with_sign),
                value->getType(), "tanh");
}

absl::StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitErf(
    PrimitiveType prim_type, llvm::Value* value) {
  if (prim_type == F64) {
    return EmitDeviceMathCall(TargetDeviceFunctionID::kErf, {value},
                              {prim_type}, prim_type);
  }
  // Upcast F16 to F32 if necessary.
  llvm::Type* type = prim_type == F16 ? b()->getFloatTy() : value->getType();
  if (type == b()->getFloatTy()) {
    llvm::Value* x = FPCast(value, type);
    auto* result = llvm_ir::EmitErfF32(b(), x);
    return FPCast(result, value->getType());
  }
  return Unimplemented("erf");
}

absl::StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitComplexAbs(
    PrimitiveType prim_type, llvm::Value* value) {
  return EmitDeviceMathCall(TargetDeviceFunctionID::kHypot,
                            {EmitExtractReal(value), EmitExtractImag(value)},
                            {prim_type, prim_type}, prim_type);
}

absl::StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitCbrt(
    PrimitiveType prim_type, llvm::Value* value) {
  return EmitDeviceMathCall(TargetDeviceFunctionID::kCbrt, {value}, {prim_type},
                            prim_type);
}

absl::StatusOr<std::vector<llvm::Value*>>
GpuElementalIrEmitter::EmitThreadLocalCall(
    const HloComputation& callee, absl::Span<llvm::Value* const> parameters,
    absl::string_view, bool /*is_reducer*/) {
  return CallNestedComputationWithScalars(b(), ir_emitter_context_, callee,
                                          parameters);
}

}  // namespace gpu
}  // namespace xla
