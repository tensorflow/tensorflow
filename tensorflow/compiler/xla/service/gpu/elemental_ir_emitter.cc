/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/elemental_ir_emitter.h"

#include <stddef.h>

#include <unordered_map>
#include <vector>

#include "llvm/IR/DerivedTypes.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
// IWYU pragma: no_include "llvm/IR/Attributes.gen.inc"
// IWYU pragma: no_include "llvm/IR/Intrinsics.gen.inc"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/APInt.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/gpu/target_util.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/llvm_ir/ir_array.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_loop.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/service/llvm_ir/math_ops.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace gpu {

using absl::StrAppend;
using llvm_ir::IrArray;
using llvm_ir::IrName;
using llvm_ir::SetToFirstInsertPoint;

namespace {
// Returns whether operand is a floating-point literal with the given value.
bool IsFPLiteralWithValue(const HloInstruction* operand, float value) {
  if (operand->opcode() == HloOpcode::kConstant &&
      operand->literal().IsAllFloat(value)) {
    return true;
  }
  return operand->opcode() == HloOpcode::kBroadcast &&
         IsFPLiteralWithValue(operand->operand(0), value);
}
}  // namespace

GpuElementalIrEmitter::GpuElementalIrEmitter(
    const HloModuleConfig& hlo_module_config, llvm::Module* module,
    llvm::IRBuilder<>* b, NestedComputer compute_nested)
    : ElementalIrEmitter(hlo_module_config, module, b),
      compute_nested_(std::move(compute_nested)) {}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitDeviceMathCall(
    TargetDeviceFunctionID funcid, absl::Span<llvm::Value* const> operands,
    absl::Span<const PrimitiveType> input_types, PrimitiveType output_type) {
  // Device functions dont have f16 math functions, so we convert the operands
  // to f32 before calling the function and then convert the result back to f16.
  bool cast_result_to_fp16 = false;
  std::vector<llvm::Value*> converted_operands(operands.begin(),
                                               operands.end());
  std::vector<PrimitiveType> converted_input_types(input_types.begin(),
                                                   input_types.end());
  switch (output_type) {
    case F16:
      cast_result_to_fp16 = true;
      for (int64 i = 0; i < operands.size(); ++i) {
        if (input_types[i] == F16) {
          converted_operands[i] =
              FPCast(converted_operands[i], b_->getFloatTy());
          converted_input_types[i] = F32;
        }
      }
      output_type = F32;
      TF_FALLTHROUGH_INTENDED;
    case F32:
      break;
    case F64:
      break;
    default:
      return Unimplemented("Bad type for device math call: %s",
                           PrimitiveType_Name(output_type));
  }
  const string& munged_callee =
      ObtainDeviceFunctionName(funcid, output_type, b_);
  llvm::Value* result = EmitMathCall(munged_callee, converted_operands,
                                     converted_input_types, output_type)
                            .ValueOrDie();
  if (cast_result_to_fp16) {
    result = FPCast(result, b_->getHalfTy());
  }
  return result;
}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitLlvmIntrinsicMathCall(
    const string& callee_name, absl::Span<llvm::Value* const> operands,
    absl::Span<const PrimitiveType> input_types, PrimitiveType output_type) {
  // llvm intrinsics differentiate between half/float/double functions via
  // the suffixes ".f16", ".f32" and ".f64".
  string munged_callee = callee_name;
  switch (output_type) {
    case F16:
      StrAppend(&munged_callee, ".f16");
      break;
    case F32:
      StrAppend(&munged_callee, ".f32");
      break;
    case F64:
      StrAppend(&munged_callee, ".f64");
      break;
    default:
      return Unimplemented("Bad type for llvm intrinsic math call: %s",
                           PrimitiveType_Name(output_type));
  }
  return EmitMathCall(munged_callee, operands, input_types, output_type);
}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitMathCall(
    const string& callee_name, absl::Span<llvm::Value* const> operands,
    absl::Span<const PrimitiveType> input_types, PrimitiveType output_type) {
  // Binary math functions transform are of type [T] -> T.
  for (PrimitiveType input_type : input_types) {
    if (output_type != input_type) {
      return Unimplemented("Input type != output type: %s != %s",
                           PrimitiveType_Name(input_type),
                           PrimitiveType_Name(output_type));
    }
  }

  return EmitDeviceFunctionCall(
      callee_name, operands, input_types, output_type,
      {llvm::Attribute::ReadNone, llvm::Attribute::NoUnwind}, b_);
}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitFloatBinaryOp(
    const HloInstruction* op, llvm::Value* lhs_value, llvm::Value* rhs_value) {
  PrimitiveType lhs_input_type = op->operand(0)->shape().element_type();
  PrimitiveType rhs_input_type = op->operand(1)->shape().element_type();
  PrimitiveType output_type = op->shape().element_type();
  HloOpcode opcode = op->opcode();

  if (hlo_module_config_.debug_options().xla_gpu_enable_fast_min_max() &&
      (opcode == HloOpcode::kMaximum || opcode == HloOpcode::kMinimum)) {
    return llvm_ir::EmitCallToIntrinsic(
        opcode == HloOpcode::kMaximum ? llvm::Intrinsic::maxnum
                                      : llvm::Intrinsic::minnum,
        {lhs_value, rhs_value}, {lhs_value->getType()}, b_);
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

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitPowerOp(
    const HloInstruction* op, llvm::Value* lhs_value, llvm::Value* rhs_value) {
  CHECK_EQ(op->opcode(), HloOpcode::kPower);
  PrimitiveType lhs_input_type = op->operand(0)->shape().element_type();
  PrimitiveType rhs_input_type = op->operand(1)->shape().element_type();
  PrimitiveType output_type = op->shape().element_type();
  return EmitDeviceMathCall(TargetDeviceFunctionID::kPow,
                            {lhs_value, rhs_value},
                            {lhs_input_type, rhs_input_type}, output_type);
}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitLog(PrimitiveType prim_type,
                                                      llvm::Value* value) {
  return EmitDeviceMathCall(TargetDeviceFunctionID::kLog, {value}, {prim_type},
                            prim_type);
}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitLog1p(PrimitiveType prim_type,
                                                        llvm::Value* value) {
  return EmitDeviceMathCall(TargetDeviceFunctionID::kLog1p, {value},
                            {prim_type}, prim_type);
}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitSin(PrimitiveType prim_type,
                                                      llvm::Value* value) {
  return EmitDeviceMathCall(TargetDeviceFunctionID::kSin, {value}, {prim_type},
                            prim_type);
}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitCos(PrimitiveType prim_type,
                                                      llvm::Value* value) {
  return EmitDeviceMathCall(TargetDeviceFunctionID::kCos, {value}, {prim_type},
                            prim_type);
}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitExp(PrimitiveType prim_type,
                                                      llvm::Value* value) {
  return EmitDeviceMathCall(TargetDeviceFunctionID::kExp, {value}, {prim_type},
                            prim_type);
}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitExpm1(PrimitiveType prim_type,
                                                        llvm::Value* value) {
  return EmitDeviceMathCall(TargetDeviceFunctionID::kExpm1, {value},
                            {prim_type}, prim_type);
}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitPow(PrimitiveType prim_type,
                                                      llvm::Value* lhs,
                                                      llvm::Value* rhs) {
  return EmitDeviceMathCall(TargetDeviceFunctionID::kPow, {lhs, rhs},
                            {prim_type, prim_type}, prim_type);
}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitSqrt(PrimitiveType prim_type,
                                                       llvm::Value* value) {
  return EmitDeviceMathCall(TargetDeviceFunctionID::kSqrt, {value}, {prim_type},
                            prim_type);
}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitRsqrt(PrimitiveType prim_type,
                                                        llvm::Value* value) {
  return EmitDeviceMathCall(TargetDeviceFunctionID::kRsqrt, {value},
                            {prim_type}, prim_type);
}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitAtan2(PrimitiveType prim_type,
                                                        llvm::Value* lhs,
                                                        llvm::Value* rhs) {
  return EmitDeviceMathCall(TargetDeviceFunctionID::kAtan2, {lhs, rhs},
                            {prim_type, prim_type}, prim_type);
}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitTanh(PrimitiveType prim_type,
                                                       llvm::Value* value) {
  // Emit a fast approximation of tanh instead of calling __nv_tanh.
  // __nv_tanh is particularly bad because it contains branches, thus
  // preventing LLVM's load-store vectorizer from working its magic across a
  // function which contains tanh calls.
  //
  // This routine isn't numerically precise, but it's good enough for ML.

  // Upcast F16 to F32 if necessary.
  llvm::Type* type = prim_type == F16 ? b_->getFloatTy() : value->getType();
  llvm::Value* input = FPCast(value, type);

  // If |value| >= kMaxValue, tanh() is set to -1.0 or 1.0.
  constexpr double kMaxValue = 20.0;
  auto max_value = llvm::ConstantFP::get(type, kMaxValue);
  llvm::Value* abs_value =
      llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::fabs, {input}, {type}, b_);

  llvm::Value* fast_tanh = llvm_ir::EmitFastTanh(b_, input);
  auto one = llvm::ConstantFP::get(type, 1.0);
  auto one_with_sign = llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::copysign,
                                                    {one, input}, {type}, b_);
  return FPCast(Select(FCmpULT(abs_value, max_value), fast_tanh, one_with_sign),
                value->getType());
}

StatusOr<llvm::Value*> GpuElementalIrEmitter::EmitComplexAbs(
    PrimitiveType prim_type, llvm::Value* value) {
  return EmitDeviceMathCall(TargetDeviceFunctionID::kHypot,
                            {EmitExtractReal(value), EmitExtractImag(value)},
                            {prim_type, prim_type}, prim_type);
}

llvm::Value* GpuElementalIrEmitter::EmitThreadId() {
  llvm::Value* block_id = IntCast(
      EmitCallToTargetIntrinsic(TargetIntrinsicID::kBlockIdx, {}, {}, b_),
      b_->getIntNTy(128), /*isSigned=*/true, "block.id");
  llvm::Value* thread_id_in_block = IntCast(
      EmitCallToTargetIntrinsic(TargetIntrinsicID::kThreadIdx, {}, {}, b_),
      b_->getIntNTy(128), /*isSigned=*/true, "thread.id");
  llvm::Value* threads_per_block = IntCast(
      EmitCallToTargetIntrinsic(TargetIntrinsicID::kBlockDimx, {}, {}, b_),
      b_->getIntNTy(128), /*isSigned=*/true, "threads_per_block");
  return NSWAdd(NSWMul(block_id, threads_per_block), thread_id_in_block);
}

llvm_ir::ElementGenerator GpuElementalIrEmitter::MakeElementGenerator(
    const HloInstruction* hlo,
    const HloToElementGeneratorMap& operand_to_generator) {
  switch (hlo->opcode()) {
    case HloOpcode::kMap:
      return [=, &operand_to_generator](
                 const IrArray::Index& index) -> StatusOr<llvm::Value*> {
        TF_RET_CHECK(!hlo->operands().empty())
            << "Zero operand map not implemented in GPU backend.";
        TF_RET_CHECK(hlo->to_apply()->num_parameters() > 0);
        std::vector<llvm::Value*> operand_elements;
        for (HloInstruction* operand : hlo->operands()) {
          TF_ASSIGN_OR_RETURN(llvm::Value * value,
                              operand_to_generator.at(operand)(index));
          operand_elements.push_back(value);
        }
        return compute_nested_(*hlo->to_apply(), operand_elements);
      };
    case HloOpcode::kReduceWindow:
      // Pseudocode:
      // for each index I in output
      //   value = init_value
      //   for each index W in window
      //     for each dimension i from 0 to rank - 1
      //       (input index I)[i] = O[i] * stride[i] + W[i] - pad_low[i]
      //     if I in bounds of input
      //       value = function(value, input[I])
      //     output[O] = value
      return [=, &operand_to_generator](
                 const IrArray::Index& index) -> StatusOr<llvm::Value*> {
        const HloInstruction* operand = hlo->operand(0);
        const Window& window = hlo->window();

        PrimitiveType operand_element_type = operand->shape().element_type();
        llvm::Value* accum_ptr = llvm_ir::EmitAllocaAtFunctionEntry(
            llvm_ir::PrimitiveTypeToIrType(operand_element_type, module_),
            "reduce_window_accum_ptr", b_);
        {
          TF_ASSIGN_OR_RETURN(llvm::Value * init_value,
                              operand_to_generator.at(hlo->operand(1))(
                                  IrArray::Index(index.GetType())));
          Store(init_value, accum_ptr);
        }

        llvm::Type* index_type = index.GetType();
        auto index_typed_const = [&](uint64 c) -> llvm::Constant* {
          return index.GetConstantWithIndexType(c);
        };

        llvm_ir::ForLoopNest loops(IrName(hlo), b_, index_type);
        std::vector<int64> window_size;
        for (const auto& dim : window.dimensions()) {
          window_size.push_back(dim.size());
        }
        const IrArray::Index window_index = loops.AddLoopsForShape(
            ShapeUtil::MakeShape(operand_element_type, window_size), "window");
        CHECK_EQ(window_index.size(), index.size());

        SetToFirstInsertPoint(loops.GetInnerLoopBodyBasicBlock(), b_);

        std::vector<llvm::Value*> input_multi_index(index.size());
        llvm::Value* in_bounds = b_->getInt1(true);
        for (size_t i = 0; i < index.size(); ++i) {
          llvm::Value* stridden_index = NSWMul(
              index[i], index_typed_const(window.dimensions(i).stride()));
          input_multi_index[i] = NSWSub(
              NSWAdd(stridden_index,
                     NSWMul(window_index[i],
                            index_typed_const(
                                window.dimensions(i).window_dilation()))),
              index_typed_const(window.dimensions(i).padding_low()));

          // We need to verify that we are not in the dilated base area.
          llvm::Value* dilation_condition = ICmpEQ(
              SRem(input_multi_index[i],
                   index_typed_const(window.dimensions(i).base_dilation())),
              index_typed_const(0));
          in_bounds = And(in_bounds, dilation_condition);

          // Apply base dilation to the index.
          input_multi_index[i] =
              SDiv(input_multi_index[i],
                   index_typed_const(window.dimensions(i).base_dilation()));

          // We must check whether 0 <= input_multi_index[i] < bound, as
          // otherwise we are in the pad and so can skip the computation. This
          // comparison is equivalent to the unsigned comparison
          // input_multi_index[i] < bound, as a negative value wraps to a large
          // positive value.
          in_bounds =
              And(in_bounds,
                  ICmpULT(input_multi_index[i],
                          index_typed_const(operand->shape().dimensions(i))));
        }

        llvm_ir::LlvmIfData if_data =
            llvm_ir::EmitIfThenElse(in_bounds, "in_bounds", b_);
        SetToFirstInsertPoint(if_data.true_block, b_);

        // We are not in pad, so do the computation.
        IrArray::Index input_index(input_multi_index, operand->shape(),
                                   index_type);
        TF_ASSIGN_OR_RETURN(llvm::Value * input_value,
                            operand_to_generator.at(operand)(input_index));
        TF_ASSIGN_OR_RETURN(
            llvm::Value * accum_value,
            compute_nested_(*hlo->to_apply(), {Load(accum_ptr), input_value}));
        Store(accum_value, accum_ptr);

        SetToFirstInsertPoint(loops.GetOuterLoopExitBasicBlock(), b_);
        return Load(accum_ptr);
      };
    case HloOpcode::kReduce:
      // TODO(b/118332391): This should be supported.
      CHECK_EQ(hlo->operand_count(), 2) << "Did not expect variadic reduce";
      return [=, &operand_to_generator](
                 const IrArray::Index& output_index) -> StatusOr<llvm::Value*> {
        const HloInstruction* operand = hlo->operand(0);
        llvm::Value* accum_ptr =
            b()->CreateAlloca(llvm_ir::PrimitiveTypeToIrType(
                hlo->shape().element_type(), module_));
        llvm::Type* index_type = output_index.GetType();
        TF_ASSIGN_OR_RETURN(llvm::Value * init_value,
                            operand_to_generator.at(hlo->operand(1))(
                                IrArray::Index(index_type)));
        b()->CreateStore(init_value, accum_ptr);

        llvm_ir::ForLoopNest loops(IrName(hlo), b_, index_type);
        std::vector<llvm::Value*> input_multi_index =
            loops.AddLoopsForShapeOnDimensions(
                operand->shape(), hlo->dimensions(), "reduction_dim");
        if (!ShapeUtil::IsScalar(hlo->shape())) {
          // Here only input_multi_index[hlo->dimensions()] are non-null, so we
          // must set the rest.
          size_t j = 0;
          for (auto& i : input_multi_index) {
            if (i == nullptr) {
              i = output_index[j++];
            }
          }
          CHECK_EQ(output_index.size(), j);
        }
        llvm_ir::IrArray::Index input_index(
            input_multi_index, hlo->operand(0)->shape(), index_type);

        SetToFirstInsertPoint(loops.GetInnerLoopBodyBasicBlock(), b());
        TF_ASSIGN_OR_RETURN(
            llvm::Value * input_value,
            operand_to_generator.at(hlo->operand(0))(input_index));
        TF_ASSIGN_OR_RETURN(
            llvm::Value * accum_value,
            compute_nested_(*hlo->to_apply(),
                            {b()->CreateLoad(accum_ptr), input_value}));
        b()->CreateStore(accum_value, accum_ptr);
        SetToFirstInsertPoint(loops.GetOuterLoopExitBasicBlock(), b());
        return b()->CreateLoad(accum_ptr);
      };
    default:
      return ElementalIrEmitter::MakeElementGenerator(hlo,
                                                      operand_to_generator);
  }
}

}  // namespace gpu
}  // namespace xla
