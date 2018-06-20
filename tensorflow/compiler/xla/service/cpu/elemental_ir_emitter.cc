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

#include "tensorflow/compiler/xla/service/cpu/elemental_ir_emitter.h"

#include <string>

#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace cpu {

StatusOr<llvm::Value*> CpuElementalIrEmitter::EmitFloatUnaryOp(
    const HloInstruction* op, llvm::Value* operand_value) const {
  switch (op->opcode()) {
    case HloOpcode::kTanh: {
      PrimitiveType element_type = op->shape().element_type();
      bool cast_result_to_fp16 = false;
      string function_name;
      switch (element_type) {
        case F16:
          cast_result_to_fp16 = true;
          operand_value = ir_builder_->CreateFPCast(operand_value,
                                                    ir_builder_->getFloatTy());
          TF_FALLTHROUGH_INTENDED;
        case F32:
          function_name = "tanhf";
          break;
        case F64:
          function_name = "tanh";
          break;
        default:
          return Unimplemented("tanh");
      }
      // Create a function declaration.
      llvm::Function* function =
          llvm::cast<llvm::Function>(module_->getOrInsertFunction(
              llvm_ir::AsStringRef(function_name), operand_value->getType(),
              operand_value->getType()));
      function->setCallingConv(llvm::CallingConv::C);
      function->setDoesNotThrow();
      function->setDoesNotAccessMemory();
      // Create an instruction to call the function.
      llvm::Value* result = ir_builder_->CreateCall(function, operand_value);
      if (cast_result_to_fp16) {
        result = ir_builder_->CreateFPCast(result, ir_builder_->getHalfTy());
      }
      return result;
    }
    default:
      return ElementalIrEmitter::EmitFloatUnaryOp(op, operand_value);
  }
}

StatusOr<llvm::Value*> CpuElementalIrEmitter::EmitAtan2(
    PrimitiveType prim_type, llvm::Value* lhs, llvm::Value* rhs) const {
  string function_name;
  bool cast_result_to_fp16 = false;
  switch (prim_type) {
    case F16:
      cast_result_to_fp16 = true;
      lhs = ir_builder_->CreateFPCast(lhs, ir_builder_->getFloatTy());
      rhs = ir_builder_->CreateFPCast(rhs, ir_builder_->getFloatTy());
      TF_FALLTHROUGH_INTENDED;
    case F32:
      function_name = "atan2f";
      break;
    case F64:
      function_name = "atan2";
      break;
    default:
      return Unimplemented("atan2");
  }
  // Create a function declaration.
  llvm::Function* function =
      llvm::cast<llvm::Function>(module_->getOrInsertFunction(
          llvm_ir::AsStringRef(function_name), lhs->getType(), lhs->getType(),
          rhs->getType()));
  function->setCallingConv(llvm::CallingConv::C);
  function->setDoesNotThrow();
  function->setDoesNotAccessMemory();
  // Create an instruction to call the function.
  llvm::Value* result = ir_builder_->CreateCall(function, {lhs, rhs});
  if (cast_result_to_fp16) {
    result = ir_builder_->CreateFPCast(result, ir_builder_->getHalfTy());
  }
  return result;
}

llvm_ir::ElementGenerator CpuElementalIrEmitter::MakeElementGenerator(
    const HloInstruction* hlo,
    const HloToElementGeneratorMap& operand_to_generator) const {
  if (hlo->opcode() == HloOpcode::kMap) {
    return [this, hlo, &operand_to_generator](
               const llvm_ir::IrArray::Index& index) -> StatusOr<llvm::Value*> {
      std::vector<llvm::Value*> operands;
      for (int i = 0; i < hlo->operand_count(); i++) {
        TF_ASSIGN_OR_RETURN(llvm::Value * operand_value,
                            operand_to_generator.at(hlo->operand(i))(
                                ElementwiseSourceIndex(index, *hlo, i)));
        operands.push_back(operand_value);
      }
      return ir_emitter_->EmitScalarCall(hlo->shape().element_type(),
                                         hlo->to_apply(), operands,
                                         llvm_ir::IrName(hlo));
    };
  }
  return ElementalIrEmitter::MakeElementGenerator(hlo, operand_to_generator);
}
}  // namespace cpu
}  // namespace xla
