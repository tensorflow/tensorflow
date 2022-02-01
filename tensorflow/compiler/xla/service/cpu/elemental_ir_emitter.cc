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
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

using xla::llvm_ir::IrArray;

namespace xla {
namespace cpu {

StatusOr<llvm::Value*> CpuElementalIrEmitter::EmitAtan2(
    PrimitiveType prim_type, llvm::Value* lhs, llvm::Value* rhs,
    absl::string_view /*name*/) {
  std::string function_name;
  bool cast_result_to_fp16 = false;
  switch (prim_type) {
    case F16:
      cast_result_to_fp16 = true;
      lhs = FPCast(lhs, b()->getFloatTy());
      rhs = FPCast(rhs, b()->getFloatTy());
      ABSL_FALLTHROUGH_INTENDED;
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
  llvm::Function* function = llvm::dyn_cast<llvm::Function>(
      module()
          ->getOrInsertFunction(function_name, lhs->getType(), lhs->getType(),
                                rhs->getType())
          .getCallee());
  function->setCallingConv(llvm::CallingConv::C);
  function->setDoesNotThrow();
  function->setDoesNotAccessMemory();
  // Create an instruction to call the function.
  llvm::Value* result = Call(function, {lhs, rhs});
  if (cast_result_to_fp16) {
    result = FPCast(result, b()->getHalfTy());
  }
  return result;
}

StatusOr<llvm::Value*> CpuElementalIrEmitter::EmitTanh(PrimitiveType prim_type,
                                                       llvm::Value* value) {
  bool cast_result_to_fp16 = false;
  std::string function_name;
  switch (prim_type) {
    case F16:
      cast_result_to_fp16 = true;
      value = FPCast(value, b()->getFloatTy());
      ABSL_FALLTHROUGH_INTENDED;
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
  llvm::Function* function = llvm::dyn_cast<llvm::Function>(
      module()
          ->getOrInsertFunction(function_name, value->getType(),
                                value->getType())
          .getCallee());
  function->setCallingConv(llvm::CallingConv::C);
  function->setDoesNotThrow();
  function->setDoesNotAccessMemory();
  // Create an instruction to call the function.
  llvm::Value* result = Call(function, value);
  if (cast_result_to_fp16) {
    result = FPCast(result, b()->getHalfTy());
  }
  return result;
}

}  // namespace cpu
}  // namespace xla
