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

#include "xla/service/cpu/elemental_math_emitter.h"

#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "xla/service/llvm_ir/math_ops.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

absl::StatusOr<llvm::Value*> EmitAtan2(llvm::Module* module,
                                       llvm::IRBuilderBase& b,
                                       PrimitiveType prim_type,
                                       llvm::Value* lhs, llvm::Value* rhs) {
  std::string function_name;
  bool cast_result_to_fp16 = false;
  switch (prim_type) {
    case F16:
      cast_result_to_fp16 = true;
      lhs = b.CreateFPCast(lhs, b.getFloatTy());
      rhs = b.CreateFPCast(rhs, b.getFloatTy());
      [[fallthrough]];
    case F32:
      function_name = "atan2f";
      break;
    case F64:
      function_name = "atan2";
      break;
    default:
      return absl::UnimplementedError("atan2");
  }
  // Create a function declaration.
  llvm::Function* function = llvm::dyn_cast<llvm::Function>(
      module
          ->getOrInsertFunction(function_name, lhs->getType(), lhs->getType(),
                                rhs->getType())
          .getCallee());
  function->setCallingConv(llvm::CallingConv::C);
  function->setDoesNotThrow();
  function->setDoesNotAccessMemory();
  // Create an instruction to call the function.
  llvm::Value* result = b.CreateCall(function, {lhs, rhs});
  if (cast_result_to_fp16) {
    result = b.CreateFPCast(result, b.getHalfTy());
  }
  return result;
}

absl::StatusOr<llvm::Value*> EmitTanh(llvm::Module* module,
                                      llvm::IRBuilderBase& b,
                                      PrimitiveType prim_type,
                                      llvm::Value* value) {
  bool cast_result_to_fp16 = false;
  std::string function_name;
  switch (prim_type) {
    case F16:
      cast_result_to_fp16 = true;
      value = b.CreateFPCast(value, b.getFloatTy());
      [[fallthrough]];
    case F32:
      function_name = "tanhf";
      break;
    case F64:
      function_name = "tanh";
      break;
    default:
      return absl::UnimplementedError("tanh");
  }
  // Create a function declaration.
  llvm::Function* function = llvm::dyn_cast<llvm::Function>(
      module
          ->getOrInsertFunction(function_name, value->getType(),
                                value->getType())
          .getCallee());
  function->setCallingConv(llvm::CallingConv::C);
  function->setDoesNotThrow();
  function->setDoesNotAccessMemory();
  // Create an instruction to call the function.
  llvm::Value* result = b.CreateCall(function, value);
  if (cast_result_to_fp16) {
    result = b.CreateFPCast(result, b.getHalfTy());
  }
  return result;
}

absl::StatusOr<llvm::Value*> EmitErf(llvm::Module* module,
                                     llvm::IRBuilderBase& b,
                                     PrimitiveType prim_type,
                                     llvm::Value* value) {
  if (prim_type == F64) {
    std::string function_name = "erf";
    // Create a function declaration.
    llvm::Function* function = llvm::dyn_cast<llvm::Function>(
        module
            ->getOrInsertFunction(function_name, value->getType(),
                                  value->getType())
            .getCallee());
    function->setCallingConv(llvm::CallingConv::C);
    function->setDoesNotThrow();
    function->setDoesNotAccessMemory();
    // Create an instruction to call the function.
    llvm::Value* result = b.CreateCall(function, value);
    return result;
  }
  // Upcast F16 to F32 if necessary.
  llvm::Type* type = prim_type == F16 ? b.getFloatTy() : value->getType();
  if (type == b.getFloatTy()) {
    llvm::Value* x = b.CreateFPCast(value, type);
    auto* result = llvm_ir::EmitErfF32(&b, x);
    return b.CreateFPCast(result, value->getType());
  }
  return absl::UnimplementedError("erf");
}

}  // namespace xla::cpu
