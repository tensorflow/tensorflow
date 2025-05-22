/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/codegen/math/simple_jit_runner.h"

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/TypeSize.h"
#include "llvm/Support/raw_ostream.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/xla_data.pb.h"

namespace xla::codegen::math {
JitRunner::JitRunner(std::unique_ptr<llvm::Module> module,
                     std::unique_ptr<llvm::LLVMContext> context) {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();
  auto jit_or_err = llvm::orc::LLJITBuilder().create();
  if (!jit_or_err) {
    llvm::report_fatal_error(
        llvm::Twine(llvm::toString(jit_or_err.takeError())));
  }
  jit_ = std::move(jit_or_err.get());
  llvm::ExitOnError exit_on_err;
  llvm::orc::ThreadSafeModule tsm(std::move(module), std::move(context));
  exit_on_err(jit_->addIRModule(std::move(tsm)));
}

JitRunner::~JitRunner() { llvm::llvm_shutdown(); }

llvm::Expected<void*> JitRunner::CreateVectorWrapperFunction(
    const std::string& original_function_name, size_t vector_size,
    PrimitiveType ret_type, std::vector<PrimitiveType> arg_types) {
  // Create a new LLVMContext and Module for the wrapper function.
  // LLJIT takes ownership of these when added.
  auto context_owner = std::make_unique<llvm::LLVMContext>();
  llvm::LLVMContext* context = context_owner.get();
  auto wrapper_module_owner = std::make_unique<llvm::Module>(
      original_function_name + "_wrapper_module", *context);
  llvm::Module* wrapper_module = wrapper_module_owner.get();
  llvm::IRBuilder<> builder(*context);

  auto vec_type = [=](PrimitiveType type) {
    return llvm::VectorType::get(llvm_ir::PrimitiveTypeToIrType(type, *context),
                                 llvm::ElementCount::getFixed(vector_size));
  };

  llvm::Type* ret_vec_type = vec_type(ret_type);
  std::vector<llvm::Type*> arg_vec_types;
  arg_vec_types.reserve(arg_types.size());
  for (PrimitiveType arg_type : arg_types) {
    arg_vec_types.push_back(vec_type(arg_type));
  }

  llvm::FunctionType* original_func_type =
      llvm::FunctionType::get(ret_vec_type, arg_vec_types, false);

  // Declare the original function in the new wrapper module.
  // The JIT's symbol resolution will find the actual definition.
  llvm::Function* original_func = llvm::Function::Create(
      original_func_type, llvm::Function::ExternalLinkage,
      original_function_name, *wrapper_module);

  std::vector<llvm::Type*> ptr_types;
  ptr_types.push_back(ret_vec_type->getScalarType()->getPointerTo());
  for (llvm::Type* arg_vec_type : arg_vec_types) {
    ptr_types.push_back(arg_vec_type->getPointerTo());
  }

  llvm::FunctionType* wrapper_llvm_func_type =
      llvm::FunctionType::get(builder.getVoidTy(), ptr_types, false);

  std::string wrapper_function_name = original_function_name + "_wrapper";
  llvm::Function* wrapper_llvm_func = llvm::Function::Create(
      wrapper_llvm_func_type, llvm::Function::ExternalLinkage,
      wrapper_function_name, *wrapper_module);

  llvm::BasicBlock* entry_block =
      llvm::BasicBlock::Create(*context, "entry", wrapper_llvm_func);
  builder.SetInsertPoint(entry_block);

  llvm::Value* ret_ptr_arg = wrapper_llvm_func->getArg(0);
  llvm::Value* ret_vec_ptr =
      builder.CreateBitCast(ret_ptr_arg, ret_vec_type->getPointerTo());

  std::vector<llvm::Value*> arg_vecs;
  for (int i = 1; i < wrapper_llvm_func->arg_size(); ++i) {
    llvm::Value* ptr_arg = wrapper_llvm_func->getArg(i);
    llvm::Type* type = arg_vec_types[i - 1];
    llvm::Value* vec_ptr = builder.CreateBitCast(ptr_arg, type->getPointerTo());
    llvm::LoadInst* arg_vec = builder.CreateLoad(arg_vec_types[i - 1], vec_ptr);
    arg_vec->setAlignment(llvm::Align(32));
    arg_vecs.push_back(arg_vec);
  }

  // Call the original vectorized function
  llvm::CallInst* result_vec = builder.CreateCall(original_func, arg_vecs);

  // Store the result back to the output pointer
  llvm::StoreInst* store_result = builder.CreateStore(result_vec, ret_vec_ptr);
  store_result->setAlignment(llvm::Align(32));

  builder.CreateRetVoid();  // Wrapper returns void

  // Verify the wrapper function
  std::string error_str;
  llvm::raw_string_ostream os(error_str);
  if (llvm::verifyFunction(*wrapper_llvm_func, &os)) {
    return llvm::make_error<llvm::StringError>(
        llvm::errc::invalid_argument,
        "Error in wrapper function IR: " + os.str());
  }

  // Add the wrapper module to the JIT.
  // LLJIT now owns wrapper_module_owner and context_owner.
  llvm::ExitOnError exit_on_err;
  exit_on_err(jit_->addIRModule(llvm::orc::ThreadSafeModule(
      std::move(wrapper_module_owner), std::move(context_owner))));

  // Look up the JITed function pointer and return it
  auto function_sym = jit_->lookup(wrapper_function_name);
  if (!function_sym) {
    return function_sym.takeError();
  }
  return reinterpret_cast<void*>(function_sym->getValue());
}

}  // namespace xla::codegen::math
