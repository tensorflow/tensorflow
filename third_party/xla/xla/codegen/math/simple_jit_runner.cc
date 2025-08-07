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
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/call_once.h"
#include "absl/log/log.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ExecutionEngine/JITEventListener.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
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
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/TypeSize.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/xla_data.pb.h"

namespace xla::codegen::math {

namespace {
void initializeNativeTargets() {
  static absl::once_flag once;
  absl::call_once(once, []() {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
  });
}
}  // namespace

JitRunner::JitRunner(std::unique_ptr<llvm::Module> module,
                     std::unique_ptr<llvm::LLVMContext> context) {
  initializeNativeTargets();
  tsc_ = std::make_unique<llvm::orc::ThreadSafeContext>(std::move(context));
  perf_listener_ = llvm::JITEventListener::createPerfJITEventListener();
  auto jit_builder = llvm::orc::LLJITBuilder();
  if (perf_listener_ != nullptr) {
    jit_builder = std::move(jit_builder.setObjectLinkingLayerCreator(
        [&](llvm::orc::ExecutionSession& ES) {
          auto obj_layer =
              std::make_unique<llvm::orc::RTDyldObjectLinkingLayer>(
                  ES, [](const llvm::MemoryBuffer& _) {
                    return std::make_unique<llvm::SectionMemoryManager>();
                  });
          obj_layer->registerJITEventListener(*perf_listener_);
          return obj_layer;
        }));
  }
  auto jit_or_err = jit_builder.create();
  if (!jit_or_err) {
    llvm::report_fatal_error(
        llvm::Twine(llvm::toString(jit_or_err.takeError())));
  }
  jit_ = std::move(jit_or_err.get());
  llvm::orc::ThreadSafeModule tsm(std::move(module), *tsc_);
  llvm::ExitOnError exit_on_err;
  exit_on_err(jit_->addIRModule(std::move(tsm)));
}

// Returns a JITed function that loops over a vectorized function.
// The original function is expected to have a signature like:
//   <VectorSize x RetType> func(<VectorSize x Arg1Type>, <VectorSize x
//   Arg2Type>, ...)
// This function creates a wrapper that bridges the gap between C++ array
// types and LLVM vector types. The returned std::function has a signature like:
// void fn(std:::array<ArgTypes, VectorSize>& return_array,
//   size_t iteration_count, size_t source_array_length, const
//   std::array<ArgTypes, VectorSize>&...)
llvm::Expected<void*> JitRunner::CreateVectorWrapperWithLoop(
    const std::string& original_function_name, size_t vector_size,
    PrimitiveType ret_type, std::vector<PrimitiveType> arg_types) {
  return tsc_->withContextDo([&](llvm::LLVMContext* ctx)
                                 -> llvm::Expected<void*> {
    auto wrapper_module_owner = std::make_unique<llvm::Module>(
        original_function_name + "_wrapper_module", *ctx);
    llvm::Module* wrapper_module = wrapper_module_owner.get();
    llvm::IRBuilder<> builder(*ctx);

    auto vec_type = [&](PrimitiveType type) {
      return llvm::VectorType::get(llvm_ir::PrimitiveTypeToIrType(type, *ctx),
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
    llvm::Function* original_func = llvm::Function::Create(
        original_func_type, llvm::Function::ExternalLinkage,
        original_function_name, *wrapper_module);

    std::vector<llvm::Type*> wrapper_arg_types;
    // 1. Pointer to write the return data
    wrapper_arg_types.push_back(ret_vec_type->getScalarType()->getPointerTo());
    // 2. Iteration count, passed by value
    wrapper_arg_types.push_back(builder.getInt32Ty());
    // 3. Data length (number of elements in source arrays), by value
    wrapper_arg_types.push_back(builder.getInt32Ty());
    // 4. Pointers for each input data array
    for (llvm::Type* arg_vec_type : arg_vec_types) {
      wrapper_arg_types.push_back(
          arg_vec_type->getScalarType()->getPointerTo());
    }

    llvm::FunctionType* wrapper_llvm_func_type =
        llvm::FunctionType::get(builder.getVoidTy(), wrapper_arg_types, false);
    std::string wrapper_function_name = original_function_name + "_wrapper";
    llvm::Function* wrapper_llvm_func = llvm::Function::Create(
        wrapper_llvm_func_type, llvm::Function::ExternalLinkage,
        wrapper_function_name, *wrapper_module);

    llvm::BasicBlock* entry_block =
        llvm::BasicBlock::Create(*ctx, "entry", wrapper_llvm_func);
    llvm::BasicBlock* loop_header =
        llvm::BasicBlock::Create(*ctx, "loop.header", wrapper_llvm_func);
    llvm::BasicBlock* loop_body =
        llvm::BasicBlock::Create(*ctx, "loop.body", wrapper_llvm_func);
    llvm::BasicBlock* loop_exit =
        llvm::BasicBlock::Create(*ctx, "loop.exit", wrapper_llvm_func);
    builder.SetInsertPoint(entry_block);
    // Use a pointer here to make the loop difficult to optimize away.
    llvm::AllocaInst* counter =
        builder.CreateAlloca(builder.getInt32Ty(), nullptr, "counter");
    builder.CreateStore(builder.getInt32(0), counter);
    builder.CreateBr(loop_header);

    builder.SetInsertPoint(loop_header);
    llvm::Value* loop_iterations = wrapper_llvm_func->getArg(1);
    llvm::Value* current_count =
        builder.CreateLoad(builder.getInt32Ty(), counter, "current_count");
    llvm::Value* condition =
        builder.CreateICmpSLT(current_count, loop_iterations, "loop_cond");
    builder.CreateCondBr(condition, loop_body, loop_exit);

    builder.SetInsertPoint(loop_body);
    llvm::Value* data_length = wrapper_llvm_func->getArg(2);

    // Calculate the number of vectors in the data array
    llvm::Value* num_vectors = builder.CreateUDiv(
        data_length, builder.getInt32(vector_size), "num_vectors");
    // Calculate the index for this iteration: current_count % num_vectors
    llvm::Value* index =
        builder.CreateURem(current_count, num_vectors, "index");

    // Load input vectors using the calculated index
    std::vector<llvm::Value*> arg_vecs;
    for (int i = 0; i < arg_types.size(); ++i) {
      llvm::Value* base_ptr = wrapper_llvm_func->getArg(i + 3);
      llvm::Value* vec_ptr =
          builder.CreateGEP(arg_vec_types[i], base_ptr, index, "vec_ptr");
      llvm::LoadInst* arg_vec =
          builder.CreateLoad(arg_vec_types[i], vec_ptr, "arg_vec");
      arg_vec->setAlignment(llvm::Align(32));
      arg_vecs.push_back(arg_vec);
    }
    llvm::CallInst* result_vec = builder.CreateCall(original_func, arg_vecs);
    llvm::Value* ret_base_ptr = wrapper_llvm_func->getArg(0);
    llvm::Value* ret_vec_ptr =
        builder.CreateGEP(ret_vec_type, ret_base_ptr, index, "ret_vec_ptr");
    llvm::StoreInst* store_result =
        builder.CreateStore(result_vec, ret_vec_ptr);
    store_result->setAlignment(llvm::Align(32));

    llvm::Value* next_count =
        builder.CreateAdd(current_count, builder.getInt32(1), "next_count");
    builder.CreateStore(next_count, counter);
    builder.CreateBr(loop_header);
    builder.SetInsertPoint(loop_exit);
    builder.CreateRetVoid();

    std::string error_str;
    llvm::raw_string_ostream os(error_str);
    if (llvm::verifyFunction(*wrapper_llvm_func, &os)) {
      return llvm::make_error<llvm::StringError>(
          llvm::errc::invalid_argument,
          "Error in wrapper function IR: " + os.str());
    }
    llvm::ExitOnError exit_on_err;
    exit_on_err(jit_->addIRModule(
        llvm::orc::ThreadSafeModule(std::move(wrapper_module_owner), *tsc_)));
    auto function_sym = jit_->lookup(wrapper_function_name);
    if (!function_sym) {
      return function_sym.takeError();
    }
    return reinterpret_cast<void*>(function_sym->getValue());
  });
}

std::unique_ptr<llvm::TargetMachine> CreateHostTargetMachine() {
  initializeNativeTargets();
  const std::string triple = llvm::sys::getDefaultTargetTriple();
  llvm::StringRef cpu = llvm::sys::getHostCPUName();
  llvm::StringMap<bool> features = llvm::sys::getHostCPUFeatures();
  std::string errors = "";
  const llvm::Target* target =
      llvm::TargetRegistry::lookupTarget(llvm::StringRef(triple), errors);
  LOG_IF(FATAL, !target) << "Failed to lookup target: " << errors;
  std::string feature_str;
  for (const auto& [feature, value] : features) {
    if (value) {
      feature_str += "+" + feature.str() + ",";
    }
  }
  llvm::TargetOptions target_options;
  std::unique_ptr<llvm::TargetMachine> target_machine(
      target->createTargetMachine(llvm::Triple(triple), cpu, feature_str,
                                  target_options, std::nullopt, std::nullopt));
  LOG_IF(FATAL, !target_machine) << "Failed to create target machine";
  return target_machine;
}
}  // namespace xla::codegen::math
