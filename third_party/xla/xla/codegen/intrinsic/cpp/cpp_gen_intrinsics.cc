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

#include "xla/codegen/intrinsic/cpp/cpp_gen_intrinsics.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "xla/codegen/intrinsic/cpp/eigen_unary_16_ll.h"
#include "xla/codegen/intrinsic/cpp/eigen_unary_32_ll.h"
#include "xla/codegen/intrinsic/cpp/eigen_unary_64_ll.h"
#include "xla/codegen/intrinsic/intrinsic.h"
#include "xla/service/llvm_ir/llvm_util.h"

namespace xla::codegen {

const std::string& GetCppGenIrString(
    const intrinsics::IntrinsicOptions& options) {
  if (options.Contains("+avx512f") && (options.prefer_vector_width >= 512 ||
                                       options.prefer_vector_width == 0)) {
    return ::llvm_ir::kEigenUnary64LlIr;
  }
  if (options.Contains("+avx")) {
    return ::llvm_ir::kEigenUnary32LlIr;
  }
  return ::llvm_ir::kEigenUnary16LlIr;
}

bool AreEigenIntrinsicsAvailable() {
  return !GetCppGenIrString(intrinsics::IntrinsicOptions()).empty();
}

namespace {

void PrepareForInlining(llvm::Function* func) {
  func->setLinkage(llvm::Function::InternalLinkage);
  if (!func->hasFnAttribute(llvm::Attribute::NoInline)) {
    func->addFnAttr(llvm::Attribute::AlwaysInline);
  }
}

// Wraps `body`, whose host C ABI signature returns through sret and/or takes
// arguments through pointers, in a function of the requested value signature.
llvm::Function* CreateDirectAdapter(llvm::Module* module, llvm::Function* body,
                                    llvm::FunctionType* type) {
  const llvm::DataLayout& data_layout = module->getDataLayout();
  std::string name = body->getName().str();
  body->setName(name + ".body");
  PrepareForInlining(body);

  llvm::Function* adapter = llvm::Function::Create(
      type, llvm::Function::InternalLinkage, name, module);
  llvm::IRBuilder<> builder(
      llvm::BasicBlock::Create(module->getContext(), "entry", adapter));
  auto create_slot = [&](llvm::Type* slot_type, llvm::MaybeAlign param_align) {
    llvm::AllocaInst* slot = builder.CreateAlloca(slot_type);
    llvm::Align alignment = data_layout.getPrefTypeAlign(slot_type);
    if (param_align) {
      alignment = std::max(alignment, *param_align);
    }
    slot->setAlignment(alignment);
    return slot;
  };

  std::vector<llvm::Value*> args;
  unsigned body_arg = 0;
  llvm::AllocaInst* ret_slot = nullptr;
  if (body->getReturnType()->isVoidTy() && body->arg_size() > 0 &&
      body->hasParamAttribute(0, llvm::Attribute::StructRet)) {
    ret_slot =
        create_slot(body->getParamStructRetType(0), body->getParamAlign(0));
    args.push_back(ret_slot);
    body_arg = 1;
  }
  args.reserve(args.size() + adapter->arg_size());
  for (llvm::Argument& arg : adapter->args()) {
    CHECK_LT(body_arg, body->arg_size())
        << "CppGen function '" << name << "' has fewer parameters than "
        << llvm_ir::DumpToString(type);
    if (body->getArg(body_arg)->getType()->isPointerTy() &&
        !arg.getType()->isPointerTy()) {
      llvm::AllocaInst* slot =
          create_slot(arg.getType(), body->getParamAlign(body_arg));
      builder.CreateStore(&arg, slot);
      args.push_back(slot);
    } else {
      args.push_back(&arg);
    }
    ++body_arg;
  }
  CHECK_EQ(body_arg, body->arg_size())
      << "CppGen function '" << name << "' has more parameters than "
      << llvm_ir::DumpToString(type);

  llvm::CallInst* call = builder.CreateCall(body, args);
  for (unsigned i = 0; i < body->arg_size(); ++i) {
    call->addParamAttrs(
        i, llvm::AttrBuilder(module->getContext(),
                             body->getAttributes().getParamAttrs(i)));
  }
  llvm::Value* result = call;
  if (ret_slot != nullptr) {
    result = builder.CreateLoad(type->getReturnType(), ret_slot);
  }
  CHECK(result->getType() == type->getReturnType())
      << "CppGen function '" << name << "' returns "
      << llvm_ir::DumpToString(body->getReturnType()) << ", expected "
      << llvm_ir::DumpToString(type->getReturnType());
  builder.CreateRet(result);
  return adapter;
}

}  // namespace

llvm::Function* GetCppGenFunction(llvm::Module* module, absl::string_view name,
                                  llvm::FunctionType* type) {
  llvm::Function* func =
      module->getFunction(llvm::StringRef(name.data(), name.size()));
  CHECK(func != nullptr)
      << "CppGen function '" << name
      << "' was not found in the module. Ensure the "
         "function name is correct and the library "
         "containing it was linked by IntrinsicFunctionLib.\n"
      << llvm_ir::DumpToString(module);

  if (func->isDeclaration()) {
    return func;
  }
  if (func->getFunctionType() != type) {
    return CreateDirectAdapter(module, func, type);
  }
  PrepareForInlining(func);
  return func;
}

std::unique_ptr<llvm::Module> ParseEmbeddedBitcode(
    llvm::LLVMContext& context, const std::string& bitcode,
    absl::string_view source_name) {
  if (bitcode.empty()) {
    LOG_FIRST_N(INFO, 1)
        << "Empty bitcode string provided for " << source_name
        << ". Optimizations relying on this IR will be disabled.";
    return std::make_unique<llvm::Module>("empty", context);
  }

  llvm::SMDiagnostic diagnostic;
  std::unique_ptr<llvm::MemoryBuffer> buffer = llvm::MemoryBuffer::getMemBuffer(
      llvm::StringRef(bitcode.data(), bitcode.size()),
      llvm::StringRef(source_name.data(), source_name.size()),
      /*RequiresNullTerminator=*/false);
  std::unique_ptr<llvm::Module> module =
      llvm::parseIR(buffer->getMemBufferRef(), diagnostic, context);

  CHECK(module != nullptr) << "Failed to parse IR: "
                           << diagnostic.getMessage().str() << "\n"
                           << bitcode;
  // Mach-O clang emits asm("name") labels as "\01name"; match ELF naming.
  for (llvm::Function& func : *module) {
    llvm::StringRef name = func.getName();
    if (name.consume_front("\01")) {
      func.setName(name);
    }
  }
  return module;
}

// The default LLVM diagnostic handler uses llvm::errs(), which is not
// thread-safe.
static void DiagnosticHandler(const llvm::DiagnosticInfo* diag_info,
                              void* context) {
  std::string error_string;
  llvm::raw_string_ostream string_printer(error_string);
  llvm::DiagnosticPrinterRawOStream diagnostic_printer(string_printer);
  diag_info->print(diagnostic_printer);

  if (diag_info->getSeverity() == llvm::DS_Error) {
    LOG(ERROR) << error_string;
  } else {
    VLOG(1) << error_string;
  }
}

void CppGenIntrinsicLibrary::LinkIntoModule(llvm::Module& dst_module) const {
  llvm::LLVMContext& context = dst_module.getContext();

  std::unique_ptr<llvm::Module> lib_module =
      ParseEmbeddedBitcode(context, ir_text_, source_name_);

  std::vector<std::string> lib_functions;
  // The linker merges same-named symbols regardless of signature.
  std::vector<std::pair<llvm::Function*, std::string>> mismatched_decls;
  for (const auto& func : *lib_module) {
    if (func.isDeclaration()) {
      continue;
    }
    lib_functions.push_back(func.getName().str());
    llvm::Function* decl = dst_module.getFunction(func.getName());
    if (decl != nullptr && decl->isDeclaration() &&
        decl->getFunctionType() != func.getFunctionType()) {
      mismatched_decls.emplace_back(decl, func.getName().str());
      decl->setName(func.getName() + ".old_decl");
    }
  }

  const llvm::DataLayout& hostDataLayout = dst_module.getDataLayout();
  lib_module->setDataLayout(hostDataLayout);

  auto old_handler = context.getDiagnosticHandlerCallBack();
  void* old_handler_context = context.getDiagnosticContext();

  context.setDiagnosticHandlerCallBack(DiagnosticHandler, nullptr);

  // Using static Linker::linkModules based on previous success, but matching
  // logic
  if (llvm::Linker::linkModules(dst_module, std::move(lib_module))) {
    LOG(FATAL) << "LLVM Linker failed to link CppGen library.";
  }

  context.setDiagnosticHandlerCallBack(old_handler, old_handler_context);

  for (const auto& func : lib_functions) {
    llvm::Function* linked_func = dst_module.getFunction(func);
    if (linked_func && !linked_func->isDeclaration()) {
      linked_func->setLinkage(llvm::Function::InternalLinkage);
      if (!linked_func->hasFnAttribute(llvm::Attribute::NoInline)) {
        linked_func->addFnAttr(llvm::Attribute::AlwaysInline);
      }
      linked_func->removeFnAttr("probe-stack");
      linked_func->removeFnAttr("target-cpu");
      linked_func->removeFnAttr("target-features");
    }
  }

  for (const auto& [decl, name] : mismatched_decls) {
    llvm::Function* adapter =
        GetCppGenFunction(&dst_module, name, decl->getFunctionType());
    decl->replaceAllUsesWith(adapter);
    decl->eraseFromParent();
  }
}

}  // namespace xla::codegen
