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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/SourceMgr.h"
#include "xla/codegen/intrinsic/cpp/eigen_unary_ll.h"
#include "xla/codegen/intrinsic/intrinsic.h"
#include "xla/service/llvm_ir/llvm_util.h"

namespace xla::codegen {

const std::string& GetCppGenIrString(
    const intrinsics::IntrinsicOptions& options) {
  return ::llvm_ir::kEigenUnaryLlIr;
}

bool AreEigenIntrinsicsAvailable() {
  return !GetCppGenIrString(intrinsics::IntrinsicOptions()).empty();
}

llvm::Function* GetCppGenFunction(llvm::Module* module,
                                  absl::string_view name) {
  llvm::Function* func =
      module->getFunction(llvm::StringRef(name.data(), name.size()));
  CHECK(func != nullptr)
      << "CppGen function '" << name
      << "' was not found in the module. Ensure the "
         "function name is correct and the library "
         "containing it was linked by IntrinsicFunctionLib.\n"
      << llvm_ir::DumpToString(module);

  if (!func->isDeclaration()) {
    func->setLinkage(llvm::Function::InternalLinkage);
    func->addFnAttr(llvm::Attribute::AlwaysInline);
  }
  return func;
}

std::unique_ptr<llvm::Module> ParseEmbeddedBitcode(
    llvm::LLVMContext& context, absl::string_view bitcode,
    absl::string_view source_name) {
  if (bitcode.empty()) {
    LOG_FIRST_N(WARNING, 1)
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
  return module;
}

void CppGenIntrinsicLibrary::LinkIntoModule(llvm::Module& dst_module) const {
  llvm::LLVMContext& context = dst_module.getContext();

  std::unique_ptr<llvm::Module> lib_module =
      ParseEmbeddedBitcode(context, ir_text_, source_name_);

  std::vector<std::string> lib_functions;
  for (const auto& func : *lib_module) {
    if (!func.isDeclaration()) {
      lib_functions.push_back(func.getName().str());
    }
  }

  const llvm::DataLayout& hostDataLayout = dst_module.getDataLayout();
  lib_module->setDataLayout(hostDataLayout);

  // Using static Linker::linkModules based on previous success, but matching
  // logic
  if (llvm::Linker::linkModules(dst_module, std::move(lib_module))) {
    LOG(FATAL) << "LLVM Linker failed to link CppGen library.";
  }

  for (const auto& func : lib_functions) {
    llvm::Function* linked_func = dst_module.getFunction(func);
    if (linked_func && !linked_func->isDeclaration()) {
      linked_func->setLinkage(llvm::Function::InternalLinkage);
      linked_func->addFnAttr(llvm::Attribute::AlwaysInline);
    }
  }
}

}  // namespace xla::codegen
