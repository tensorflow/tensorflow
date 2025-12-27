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

void CppGenIntrinsicLibrary::LinkIntoModule(llvm::Module& dst_module) const {
  llvm::SMDiagnostic err;
  llvm::LLVMContext& context = dst_module.getContext();

  // Use ir_text_ member variable
  std::unique_ptr<llvm::Module> lib_module = llvm::parseIR(
      llvm::MemoryBufferRef(ir_text_, source_name_), err, context);

  CHECK(lib_module != nullptr)
      << "Failed to parse IR: " << err.getMessage().str() << "\n"
      << ir_text_;

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
