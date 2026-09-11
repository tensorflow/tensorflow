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
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "xla/codegen/intrinsic/cpp/eigen_unary_32_ll.h"
#include "xla/codegen/intrinsic/cpp/eigen_unary_64_ll.h"
#include "xla/codegen/intrinsic/cpp/expm1_32_ll.h"
#include "xla/codegen/intrinsic/cpp/expm1_64_ll.h"
#include "xla/codegen/intrinsic/intrinsic.h"
#include "xla/service/llvm_ir/llvm_util.h"

namespace xla::codegen {

namespace {

const std::string& SelectIrString(const intrinsics::IntrinsicOptions& options,
                                  const std::string& ir_32,
                                  const std::string& ir_64) {
  if (options.Contains("+avx512f")) {
    return ir_64;
  }
  return ir_32;
}

}  // namespace

const std::string& GetCppGenIrString(
    const intrinsics::IntrinsicOptions& options) {
  return SelectIrString(options, ::llvm_ir::kEigenUnary32LlIr,
                        ::llvm_ir::kEigenUnary64LlIr);
}

bool AreCppGenIntrinsicsAvailable() {
  return !GetCppGenIrString(intrinsics::IntrinsicOptions()).empty();
}

std::vector<CppGenIntrinsicLibrary> GetCppGenLibraries(
    const intrinsics::IntrinsicOptions& options) {
  return {
      CppGenIntrinsicLibrary(
          SelectIrString(options, ::llvm_ir::kEigenUnary32LlIr,
                         ::llvm_ir::kEigenUnary64LlIr),
          "eigen"),
      CppGenIntrinsicLibrary(SelectIrString(options, ::llvm_ir::kExpm132LlIr,
                                            ::llvm_ir::kExpm164LlIr),
                             "expm1"),
  };
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
    if (!func->hasFnAttribute(llvm::Attribute::NoInline)) {
      func->addFnAttr(llvm::Attribute::AlwaysInline);
    }
  }
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
  for (const auto& func : *lib_module) {
    if (!func.isDeclaration()) {
      lib_functions.push_back(func.getName().str());
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
    }
  }
}

}  // namespace xla::codegen
