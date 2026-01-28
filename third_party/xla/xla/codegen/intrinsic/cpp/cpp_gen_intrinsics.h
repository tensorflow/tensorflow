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

#ifndef XLA_CODEGEN_INTRINSIC_CPP_CPP_GEN_INTRINSICS_H_
#define XLA_CODEGEN_INTRINSIC_CPP_CPP_GEN_INTRINSICS_H_

#include <memory>
#include <string>

#include "absl/strings/string_view.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include "xla/codegen/intrinsic/intrinsic.h"

namespace xla::codegen {

const std::string& GetCppGenIrString(
    const intrinsics::IntrinsicOptions& options);

// Helper to parse embedded bitcode into a module.
// Wraps llvm::parseIR and handles initialization of MemoryBuffer.
std::unique_ptr<llvm::Module> ParseEmbeddedBitcode(
    llvm::LLVMContext& context, absl::string_view bitcode,
    absl::string_view source_name = "embedded_module");

// Returns true if the Eigen C++ intrinsics were compiled and are available.
// If the compiler does not support vector extensions, this will return false.
bool AreEigenIntrinsicsAvailable();

// Helper for Intrinsic<T> classes that use CppGen backend for some types.
// Looks up a function by name in the module (assuming it was linked from
// a CppGen library) and configures its linkage and attributes for inlining.
llvm::Function* GetCppGenFunction(llvm::Module* module, absl::string_view name);

class CppGenIntrinsicLibrary {
 public:
  explicit CppGenIntrinsicLibrary(absl::string_view ir_text,
                                  absl::string_view source_name)
      : ir_text_(ir_text), source_name_(source_name) {}

  // Links the CppGen library into the given module. This will insert all
  // of the function definitions in the ir_text into the dst_module and set
  // internal linkage and the alwaysinline attribute on each of them so they
  // can be inlined and removed later.
  void LinkIntoModule(llvm::Module& dst_module) const;

 private:
  std::string ir_text_;
  std::string source_name_;
};

}  // namespace xla::codegen

#endif  // XLA_CODEGEN_INTRINSIC_CPP_CPP_GEN_INTRINSICS_H_
