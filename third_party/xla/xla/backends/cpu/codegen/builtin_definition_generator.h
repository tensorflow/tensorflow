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

#ifndef XLA_BACKENDS_CPU_CODEGEN_BUILTIN_DEFINITION_GENERATOR_H_
#define XLA_BACKENDS_CPU_CODEGEN_BUILTIN_DEFINITION_GENERATOR_H_

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Support/Error.h"

namespace xla::cpu {

// Generates symbol definitions for builtin XLA runtime symbols, which are
// looked up at run time in the parent process:
//
//   - libc symbols (e.g. memcpy, memmove, memset)
//   - libm symbols (e.g. sin, cos, etc.)
//   - compiler-rt symbols (e.g. __msan_unpoison)
//   - custom XLA symbols (e.g. __truncsfbf2)
//
// We keep the list of definitions short, and prefer to compile math functions
// into generated XLA:CPU executables via intrinsics, as it allows the LLVM
// optimizer to inline them and optimize across function call boundaries.
class BuiltinDefinitionGenerator : public llvm::orc::DefinitionGenerator {
 public:
  explicit BuiltinDefinitionGenerator(llvm::DataLayout data_layout);

  llvm::Error tryToGenerate(llvm::orc::LookupState&, llvm::orc::LookupKind,
                            llvm::orc::JITDylib& jit_dylib,
                            llvm::orc::JITDylibLookupFlags,
                            const llvm::orc::SymbolLookupSet& names) final;

 private:
  llvm::DataLayout data_layout_;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_CODEGEN_BUILTIN_DEFINITION_GENERATOR_H_
