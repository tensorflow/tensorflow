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

#include "xla/backends/cpu/codegen/jit_compiler.h"

#include <cstddef>
#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Target/TargetOptions.h"
#include "xla/backends/cpu/codegen/function_library.h"
#include "xla/util.h"

namespace xla::cpu {
namespace {

// XLA JIT compiler built on top of LLVM ORC APIs.
class LlvmOrcJitCompiler : public JitCompiler {
 public:
  LlvmOrcJitCompiler(llvm::TargetOptions target_options,
                     llvm::CodeGenOptLevel opt_level, const Options& options);

  absl::Status AddModule(llvm::orc::ThreadSafeModule module,
                         size_t dylib_index) final;

  absl::StatusOr<std::unique_ptr<FunctionLibrary>> Compile() && final;

 private:
};

// XLA function library compiled from LLVM module(s) using ORC APIs.
class LlvmOrcFunctionLibrary : public FunctionLibrary {
 public:
};

}  // namespace

absl::StatusOr<std::unique_ptr<JitCompiler>> JitCompiler::Create(
    llvm::TargetOptions target_options, llvm::CodeGenOptLevel opt_level,
    const Options& options) {
  return std::make_unique<LlvmOrcJitCompiler>(std::move(target_options),
                                              opt_level, options);
}

LlvmOrcJitCompiler::LlvmOrcJitCompiler(llvm::TargetOptions target_options,
                                       llvm::CodeGenOptLevel opt_level,
                                       const Options& options) {}

absl::Status LlvmOrcJitCompiler::AddModule(llvm::orc::ThreadSafeModule module,
                                           size_t dylib_index) {
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<FunctionLibrary>>
LlvmOrcJitCompiler::Compile() && {
  return Unimplemented("Not implemented yet");
}

}  // namespace xla::cpu
