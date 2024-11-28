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

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "llvm/AsmParser/Parser.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Target/TargetOptions.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla::cpu {

TEST(JitCompilerTest, Compile) {
  auto context = std::make_unique<llvm::LLVMContext>();

  constexpr std::string_view ir = R"(
    define void @AddInplace(ptr %arg) {
      %v0 = load float, ptr %arg
      %v1 = fadd float %v0, %v0
      store float %v1, ptr %arg
      ret void
    })";

  llvm::SMDiagnostic diagnostic;
  llvm::MemoryBufferRef ir_buffer(ir, "AddInplace");

  auto module = llvm::parseAssembly(ir_buffer, diagnostic, *context);
  ASSERT_TRUE(module) << "Failed to parse LLVM IR: "
                      << diagnostic.getMessage().str();

  JitCompiler::Options options;

  TF_ASSERT_OK_AND_ASSIGN(
      auto compiler, JitCompiler::Create(llvm::TargetOptions(),
                                         llvm::CodeGenOptLevel::None, options));

  llvm::orc::ThreadSafeModule tsm(std::move(module), std::move(context));
  TF_ASSERT_OK(compiler.AddModule(std::move(tsm)));

  std::vector<std::string> function_names = {"AddInplace"};
  TF_ASSERT_OK_AND_ASSIGN(auto function_library,
                          std::move(compiler).Compile(function_names));

  TF_ASSERT_OK_AND_ASSIGN(
      auto* add_in_place,
      function_library->ResolveFunction<void(float*)>("AddInplace"));

  float value = 1.0f;
  add_in_place(&value);
  EXPECT_EQ(value, 2.0f);
}

}  // namespace xla::cpu
