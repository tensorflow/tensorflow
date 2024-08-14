/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/tests/llvm_irgen_test_base.h"

#include <functional>
#include <utility>

#include "absl/status/status.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/tests/filecheck.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tsl/platform/test.h"

namespace xla {

void LlvmIrGenTestBase::SetIrHook(bool match_optimized_ir) {
  auto llvm_compiler = GetLLVMCompiler();
  using std::placeholders::_1;

  // Add the IR inspection hook to the LLVM compiler.
  if (match_optimized_ir) {
    llvm_compiler->SetPostOptimizationHook(
        std::bind(&LlvmIrGenTestBase::IrHook, this, _1));
  } else {
    llvm_compiler->SetPreOptimizationHook(
        std::bind(&LlvmIrGenTestBase::IrHook, this, _1));
  }
}

void LlvmIrGenTestBase::ResetIrHook() {
  auto llvm_compiler = GetLLVMCompiler();

  llvm_compiler->RemovePreOptimizationHook();
  llvm_compiler->RemovePostOptimizationHook();
}

void LlvmIrGenTestBase::CompileAndVerifyIr(
    std::unique_ptr<HloModule> hlo_module, const std::string& pattern,
    bool match_optimized_ir, bool run_optimization_passes) {
  SetIrHook(match_optimized_ir);
  absl::Status status =
      CompileToExecutable(std::move(hlo_module), run_optimization_passes)
          .status();
  ResetIrHook();
  TF_ASSERT_OK(status);

  absl::StatusOr<bool> filecheck_result = RunFileCheck(ir_, pattern);
  TF_ASSERT_OK(filecheck_result.status());
  EXPECT_TRUE(filecheck_result.value()) << "Full IR: " << ir_;
}

void LlvmIrGenTestBase::CompileAndVerifyIr(const std::string& hlo_text,
                                           const std::string& expected_llvm_ir,
                                           bool match_optimized_ir,
                                           bool run_optimization_passes) {
  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text, config));
  CompileAndVerifyIr(std::move(module), expected_llvm_ir, match_optimized_ir,
                     run_optimization_passes);
}

void LlvmIrGenTestBase::CompileAheadOfTimeAndVerifyIr(
    std::unique_ptr<HloModule> hlo_module, const AotCompilationOptions& options,
    const std::string& pattern, bool match_optimized_ir) {
  SetIrHook(match_optimized_ir);
  absl::Status status =
      CompileToAotCompilationResult(std::move(hlo_module), options).status();
  ResetIrHook();
  TF_ASSERT_OK(status);

  absl::StatusOr<bool> filecheck_result = RunFileCheck(ir_, pattern);
  ASSERT_TRUE(filecheck_result.ok());
  EXPECT_TRUE(filecheck_result.value()) << "Full IR: " << ir_;
}

LLVMCompiler* LlvmIrGenTestBase::GetLLVMCompiler() {
  return static_cast<LLVMCompiler*>(backend().compiler());
}

absl::Status LlvmIrGenTestBase::IrHook(const llvm::Module& module) {
  ir_ = llvm_ir::DumpToString(&module);
  return absl::OkStatus();
}

}  // namespace xla
