/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/tests/llvm_irgen_test_base.h"

#include <functional>
#include <utility>

#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/tests/filecheck.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

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
    std::unique_ptr<HloModule> hlo_module, const string& pattern,
    bool match_optimized_ir) {
  SetIrHook(match_optimized_ir);
  Status status = CompileToExecutable(std::move(hlo_module)).status();
  ResetIrHook();
  TF_ASSERT_OK(status);

  StatusOr<bool> filecheck_result = RunFileCheck(ir_, pattern);
  TF_ASSERT_OK(filecheck_result.status());
  EXPECT_TRUE(filecheck_result.ValueOrDie()) << "Full IR: " << ir_;
}

void LlvmIrGenTestBase::CompileAndVerifyIr(const string& hlo_text,
                                           const string& expected_llvm_ir,
                                           bool match_optimized_ir) {
  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text, config));
  CompileAndVerifyIr(std::move(module), expected_llvm_ir, match_optimized_ir);
}

void LlvmIrGenTestBase::CompileAheadOfTimeAndVerifyIr(
    std::unique_ptr<HloModule> hlo_module, const AotCompilationOptions& options,
    const string& pattern, bool match_optimized_ir) {
  SetIrHook(match_optimized_ir);
  Status status =
      CompileToAotCompilationResult(std::move(hlo_module), options).status();
  ResetIrHook();
  TF_ASSERT_OK(status);

  StatusOr<bool> filecheck_result = RunFileCheck(ir_, pattern);
  ASSERT_TRUE(filecheck_result.ok());
  EXPECT_TRUE(filecheck_result.ValueOrDie()) << "Full IR: " << ir_;
}

void LlvmIrGenTestBase::MatchOptimizedHlo(absl::string_view hlo,
                                          absl::string_view pattern,
                                          bool print_operand_shape) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized_module,
                          GetOptimizedModule(hlo));
  HloPrintOptions print_opts;
  print_opts.set_print_operand_shape(print_operand_shape);
  StatusOr<bool> filecheck_result =
      RunFileCheck(optimized_module->ToString(print_opts), pattern);
  TF_ASSERT_OK(filecheck_result.status());
  EXPECT_TRUE(filecheck_result.ValueOrDie());
}

StatusOr<std::unique_ptr<HloModule>> LlvmIrGenTestBase::GetOptimizedModule(
    absl::string_view hlo) {
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnVerifiedModule(hlo, GetModuleConfigForTest()));
  return backend().compiler()->RunHloPasses(
      std::move(module), backend().default_stream_executor(),
      backend().default_stream_executor()->GetAllocator());
}

LLVMCompiler* LlvmIrGenTestBase::GetLLVMCompiler() {
  return static_cast<LLVMCompiler*>(backend().compiler());
}

Status LlvmIrGenTestBase::IrHook(const llvm::Module& module) {
  ir_ = llvm_ir::DumpModuleToString(module);
  return Status::OK();
}

}  // namespace xla
