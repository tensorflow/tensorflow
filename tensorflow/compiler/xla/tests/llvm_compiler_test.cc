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

#include "tensorflow/compiler/xla/service/llvm_compiler.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

class LLVMCompilerTest : public HloTestBase {};

XLA_TEST_F(LLVMCompilerTest, CompilerHooks) {
  int pre_opt_hook_call_count = 0;
  int post_opt_hook_call_count = 0;

  auto pre_opt_hook = [&pre_opt_hook_call_count](const llvm::Module &) {
    ++pre_opt_hook_call_count;
    return Status::OK();
  };
  auto post_opt_hook = [&post_opt_hook_call_count](const llvm::Module &) {
    ++post_opt_hook_call_count;
    return Status::OK();
  };

  // Create HLO module, and run the compiler.
  auto builder = HloComputation::Builder(TestName());
  builder.AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0<float>(42.0)));

  auto hlo_module = CreateNewModule();
  hlo_module->AddEntryComputation(builder.Build());

  auto compiler = static_cast<LLVMCompiler *>(backend_->compiler());
  compiler->SetPreOptimizationHook(pre_opt_hook);
  compiler->SetPostOptimizationHook(post_opt_hook);

  ASSERT_TRUE(
      compiler
          ->Compile(std::move(hlo_module), backend_->default_stream_executor())
          .ok());

  // Test that hooks were called.
  EXPECT_EQ(1, pre_opt_hook_call_count);
  EXPECT_EQ(1, post_opt_hook_call_count);
}

}  // namespace
}  // namespace xla
