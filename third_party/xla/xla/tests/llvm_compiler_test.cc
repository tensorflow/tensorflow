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

#include "xla/service/llvm_compiler.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "llvm/IR/Module.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module_group.h"
#include "xla/literal_util.h"
#include "xla/service/backend.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/test_helpers.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/threadpool.h"

namespace xla {
namespace {

using LLVMCompilerTest = HloTestBase;

const char* const kHloText = R"(
HloModule Constant

ENTRY main  {
  ROOT constant = f32[] constant(42.0)
}
)";

TEST_F(LLVMCompilerTest, HooksTest) {
  int pre_opt_hook_call_count = 0;
  int post_opt_hook_call_count = 0;

  auto pre_opt_hook = [&pre_opt_hook_call_count](const llvm::Module&) {
    ++pre_opt_hook_call_count;
    return absl::OkStatus();
  };
  auto post_opt_hook = [&post_opt_hook_call_count](const llvm::Module&) {
    ++post_opt_hook_call_count;
    return absl::OkStatus();
  };

  // Create HLO module, and run the compiler.
  auto hlo_module = ParseAndReturnVerifiedModule(kHloText).value();
  LLVMCompiler* compiler =
      tensorflow::down_cast<xla::LLVMCompiler*>(backend().compiler());
  compiler->SetPreOptimizationHook(pre_opt_hook);
  compiler->SetPostOptimizationHook(post_opt_hook);

  ASSERT_TRUE(compiler
                  ->RunBackend(std::move(hlo_module),
                               backend().default_stream_executor(),
                               /*device_allocator=*/nullptr)
                  .ok());

  // Test that hooks were called.
  EXPECT_EQ(1, pre_opt_hook_call_count);
  EXPECT_EQ(1, post_opt_hook_call_count);
}

TEST_F(LLVMCompilerTest, DISABLED_MultiModuleCompilation) {
  auto hlo_module = ParseAndReturnVerifiedModule(kHloText).value();
  auto hlo_module2 = ParseAndReturnVerifiedModule(kHloText).value();
  std::vector<std::unique_ptr<HloModule>> modules;
  modules.push_back(std::move(hlo_module));
  modules.push_back(std::move(hlo_module2));
  auto module_group =
      std::make_unique<HloModuleGroup>("test_module_group", std::move(modules));

  std::vector<std::vector<se::StreamExecutor*>> executors;
  executors.push_back({backend().default_stream_executor()});
  executors.push_back({backend().default_stream_executor()});

  EXPECT_IS_OK(backend().compiler()->Compile(std::move(module_group),
                                             std::move(executors),
                                             backend().memory_allocator()));
}

}  // namespace
}  // namespace xla
