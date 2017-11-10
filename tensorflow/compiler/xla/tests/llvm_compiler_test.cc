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
#include "tensorflow/compiler/xla/service/backend.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_compiler.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_compiler.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/stream_executor/stream_executor.h"

namespace xla {
namespace {

class LLVMCompilerTest : public ::testing::Test {
 public:
  void SetUp() override {
    Platform *platform = FindPlatform();
    ASSERT_NE(platform, nullptr);

    BackendOptions backend_options;
    backend_options.set_platform(platform);
    StatusOr<std::unique_ptr<Backend>> backend_or_status =
        Backend::CreateBackend(backend_options);
    ASSERT_IS_OK(backend_or_status.status());
    backend_ = backend_or_status.ConsumeValueOrDie();
  }

  ~LLVMCompilerTest() override {}

 protected:
  using Platform = ::perftools::gputools::Platform;

  explicit LLVMCompilerTest(string platform_name)
      : platform_name_(std::move(platform_name)) {}

  void TestCompilerHooks(LLVMCompiler *compiler) {
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

    compiler->SetPreOptimizationHook(pre_opt_hook);
    compiler->SetPostOptimizationHook(post_opt_hook);

    ASSERT_TRUE(compiler
                    ->Compile(std::move(hlo_module),
                              backend_->default_stream_executor())
                    .ok());

    // Test that hooks were called.
    EXPECT_EQ(1, pre_opt_hook_call_count);
    EXPECT_EQ(1, post_opt_hook_call_count);
  }

 private:
  Platform *FindPlatform() {
    for (Platform *platform :
         PlatformUtil::GetSupportedPlatforms().ConsumeValueOrDie()) {
      if (platform->Name() == platform_name_) {
        return platform;
      }
    }
    return nullptr;
  }

  string platform_name_;
  std::unique_ptr<Backend> backend_;

  static string TestName() {
    return ::testing::UnitTest::GetInstance()->current_test_info()->name();
  }

  static std::unique_ptr<HloModule> CreateNewModule() {
    HloModuleConfig config;
    config.set_debug_options(legacy_flags::GetDebugOptionsFromFlags());
    return MakeUnique<HloModule>(TestName(), VersionedComputationHandle(),
                                 config);
  }
};

class CpuCompilerTest : public LLVMCompilerTest {
 public:
  CpuCompilerTest() : LLVMCompilerTest("Host") {}
};

class GpuCompilerTest : public LLVMCompilerTest {
 public:
  GpuCompilerTest() : LLVMCompilerTest("CUDA") {}
};

TEST_F(CpuCompilerTest, HooksTest) {
  cpu::CpuCompiler compiler;
  TestCompilerHooks(&compiler);
}

TEST_F(GpuCompilerTest, HooksTest) {
  gpu::GpuCompiler compiler;
  TestCompilerHooks(&compiler);
}

}  // namespace
}  // namespace xla
