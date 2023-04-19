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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/backend.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_compiler.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_compiler.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/stream_executor/device_description.h"
#include "tensorflow/compiler/xla/stream_executor/stream_executor.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/verified_hlo_module.h"
#include "tensorflow/tsl/platform/test.h"

namespace xla {
namespace gpu {

// Creating dummy data structure needed to initialize a GpuDummyCompiler
PLATFORM_DEFINE_ID(kDummyTestId);
constexpr char kDummyTriple[] = "dummy-triple";
constexpr char kDummyLayout[] = "e";

// This class is a dummy implementation of GpuCompiler and is targeted for unit
// test only
class GpuDummyCompiler : public GpuCompiler {
 public:
  GpuDummyCompiler() : GpuCompiler(kDummyTestId, kDummyTriple, kDummyLayout) {}

  Status OptimizeHloConvolutionCanonicalization(
      HloModule* hlo_module, GpuVersion gpu_version,
      se::dnn::VersionInfo dnn_version,
      se::DeviceMemoryAllocator* device_allocator) {
    return OkStatus();
  }

  Status OptimizeHloPostLayoutAssignment(
      HloModule* hlo_module, se::StreamExecutor* stream_executor,
      const CompileOptions& options, const GpuTargetConfig& gpu_target_config,
      const AutotuneResults* autotune_results) override {
    return OkStatus();
  }

  GpuVersion GetGpuVersion(se::StreamExecutor*) override {
#if GOOGLE_CUDA
    return se::CudaComputeCapability{0, 0};
#elif TENSORFLOW_USE_ROCM
    return se::RocmComputeCapability{"gfx908"};    
#endif      
  }

  StatusOr<std::pair<std::string, std::vector<uint8_t>>> CompileTargetBinary(
      const HloModuleConfig& module_config, llvm::Module* llvm_module,
      GpuVersion gpu_version, bool relocatable,
      const HloModule* debug_module) override {
    std::vector<uint8_t> compiled_results;
    return std::pair<std::string, std::vector<uint8_t>>(
        "", std::move(compiled_results));
  }
};
}  // namespace gpu

namespace {

class LLVMCompilerTest : public ::testing::Test {
 public:
  void SetUp() override {
    Platform* platform = FindPlatform();
    ASSERT_NE(platform, nullptr);

    BackendOptions backend_options;
    backend_options.set_platform(platform);
    StatusOr<std::unique_ptr<Backend>> backend_or_status =
        Backend::CreateBackend(backend_options);
    ASSERT_IS_OK(backend_or_status.status());
    backend_ = std::move(backend_or_status).value();
  }

  ~LLVMCompilerTest() override {}

 protected:
  using Platform = se::Platform;

  explicit LLVMCompilerTest(std::string platform_name)
      : platform_name_(std::move(platform_name)) {}

  void TestCompilerHooks(LLVMCompiler* compiler) {
    int pre_opt_hook_call_count = 0;
    int post_opt_hook_call_count = 0;

    auto pre_opt_hook = [&pre_opt_hook_call_count](const llvm::Module&) {
      ++pre_opt_hook_call_count;
      return OkStatus();
    };
    auto post_opt_hook = [&post_opt_hook_call_count](const llvm::Module&) {
      ++post_opt_hook_call_count;
      return OkStatus();
    };

    // Create HLO module, and run the compiler.
    auto builder = HloComputation::Builder(TestName());
    builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0)));

    auto hlo_module = CreateNewVerifiedModule();
    hlo_module->AddEntryComputation(builder.Build());

    compiler->SetPreOptimizationHook(pre_opt_hook);
    compiler->SetPostOptimizationHook(post_opt_hook);

    ASSERT_TRUE(compiler
                    ->RunBackend(std::move(hlo_module),
                                 backend_->default_stream_executor(),
                                 /*device_allocator=*/nullptr)
                    .ok());

    // Test that hooks were called.
    EXPECT_EQ(1, pre_opt_hook_call_count);
    EXPECT_EQ(1, post_opt_hook_call_count);
  }

  void TestMultiModuleCompilation(LLVMCompiler* compiler) {
    HloComputation::Builder builder(TestName());
    builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0)));

    std::unique_ptr<HloModule> hlo_module = CreateNewVerifiedModule();
    hlo_module->AddEntryComputation(builder.Build());

    auto module_group = std::make_unique<HloModuleGroup>("test_module_group");
    module_group->push_back(hlo_module->Clone());
    module_group->push_back(std::move(hlo_module));

    std::vector<std::vector<se::StreamExecutor*>> executors;
    executors.push_back({backend_->default_stream_executor()});
    executors.push_back({backend_->default_stream_executor()});

    EXPECT_IS_OK(compiler->Compile(std::move(module_group),
                                   std::move(executors),
                                   /*device_allocator=*/nullptr));
  }

 private:
  Platform* FindPlatform() {
    auto status_or_platform = PlatformUtil::GetPlatform(platform_name_);
    return status_or_platform.ok() ? status_or_platform.value() : nullptr;
  }

  std::string platform_name_;
  std::unique_ptr<Backend> backend_;

  static std::string TestName() {
    return ::testing::UnitTest::GetInstance()->current_test_info()->name();
  }

  std::unique_ptr<HloModule> CreateNewVerifiedModule() {
    HloModuleConfig config;
    config.set_debug_options(GetDebugOptionsFromFlags());
    return std::make_unique<VerifiedHloModule>(
        TestName(), config, /*verifier_layout_sensitive=*/false,
        /*allow_mixed_precision_in_hlo_verifier=*/true,
        backend_->compiler()->ShapeSizeBytesFunction());
  }
};

class CpuCompilerTest : public LLVMCompilerTest {
 public:
  CpuCompilerTest() : LLVMCompilerTest("Host") {}
};

class GpuCompilerTest : public LLVMCompilerTest {
 public:
  GpuCompilerTest() : LLVMCompilerTest("GPU") {}
};

TEST_F(CpuCompilerTest, HooksTest) {
  cpu::CpuCompiler compiler;
  TestCompilerHooks(&compiler);
}

TEST_F(GpuCompilerTest, HooksTest) {
  gpu::GpuDummyCompiler compiler;
  TestCompilerHooks(&compiler);
}

TEST_F(CpuCompilerTest, CpuMultiModuleCompilation) {
  cpu::CpuCompiler compiler;
  TestMultiModuleCompilation(&compiler);
}

TEST_F(GpuCompilerTest, GpuMultModuleCompilation) {
  gpu::GpuDummyCompiler compiler;
  TestMultiModuleCompilation(&compiler);
}
}  // namespace
}  // namespace xla
