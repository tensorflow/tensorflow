/* Copyright 2026 The OpenXLA Authors.

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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/tests/hlo_legacy_gpu_test_base.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/compiled_module.h"
#include "xla/service/compiler.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/gpu_compiler.h"
#include "xla/service/gpu_topology.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_runner.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/service/llvm_ir/llvm_command_line_options.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/test.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

using ::testing::AssertionResult;
using ::testing::NotNull;
using ::testing::Pointee;
using ::testing::Property;
using ::testing::SizeIs;

se::Platform* GpuPlatform() {
  std::string name =
      absl::AsciiStrToUpper(PlatformUtil::CanonicalPlatformName("gpu").value());
  return se::PlatformManager::PlatformWithName(name).value();
}

class AotCompilationTest : public HloLegacyGpuTestBase,
                           public ::testing::WithParamInterface<bool> {
 protected:
  AotCompilationTest()
      : aot_options_(
            std::make_unique<AotCompilationOptions>(compiler()->PlatformId())) {
    GpuTopology gpu_topology = GetSingleDeviceGpuTopology(
        /*platform_version=*/"", gpu_target_config());
    aot_options_->set_gpu_topology(gpu_topology);
  }

  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options =
        HloHardwareIndependentTestBase::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_experimental_aot_compiled_thunks(GetParam());
    return debug_options;
  }

  std::unique_ptr<AotCompilationOptions> aot_options_;
};

INSTANTIATE_TEST_SUITE_P(NewAotFlow, AotCompilationTest, ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool>& info) {
                           return info.param ? "NewAotFlowEnabled"
                                             : "NewAotFlowDisabled";
                         });

TEST_P(AotCompilationTest, CompileAndLoadAotResult) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> add_1_hlo,
                       ParseAndReturnVerifiedModule(R"hlo(
    add1 {
      p = s32[] parameter(0)
      c = s32[] constant(1)
      ROOT a = s32[] add(p, c)
    }

    ENTRY e {
      p = s32[] parameter(0)
      ROOT r = s32[] fusion(p), kind=kLoop, calls=add1
    })hlo",
                                                    GetModuleConfigForTest()));

  ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<CompiledModule>> aot_results,
      compiler()->CompileAheadOfTime(std::move(add_1_hlo), *aot_options_));
  ASSERT_THAT(aot_results, SizeIs(1));

  ASSERT_OK_AND_ASSIGN(std::string serialized_aot_result,
                       std::move(aot_results[0])->SerializeAsString());
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<CompiledModule> aot_result,
      compiler()->LoadAotCompilationResult(serialized_aot_result));

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Executable> executable,
      std::move(*aot_result)
          .LoadExecutable(compiler()->PlatformId(), device_description()));
  auto hlo_runner = std::make_unique<HloRunner>(GpuPlatform());
  std::unique_ptr<OpaqueExecutable> wrapped_executable =
      hlo_runner->WrapExecutable(std::move(executable));

  const xla::Literal literal_input = xla::LiteralUtil::CreateR0<int32_t>(1);
  const xla::Literal literal_expected_result =
      xla::LiteralUtil::CreateR0<int32_t>(2);
  ASSERT_OK_AND_ASSIGN(Literal result,
                       hlo_runner->ExecuteWithExecutable(
                           wrapped_executable.get(), {&literal_input}));
  EXPECT_TRUE(LiteralTestUtil::Equal(result, literal_expected_result));
}

TEST_P(AotCompilationTest, ExportAndImportAotResult) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> add_1_hlo,
                       ParseAndReturnVerifiedModule(R"hlo(
    add1 {
      p = s32[] parameter(0)
      c = s32[] constant(1)
      ROOT a = s32[] add(p, c)
    }

    ENTRY e {
      p = s32[] parameter(0)
      ROOT r = s32[] fusion(p), kind=kLoop, calls=add1
    })hlo",
                                                    GetModuleConfigForTest()));

  Compiler::CompileOptions compile_options;
  compile_options.gpu_topology = aot_options_->gpu_topology();
  compile_options.early_exit_with_layouts = false;
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Executable> executable,
      compiler()->RunBackend(std::move(add_1_hlo), /*executor=*/nullptr,
                             compile_options));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<CompiledModule> aot_result,
                       compiler()->Export(executable.get()));

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Executable> new_executable,
      std::move(*aot_result)
          .LoadExecutable(compiler()->PlatformId(), device_description()));
  auto hlo_runner = std::make_unique<HloRunner>(GpuPlatform());
  std::unique_ptr<OpaqueExecutable> wrapped_executable =
      hlo_runner->WrapExecutable(std::move(new_executable));

  const xla::Literal literal_input = xla::LiteralUtil::CreateR0<int32_t>(1);
  const xla::Literal literal_expected_result =
      xla::LiteralUtil::CreateR0<int32_t>(2);
  ASSERT_OK_AND_ASSIGN(Literal result,
                       hlo_runner->ExecuteWithExecutable(
                           wrapped_executable.get(), {&literal_input}));
  EXPECT_TRUE(LiteralTestUtil::Equal(result, literal_expected_result));
}

TEST_P(AotCompilationTest, EarlyExitWithLayouts) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> add_1_hlo,
                       ParseAndReturnVerifiedModule(R"hlo(
    add1 {
      p = s32[] parameter(0)
      c = s32[] constant(1)
      ROOT a = s32[] add(p, c)
    }

    ENTRY e {
      p = s32[] parameter(0)
      ROOT r = s32[] fusion(p), kind=kLoop, calls=add1
    })hlo",
                                                    GetModuleConfigForTest()));

  aot_options_->set_early_exit_point(
      AotCompilationOptions::EarlyExitPoint::kAfterLayoutAssignment);
  ASSERT_OK_AND_ASSIGN(
      std::vector<std::unique_ptr<CompiledModule>> aot_results,
      compiler()->CompileAheadOfTime(std::move(add_1_hlo), *aot_options_));
  EXPECT_THAT(aot_results, ElementsAre(Pointee(Property(
                               &CompiledModule::optimized_module, NotNull()))));
}

class KernelCacheTest : public HloLegacyGpuTestBase {
 public:
  void SetUp() override {
    CHECK(tsl::Env::Default()->LocalTempFilename(&cache_file_name_));
    HloModuleConfig config;
    config.set_debug_options(GetDebugOptionsForTest());
    ASSERT_OK_AND_ASSIGN(bool can_use_link_modules,
                         dynamic_cast<GpuCompiler*>(compiler())
                             ->CanUseLinkModules(config, device_description()));
    if (!can_use_link_modules) {
      GTEST_SKIP() << "Caching compiled kernels requires support of linking.";
    }
  }

  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options =
        HloHardwareIndependentTestBase::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_kernel_cache_file(cache_file_name_);
    debug_options.set_xla_gpu_enable_llvm_module_compilation_parallelism(true);
    return debug_options;
  }

  bool CacheFileExists() {
    if (!tsl::Env::Default()->FileExists(cache_file_name_).ok()) {
      return false;
    }
    return true;
  }

  int CacheEntryCount() {
    if (!CacheFileExists()) {
      return 0;
    }
    std::string serialized;
    TF_EXPECT_OK(tsl::ReadFileToString(tsl::Env::Default(), cache_file_name_,
                                       &serialized));
    CompilationCacheProto proto;
    EXPECT_TRUE(proto.ParseFromString(serialized));
    return proto.entries_size();
  }

  std::string cache_file_name_;
  static constexpr absl::string_view kHloText = R"(
  ENTRY e {
    p = s8[] parameter(0)
    c = s8[] constant(8)
    ROOT _ = s8[] add(p, c)
  })";
};

TEST_F(KernelCacheTest, CacheIsGenerated) {
  // First run - no cache file
  EXPECT_FALSE(CacheFileExists());
  EXPECT_TRUE(Run(kHloText, /*run_hlo_passes=*/false));
  // First run generates a cache
  EXPECT_EQ(CacheEntryCount(), 1);
  // Second run - with cache file
  EXPECT_TRUE(Run(kHloText, /*run_hlo_passes=*/false));
  EXPECT_EQ(CacheEntryCount(), 1);
}

TEST_F(KernelCacheTest, NoCacheIsGeneratedWithoutCompiledKernels) {
  EXPECT_FALSE(CacheFileExists());
  EXPECT_TRUE(Run(R"(
  ENTRY e {
    a = f32[5,5] parameter(0)
    ROOT _ = f32[5,5] custom-call(a, a), custom_call_target="__cublas$gemm",
      backend_config="{ \"gemm_backend_config\": {\"alpha_real\":1,\"beta\":0,\"dot_dimension_numbers\":{\"lhs_contracting_dimensions\":[\"1\"],\"rhs_contracting_dimensions\":[\"0\"],\"lhs_batch_dimensions\":[],\"rhs_batch_dimensions\":[]},\"alpha_imag\":0,\"precision_config\":{\"operand_precision\":[\"DEFAULT\",\"DEFAULT\"]},\"epilogue\":\"DEFAULT\"}}"
  })",
                  /*run_hlo_passes=*/false));
  EXPECT_FALSE(CacheFileExists());
}

TEST_F(KernelCacheTest, CacheGrowsWithNewKernels) {
  EXPECT_FALSE(CacheFileExists());
  EXPECT_TRUE(Run(kHloText, /*run_hlo_passes=*/false));
  EXPECT_EQ(CacheEntryCount(), 1);
  // Second run - with cache file and another HLO
  EXPECT_TRUE(Run(R"(
  ENTRY e {
    p = s8[] parameter(0)
    ROOT _ = s8[] multiply(p, p)
  })",
                  /*run_hlo_passes=*/false));
  EXPECT_EQ(CacheEntryCount(), 2);
}

TEST_F(KernelCacheTest, AllKernelsAreCachedBecauseSplitModuleUsesRoundRobin) {
  EXPECT_FALSE(CacheFileExists());
  EXPECT_TRUE(Run(R"(
  ENTRY e {
    p = s8[] parameter(0)
    n = s8[] negate(p)
    a = s8[] add(n, n)
    s = s8[] subtract(p, a)
    ROOT _ = s8[] multiply(s, p)
  })",
                  /*run_hlo_passes=*/false));
  EXPECT_EQ(CacheEntryCount(), 4);
}

TEST_F(KernelCacheTest, CachingWorksWithLoadedExecutables) {
  const std::string kHloAdd1 = R"(
add1 {
  p = s32[] parameter(0)
  c = s32[] constant(1)
  ROOT a = s32[] add(p, c)
}

ENTRY e {
  p = s32[] parameter(0)
  ROOT r = s32[] fusion(p), kind=kLoop, calls=add1
})";

  const std::string kHloAdd2 = R"(
add2 {
  p = s32[] parameter(0)
  c = s32[] constant(2)
  ROOT a = s32[] add(p, c)
}

ENTRY e {
  p = s32[] parameter(0)
  ROOT r = s32[] fusion(p), kind=kLoop, calls=add2
})";

  AotCompilationOptions aot_options(compiler()->PlatformId());
  GpuTopology gpu_topology = GetSingleDeviceGpuTopology(
      /*platform_version=*/"", gpu_target_config());
  aot_options.set_gpu_topology(gpu_topology);

  auto test = [this, &aot_options](absl::string_view hlo, int input,
                                   int expected_result) {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         ParseAndReturnVerifiedModule(hlo));
    ASSERT_OK_AND_ASSIGN(
        std::vector<std::unique_ptr<CompiledModule>> aot_results,
        compiler()->CompileAheadOfTime(std::move(module), aot_options));

    ASSERT_OK_AND_ASSIGN(std::string serialized_aot_result,
                         aot_results[0]->SerializeAsString());
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<CompiledModule> aot_result,
        compiler()->LoadAotCompilationResult(serialized_aot_result));

    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<Executable> executable,
        std::move(*aot_result)
            .LoadExecutable(compiler()->PlatformId(), device_description()));
    auto hlo_runner = std::make_unique<HloRunner>(GpuPlatform());
    std::unique_ptr<OpaqueExecutable> wrapped_executable =
        hlo_runner->WrapExecutable(std::move(executable));

    const xla::Literal literal_input =
        xla::LiteralUtil::CreateR0<int32_t>(input);
    const xla::Literal literal_expected_result =
        xla::LiteralUtil::CreateR0<int32_t>(expected_result);

    ASSERT_OK_AND_ASSIGN(Literal result,
                         hlo_runner->ExecuteWithExecutable(
                             wrapped_executable.get(), {&literal_input}));

    EXPECT_TRUE(LiteralTestUtil::Equal(result, literal_expected_result));
  };

  test(kHloAdd1, 1, 2);
  test(kHloAdd2, 1, 3);
  // The test used to fail on the second execution of the second module when it
  // was already cached.
  test(kHloAdd2, 1, 3);
}

class KernelCacheTestSingleThreaded : public KernelCacheTest {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = KernelCacheTest::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_force_compilation_parallelism(1);
    return debug_options;
  }
};

TEST_F(KernelCacheTestSingleThreaded, CacheIsGenerated) {
  EXPECT_FALSE(CacheFileExists());
  EXPECT_TRUE(Run(kHloText, /*run_hlo_passes=*/false));
  EXPECT_EQ(CacheEntryCount(), 1);
  EXPECT_TRUE(Run(kHloText, /*run_hlo_passes=*/false));
  EXPECT_EQ(CacheEntryCount(), 1);
}

class NoKernelCacheTest : public KernelCacheTest {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = KernelCacheTest::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_enable_llvm_module_compilation_parallelism(false);
    return debug_options;
  }
};

TEST_F(NoKernelCacheTest, NoCacheWithoutCompilationParallelism) {
  EXPECT_TRUE(Run(kHloText, /*run_hlo_passes=*/false));
  EXPECT_FALSE(CacheFileExists());
}

}  // namespace
}  // namespace gpu
}  // namespace xla
