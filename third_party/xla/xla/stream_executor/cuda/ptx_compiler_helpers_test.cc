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

#include "xla/stream_executor/cuda/ptx_compiler_helpers.h"

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/attributes.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/gpu/gpu_asm_opts.h"
#include "xla/stream_executor/kernel_stats.h"

namespace stream_executor {
namespace {

using ::testing::ElementsAreArray;

// When the compilation succeeds, then the error log is empty.
constexpr absl::string_view kPtxasLogSuccessfulCompilation = R"(
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function 'input_concatenate_fusion' for 'sm_80'
ptxas info    : Function properties for input_concatenate_fusion
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 10 registers, 368 bytes cmem[0]
)";

constexpr absl::string_view kPtxasLogTooOldError = R"(
// Something in the log before the error.
ptxas fatal   : Value 'sm_80' is not defined for option 'gpu-name'
ptxas fatal   : Ptx assembly aborted due to errors
// Something in the log after the error.
)";

constexpr absl::string_view kPtxasLogRegisterAllocationError = R"(
// Something in the log before the error.
ptxas fatal   : (C7600) Register allocation failed with register count of '64'. Compile the program with a higher register target
ptxas fatal   : Ptx assembly aborted due to errors
// Something in the log after the error.
)";

constexpr absl::string_view kPtxasLogRegisterSpillWarning = R"(
// Something in the log before the warning.
ptxas warning : Registers are spilled to local memory in function '__kernel', 18 bytes spill stores, 8 bytes spill loads
// Something in the log after the warning.
)";

constexpr absl::string_view kPtxasLogMultipleKernelsSpillingRegisters = R"(
// Something in the log before the warning.
ptxas warning : Registers are spilled to local memory in function '__kernel', 18 bytes spill stores, 8 bytes spill loads
// Log log log.
ptxas warning : Registers are spilled to local memory in function '__kernel2', 1024 bytes spill stores, 1099 bytes spill loads
// Something in the log after the warning.
)";

TEST(PtxCompilerHelpersTest, IsPtxRegisterAllocationError) {
  EXPECT_TRUE(IsPtxRegisterAllocationError(kPtxasLogRegisterAllocationError));
  EXPECT_FALSE(IsPtxRegisterAllocationError(kPtxasLogRegisterSpillWarning));
}

constexpr absl::string_view kDefaultArchitecture = "sm_80";

TEST(PtxCompilerHelpersTest, CreateErrorFromPTXASLogNoError) {
  EXPECT_THAT(CreateErrorFromPTXASLog(kPtxasLogSuccessfulCompilation,
                                      kDefaultArchitecture,
                                      /*cancel_if_reg_spill=*/true),
              absl_testing::IsOk());
}

TEST(PtxCompilerHelpersTest,
     CreateErrorFromPTXASLogDetectsRegisterAllocationError) {
  EXPECT_THAT(CreateErrorFromPTXASLog(kPtxasLogRegisterAllocationError,
                                      kDefaultArchitecture,
                                      /*cancel_if_reg_spill=*/true),
              absl_testing::StatusIs(absl::StatusCode::kResourceExhausted));
}

TEST(PtxCompilerHelpersTest, CreateErrorFromPTXASLogDetectsPtxAsTooOldError) {
  EXPECT_THAT(
      CreateErrorFromPTXASLog(kPtxasLogTooOldError, kDefaultArchitecture,
                              /*cancel_if_reg_spill=*/true),
      absl_testing::StatusIs(absl::StatusCode::kUnimplemented));
}

TEST(PtxCompilerHelpersTest,
     CreateErrorFromPTXASLogDetectsRegisterSpillWarning) {
  EXPECT_THAT(CreateErrorFromPTXASLog(kPtxasLogRegisterSpillWarning,
                                      kDefaultArchitecture,
                                      /*cancel_if_reg_spill=*/true),
              absl_testing::StatusIs(absl::StatusCode::kCancelled));
}

TEST(PtxCompilerHelpersTest,
     CreateErrorFromPTXASLogIgnoresRegisterSpillWarningIfNotRequested) {
  EXPECT_THAT(CreateErrorFromPTXASLog(kPtxasLogRegisterSpillWarning,
                                      kDefaultArchitecture,
                                      /*cancel_if_reg_spill=*/false),
              absl_testing::IsOk());
}

TEST(PtxCompilerHelpersTest, IsPtxRegisterAllocationErrorStatus) {
  EXPECT_TRUE(IsPtxRegisterAllocationError(
      PtxRegisterAllocationError("Register allocation failed")));
  EXPECT_FALSE(
      IsPtxRegisterAllocationError(absl::ResourceExhaustedError("OOM")));
  EXPECT_FALSE(IsPtxRegisterAllocationError(absl::OkStatus()));
}

TEST(PtxCompilerHelpersTest, ModuleStatsAreCorrectlyExtractedFromLog) {
  ModuleStats kernel_stats_map =
      ExtractModuleStatsFromLog(kPtxasLogRegisterSpillWarning);
  EXPECT_EQ(kernel_stats_map.size(), 1);
  EXPECT_EQ(kernel_stats_map["__kernel"].store_bytes_spilled, 18);
  EXPECT_EQ(kernel_stats_map["__kernel"].load_bytes_spilled, 8);
  kernel_stats_map =
      ExtractModuleStatsFromLog(kPtxasLogMultipleKernelsSpillingRegisters);
  EXPECT_EQ(kernel_stats_map.size(), 2);
  EXPECT_EQ(kernel_stats_map["__kernel"].store_bytes_spilled, 18);
  EXPECT_EQ(kernel_stats_map["__kernel"].load_bytes_spilled, 8);
  EXPECT_EQ(kernel_stats_map["__kernel2"].store_bytes_spilled, 1024);
  EXPECT_EQ(kernel_stats_map["__kernel2"].load_bytes_spilled, 1099);
  kernel_stats_map = ExtractModuleStatsFromLog(kPtxasLogSuccessfulCompilation);
  EXPECT_EQ(kernel_stats_map.size(), 0);
}

struct AppendPtxFlagsTestParam {
  std::string test_name ABSL_REQUIRE_EXPLICIT_INIT;
  bool disable_gpuasm_optimizations ABSL_REQUIRE_EXPLICIT_INIT;
  std::vector<std::string> extra_flags ABSL_REQUIRE_EXPLICIT_INIT;
  std::vector<std::string> expected_flags ABSL_REQUIRE_EXPLICIT_INIT;
};

using AppendPtxFlagsTest = testing::TestWithParam<AppendPtxFlagsTestParam>;

TEST_P(AppendPtxFlagsTest, GeneratesExpectedFlags) {
  const AppendPtxFlagsTestParam& param = GetParam();
  GpuAsmOpts options;
  options.disable_gpuasm_optimizations = param.disable_gpuasm_optimizations;
  options.extra_flags = param.extra_flags;
  std::vector<std::string> flags;
  AppendPtxCompilerFlags(options, flags);
  EXPECT_THAT(flags, ElementsAreArray(param.expected_flags));
}

INSTANTIATE_TEST_SUITE_P(
    AppendPtxFlagsTests, AppendPtxFlagsTest,
    testing::Values(
        AppendPtxFlagsTestParam{/*test_name=*/"Default",
                                /*disable_gpuasm_optimizations=*/false,
                                /*extra_flags=*/{},
                                /*expected_flags=*/{}},
        AppendPtxFlagsTestParam{/*test_name=*/"DisableOptimizations",
                                /*disable_gpuasm_optimizations=*/true,
                                /*extra_flags=*/{},
                                /*expected_flags=*/{"-O0"}},
        AppendPtxFlagsTestParam{/*test_name=*/"ExtraFlags",
                                /*disable_gpuasm_optimizations=*/false,
                                /*extra_flags=*/{"--foo", "-bar"},
                                /*expected_flags=*/{"--foo", "-bar"}},
        AppendPtxFlagsTestParam{
            /*test_name=*/"DisableOptimizationsAndExtraFlags",
            /*disable_gpuasm_optimizations=*/true,
            /*extra_flags=*/{"--foo", "-bar"},
            /*expected_flags=*/{"-O0", "--foo", "-bar"}}),
    [](const testing::TestParamInfo<AppendPtxFlagsTestParam>& info) {
      return info.param.test_name;
    });

struct AppendArchFlagsTestParam {
  std::string test_name ABSL_REQUIRE_EXPLICIT_INIT;
  CudaComputeCapability cc ABSL_REQUIRE_EXPLICIT_INIT;
  bool disable_gpuasm_optimizations ABSL_REQUIRE_EXPLICIT_INIT;
  std::vector<std::string> extra_flags ABSL_REQUIRE_EXPLICIT_INIT;
  bool dump_compilation_log ABSL_REQUIRE_EXPLICIT_INIT;
  std::vector<std::string> expected_flags ABSL_REQUIRE_EXPLICIT_INIT;
};

using AppendArchFlagsTest = testing::TestWithParam<AppendArchFlagsTestParam>;

TEST_P(AppendArchFlagsTest, GeneratesExpectedFlags) {
  const AppendArchFlagsTestParam& param = GetParam();
  GpuAsmOpts options;
  options.disable_gpuasm_optimizations = param.disable_gpuasm_optimizations;
  options.extra_flags = param.extra_flags;
  std::vector<std::string> flags;
  AppendArchitectureSpecificPtxCompilerFlags(param.cc, options,
                                             param.dump_compilation_log, flags);
  EXPECT_THAT(flags, ElementsAreArray(param.expected_flags));
}

INSTANTIATE_TEST_SUITE_P(
    AppendArchFlagsTests, AppendArchFlagsTest,
    testing::Values(
        AppendArchFlagsTestParam{
            /*test_name=*/"AmpereDefault",
            /*cc=*/CudaComputeCapability::Ampere(),
            /*disable_gpuasm_optimizations=*/false,
            /*extra_flags=*/{},
            /*dump_compilation_log=*/false,
            /*expected_flags=*/{"-arch=sm_80", "--warn-on-spills"}},
        AppendArchFlagsTestParam{
            /*test_name=*/"AmpereDumpLog",
            /*cc=*/CudaComputeCapability::Ampere(),
            /*disable_gpuasm_optimizations=*/false,
            /*extra_flags=*/{},
            /*dump_compilation_log=*/true,
            /*expected_flags=*/{"-arch=sm_80", "--warn-on-spills", "-v"}},
        AppendArchFlagsTestParam{
            /*test_name=*/"AmpereDisableOptimizations",
            /*cc=*/CudaComputeCapability::Ampere(),
            /*disable_gpuasm_optimizations=*/true,
            /*extra_flags=*/{},
            /*dump_compilation_log=*/false,
            /*expected_flags=*/{"-arch=sm_80", "--warn-on-spills", "-O0"}},
        AppendArchFlagsTestParam{
            /*test_name=*/"AmpereDumpLogAndDisableOptimizations",
            /*cc=*/CudaComputeCapability::Ampere(),
            /*disable_gpuasm_optimizations=*/true,
            /*extra_flags=*/{},
            /*dump_compilation_log=*/true,
            /*expected_flags=*/
            {"-arch=sm_80", "--warn-on-spills", "-v", "-O0"}},
        AppendArchFlagsTestParam{
            /*test_name=*/"AmpereExtraFlags",
            /*cc=*/CudaComputeCapability::Ampere(),
            /*disable_gpuasm_optimizations=*/false,
            /*extra_flags=*/{"--foo", "-bar"},
            /*dump_compilation_log=*/false,
            /*expected_flags=*/
            {"-arch=sm_80", "--warn-on-spills", "--foo", "-bar"}},
        AppendArchFlagsTestParam{
            /*test_name=*/"AmpereDumpLogAndExtraFlags",
            /*cc=*/CudaComputeCapability::Ampere(),
            /*disable_gpuasm_optimizations=*/false,
            /*extra_flags=*/{"--foo", "-bar"},
            /*dump_compilation_log=*/true,
            /*expected_flags=*/
            {"-arch=sm_80", "--warn-on-spills", "-v", "--foo", "-bar"}},
        AppendArchFlagsTestParam{
            /*test_name=*/"AmpereDisableOptAndExtraFlags",
            /*cc=*/CudaComputeCapability::Ampere(),
            /*disable_gpuasm_optimizations=*/true,
            /*extra_flags=*/{"--foo", "-bar"},
            /*dump_compilation_log=*/false,
            /*expected_flags=*/
            {"-arch=sm_80", "--warn-on-spills", "-O0", "--foo", "-bar"}},
        AppendArchFlagsTestParam{
            /*test_name=*/"AmpereAllOptions",
            /*cc=*/CudaComputeCapability::Ampere(),
            /*disable_gpuasm_optimizations=*/true,
            /*extra_flags=*/{"--foo", "-bar"},
            /*dump_compilation_log=*/true,
            /*expected_flags=*/
            {"-arch=sm_80", "--warn-on-spills", "-v", "-O0", "--foo", "-bar"}},
        AppendArchFlagsTestParam{
            /*test_name=*/"HopperDefault",
            /*cc=*/CudaComputeCapability::Hopper(),
            /*disable_gpuasm_optimizations=*/false,
            /*extra_flags=*/{},
            /*dump_compilation_log=*/false,
            /*expected_flags=*/{"-arch=sm_90", "--warn-on-spills"}},
        AppendArchFlagsTestParam{
            /*test_name=*/"HopperAllOptions",
            /*cc=*/CudaComputeCapability::Hopper(),
            /*disable_gpuasm_optimizations=*/true,
            /*extra_flags=*/{"--foo", "-bar"},
            /*dump_compilation_log=*/true,
            /*expected_flags=*/
            {"-arch=sm_90", "--warn-on-spills", "-v", "-O0", "--foo", "-bar"}},
        AppendArchFlagsTestParam{
            /*test_name=*/"BlackwellDefault",
            /*cc=*/CudaComputeCapability::Blackwell(),
            /*disable_gpuasm_optimizations=*/false,
            /*extra_flags=*/{},
            /*dump_compilation_log=*/false,
            /*expected_flags=*/{"-arch=sm_100", "--warn-on-spills"}}),
    [](const testing::TestParamInfo<AppendArchFlagsTestParam>& info) {
      return info.param.test_name;
    });

}  // namespace
}  // namespace stream_executor
