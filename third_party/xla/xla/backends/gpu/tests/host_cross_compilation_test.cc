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
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/cpu/target_machine_options.h"
#include "xla/backends/gpu/ffi.h"
#include "xla/backends/gpu/target_config/target_config.h"
#include "xla/ffi/ffi.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/compiler.h"
#include "xla/service/gpu_topology.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/stream_executor/platform_id.h"

namespace xla::gpu {
namespace {
using ::absl_testing::StatusIs;

absl::StatusOr<std::unique_ptr<Compiler>> GetGpuCompiler() {
  ASSIGN_OR_RETURN(stream_executor::PlatformId platform_id,
                   PlatformUtil::GetPlatformIdFromCanonicalName("CUDA"));
  return Compiler::GetForPlatform(platform_id);
}

absl::StatusOr<std::unique_ptr<HloModule>> GetHloModule() {
  constexpr absl::string_view hlo_text = {R"(
      HloModule m

      ENTRY test_computation {
        ROOT root = u32[] custom-call(),
          custom_call_target="__xla_test$$verify_cpu_target_machine_options",
          api_version=API_VERSION_TYPED_FFI
      }
    )"};
  auto hlo_module = std::make_unique<VerifiedHloModule>("m", HloModuleConfig(),
                                                        false, false, nullptr);
  RETURN_IF_ERROR(hlo_module->ParseHloStringAndVerifyModule(hlo_text));
  return hlo_module;
}

static bool passes_cpu_target_machine_options_instantiate_called = false;

absl::Status VerifyCpuTargetMachineOptionsInstantiate(
    const xla::cpu::TargetMachineOptions* opts) {
  passes_cpu_target_machine_options_instantiate_called = true;
  if (opts == nullptr) {
    return absl::InvalidArgumentError(
        "No CPU target machine options passed in.");
  }
  if (opts->triple() != "test-triple") {
    return absl::InvalidArgumentError(absl::StrCat(
        "Expected triple to be test-triple, but got ", opts->triple()));
  }
  if (opts->cpu() != "test-cpu") {
    return absl::InvalidArgumentError(
        absl::StrCat("Expected cpu to be test-cpu, but got ", opts->cpu()));
  }
  return absl::OkStatus();
}

// Execute stage is required for CustomCallThunk, so we do an empty one.
absl::Status VerifyCpuTargetMachineOptionsExecute(
    ffi::Result<ffi::BufferR0<U32>> output) {
  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER(kVerifyCpuTargetMachineOptionsExecute,
                       VerifyCpuTargetMachineOptionsExecute,
                       ffi::Ffi::Bind().Ret<ffi::BufferR0<U32>>());

XLA_FFI_DEFINE_HANDLER(kVerifyCpuTargetMachineOptionsInstantiate,
                       VerifyCpuTargetMachineOptionsInstantiate,
                       ffi::Ffi::Bind<ffi::ExecutionStage::kInstantiate>()
                           .Ctx<ffi::CpuTargetMachineOptions>());

constexpr absl::string_view kVerifyCpuTargetMachineOptionsCustomCallName =
    "__xla_test$$verify_cpu_target_machine_options";

XLA_FFI_REGISTER_HANDLER(
    ffi::GetXlaFfiApi(), kVerifyCpuTargetMachineOptionsCustomCallName, "CUDA",
    {kVerifyCpuTargetMachineOptionsInstantiate, nullptr, nullptr,
     kVerifyCpuTargetMachineOptionsExecute},
    static_cast<uint32_t>(ffi::Traits::kCmdBufferCompatible));

// We have 4 versions to test:
// 1. Compile
// 1a. TargetMachineOptions passed in CompileOptions
// 1b. TargetMachineOptions passed in GpuTopology
// 2. CompileAheadOfTime
// 2a. TargetMachineOptions passed in AotCompilationOptions.gpu_topology
// 3. CompileAheadOfTime with xla_gpu_experimental_aot_compiled_thunks=false aka
//    LegacyCompileAheadOfTime
// 3a. TargetMachineOptions passed in AotCompilationOptions.gpu_topology
//
// We also test whether passing invalid TargetMachineOptions fails compilation
// to mak sure the FFI handlers are actually getting called.

TEST(HostCrossCompilationTest,
     CompileCallConsidersTargetMachineOptionsInCompileOptions) {
  passes_cpu_target_machine_options_instantiate_called = false;

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module, GetHloModule());

  ASSERT_OK_AND_ASSIGN(
      stream_executor::GpuTargetConfigProto gpu_target_config_proto,
      GetGpuTargetConfig(GpuModel::A6000));
  ASSERT_OK_AND_ASSIGN(
      gpu::GpuTargetConfig gpu_target_config,
      gpu::GpuTargetConfig::FromProto(gpu_target_config_proto));

  Compiler::CompileOptions options;
  options.gpu_topology = GetSingleDeviceGpuTopology("gpu", gpu_target_config);
  options.cpu_target_config =
      Compiler::CpuTargetConfig{cpu::TargetMachineOptions{
          /*triple=*/"test-triple",
          /*cpu=*/"test-cpu",
          /*features=*/"",
      }};

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Compiler> compiler, GetGpuCompiler());
  EXPECT_OK(compiler->Compile(std::move(hlo_module), /*stream_exec=*/{nullptr},
                              options));
  EXPECT_TRUE(passes_cpu_target_machine_options_instantiate_called);
}

TEST(HostCrossCompilationTest,
     CompileCallReturnsErrorIfTargetMachineOptionsIsInvalid) {
  passes_cpu_target_machine_options_instantiate_called = false;

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module, GetHloModule());

  ASSERT_OK_AND_ASSIGN(
      stream_executor::GpuTargetConfigProto gpu_target_config_proto,
      GetGpuTargetConfig(GpuModel::A6000));
  ASSERT_OK_AND_ASSIGN(
      gpu::GpuTargetConfig gpu_target_config,
      gpu::GpuTargetConfig::FromProto(gpu_target_config_proto));

  Compiler::CompileOptions options;
  options.gpu_topology = GetSingleDeviceGpuTopology("gpu", gpu_target_config);
  options.cpu_target_config =
      Compiler::CpuTargetConfig{cpu::TargetMachineOptions{
          /*triple=*/"some-invalid-triple",
          /*cpu=*/"test-cpu",
          /*features=*/"",
      }};

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Compiler> compiler, GetGpuCompiler());
  EXPECT_THAT(compiler->Compile(std::move(hlo_module),
                                /*stream_exec=*/{nullptr}, options),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_TRUE(passes_cpu_target_machine_options_instantiate_called);
}

TEST(HostCrossCompilationTest,
     CompileCallConsidersTargetMachineOptionsInGpuTopology) {
  passes_cpu_target_machine_options_instantiate_called = false;

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module, GetHloModule());

  ASSERT_OK_AND_ASSIGN(
      stream_executor::GpuTargetConfigProto gpu_target_config_proto,
      GetGpuTargetConfig(GpuModel::A6000));
  ASSERT_OK_AND_ASSIGN(
      gpu::GpuTargetConfig gpu_target_config,
      gpu::GpuTargetConfig::FromProto(gpu_target_config_proto));

  Compiler::CompileOptions options;
  options.gpu_topology =
      GetSingleDeviceGpuTopology("gpu", gpu_target_config,
                                 cpu::TargetMachineOptions{
                                     /*triple=*/"test-triple",
                                     /*cpu=*/"test-cpu",
                                     /*features=*/"",
                                 });

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Compiler> compiler, GetGpuCompiler());
  EXPECT_OK(compiler->Compile(std::move(hlo_module), /*stream_exec=*/{nullptr},
                              options));
  EXPECT_TRUE(passes_cpu_target_machine_options_instantiate_called);
}

TEST(HostCrossCompilationTest,
     LegacyCompileAheadOfTimeCallConsidersTargetMachineOptionsInGpuTopology) {
  passes_cpu_target_machine_options_instantiate_called = false;

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module, GetHloModule());

  ASSERT_OK_AND_ASSIGN(
      stream_executor::GpuTargetConfigProto gpu_target_config_proto,
      GetGpuTargetConfig(GpuModel::A6000));
  ASSERT_OK_AND_ASSIGN(
      gpu::GpuTargetConfig gpu_target_config,
      gpu::GpuTargetConfig::FromProto(gpu_target_config_proto));

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Compiler> compiler, GetGpuCompiler());
  AotCompilationOptions options(compiler->PlatformId());
  options.set_gpu_topology(
      GetSingleDeviceGpuTopology("gpu", gpu_target_config,
                                 cpu::TargetMachineOptions{
                                     /*triple=*/"test-triple",
                                     /*cpu=*/"test-cpu",
                                     /*features=*/"",
                                 }));

  EXPECT_OK(compiler->CompileAheadOfTime(std::move(hlo_module), options));
  EXPECT_TRUE(passes_cpu_target_machine_options_instantiate_called);
}

TEST(HostCrossCompilationTest,
     LegacyCompileAheadOfTimeReturnsErrorIfTargetMachineOptionsIsInvalid) {
  passes_cpu_target_machine_options_instantiate_called = false;

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module, GetHloModule());

  ASSERT_OK_AND_ASSIGN(
      stream_executor::GpuTargetConfigProto gpu_target_config_proto,
      GetGpuTargetConfig(GpuModel::A6000));
  ASSERT_OK_AND_ASSIGN(
      gpu::GpuTargetConfig gpu_target_config,
      gpu::GpuTargetConfig::FromProto(gpu_target_config_proto));

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Compiler> compiler, GetGpuCompiler());
  AotCompilationOptions options(compiler->PlatformId());
  options.set_gpu_topology(
      GetSingleDeviceGpuTopology("gpu", gpu_target_config,
                                 cpu::TargetMachineOptions{
                                     /*triple=*/"some-invalid-triple",
                                     /*cpu=*/"test-cpu",
                                     /*features=*/"",
                                 }));

  EXPECT_THAT(compiler->CompileAheadOfTime(std::move(hlo_module), options),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_TRUE(passes_cpu_target_machine_options_instantiate_called);
}

TEST(HostCrossCompilationTest,
     CompileAheadOfTimeCallConsidersTargetMachineOptionsInGpuTopology) {
  passes_cpu_target_machine_options_instantiate_called = false;

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module, GetHloModule());
  ASSERT_OK_AND_ASSIGN(
      stream_executor::GpuTargetConfigProto gpu_target_config_proto,
      GetGpuTargetConfig(GpuModel::A6000));
  ASSERT_OK_AND_ASSIGN(
      gpu::GpuTargetConfig gpu_target_config,
      gpu::GpuTargetConfig::FromProto(gpu_target_config_proto));

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Compiler> compiler, GetGpuCompiler());
  AotCompilationOptions options(compiler->PlatformId());
  options.set_gpu_topology(
      GetSingleDeviceGpuTopology("gpu", gpu_target_config,
                                 cpu::TargetMachineOptions{
                                     /*triple=*/"test-triple",
                                     /*cpu=*/"test-cpu",
                                     /*features=*/"",
                                 }));

  EXPECT_OK(compiler->CompileAheadOfTime(std::move(hlo_module), options));
  EXPECT_TRUE(passes_cpu_target_machine_options_instantiate_called);
}

TEST(HostCrossCompilationTest,
     CompileAheadOfTimeCallReturnsErrorIfTargetMachineOptionsIsInvalid) {
  passes_cpu_target_machine_options_instantiate_called = false;

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module, GetHloModule());
  ASSERT_OK_AND_ASSIGN(
      stream_executor::GpuTargetConfigProto gpu_target_config_proto,
      GetGpuTargetConfig(GpuModel::A6000));
  ASSERT_OK_AND_ASSIGN(
      gpu::GpuTargetConfig gpu_target_config,
      gpu::GpuTargetConfig::FromProto(gpu_target_config_proto));

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Compiler> compiler, GetGpuCompiler());
  AotCompilationOptions options(compiler->PlatformId());
  options.set_gpu_topology(
      GetSingleDeviceGpuTopology("gpu", gpu_target_config,
                                 cpu::TargetMachineOptions{
                                     /*triple=*/"some-invalid-triple",
                                     /*cpu=*/"test-cpu",
                                     /*features=*/"",
                                 }));

  EXPECT_THAT(compiler->CompileAheadOfTime(std::move(hlo_module), options),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_TRUE(passes_cpu_target_machine_options_instantiate_called);
}

}  // namespace
}  // namespace xla::gpu
