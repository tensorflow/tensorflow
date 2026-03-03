/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/pjrt/cpu/cpu_pjrt_compiler.h"

#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "xla/backends/cpu/codegen/cpu_features.h"
#include "xla/backends/cpu/target_machine_options.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/pjrt/cpu/cpu_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_topology.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_topology_description.h"
#include "xla/pjrt/proto/compile_options.pb.h"
#include "xla/service/cpu/executable.pb.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::cpu {
namespace {

constexpr absl::string_view kProgram = R"(HloModule Computation

ENTRY Computation() -> s32[] {
  ROOT result = s32[] constant(2)
})";

constexpr absl::string_view kMlirProgram = R"mlir(
  module {
    func.func @main() -> tensor<i32> {
      %0 = mhlo.constant dense<2> : tensor<i32>
      return %0 : tensor<i32>
    }
  })mlir";

using CpuPjrtCompilerTest = xla::HloHardwareIndependentTestBase;

std::unique_ptr<CpuTopologyDescription> GetDefaultCpuTopologyDescription() {
  constexpr int kCpuDeviceCount = 1;
  constexpr int kProcessId = 0;
  std::vector<CpuTopology::CpuDevice> cpu_topology_devices;
  cpu_topology_devices.reserve(kCpuDeviceCount);
  for (int i = 0; i < kCpuDeviceCount; ++i) {
    cpu_topology_devices.push_back(CpuTopology::CpuDevice{kProcessId, i});
  }

  return std::make_unique<CpuTopologyDescription>(
      xla::CpuPlatformId(), xla::CpuPlatformName(), xla::CpuPlatformVersion(),
      CpuTopology(cpu_topology_devices,
                  xla::cpu::TargetMachineOptions(GetDebugOptionsFromFlags())));
}
TEST_F(CpuPjrtCompilerTest, CompileXlaComputationSuccess) {
  xla::CompileOptions options;
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kProgram));
  xla::XlaComputation computation(module->ToProto());

  auto topology_description = GetDefaultCpuTopologyDescription();

  xla::cpu::CpuPjRtCompiler compiler;
  TF_ASSERT_OK_AND_ASSIGN(
      auto executable,
      compiler.Compile(options, computation, *topology_description,
                       /*client=*/nullptr));
}

TEST_F(CpuPjrtCompilerTest, CompileMlirOpSuccess) {
  xla::CompileOptions options;
  mlir::MLIRContext context;
  context.loadDialect<mlir::func::FuncDialect, mlir::mhlo::MhloDialect>();
  auto mlir_module =
      mlir::parseSourceString<mlir::ModuleOp>(kMlirProgram, &context);

  auto topology_description = GetDefaultCpuTopologyDescription();

  xla::cpu::CpuPjRtCompiler compiler;
  TF_ASSERT_OK_AND_ASSIGN(
      auto executable,
      compiler.Compile(options, *mlir_module, *topology_description,
                       /*client=*/nullptr));
}

TEST_F(CpuPjrtCompilerTest, CompileXlaComputationWithAvx512FeatureOn) {
  xla::CompileOptions options;
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kProgram));
  xla::XlaComputation computation(module->ToProto());

  constexpr int kCpuDeviceCount = 1;
  constexpr int kProcessId = 0;
  std::vector<CpuTopology::CpuDevice> cpu_topology_devices;
  cpu_topology_devices.reserve(kCpuDeviceCount);
  for (int i = 0; i < kCpuDeviceCount; ++i) {
    cpu_topology_devices.push_back(CpuTopology::CpuDevice{kProcessId, i});
  }

  // Set custom topology.
  auto topology_description = std::make_unique<CpuTopologyDescription>(
      xla::CpuPlatformId(), xla::CpuPlatformName(), xla::CpuPlatformVersion(),
      CpuTopology(cpu_topology_devices,
                  xla::cpu::TargetMachineOptions(/*triple=*/"", /*cpu=*/"",
                                                 /*features=*/"+avx512")));

  xla::cpu::CpuPjRtCompiler compiler;
  TF_ASSERT_OK_AND_ASSIGN(
      auto executable,
      compiler.Compile(options, computation, *topology_description,
                       /*client=*/nullptr));

  // We serialize and then load the executable to confirm that the target
  // machine options were set correctly.

  TF_ASSERT_OK_AND_ASSIGN(auto serialized_executable,
                          executable->SerializeExecutable());

  ExecutableAndOptionsProto proto;
  EXPECT_TRUE(proto.ParseFromString(serialized_executable));

  xla::cpu::CompilationResultProto compilation_result_proto;

  EXPECT_TRUE(
      compilation_result_proto.ParseFromString(proto.serialized_executable()));

  EXPECT_THAT(compilation_result_proto.target_machine_options().features(),
              testing::HasSubstr("+avx512"));
}

}  // namespace
}  // namespace xla::cpu
