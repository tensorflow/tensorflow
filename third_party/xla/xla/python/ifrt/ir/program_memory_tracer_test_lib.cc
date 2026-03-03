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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/ir/atom_program_compiler.h"
#include "xla/python/ifrt/ir/ifrt_ir_loaded_executable.h"
#include "xla/python/ifrt/ir/ifrt_ir_program.h"
#include "xla/python/ifrt/ir/program_memory_tracer.h"
#include "xla/python/ifrt/ir/tests/executable_impl_test_base.h"
#include "xla/python/ifrt/support/module_parsing.h"
#include "xla/python/ifrt/test_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace ifrt {
namespace {

typedef absl::flat_hash_map<std::string, xla::CompiledMemoryStats>
    MpmdCompiledMemoryStats;
typedef absl::flat_hash_map<std::string, absl::Span<xla::ifrt::Device* const>>
    MpmdAddressableDevices;

constexpr int64_t kGiB = 1024 * 1024 * 1024;

class ProgramMemoryTracerTest
    : public xla::ifrt::test_util::IfrtIrExecutableImplTestBase {
 public:
  ProgramMemoryTracerTest() {
    xla::ifrt::support::RegisterMlirDialects(mlir_context_);
  }

  void SetUp() override {
    ASSERT_OK_AND_ASSIGN(client_, xla::ifrt::test_util::GetClient());
  }

 protected:
  std::unique_ptr<xla::ifrt::IfrtIRCompileOptions> CreateCompileOptions(
      const xla::ifrt::DeviceListRef& devices) {
    std::vector<xla::ifrt::DeviceId> device_assignments;
    device_assignments.reserve(devices->size());
    for (const auto& device : devices->devices()) {
      device_assignments.push_back(device->Id());
    }
    return std::make_unique<xla::ifrt::IfrtIRCompileOptions>(
        std::move(device_assignments), xla::ifrt::AtomExecutableMap(),
        std::make_shared<absl::flat_hash_map<
            std::string, std::unique_ptr<xla::ifrt::CompileOptions>>>());
  }

  absl::StatusOr<xla::ifrt::DeviceListRef> PickDevices(int count) {
    absl::Span<xla::ifrt::Device* const> devices = client_->devices();
    TF_RET_CHECK(count <= devices.size())
        << "Requested " << count << " devices. Only have " << devices.size();
    return client_->MakeDeviceList(devices.first(count));
  }

  absl::StatusOr<std::shared_ptr<IfrtIrLoadedExecutable>> CompileAndLoad(
      mlir::ModuleOp mlir_module, xla::ifrt::DeviceListRef devices) {
    auto program = std::make_unique<xla::ifrt::IfrtIRProgram>(mlir_module);
    auto options = CreateCompileOptions(devices);
    TF_ASSIGN_OR_RETURN(
        xla::ifrt::LoadedExecutableRef executable,
        client_->GetDefaultCompiler()
            ->CompileAndLoad(std::move(program), std::move(options))
            .Await());
    return std::static_pointer_cast<IfrtIrLoadedExecutable>(
        std::move(executable));
  }
};

TEST_F(ProgramMemoryTracerTest, IfrtIrProgramMemoryStatsWithCopyArrays) {
  std::string source = R"(
!array0 = !ifrt.array<tensor<1024x1024x768xi32>,
                      #ifrt.sharding_param<1x1x1 to [0] on 1>, [0]>
!array1 = !ifrt.array<tensor<1024x1024x768xi32>,
                      #ifrt.sharding_param<1x1x1 to [0] on 1>, [1]>
module {
  func.func @main(%arg0: !array0 {ifrt.donated}) -> (!array0, !array0)
      attributes {ifrt.function} {
    %out_0, %ctrl_0 = ifrt.CopyArrays(%arg0) : (!array0) -> !array1

    // The output array should be deleted as soon as the op completes.
    %out_1, %ctrl_1 = ifrt.CopyArrays(%out_0) : (!array1) -> !array0

    %out_2, %ctrl_2 = ifrt.CopyArrays(%out_0) {donated=true}
      : (!array1) -> !array0

    // Should not increase peak memory usage on device 1 because the other array
    // has been donated.
    %out_3, %ctrl_3 = ifrt.CopyArrays(%arg0) : (!array0) -> !array1

    // The argument should be kept alive after the end of the program.
    return %arg0, %out_2 : !array0, !array0
  }
}
  )";
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
                       LoadFromSource(source));
  ASSERT_OK_AND_ASSIGN(xla::ifrt::DeviceListRef devices, PickDevices(2));
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<IfrtIrLoadedExecutable> executable,
                       CompileAndLoad(mlir_module.get(), devices));
  ASSERT_OK_AND_ASSIGN(xla::ifrt::IfrtIrProgramMemoryStats memory_stats,
                       executable->GetIfrtIrProgramMemoryStats());
  EXPECT_EQ(memory_stats.output_size_in_bytes, 6 * kGiB);
  EXPECT_EQ(memory_stats.argument_size_in_bytes, 3 * kGiB);
  EXPECT_EQ(memory_stats.host_argument_size_in_bytes, 0);
  EXPECT_EQ(memory_stats.host_output_size_in_bytes, 0);
  EXPECT_EQ(memory_stats
                .device_to_peak_bytes_used[devices->devices()[0]->Id().value()],
            6 * kGiB);
  EXPECT_EQ(memory_stats
                .device_to_peak_bytes_used[devices->devices()[1]->Id().value()],
            3 * kGiB);
}

TEST_F(ProgramMemoryTracerTest, IfrtIrProgramMemoryStatsWithCallOps) {
  std::string source = R"(
!input = !ifrt.array<tensor<1x1xi32>,
                     #ifrt.sharding_param<1x1 to [0] on 1>, [0]>
!array0 = !ifrt.array<tensor<1024x1024x1280xi32>,
                      #ifrt.sharding_param<1x1x1 to [0] on 1>, [0]>
!array1 = !ifrt.array<tensor<1024x1024x1280xi32>,
                      #ifrt.sharding_param<1x1x1 to [0] on 1>, [1]>
module {
  func.func @main(%arg0: !input) -> (!array0) attributes {ifrt.function} {
    %out_0, %ctrl_0 = ifrt.Call @generate_data(%arg0) on devices [0]
      : (!input) -> !array0
    %out_1, %ctrl_1 = ifrt.CopyArrays(%out_0) {donated=true}
      : (!array0) -> !array1

    // The input array is donated so peak on device 1 should only increase by
    // the code size.
    %out_2, %ctrl_2 = ifrt.Call @identity(%out_1) on devices [1]
      {io_aliases = [array<i32: 0, 0>]} : (!array1) -> !array1

    %out_3, %ctrl_3 = ifrt.CopyArrays(%out_2) {donated=true}
      : (!array1) -> !array0

    return %out_3 : !array0
  }

  func.func private @generate_data(%arg0: tensor<1x1xi32>)
      -> (tensor<1024x1024x1280xi32>) {
    %1 = stablehlo.reshape %arg0 : (tensor<1x1xi32>) -> tensor<1x1x1xi32>
    %2 = "stablehlo.broadcast_in_dim"(%1)
      { broadcast_dimensions = array<i64: 0, 1, 2> }
      : (tensor<1x1x1xi32>) -> tensor<1024x1024x1280xi32>
    return %2 : tensor<1024x1024x1280xi32>
  }

  func.func private @identity(%arg0: tensor<1024x1024x1280xi32>)
      -> tensor<1024x1024x1280xi32> {
    return %arg0 : tensor<1024x1024x1280xi32>
  }
}
  )";
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
                       LoadFromSource(source));
  ASSERT_OK_AND_ASSIGN(xla::ifrt::DeviceListRef devices, PickDevices(2));
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<IfrtIrLoadedExecutable> executable,
                       CompileAndLoad(mlir_module.get(), devices));
  ASSERT_OK_AND_ASSIGN(xla::ifrt::IfrtIrProgramMemoryStats memory_stats,
                       executable->GetIfrtIrProgramMemoryStats());
  EXPECT_EQ(memory_stats.output_size_in_bytes, 5 * kGiB);
  EXPECT_EQ(memory_stats.argument_size_in_bytes, 1024);
  EXPECT_EQ(memory_stats.host_argument_size_in_bytes, 0);
  EXPECT_EQ(memory_stats.host_output_size_in_bytes, 0);

  ASSERT_OK_AND_ASSIGN(MpmdCompiledMemoryStats executable_memory_stats,
                       executable->GetMpmdCompiledMemoryStats());
  absl::flat_hash_map<int32_t, int64_t> expected_memory;
  ASSERT_OK_AND_ASSIGN(MpmdAddressableDevices mpmd_addressable_devices,
                       executable->GetMpmdAddressableDevices());
  for (const auto& [name, stats] : executable_memory_stats) {
    ASSERT_TRUE(mpmd_addressable_devices.contains(name));
    absl::Span<xla::ifrt::Device* const> devices =
        mpmd_addressable_devices[name];
    for (const auto& device : devices) {
      expected_memory[device->Id().value()] = std::max(
          expected_memory[device->Id().value()],
          stats.temp_size_in_bytes + stats.generated_code_size_in_bytes);
    }
  }
  EXPECT_EQ(
      memory_stats
          .device_to_peak_bytes_used[devices->devices()[0]->Id().value()],
      1024 + 5 * kGiB + expected_memory[devices->devices()[0]->Id().value()]);
  EXPECT_EQ(memory_stats
                .device_to_peak_bytes_used[devices->devices()[1]->Id().value()],
            5 * kGiB + expected_memory[devices->devices()[1]->Id().value()]);
}

TEST_F(ProgramMemoryTracerTest, IfrtIrShardedProgramMemoryStats) {
  std::string source = R"(
!input = !ifrt.array<tensor<1x1xi32>,
                     #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>
!array = !ifrt.array<tensor<1024x1024x1536xi32>,
                     #ifrt.sharding_param<2x1x1 to [0] on 2>, [0, 1]>
module {
  func.func @main(%arg0: !input) -> (!array) attributes {ifrt.function} {
    %out_0, %ctrl_0 = ifrt.Call @generate_data(%arg0) on devices [0, 1]
      : (!input) -> !array
    %out_1, %ctrl_1 = ifrt.CopyArrays(%out_0) {donated=true}
      : (!array) -> !array

    // The input array is donated so peak should only increase by code size.
    %out_2, %ctrl_2 = ifrt.Call @identity(%out_1) on devices [0, 1]
      {io_aliases = [array<i32: 0, 0>]} : (!array) -> !array

    return %out_2 : !array
  }

  func.func private @generate_data(%arg0: tensor<1x1xi32>)
      -> (tensor<1024x1024x1536xi32>) {
    %1 = stablehlo.reshape %arg0 : (tensor<1x1xi32>) -> tensor<1x1x1xi32>
    %2 = "stablehlo.broadcast_in_dim"(%1)
      { broadcast_dimensions = array<i64: 0, 1, 2> }
      : (tensor<1x1x1xi32>) -> tensor<1024x1024x1536xi32>
    return %2 : tensor<1024x1024x1536xi32>
  }

  func.func private @identity(%arg0: tensor<1024x1024x1536xi32>)
      -> tensor<1024x1024x1536xi32> {
    return %arg0 : tensor<1024x1024x1536xi32>
  }
}
  )";
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
                       LoadFromSource(source));
  ASSERT_OK_AND_ASSIGN(xla::ifrt::DeviceListRef devices, PickDevices(2));
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<IfrtIrLoadedExecutable> executable,
                       CompileAndLoad(mlir_module.get(), devices));
  ASSERT_OK_AND_ASSIGN(xla::ifrt::IfrtIrProgramMemoryStats memory_stats,
                       executable->GetIfrtIrProgramMemoryStats());
  EXPECT_EQ(memory_stats.output_size_in_bytes, 3 * kGiB);
  EXPECT_EQ(memory_stats.argument_size_in_bytes, 1024);
  EXPECT_EQ(memory_stats.host_argument_size_in_bytes, 0);
  EXPECT_EQ(memory_stats.host_output_size_in_bytes, 0);

  ASSERT_OK_AND_ASSIGN(MpmdCompiledMemoryStats executable_memory_stats,
                       executable->GetMpmdCompiledMemoryStats());
  absl::flat_hash_map<int32_t, int64_t> expected_memory;
  ASSERT_OK_AND_ASSIGN(MpmdAddressableDevices mpmd_addressable_devices,
                       executable->GetMpmdAddressableDevices());
  for (const auto& [name, stats] : executable_memory_stats) {
    ASSERT_TRUE(mpmd_addressable_devices.contains(name));
    absl::Span<xla::ifrt::Device* const> devices =
        mpmd_addressable_devices[name];
    expected_memory[devices[0]->Id().value()] =
        std::max(expected_memory[devices[0]->Id().value()],
                 stats.temp_size_in_bytes + stats.generated_code_size_in_bytes);
  }
  EXPECT_EQ(
      memory_stats
          .device_to_peak_bytes_used[devices->devices()[0]->Id().value()],
      1024 + 3 * kGiB + expected_memory[devices->devices()[0]->Id().value()]);
}

TEST_F(ProgramMemoryTracerTest, IfrtIrProgramMemoryStatsWithOffloadedInput) {
  std::string source = R"(
!array_host = !ifrt.array<tensor<16xf32>,
                          #ifrt.sharding_param<2 to [0] on 2>, [0, 1],
                          memory_kind = "pinned_host">
!array = !ifrt.array<tensor<16xf32>,
                     #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
module @sin_from_offloaded_arg {
  func.func public @main(%arg0: !array_host) -> (!array)
      attributes {ifrt.function} {
    %out, %ctrl = ifrt.Call @sin::@main(%arg0) on devices [0, 1]
      : (!array_host) -> !array
    return %out : !array
  }

  module @sin attributes {sym_visibility = "private"} {
    func.func @main(%arg0: tensor<16xf32> {mhlo.memory_kind = "pinned_host"})
        -> tensor<16xf32> {
      %0 = stablehlo.custom_call @annotate_device_placement(%arg0) {
        backend_config = "",
        has_side_effect = true,
        mhlo.frontend_attributes = {_xla_buffer_placement = "device"}}
        : (tensor<16xf32>) -> tensor<16xf32>
      %1 = stablehlo.sine %0 : tensor<16xf32>
      return %1 : tensor<16xf32>
    }
  }
}
  )";
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
                       LoadFromSource(source));
  ASSERT_OK_AND_ASSIGN(xla::ifrt::DeviceListRef devices, PickDevices(2));
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<IfrtIrLoadedExecutable> executable,
                       CompileAndLoad(mlir_module.get(), devices));
  ASSERT_OK_AND_ASSIGN(xla::ifrt::IfrtIrProgramMemoryStats memory_stats,
                       executable->GetIfrtIrProgramMemoryStats());
  // Arrays allocated on host memory should not be counted in the memory stats.
  EXPECT_EQ(memory_stats.argument_size_in_bytes, 0);
  EXPECT_EQ(memory_stats.output_size_in_bytes, 1024);
  EXPECT_EQ(memory_stats.host_argument_size_in_bytes, 32);
  EXPECT_EQ(memory_stats.host_output_size_in_bytes, 0);
}

TEST_F(ProgramMemoryTracerTest, IfrtIrProgramMemoryStatsWithPaddingAndLayout) {
  std::string source = R"(
!array0 = !ifrt.array<tensor<12x16xf32>,
                      #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1],
                      layout = "{1,0:T(1,128)}">
!array1 = !ifrt.array<tensor<12x16xf32>,
                      #ifrt.sharding_param<2x1 to [0] on 2>, [0, 1],
                      layout = "{0,1:T(1,128)}">
module @padded_arrays_with_layouts {
  func.func public @main(%arg0: !array0, %arg1: !array1) -> (!array0)
      attributes {ifrt.function} {
    return %arg0: !array0
  }
}
  )";
  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
                       LoadFromSource(source));
  ASSERT_OK_AND_ASSIGN(xla::ifrt::DeviceListRef devices, PickDevices(2));
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<IfrtIrLoadedExecutable> executable,
                       CompileAndLoad(mlir_module.get(), devices));
  ASSERT_OK_AND_ASSIGN(xla::ifrt::IfrtIrProgramMemoryStats memory_stats,
                       executable->GetIfrtIrProgramMemoryStats());
  EXPECT_EQ(memory_stats.argument_size_in_bytes, 11264);
  // The second dimension is padded to 128.
  EXPECT_EQ(memory_stats.output_size_in_bytes, 3072);
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
