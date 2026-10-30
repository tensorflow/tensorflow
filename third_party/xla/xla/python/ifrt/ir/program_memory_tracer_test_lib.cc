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
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/pjrt/compiled_memory_stats.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/ir/ifrt_ir_loaded_executable.h"
#include "xla/python/ifrt/ir/ifrt_ir_loaded_executable_test_base.h"
#include "xla/python/ifrt/ir/program_memory_tracer.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"

namespace xla {
namespace ifrt {
namespace {

typedef absl::flat_hash_map<std::string, xla::CompiledMemoryStats>
    MpmdCompiledMemoryStats;
typedef absl::flat_hash_map<std::string, absl::Span<Device* const>>
    MpmdAddressableDevices;

constexpr int64_t kGiB = 1024 * 1024 * 1024;

class ProgramMemoryTracerTest
    : public xla::ifrt::test_util::IfrtIrLoadedExecutableTestBase {
 protected:
  absl::StatusOr<std::shared_ptr<IfrtIrLoadedExecutable>> GetIfrtIrExecutable(
      absl::string_view source, DeviceListRef devices) {
    ABSL_ASSIGN_OR_RETURN(LoadedExecutableRef executable,
                     CompileProgram(source, devices));
    return std::static_pointer_cast<IfrtIrLoadedExecutable>(
        std::move(executable));
  }
};

TEST_F(ProgramMemoryTracerTest, IfrtIrProgramMemoryStatsWithCopyArrays) {
  std::string source = R"mlir(
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

    %out_2, %ctrl_2 = ifrt.CopyArrays(%out_0) <donated=true>
      : (!array1) -> !array0

    // Should not increase peak memory usage on device 1 because the other array
    // has been donated.
    %out_3, %ctrl_3 = ifrt.CopyArrays(%arg0) : (!array0) -> !array1

    // The argument should be kept alive after the end of the program.
    return %arg0, %out_2 : !array0, !array0
  }
}
  )mlir";
  ASSERT_OK_AND_ASSIGN(DeviceListRef devices, PickDevices(2));
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<IfrtIrLoadedExecutable> executable,
                       GetIfrtIrExecutable(source, devices));
  ASSERT_OK_AND_ASSIGN(IfrtIrProgramMemoryStats memory_stats,
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

TEST_F(ProgramMemoryTracerTest, BitcastArraysDoesntChangeMemoryStats) {
  std::string source = R"mlir(
!array0 = !ifrt.array<tensor<1024x1024x768xi32>,
                      #ifrt.sharding_param<1x1x1 to [0] on 1>, [0]>
!array1 = !ifrt.array<tensor<1x1024x1024x768xi32>,
                      #ifrt.sharding_param<1x1x1x1 to [0] on 1>, [0]>
module {
  func.func @main(
      %arg0: !array0 {ifrt.donated}, %arg1: !array1 {ifrt.donated}) -> (!array1)
      attributes {ifrt.function} {
    %0, %1, %ctrl_0 = ifrt.BitcastArrays(%arg0, %arg1)
      <donated=true> : (!array0, !array1) -> (!array1, !array0)
    return %0 : !array1
  }
}
  )mlir";
  ASSERT_OK_AND_ASSIGN(DeviceListRef devices, PickDevices(1));
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<IfrtIrLoadedExecutable> executable,
                       GetIfrtIrExecutable(source, devices));
  ASSERT_OK_AND_ASSIGN(IfrtIrProgramMemoryStats memory_stats,
                       executable->GetIfrtIrProgramMemoryStats());
  EXPECT_EQ(memory_stats.argument_size_in_bytes, 6 * kGiB);
  EXPECT_EQ(memory_stats.output_size_in_bytes, 3 * kGiB);
  EXPECT_EQ(memory_stats.host_argument_size_in_bytes, 0);
  EXPECT_EQ(memory_stats.host_output_size_in_bytes, 0);
  EXPECT_EQ(memory_stats
                .device_to_peak_bytes_used[devices->devices()[0]->Id().value()],
            6 * kGiB);
}

TEST_F(ProgramMemoryTracerTest, ConcurrentIfrtIrProgramMemoryStats) {
  std::string source = R"mlir(
!array0 = !ifrt.array<tensor<1024x1024x768xi32>,
                      #ifrt.sharding_param<1x1x1 to [0] on 1>, [0]>
!array1 = !ifrt.array<tensor<1024x1024x768xi32>,
                      #ifrt.sharding_param<1x1x1 to [0] on 1>, [1]>
module {
  func.func @main(%arg0: !array0 {ifrt.donated}) -> (!array0, !array0)
      attributes {ifrt.function} {
    %out_0, %ctrl_0 = ifrt.CopyArrays(%arg0) : (!array0) -> !array1
    %out_1, %ctrl_1 = ifrt.CopyArrays(%out_0) <donated=true>
      : (!array1) -> !array0
    return %arg0, %out_1 : !array0, !array0
  }
}
  )mlir";
  ASSERT_OK_AND_ASSIGN(DeviceListRef devices, PickDevices(2));
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<IfrtIrLoadedExecutable> executable,
                       GetIfrtIrExecutable(source, devices));

  // The first calls race to build the cached memory trace.
  constexpr int kNumThreads = 8;
  std::vector<absl::StatusOr<IfrtIrProgramMemoryStats>> results(kNumThreads);
  {
    // The destructor waits for all scheduled calls to finish.
    tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "memory_stats",
                                        kNumThreads);
    for (int i = 0; i < kNumThreads; ++i) {
      thread_pool.Schedule([&executable, &results, i] {
        results[i] = executable->GetIfrtIrProgramMemoryStats();
      });
    }
  }
  for (const absl::StatusOr<IfrtIrProgramMemoryStats>& result : results) {
    ASSERT_TRUE(result.ok()) << result.status();
    EXPECT_EQ(result->argument_size_in_bytes, 3 * kGiB);
    EXPECT_EQ(result->output_size_in_bytes, 6 * kGiB);
    EXPECT_EQ(result->device_to_peak_bytes_used.at(
                  devices->devices()[0]->Id().value()),
              6 * kGiB);
  }
}

TEST_F(ProgramMemoryTracerTest, IfrtIrProgramMemoryStatsWithCallOps) {
  std::string source = R"mlir(
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
    %out_1, %ctrl_1 = ifrt.CopyArrays(%out_0) <donated=true>
      : (!array0) -> !array1

    // The input array is donated so peak on device 1 should only increase by
    // the code size.
    %out_2, %ctrl_2 = ifrt.Call @identity(%out_1) on devices [1]
      <io_aliases = [array<i32: 0, 0>]> : (!array1) -> !array1

    %out_3, %ctrl_3 = ifrt.CopyArrays(%out_2) <donated=true>
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
  )mlir";
  ASSERT_OK_AND_ASSIGN(DeviceListRef devices, PickDevices(2));
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<IfrtIrLoadedExecutable> executable,
                       GetIfrtIrExecutable(source, devices));
  ASSERT_OK_AND_ASSIGN(IfrtIrProgramMemoryStats memory_stats,
                       executable->GetIfrtIrProgramMemoryStats());
  const int64_t expected_arg_bytes =
      xla::IsTpuId(client_->platform_id()) ? 1024 : 4;
  EXPECT_EQ(memory_stats.output_size_in_bytes, 5 * kGiB);
  EXPECT_EQ(memory_stats.argument_size_in_bytes, expected_arg_bytes);
  EXPECT_EQ(memory_stats.host_argument_size_in_bytes, 0);
  EXPECT_EQ(memory_stats.host_output_size_in_bytes, 0);

  ASSERT_OK_AND_ASSIGN(MpmdCompiledMemoryStats executable_memory_stats,
                       executable->GetMpmdCompiledMemoryStats());
  absl::flat_hash_map<int32_t, int64_t> expected_memory;
  ASSERT_OK_AND_ASSIGN(MpmdAddressableDevices mpmd_addressable_devices,
                       executable->GetMpmdAddressableDevices());
  for (const auto& [name, stats] : executable_memory_stats) {
    ASSERT_TRUE(mpmd_addressable_devices.contains(name));
    absl::Span<Device* const> devices = mpmd_addressable_devices[name];
    for (const auto& device : devices) {
      expected_memory[device->Id().value()] = std::max(
          expected_memory[device->Id().value()],
          stats.temp_size_in_bytes + stats.generated_code_size_in_bytes);
    }
  }
  EXPECT_EQ(memory_stats
                .device_to_peak_bytes_used[devices->devices()[0]->Id().value()],
            expected_arg_bytes + 5 * kGiB +
                expected_memory[devices->devices()[0]->Id().value()]);
  EXPECT_EQ(memory_stats
                .device_to_peak_bytes_used[devices->devices()[1]->Id().value()],
            5 * kGiB + expected_memory[devices->devices()[1]->Id().value()]);
}

TEST_F(ProgramMemoryTracerTest, IfrtIrShardedProgramMemoryStats) {
  std::string source = R"mlir(
!input = !ifrt.array<tensor<1x1xi32>,
                     #ifrt.sharding_param<1x1 to [0] on 2>, [0, 1]>
!array = !ifrt.array<tensor<1024x1024x1536xi32>,
                     #ifrt.sharding_param<2x1x1 to [0] on 2>, [0, 1]>
module {
  func.func @main(%arg0: !input) -> (!array) attributes {ifrt.function} {
    %out_0, %ctrl_0 = ifrt.Call @generate_data(%arg0) on devices [0, 1]
      : (!input) -> !array
    %out_1, %ctrl_1 = ifrt.CopyArrays(%out_0) <donated=true>
      : (!array) -> !array

    // The input array is donated so peak should only increase by code size.
    %out_2, %ctrl_2 = ifrt.Call @identity(%out_1) on devices [0, 1]
      <io_aliases = [array<i32: 0, 0>]> : (!array) -> !array

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
  )mlir";
  ASSERT_OK_AND_ASSIGN(DeviceListRef devices, PickDevices(2));
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<IfrtIrLoadedExecutable> executable,
                       GetIfrtIrExecutable(source, devices));
  ASSERT_OK_AND_ASSIGN(IfrtIrProgramMemoryStats memory_stats,
                       executable->GetIfrtIrProgramMemoryStats());
  const int64_t expected_arg_bytes =
      xla::IsTpuId(client_->platform_id()) ? 1024 : 4;
  EXPECT_EQ(memory_stats.output_size_in_bytes, 3 * kGiB);
  EXPECT_EQ(memory_stats.argument_size_in_bytes, expected_arg_bytes);
  EXPECT_EQ(memory_stats.host_argument_size_in_bytes, 0);
  EXPECT_EQ(memory_stats.host_output_size_in_bytes, 0);

  ASSERT_OK_AND_ASSIGN(MpmdCompiledMemoryStats executable_memory_stats,
                       executable->GetMpmdCompiledMemoryStats());
  absl::flat_hash_map<int32_t, int64_t> expected_memory;
  ASSERT_OK_AND_ASSIGN(MpmdAddressableDevices mpmd_addressable_devices,
                       executable->GetMpmdAddressableDevices());
  for (const auto& [name, stats] : executable_memory_stats) {
    ASSERT_TRUE(mpmd_addressable_devices.contains(name));
    absl::Span<Device* const> devices = mpmd_addressable_devices[name];
    expected_memory[devices[0]->Id().value()] =
        std::max(expected_memory[devices[0]->Id().value()],
                 stats.temp_size_in_bytes + stats.generated_code_size_in_bytes);
  }
  EXPECT_EQ(memory_stats
                .device_to_peak_bytes_used[devices->devices()[0]->Id().value()],
            expected_arg_bytes + 3 * kGiB +
                expected_memory[devices->devices()[0]->Id().value()]);
}

TEST_F(ProgramMemoryTracerTest, IfrtIrProgramMemoryStatsWithOffloadedInput) {
  std::string source = R"mlir(
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

  module @sin <sym_visibility = "private"> {
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
  )mlir";
  ASSERT_OK_AND_ASSIGN(DeviceListRef devices, PickDevices(2));
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<IfrtIrLoadedExecutable> executable,
                       GetIfrtIrExecutable(source, devices));
  ASSERT_OK_AND_ASSIGN(IfrtIrProgramMemoryStats memory_stats,
                       executable->GetIfrtIrProgramMemoryStats());
  const int64_t expected_bytes =
      xla::IsTpuId(client_->platform_id()) ? 1024 : 32;
  // Arrays allocated on host memory should not be counted in the device memory
  // stats.
  EXPECT_EQ(memory_stats.argument_size_in_bytes, 0);
  EXPECT_EQ(memory_stats.output_size_in_bytes, expected_bytes);
  EXPECT_EQ(memory_stats.host_argument_size_in_bytes, expected_bytes);
  EXPECT_EQ(memory_stats.host_output_size_in_bytes, 0);
}

TEST_F(ProgramMemoryTracerTest, IfrtIrProgramMemoryStatsWithPaddingAndLayout) {
  std::string source = R"mlir(
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
  )mlir";
  ASSERT_OK_AND_ASSIGN(DeviceListRef devices, PickDevices(2));
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<IfrtIrLoadedExecutable> executable,
                       GetIfrtIrExecutable(source, devices));
  ASSERT_OK_AND_ASSIGN(IfrtIrProgramMemoryStats memory_stats,
                       executable->GetIfrtIrProgramMemoryStats());
  EXPECT_EQ(memory_stats.argument_size_in_bytes, 11264);
  // The second dimension is padded to 128.
  EXPECT_EQ(memory_stats.output_size_in_bytes, 3072);
}

TEST_F(ProgramMemoryTracerTest, IfrtIrProgramMemoryStatsWithSubByteTypes) {
  std::string source = R"mlir(
!array_dev = !ifrt.array<tensor<1024x1024x768xf4E2M1FN>,
                         #ifrt.sharding_param<1x1x1 to [0] on 1>, [0]>
!array_host = !ifrt.array<tensor<1024x1024x768xf4E2M1FN>,
                          #ifrt.sharding_param<1x1x1 to [0] on 1>, [0],
                          memory_kind = "pinned_host">
module @sub_byte_types {
  func.func public @main(%arg0: !array_dev, %arg1: !array_host)
      -> (!array_dev, !array_host) attributes {ifrt.function} {
    return %arg0, %arg1 : !array_dev, !array_host
  }
}
  )mlir";
  ASSERT_OK_AND_ASSIGN(DeviceListRef devices, PickDevices(/*count=*/1));
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<IfrtIrLoadedExecutable> executable,
                       GetIfrtIrExecutable(source, devices));
  ASSERT_OK_AND_ASSIGN(IfrtIrProgramMemoryStats memory_stats,
                       executable->GetIfrtIrProgramMemoryStats());
  // 1024 * 1024 * 768 * 4 bits = 384 MiB when packed on GPU; simulated TPU does
  // not pack `f4E2M1FN` and use 1 byte/element (768 MiB) for both device and
  // `pinned_host` memory.
  const int64_t expected_bytes = xla::IsTpuId(client_->platform_id())
                                     ? 768LL * 1024 * 1024
                                     : 384LL * 1024 * 1024;
  EXPECT_EQ(memory_stats.host_argument_size_in_bytes, expected_bytes);
  EXPECT_EQ(memory_stats.host_output_size_in_bytes, expected_bytes);
  EXPECT_EQ(memory_stats.argument_size_in_bytes, expected_bytes);
  EXPECT_EQ(memory_stats.output_size_in_bytes, expected_bytes);
  EXPECT_EQ(
      memory_stats
          .device_to_peak_bytes_used[devices->devices().front()->Id().value()],
      expected_bytes);
}

TEST_F(ProgramMemoryTracerTest,
       IfrtIrProgramMemoryStatsWithUnpinnedHostSubByteTypes) {
  std::string source = R"mlir(
!array_host = !ifrt.array<tensor<1024x1024x768xf4E2M1FN>,
                          #ifrt.sharding_param<1x1x1 to [0] on 1>, [0],
                          memory_kind = "unpinned_host">
module @unpinned_host_sub_byte_types {
  func.func public @main(%arg0: !array_host)
      -> (!array_host) attributes {ifrt.function} {
    return %arg0 : !array_host
  }
}
  )mlir";
  ASSERT_OK_AND_ASSIGN(DeviceListRef devices, PickDevices(/*count=*/1));
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<IfrtIrLoadedExecutable> executable,
                       GetIfrtIrExecutable(source, devices));
  ASSERT_OK_AND_ASSIGN(IfrtIrProgramMemoryStats memory_stats,
                       executable->GetIfrtIrProgramMemoryStats());
  // `unpinned_host` uses an untiled descending layout with 4 bits/element
  // (1024 * 1024 * 768 * 4 bits = 384 MiB).
  constexpr int64_t kExpectedHostBytes = 384LL * 1024 * 1024;
  EXPECT_EQ(memory_stats.host_argument_size_in_bytes, kExpectedHostBytes);
  EXPECT_EQ(memory_stats.host_output_size_in_bytes, kExpectedHostBytes);
  EXPECT_EQ(memory_stats.argument_size_in_bytes, 0);
  EXPECT_EQ(memory_stats.output_size_in_bytes, 0);
}

TEST_F(ProgramMemoryTracerTest, IfrtIrProgramMemoryStatsWithAutoLayout) {
  std::string source = R"mlir(
!array = !ifrt.array<tensor<1024x1024xf32>,
                     #ifrt.sharding_param<1x1 to [0] on 1>, [0],
                     layout = "auto">
module @auto_layout {
  func.func public @main(%arg0: !array) -> (!array) attributes {ifrt.function} {
    %out, %ctrl = ifrt.Call @identity(%arg0) on devices [0]
      : (!array) -> !array
    return %out : !array
  }

  func.func private @identity(%arg0: tensor<1024x1024xf32>)
      -> tensor<1024x1024xf32> {
    return %arg0 : tensor<1024x1024xf32>
  }
}
  )mlir";
  ASSERT_OK_AND_ASSIGN(DeviceListRef devices, PickDevices(/*count=*/1));
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<IfrtIrLoadedExecutable> executable,
                       GetIfrtIrExecutable(source, devices));
  ASSERT_OK_AND_ASSIGN(IfrtIrProgramMemoryStats memory_stats,
                       executable->GetIfrtIrProgramMemoryStats());
  constexpr int64_t kExpectedBytes = 4LL * 1024 * 1024;
  EXPECT_EQ(memory_stats.argument_size_in_bytes, kExpectedBytes);
  EXPECT_EQ(memory_stats.output_size_in_bytes, kExpectedBytes);
  EXPECT_EQ(memory_stats.host_argument_size_in_bytes, 0);
  EXPECT_EQ(memory_stats.host_output_size_in_bytes, 0);
}

TEST_F(ProgramMemoryTracerTest,
       IfrtIrProgramMemoryStatsWithCpuAndDeviceInputs) {
  absl::StatusOr<DeviceListRef> cpu_device =
      PickDevices(/*count=*/1, /*platform_name=*/"cpu");
  if (!cpu_device.ok()) {
    GTEST_SKIP() << "Test requires at least one cpu device";
  }
  std::string source = R"mlir(
!accel_array = !ifrt.array<tensor<1x1xi32>,
                           #ifrt.sharding_param<1x1 to [0] on 1>, [0]>
!cpu_array = !ifrt.array<tensor<1x1xi32>,
                         #ifrt.sharding_param<1x1 to [0] on 1>, [1]>
module @cpu_and_device_inputs {
  func.func public @main(%arg0: !accel_array, %arg1: !cpu_array)
      -> (!accel_array) attributes {ifrt.function} {
    return %arg0 : !accel_array
  }
}
  )mlir";
  ASSERT_OK_AND_ASSIGN(DeviceListRef accel_device, PickDevices(/*count=*/1));
  ASSERT_OK_AND_ASSIGN(
      DeviceListRef devices,
      client_->MakeDeviceList(
          {accel_device->devices().front(), (*cpu_device)->devices().front()}));
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<IfrtIrLoadedExecutable> executable,
                       GetIfrtIrExecutable(source, devices));
  ASSERT_OK_AND_ASSIGN(IfrtIrProgramMemoryStats memory_stats,
                       executable->GetIfrtIrProgramMemoryStats());
  const int64_t expected_accel_bytes =
      xla::IsTpuId(client_->platform_id()) ? 1024 : 4;
  constexpr int64_t kExpectedCpuBytes = 4;
  EXPECT_EQ(memory_stats.argument_size_in_bytes,
            expected_accel_bytes + kExpectedCpuBytes);
  EXPECT_EQ(memory_stats.output_size_in_bytes, expected_accel_bytes);
  EXPECT_EQ(memory_stats.host_argument_size_in_bytes, 0);
  EXPECT_EQ(memory_stats.host_output_size_in_bytes, 0);
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
