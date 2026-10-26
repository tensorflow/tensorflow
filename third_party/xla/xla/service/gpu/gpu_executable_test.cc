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

#include "xla/service/gpu/gpu_executable.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "riegeli/bytes/cfile_reader.h"
#include "riegeli/bytes/string_reader.h"
#include "xla/backends/cpu/target_machine_options.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/loopback_collectives.h"
#include "xla/backends/gpu/runtime/collective_clique_requests.h"
#include "xla/backends/gpu/runtime/command_buffer_cmd_emitter.h"
#include "xla/backends/gpu/runtime/command_buffer_thunk.h"
#include "xla/backends/gpu/runtime/command_executor.h"
#include "xla/backends/gpu/runtime/custom_kernel_thunk.h"
#include "xla/backends/gpu/runtime/device_to_device_copy_thunk.h"
#include "xla/backends/gpu/runtime/kernel_thunk.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/backends/gpu/runtime/thunk_executor.h"
#include "xla/backends/gpu/tests/hlo_pjrt_gpu_test_base.h"
#include "xla/client/executable_build_options.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/debug_options_flags.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/literal_util.h"
#include "xla/pjrt/proto/compile_options.pb.h"
#include "xla/runtime/device_id.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/buffer_value.h"
#include "xla/service/device_assignment.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/dense_data_intermediate.h"
#include "xla/service/gpu/gpu_executable.pb.h"
#include "xla/service/gpu/gpu_executable_buffer_allocator.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/service/gpu/gpu_module_globals.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu_topology.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/logical_buffer.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_slice.h"
#include "xla/service/xla_debug_info_manager.h"
#include "xla/shape.h"
#include "xla/shape_layout.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/abi/executable_abi_version.h"
#include "xla/stream_executor/abi/executable_abi_version.pb.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/tma_metadata.h"
#include "xla/stream_executor/mock_stream.h"
#include "xla/stream_executor/mock_stream_executor.h"
#include "xla/stream_executor/module_spec.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/rocm/rocm_compute_capability.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/tsl/testing/temporary_directory.h"
#include "xla/tsl/util/proto/parse_text_proto.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/util/split_proto/split_proto_reader.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/path.h"

namespace xla::gpu {
namespace {
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::Field;
using ::testing::Optional;
using ::testing::Pair;
using ::testing::Pointee;
using ::testing::Property;
using ::testing::Return;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAre;
using ::tsl::proto_testing::EqualsProto;
using ::tsl::proto_testing::ParseTextProtoOrDie;
using ::tsl::proto_testing::Partially;
using ::tsl::testing::TemporaryDirectory;

std::string ToString(absl::Span<const uint8_t> data) {
  return std::string(reinterpret_cast<const char*>(data.data()), data.size());
}

using GpuExecutableTest = HloPjRtGpuTestBase;

void SetDummyBufferAssignment(GpuExecutable::Params& params) {
  params.buffer_assignment_proto = BufferAssignmentProto();
  params.buffer_allocations_debug_summary = "dummy";
  if (!params.gpu_topology.has_value()) {
    params.gpu_topology = GpuTopology(
        /*platform_version=*/"", /*num_partitions=*/1,
        /*num_hosts_per_partition=*/1, /*num_devices_per_host=*/1);
  }
}

Thunk::ThunkInfo ThunkInfoWithId(int thunk_id) {
  Thunk::ThunkInfo thunk_info;
  thunk_info.thunk_id = thunk_id;
  return thunk_info;
}

TEST(GpuModuleGlobalsTest, EmptyBinarySkipsModuleLoading) {
  std::vector<uint8_t> binary;
  std::vector<GpuModuleGlobals::ConstantInfo> constants;
  se::MockStreamExecutor executor;
  se::MockStream stream;
  GpuModuleGlobals module_globals(binary, constants);

  EXPECT_CALL(stream, parent()).WillOnce(Return(&executor));
  EXPECT_CALL(executor, device_ordinal()).WillRepeatedly(Return(0));
  EXPECT_CALL(executor, GetPlatform()).Times(0);
  EXPECT_CALL(executor, LoadModule(::testing::_)).Times(0);

  ASSERT_OK_AND_ASSIGN(const auto* resolved, module_globals.Resolve(&stream));

  EXPECT_TRUE(resolved->empty());
}

TEST(GpuModuleGlobalsTest, ConcurrentResolveAcrossDevices) {
  constexpr int kNumDevices = 8;
  constexpr int kThreadsPerDevice = 4;

  std::vector<uint8_t> binary = {1, 2, 3, 4};
  std::vector<GpuModuleGlobals::ConstantInfo> constants = {
      GpuModuleGlobals::ConstantInfo{
          /*symbol_name=*/"const_0",
          /*content=*/DenseDataIntermediate::Own({10, 20, 30, 40}),
          /*allocation_index=*/0}};

  std::vector<std::unique_ptr<se::MockStreamExecutor>> executors;
  std::vector<std::unique_ptr<se::MockStream>> streams;
  for (int d = 0; d < kNumDevices; ++d) {
    auto executor = std::make_unique<se::MockStreamExecutor>();
    auto stream = std::make_unique<se::MockStream>();
    const void* handle_id =
        reinterpret_cast<const void*>(static_cast<uintptr_t>(0x1000 + d));
    se::ModuleHandle module_handle(handle_id);
    se::DeviceAddressBase symbol_addr(
        reinterpret_cast<void*>(static_cast<uintptr_t>(0x2000 + d * 16)), 4);

    EXPECT_CALL(*stream, parent()).WillRepeatedly(Return(executor.get()));
    EXPECT_CALL(*executor, device_ordinal()).WillRepeatedly(Return(d));
    EXPECT_CALL(*executor, LoadModule(::testing::_))
        .Times(1)
        .WillOnce(Return(module_handle));
    EXPECT_CALL(*executor, GetSymbol("const_0", module_handle))
        .Times(1)
        .WillOnce(Return(symbol_addr));
    EXPECT_CALL(*stream, Memcpy(::testing::A<se::DeviceAddressBase*>(),
                                ::testing::A<const void*>(), 4))
        .Times(1)
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*stream, BlockHostUntilDone())
        .Times(1)
        .WillOnce(Return(absl::OkStatus()));
    EXPECT_CALL(*executor, UnloadModule(module_handle))
        .Times(1)
        .WillOnce(Return(true));

    executors.push_back(std::move(executor));
    streams.push_back(std::move(stream));
  }

  GpuModuleGlobals module_globals(binary, constants,
                                  /*devices_per_host=*/kNumDevices);

  tsl::thread::ThreadPool pool(tsl::Env::Default(), "resolve_test",
                               kNumDevices * kThreadsPerDevice);
  absl::BlockingCounter ready(kNumDevices * kThreadsPerDevice);
  absl::Notification go;
  absl::BlockingCounter done(kNumDevices * kThreadsPerDevice);

  std::vector<const GpuModuleGlobals::BufferAllocToDeviceMemoryMap*> results(
      kNumDevices * kThreadsPerDevice, nullptr);

  for (int d = 0; d < kNumDevices; ++d) {
    for (int t = 0; t < kThreadsPerDevice; ++t) {
      int idx = d * kThreadsPerDevice + t;
      pool.Schedule([&, d, idx]() {
        ready.DecrementCount();
        go.WaitForNotification();
        ASSERT_OK_AND_ASSIGN(results[idx],
                             module_globals.Resolve(streams[d].get()));
        done.DecrementCount();
      });
    }
  }

  ready.Wait();
  go.Notify();
  done.Wait();

  for (int d = 0; d < kNumDevices; ++d) {
    const auto* base = results[d * kThreadsPerDevice];
    ASSERT_NE(base, nullptr);
    ASSERT_TRUE(base->contains(0));
    EXPECT_EQ(base->at(0).opaque(),
              reinterpret_cast<void*>(static_cast<uintptr_t>(0x2000 + d * 16)));
    for (int t = 1; t < kThreadsPerDevice; ++t) {
      EXPECT_EQ(results[d * kThreadsPerDevice + t], base);
    }
  }
}

TEST_F(GpuExecutableTest, ExecuteWithWatchdogs) {
  constexpr absl::string_view kHlo = R"(
    HloModule watchdog_test
    ENTRY main {
      input = f32[4] parameter(0)
      ROOT result = f32[4] add(input, input)
    }
  )";

  HloModuleConfig config = GetModuleConfigForTest();
  config.mutable_debug_options().set_xla_gpu_execution_terminate_timeout("10s");
  config.mutable_debug_options().set_xla_gpu_device_execution_terminate_timeout(
      "10s");

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo, config));
  EXPECT_TRUE(Run(std::move(module), /*run_hlo_passes=*/true));
}

TEST_F(GpuExecutableTest, OutputInfoToAndFromProto) {
  const GpuExecutable::OutputInfo output_info0{/*allocation_index=*/42,
                                               /*passthrough=*/true,
                                               /*alias_config=*/std::nullopt};
  EXPECT_THAT(output_info0.ToProto(), EqualsProto(R"pb(
                allocation_index: 42,
                passthrough: true
              )pb"));
  EXPECT_THAT(GpuExecutable::OutputInfo::FromProto(output_info0.ToProto()),
              absl_testing::IsOkAndHolds(output_info0));

  const GpuExecutable::OutputInfo output_info1{
      /*allocation_index=*/43,
      /*passthrough=*/false,
      /*alias_config=*/
      HloInputOutputAliasConfig::Alias{
          /*parameter_number=*/89, /*parameter_index=*/ShapeIndex{1, 2, 3, 4},
          /*kind=*/HloInputOutputAliasConfig::kMustAlias}};
  EXPECT_THAT(output_info1.ToProto(), EqualsProto(R"pb(
                allocation_index: 43,
                alias_config {
                  parameter_number: 89,
                  parameter_shape_index: [ 1, 2, 3, 4 ],
                  kind: MUST_ALIAS
                }
              )pb"));
  EXPECT_THAT(GpuExecutable::OutputInfo::FromProto(output_info1.ToProto()),
              absl_testing::IsOkAndHolds(output_info1));

  const GpuExecutable::OutputInfo output_info2{
      /*allocation_index=*/44,
      /*passthrough=*/true,
      /*alias_config=*/
      HloInputOutputAliasConfig::Alias{
          /*parameter_number=*/0, /*parameter_index=*/ShapeIndex{},
          /*kind=*/HloInputOutputAliasConfig::kMayAlias}};
  EXPECT_THAT(output_info2.ToProto(), EqualsProto(R"pb(
                allocation_index: 44,
                passthrough: true,
                alias_config { kind: MAY_ALIAS }
              )pb"));
  EXPECT_THAT(GpuExecutable::OutputInfo::FromProto(output_info2.ToProto()),
              absl_testing::IsOkAndHolds(output_info2));
}

TEST_F(GpuExecutableTest, RunThunkPasses) {
  ASSERT_OK_AND_ASSIGN(
      tsl::testing::TemporaryDirectory dump_dir,
      tsl::testing::TemporaryDirectory::CreateForCurrentTestcase());
  DebugOptions debug_options = GetDebugOptionsFromFlags();
  debug_options.set_xla_dump_to(dump_dir.path());
  debug_options.set_xla_gpu_graph_min_graph_size(1);
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::FUSION);

  int execution_count = 0;
  auto create_executable = [&]() {
    Thunk::ThunkInfo thunk_info;
    BufferAllocation alloc(0, 1024, 0);
    Shape shape = ShapeUtil::MakeShape(S32, {256});
    BufferAllocation::Slice slice(&alloc, 0, 1024);

    ThunkSequence thunk_sequence;
    thunk_sequence.Emplace<KernelThunk>(
        thunk_info,
        /*kernel_name=*/"test_kernel",
        /*kernel_arguments=*/emitters::KernelArguments({}),
        /*launch_dimensions=*/LaunchDimensions(),
        /*cluster_dim=*/std::nullopt,
        /*shmem_bytes=*/0,
        /*tma_metadata=*/se::gpu::TmaMetadata(),
        /*devices_per_host=*/1);
    thunk_sequence.Emplace<DeviceToDeviceCopyThunk>(
        thunk_info, ShapedSlice{slice, shape}, ShapedSlice{slice, shape}, 1024);

    GpuExecutable::Params params;
    params.executable =
        std::make_unique<ThunkExecutor>(std::move(thunk_sequence));
    params.debug_options = debug_options;
    SetDummyBufferAssignment(params);

    params.module_name = absl::StrCat("test_module", execution_count++);
    se::DeviceDescription device_description;
    device_description.set_gpu_compute_capability(
        se::GpuComputeCapability{se::CudaComputeCapability::Volta()});
    device_description.set_driver_version({12, 3, 0});
    device_description.set_runtime_version({12, 3, 0});
    params.device_description = device_description;
    params.enable_debug_info_manager = false;
    params.debug_module =
        std::make_unique<HloModule>(params.module_name, HloModuleConfig());
    params.debug_module->mutable_config().set_debug_options(debug_options);
    return GpuExecutable::Create(std::move(params));
  };

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GpuExecutable> executable,
                       create_executable());
  const ThunkSequence& thunks = executable->thunk_executor().thunks();
  EXPECT_THAT(
      thunks,
      ElementsAre(Pointee(Property(&Thunk::kind, Thunk::kCommandBuffer))));

  std::vector<std::string> dump_files;
  ASSERT_OK(tsl::Env::Default()->GetMatchingPaths(
      tsl::io::JoinPath(dump_dir.path(),
                        "*thunk_sequence_after_thunk_passes*.txt"),
      &dump_files));

  EXPECT_EQ(dump_files.size(), 1);
}

TEST_F(GpuExecutableTest, CommandBufferAllocationPolicy) {
  std::vector<BufferAllocation> allocations;
  allocations.reserve(5);
  allocations.emplace_back(0, 4, 0);
  allocations.emplace_back(1, 4, 0);
  allocations.back().set_constant(true);
  allocations.emplace_back(2, 4, 0);
  allocations.back().set_entry_computation_parameter(
      /*parameter_number=*/0, /*param_shape_index=*/{},
      /*parameter_aliased_with_output=*/false);
  allocations.emplace_back(3, 4, 0);
  allocations.back().set_maybe_live_out(true);
  allocations.emplace_back(4, 0, 0);

  Shape shape = ShapeUtil::MakeShape(S32, {});
  BufferAllocation::Slice temp_slice(&allocations[0], 0, 4);
  BufferAllocation::Slice constant_slice(&allocations[1], 0, 4);
  BufferAllocation::Slice parameter_slice(&allocations[2], 0, 4);
  BufferAllocation::Slice live_out_slice(&allocations[3], 0, 4);
  BufferAllocation::Slice zero_sized_temp_slice(&allocations[4], 0, 0);

  emitters::KernelArgument constant_arg(shape, constant_slice);
  constant_arg.set_written(false);
  emitters::KernelArgument temp_arg(shape, temp_slice);
  emitters::KernelArgument parameter_arg(shape, parameter_slice);
  parameter_arg.set_written(false);
  emitters::KernelArgument live_out_arg(shape, live_out_slice);
  emitters::KernelArgument zero_sized_temp_arg(shape, zero_sized_temp_slice);

  ThunkSequence command_thunks = ThunkSequence::Of<KernelThunk>(
      ThunkInfoWithId(123),
      /*kernel_name=*/"test_kernel",
      /*kernel_arguments=*/
      emitters::KernelArguments(std::vector<emitters::KernelArgument>{
          constant_arg, temp_arg, parameter_arg, live_out_arg,
          zero_sized_temp_arg}),
      /*launch_dimensions=*/LaunchDimensions(),
      /*cluster_dim=*/std::nullopt,
      /*shmem_bytes=*/0,
      /*tma_metadata=*/se::gpu::TmaMetadata(),
      /*devices_per_host=*/1);

  ASSERT_OK_AND_ASSIGN(
      CommandExecutor commands,
      ConvertToCommands(command_thunks, ConvertToCommandsOptions{}));
  auto sequential_thunk = std::make_unique<SequentialThunk>(
      Thunk::ThunkInfo(), std::move(command_thunks));
  ThunkSequence thunk_sequence = ThunkSequence::Of<CommandBufferThunk>(
      std::move(commands), Thunk::ThunkInfo(), /*devices_in_process=*/1,
      std::move(sequential_thunk));
  ThunkExecutor thunk_executor(std::move(thunk_sequence));

  std::vector<const BufferAllocation*> allocation_ptrs;
  allocation_ptrs.reserve(allocations.size());
  for (const BufferAllocation& allocation : allocations) {
    allocation_ptrs.push_back(&allocation);
  }

  struct AllocationPolicy {
    size_t command_buffer_allocation_count;
    std::optional<std::vector<BufferAllocation::Index>>
        persistent_alloc_indices;
  };
  auto get_allocation_policy_without_vmm =
      [&](DebugOptions::CommandBufferUpdateMode update_mode)
      -> absl::StatusOr<AllocationPolicy> {
    DebugOptions debug_options;
    debug_options.set_xla_gpu_command_buffer_update_mode(update_mode);
    std::unique_ptr<GpuExecutableBufferAllocator> buffer_allocator =
        GpuExecutableBufferAllocator::Create("test", allocation_ptrs, shape,
                                             &debug_options, &thunk_executor,
                                             /*num_devices=*/1);
    ServiceExecutableRunOptions run_options;
    ABSL_ASSIGN_OR_RETURN(
        std::unique_ptr<GpuExecutableBufferAllocator::ExecutionScope>
            allocation_scope,
        buffer_allocator->CreateExecutionScope(
            &run_options, /*memory_allocator=*/nullptr, /*device_ordinal=*/0));

    std::vector<se::DeviceAddressBase> buffers;
    BufferAllocations buffer_allocations(buffers, /*device_ordinal=*/0,
                                         /*memory_allocator=*/nullptr);
    std::optional<std::vector<BufferAllocation::Index>> persistent_indices;
    ABSL_RETURN_IF_ERROR(allocation_scope->ExecuteWithBufferAllocations(
        buffer_allocations, /*device_ordinal=*/0,
        [&](const BufferAllocations&,
            std::optional<absl::Span<const BufferAllocation::Index>> indices) {
          if (indices.has_value()) {
            persistent_indices.emplace(indices->begin(), indices->end());
          }
          return absl::OkStatus();
        }));
    return AllocationPolicy{buffer_allocator->command_buffer_allocation_count(),
                            std::move(persistent_indices)};
  };

  ASSERT_OK_AND_ASSIGN(
      auto always_update_policy,
      get_allocation_policy_without_vmm(DebugOptions::ALWAYS_UPDATE));
  EXPECT_EQ(always_update_policy.command_buffer_allocation_count, 1);
  EXPECT_THAT(always_update_policy.persistent_alloc_indices,
              Optional(ElementsAre(1)));

  ASSERT_OK_AND_ASSIGN(auto skip_temp_policy, get_allocation_policy_without_vmm(
                                                  DebugOptions::SKIP_TEMP));
  EXPECT_EQ(skip_temp_policy.command_buffer_allocation_count, 2);
  EXPECT_THAT(skip_temp_policy.persistent_alloc_indices,
              Optional(ElementsAre(1)));

  // Without a VMM allocator, SKIP_PROFILED falls back to the base scope and
  // passes only the constant (1) as persistent.
  ASSERT_OK_AND_ASSIGN(
      auto skip_profiled_policy,
      get_allocation_policy_without_vmm(DebugOptions::SKIP_PROFILED));
  EXPECT_EQ(skip_profiled_policy.command_buffer_allocation_count, 2);
  EXPECT_THAT(skip_profiled_policy.persistent_alloc_indices,
              Optional(ElementsAre(1)));
}

TEST_F(GpuExecutableTest, ComputeComputationLayout) {
  GpuExecutable::Params params;
  params.module_name = "test_module";
  params.program_shape.AddParameter(ShapeUtil::MakeShape(F32, {1, 2, 3}), "p0");
  params.program_shape.AddParameter(ShapeUtil::MakeShape(U8, {1}), "p1");
  *params.program_shape.mutable_result() = ShapeUtil::MakeShape(F64, {2});
  params.executable = std::make_unique<ThunkExecutor>(ThunkSequence{});
  SetDummyBufferAssignment(params);
  params.allocations = std::vector<BufferAllocation>{BufferAllocation(0, 8, 0)};
  params.output_info[{}] = GpuExecutable::OutputInfo{/*allocation_index=*/0};

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GpuExecutable> executable,
                       GpuExecutable::Create(std::move(params)));
  EXPECT_THAT(executable->compute_computation_layout().parameter_layouts(),
              ElementsAre(ShapeLayout(ShapeUtil::MakeShape(F32, {1, 2, 3})),
                          ShapeLayout(ShapeUtil::MakeShape(U8, {1}))));
  EXPECT_EQ(executable->compute_computation_layout().result_layout(),
            ShapeLayout(ShapeUtil::MakeShape(F64, {2})));
}

TEST_F(GpuExecutableTest, ExecutableName) {
  GpuExecutable::Params params;
  params.module_name = "test_module";
  params.executable = std::make_unique<ThunkExecutor>(ThunkSequence{});
  SetDummyBufferAssignment(params);

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GpuExecutable> executable,
                       GpuExecutable::Create(std::move(params)));
  EXPECT_THAT(executable->name(), "test_module");
}

TEST_F(GpuExecutableTest, GetMlirAllocations) {
  GpuExecutable::Params params;
  params.module_name = "test_module";
  params.executable = std::make_unique<ThunkExecutor>(ThunkSequence{});

  std::vector<BufferAllocation> allocations;
  allocations.emplace_back(0, 1024, 0);
  allocations.emplace_back(1, 2048, 0);

  const BufferAllocation* expected_ptr0 = &allocations[0];
  const BufferAllocation* expected_ptr1 = &allocations[1];

  params.allocations = std::move(allocations);
  SetDummyBufferAssignment(params);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GpuExecutable> executable,
                       GpuExecutable::Create(std::move(params)));

  // The pointers must match exactly because the allocations may have Slice
  // objects which hold pointers to the parent allocations.
  EXPECT_THAT(executable->GetAllocations(),
              ElementsAre(expected_ptr0, expected_ptr1));
}

// A struct that owns a BufferAssignment and the HLO module it came from.
// Keeping the HLO module alive is necessary to keep the BufferAssignment
// valid.
struct ScopedBufferAssignment {
  std::unique_ptr<BufferAssignment> buffer_assignment;
  std::unique_ptr<HloModule> hlo_module;
};

absl::StatusOr<ScopedBufferAssignment> MakeNonEmptyBufferAssignment() {
  ScopedBufferAssignment holder;

  const char* hlo_text = R"(
    HloModule m
    ENTRY main {
      a = f32[128] parameter(0)
      b = f32[128] parameter(1)
      ROOT c = f32[128] add(a, b)
    })";
  ABSL_ASSIGN_OR_RETURN(holder.hlo_module, ParseAndReturnUnverifiedModule(hlo_text));

  AliasInfo alias_info;
  ABSL_ASSIGN_OR_RETURN(
      holder.buffer_assignment,
      BufferAssigner::Run(
          holder.hlo_module.get(),
          std::make_unique<DependencyHloOrdering>(holder.hlo_module.get()),
          [](const BufferValue& buffer) {
            return ShapeUtil::ByteSizeOf(buffer.shape(), sizeof(void*));
          },
          &alias_info, [](LogicalBuffer::Color) { return /*alignment=*/1; },
          BufferAssigner::Options{}));
  EXPECT_FALSE(holder.buffer_assignment->Allocations().empty());
  return holder;
}

TEST_F(GpuExecutableTest, GetBufferAssignmentAllocations) {
  ASSERT_OK_AND_ASSIGN(ScopedBufferAssignment holder,
                       MakeNonEmptyBufferAssignment());
  std::unique_ptr<BufferAssignment>& buffer_assignment =
      holder.buffer_assignment;

  GpuExecutable::Params params;
  params.module_name = "test_module";
  params.executable = std::make_unique<ThunkExecutor>(ThunkSequence{});

  params.buffer_allocations_debug_summary = buffer_assignment->ToVerboseString(
      params.alias_info.get(),
      params.debug_options.xla_debug_buffer_assignment_show_max());
  params.buffer_assignment_proto = buffer_assignment->ToProto();
  params.allocations = std::move(*buffer_assignment).TakeAllocations();

  std::vector<const BufferAllocation*> expected_allocs;
  expected_allocs.reserve(params.allocations.size());
  for (const auto& alloc : params.allocations) {
    expected_allocs.push_back(&alloc);
  }

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GpuExecutable> executable,
                       GpuExecutable::Create(std::move(params)));

  // The pointers must match exactly because the allocations may have Slice
  // objects which hold pointers to the parent allocations.
  EXPECT_THAT(executable->GetAllocations(), ElementsAreArray(expected_allocs));
}

TEST_F(GpuExecutableTest, ThunkChecksumPassAddsAllocation) {
  BufferAllocation alloc(0, 1024, 0);
  BufferAllocation::Slice slice(&alloc, 0, 1024);

  // Set up a thunk graph with a kernel that has some buffers that should be
  // checked, otherwise the pass is a no-op and doesn't need to allocate.
  auto make_test_thunk_sequence = [&]() {
    Thunk::ThunkInfo thunk_info;
    ThunkSequence thunk_sequence = ThunkSequence::Of<KernelThunk>(
        thunk_info,
        /*kernel_name=*/"test_kernel",
        /*kernel_arguments=*/
        emitters::KernelArguments({
            emitters::KernelArgument(
                ShapeUtil::MakeShape(F32, /*dimensions=*/{16}), slice),
        }),
        /*launch_dimensions=*/LaunchDimensions(),
        /*cluster_dim=*/std::nullopt,
        /*shmem_bytes=*/0,
        /*tma_metadata=*/se::gpu::TmaMetadata(),
        /*devices_per_host=*/1);
    return thunk_sequence;
  };
  auto make_test_hlo_module = []() {
    HloComputation::Builder builder("test_computation");
    HloInstruction* root = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0(1)));
    auto hlo_module =
        std::make_unique<HloModule>("test_module", HloModuleConfig());
    hlo_module->AddEntryComputation(builder.Build(/*root_instruction=*/root));
    return hlo_module;
  };

  GpuExecutable::Params params_without_pass;
  params_without_pass.debug_module = make_test_hlo_module();
  params_without_pass.executable =
      std::make_unique<ThunkExecutor>(make_test_thunk_sequence());
  SetDummyBufferAssignment(params_without_pass);

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GpuExecutable> executable_without_pass,
                       GpuExecutable::Create(std::move(params_without_pass)));
  size_t allocations_without_pass =
      executable_without_pass->GetAllocations().size();

  GpuExecutable::Params params_with_pass;
  params_with_pass.debug_module = make_test_hlo_module();
  params_with_pass.executable =
      std::make_unique<ThunkExecutor>(make_test_thunk_sequence());
  params_with_pass.debug_options
      .set_xla_gpu_experimental_enable_checksum_tracing_on_thunks(true);
  SetDummyBufferAssignment(params_with_pass);

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GpuExecutable> executable_with_pass,
                       GpuExecutable::Create(std::move(params_with_pass)));
  EXPECT_EQ(executable_with_pass->GetAllocations().size(),
            allocations_without_pass + 1);
}

TEST_F(GpuExecutableTest, DumpsMetadataListProto) {
  ASSERT_OK_AND_ASSIGN(
      tsl::testing::TemporaryDirectory dump_dir,
      tsl::testing::TemporaryDirectory::CreateForCurrentTestcase());
  DebugOptions debug_options = GetDebugOptionsFromFlags();
  debug_options.set_xla_dump_to(dump_dir.path());

  int execution_count = 0;
  auto create_executable = [&]() {
    BufferAllocation alloc(0, 1024, 0);
    BufferAllocation::Slice slice(&alloc, 0, 1024);
    Shape shape = ShapeUtil::MakeShape(S32, {256});

    ThunkSequence thunk_sequence;
    thunk_sequence.Emplace<KernelThunk>(
        ThunkInfoWithId(123),
        /*kernel_name=*/"test_kernel",
        /*kernel_arguments=*/emitters::KernelArguments({}),
        /*launch_dimensions=*/LaunchDimensions(),
        /*cluster_dim=*/std::nullopt,
        /*shmem_bytes=*/0,
        /*tma_metadata=*/se::gpu::TmaMetadata(),
        /*devices_per_host=*/1);
    thunk_sequence.Emplace<DeviceToDeviceCopyThunk>(
        ThunkInfoWithId(456), ShapedSlice{slice, shape},
        ShapedSlice{slice, shape}, 1024);

    GpuExecutable::Params params;
    params.executable =
        std::make_unique<ThunkExecutor>(std::move(thunk_sequence));
    params.debug_options = debug_options;

    params.module_name = absl::StrCat("test_module", execution_count++);
    se::DeviceDescription device_description;
    device_description.set_gpu_compute_capability(
        se::GpuComputeCapability{se::CudaComputeCapability::Volta()});
    device_description.set_driver_version({12, 3, 0});
    device_description.set_runtime_version({12, 3, 0});
    params.device_description = device_description;
    params.enable_debug_info_manager = false;
    params.debug_module =
        std::make_unique<HloModule>(params.module_name, HloModuleConfig());
    params.debug_module->mutable_config().set_debug_options(debug_options);
    SetDummyBufferAssignment(params);
    return GpuExecutable::Create(std::move(params));
  };

  ASSERT_OK(create_executable());

  std::vector<std::string> dump_files;
  ASSERT_OK(tsl::Env::Default()->GetMatchingPaths(
      tsl::io::JoinPath(dump_dir.path(), "*thunk_metadata.txt"), &dump_files));
  ASSERT_THAT(dump_files, SizeIs(1));

  ThunkMetadataListProto metadata_list_proto;
  ASSERT_OK(tsl::ReadTextProto(tsl::Env::Default(), dump_files.front(),
                               &metadata_list_proto));

  EXPECT_THAT(metadata_list_proto, EqualsProto(R"pb(
                thunk_metadata {
                  thunk_info { thunk_id: 123 }
                  thunk_kind: "kKernel"
                }
                thunk_metadata {
                  thunk_info { thunk_id: 456 }
                  thunk_kind: "kCopy"
                }
              )pb"));
}

TEST_F(GpuExecutableTest, ProtoConversion) {
  se::DeviceDescription device_description;
  device_description.set_gpu_compute_capability(
      se::GpuComputeCapability{se::CudaComputeCapability::Volta()});
  device_description.set_driver_version({12, 3, 0});
  device_description.set_runtime_version({12, 3, 0});

  ThunkSequence thunk_sequence = ThunkSequence::Of<KernelThunk>(
      ThunkInfoWithId(123),
      /*kernel_name=*/"test_kernel", emitters::KernelArguments({}),
      LaunchDimensions(),
      /*cluster_dim=*/std::nullopt,
      /*shmem_bytes=*/0, se::gpu::TmaMetadata(),
      /*devices_per_host=*/1);

  GpuExecutable::Params params;
  params.binary = {1, 2, 3};
  params.dnn_compiled_graphs = {{"test_dnn_compiled_graph", "test_json"}};

  params.executable =
      std::make_unique<ThunkExecutor>(std::move(thunk_sequence));
  params.device_description = device_description;

  params.module_name = "test_module";
  params.enable_debug_info_manager = false;
  params.allocations = {BufferAllocation(0, 1024, 0)};
  params.cpu_target_machine_options = xla::cpu::TargetMachineOptions(
      "test_triple", "test_cpu", "+test_features");
  params.gpu_topology =
      GpuTopology("test_platform", /*num_partitions=*/1,
                  /*num_hosts_per_partition=*/2, /*num_devices_per_host=*/4);
  SetDummyBufferAssignment(params);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GpuExecutable> reference_executable,
                       GpuExecutable::Create(std::move(params)));
  ASSERT_OK_AND_ASSIGN(GpuExecutableProto proto,
                       reference_executable->ToProto());

  DebugOptions debug_options = GetDebugOptionsFromFlags();
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GpuExecutable> reconstructed_executable,
      GpuExecutable::FromProto(proto, device_description, "TEST_PLATFORM",
                               debug_options));
  EXPECT_THAT(reconstructed_executable->binary(), ElementsAre(1, 2, 3));
  EXPECT_THAT(
      reconstructed_executable->dnn_compiled_graphs(),
      UnorderedElementsAre(Pair("test_dnn_compiled_graph", "test_json")));
  EXPECT_THAT(reconstructed_executable->thunk_executor().thunks(),
              ElementsAre(Pointee(Property(&Thunk::kind, Thunk::kKernel))));
  EXPECT_THAT(reconstructed_executable->GetAllocations(),
              ElementsAre(Pointee(Property(&BufferAllocation::size, 1024))));
  EXPECT_THAT(reconstructed_executable->name(), "test_module");
  ASSERT_TRUE(
      reconstructed_executable->cpu_target_machine_options().has_value());
  EXPECT_EQ(reconstructed_executable->cpu_target_machine_options().value(),
            xla::cpu::TargetMachineOptions("test_triple", "test_cpu",
                                           "+test_features"));
  ASSERT_TRUE(reconstructed_executable->gpu_topology().has_value());
  EXPECT_EQ(reconstructed_executable->gpu_topology()->platform_version(),
            "test_platform");
  EXPECT_EQ(reconstructed_executable->gpu_topology()->number_of_hosts(), 2);
  EXPECT_EQ(reconstructed_executable->gpu_topology()->num_devices_per_host(),
            4);
}

TEST_F(GpuExecutableTest, FromProtoWithoutGpuTopology) {
  se::DeviceDescription device_description;
  device_description.set_gpu_compute_capability(
      se::GpuComputeCapability{se::CudaComputeCapability::Volta()});
  device_description.set_driver_version({12, 3, 0});
  device_description.set_runtime_version({12, 3, 0});

  GpuExecutable::Params params;
  params.executable =
      std::make_unique<ThunkExecutor>(ThunkSequence::Of<KernelThunk>(
          ThunkInfoWithId(123),
          /*kernel_name=*/"test_kernel", emitters::KernelArguments({}),
          LaunchDimensions(),
          /*cluster_dim=*/std::nullopt,
          /*shmem_bytes=*/0, se::gpu::TmaMetadata(),
          /*devices_per_host=*/1));
  params.device_description = device_description;
  params.module_name = "test_module";
  params.enable_debug_info_manager = false;
  SetDummyBufferAssignment(params);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GpuExecutable> reference_executable,
                       GpuExecutable::Create(std::move(params)));
  ASSERT_OK_AND_ASSIGN(GpuExecutableProto proto,
                       reference_executable->ToProto());
  // Executables serialized before the `gpu_topology` field was introduced
  // don't have it set. Loading them must still succeed.
  proto.clear_gpu_topology();

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GpuExecutable> reconstructed_executable,
      GpuExecutable::FromProto(proto, device_description, "TEST_PLATFORM",
                               GetDebugOptionsFromFlags()));
  EXPECT_FALSE(reconstructed_executable->gpu_topology().has_value());
}

TEST_F(GpuExecutableTest, ToProtoFailsWithoutGpuTopology) {
  se::DeviceDescription device_description;
  device_description.set_gpu_compute_capability(
      se::GpuComputeCapability{se::CudaComputeCapability::Volta()});

  GpuExecutable::Params params;
  params.module_name = "test_module";
  params.executable = std::make_unique<ThunkExecutor>(ThunkSequence{});
  params.device_description = device_description;
  params.enable_debug_info_manager = false;
  SetDummyBufferAssignment(params);
  params.gpu_topology = std::nullopt;

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GpuExecutable> executable,
                       GpuExecutable::Create(std::move(params)));
  EXPECT_THAT(executable->ToProto(),
              absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST_F(GpuExecutableTest, ProtoConversionWithBackendConfigInterning) {
  se::DeviceDescription device_description;
  device_description.set_gpu_compute_capability(
      se::GpuComputeCapability{se::CudaComputeCapability::Volta()});

  GpuExecutable::Params params;
  params.module_name = "test_module";
  params.executable = std::make_unique<ThunkExecutor>(ThunkSequence{});
  params.device_description = device_description;
  params.enable_debug_info_manager = false;

  // Create a debug module with some instructions sharing backend config using
  // parser.
  const char* hlo_text = R"(
    HloModule test_module
    ENTRY comp {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add = f32[] add(p0, p1)
    })";
  ASSERT_OK_AND_ASSIGN(auto debug_module,
                       ParseAndReturnUnverifiedModule(hlo_text));
  HloInstruction* p0 =
      debug_module->entry_computation()->GetInstructionWithName("p0");
  HloInstruction* p1 =
      debug_module->entry_computation()->GetInstructionWithName("p1");

  p0->set_raw_backend_config_string("tokamax:{\"data\": 1}");
  p1->CopyBackendConfigFrom(p0);  // Force in-memory sharing to test fast path.
  params.debug_module = std::move(debug_module);
  SetDummyBufferAssignment(params);

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GpuExecutable> executable,
                       GpuExecutable::Create(std::move(params)));
  ASSERT_OK_AND_ASSIGN(GpuExecutableProto proto, executable->ToProto());

  // Verify that the serialized HLO module has interned backend configs using
  // Partially(EqualsProto).
  EXPECT_THAT(proto, Partially(EqualsProto(R"pb(
                hlo_module_with_config {
                  hlo_module {
                    payloads: "tokamax:{\"data\": 1}"
                    computations {
                      instructions {
                        name: "p0"
                        backend_config_payload { id: 0 }
                        backend_config: ""
                      }
                      instructions {
                        name: "p1"
                        backend_config_payload { id: 0 }
                        backend_config: ""
                      }
                      instructions { name: "add" }
                    }
                  }
                }
              )pb")));
}

TEST_F(GpuExecutableTest, GpuExecutableDump) {
  tsl::Env* env = tsl::Env::Default();

  DebugOptions debug_options = GetDebugOptionsFromFlags();
  debug_options.set_xla_gpu_experimental_dump_gpu_executable(true);
  ASSERT_OK_AND_ASSIGN(TemporaryDirectory temp_dir,
                       TemporaryDirectory::CreateForCurrentTestcase());
  debug_options.set_xla_dump_to(temp_dir.path());
  debug_options.set_xla_enable_dumping(true);

  BufferAllocation alloc(0, 1024, 0);
  auto create_executable = [&]() {
    ThunkSequence thunk_sequence;
    BufferAllocation::Slice slice(&alloc, 0, 1024);
    Shape shape = ShapeUtil::MakeShape(S32, {256});

    thunk_sequence.Emplace<KernelThunk>(
        ThunkInfoWithId(123),
        /*kernel_name=*/"test_kernel",
        /*kernel_arguments=*/emitters::KernelArguments({}),
        /*launch_dimensions=*/LaunchDimensions(),
        /*cluster_dim=*/std::nullopt,
        /*shmem_bytes=*/0,
        /*tma_metadata=*/se::gpu::TmaMetadata(),
        /*devices_per_host=*/1);
    thunk_sequence.Emplace<DeviceToDeviceCopyThunk>(
        ThunkInfoWithId(456), ShapedSlice{slice, shape},
        ShapedSlice{slice, shape}, 1024);

    GpuExecutable::Params params;
    params.executable =
        std::make_unique<ThunkExecutor>(std::move(thunk_sequence));
    params.debug_options = debug_options;

    params.module_name = "test_module";
    params.debug_module =
        std::make_unique<HloModule>(params.module_name, HloModuleConfig());
    params.debug_module->mutable_config().set_debug_options(debug_options);
    SetDummyBufferAssignment(params);
    return GpuExecutable::Create(std::move(params));
  };
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GpuExecutable> executable,
                       create_executable());
  ExecutableBuildOptions build_options;
  build_options.set_num_replicas(2);
  build_options.set_num_partitions(1);
  // Thread pool is not serializable, and should be ignored in the dump.
  tsl::thread::ThreadPool pool(tsl::Env::Default(), "test_pool", 1);
  build_options.set_compile_thread_pool(&pool);
  ASSERT_OK(executable->DumpExecutableIfEnabled(build_options, debug_options));

  std::vector<std::string> dump_files;
  ASSERT_OK(env->GetMatchingPaths(
      tsl::io::JoinPath(debug_options.xla_dump_to(), "*gpu_executable.riegeli"),
      &dump_files));
  ASSERT_EQ(dump_files.size(), 1);

  ExecutableAndOptionsProto dump_content;
  auto reader = std::make_unique<riegeli::CFileReader<>>(dump_files[0]);
  ASSERT_OK(ReadSplitProto(std::move(reader), dump_content));
  EXPECT_THAT(dump_content.compile_options().executable_build_options(),
              Partially(EqualsProto(R"pb(
                num_replicas: 2 num_partitions: 1
              )pb")));

  GpuExecutableProto gpu_executable_proto;
  auto executable_reader = std::make_unique<riegeli::StringReader<>>(
      dump_content.serialized_executable());
  ASSERT_OK(ReadSplitProto(std::move(executable_reader), gpu_executable_proto));
  ASSERT_THAT(gpu_executable_proto, Partially(EqualsProto(R"pb(
                module_name: "test_module"
                thunks: {
                  thunk_info: { thunk_id: 123 }
                  kernel_thunk: { kernel_name: "test_kernel" }
                }
                thunks: {
                  thunk_info: { thunk_id: 456 }
                  device_to_device_copy_thunk: {}
                }
              )pb")));
}

void* InventPointerToCudaKernel(uint64_t address) {
  return reinterpret_cast<void*>(address);
}

TEST_F(GpuExecutableTest, FromProtoWithSymbolResolver) {
  const auto proto = ParseTextProtoOrDie<GpuExecutableProto>(R"pb(
    module_name: "test_module"
    buffer_assignment {}
    buffer_allocations_debug_summary: "dummy"
    gpu_compute_capability: {
      cuda_compute_capability: { major: 9 minor: 0 feature_extension: NONE }
    }
    thunks {
      thunk_info { thunk_id: 2 }
      custom_kernel_thunk {
        custom_kernel {
          kernel_spec {
            in_process_symbol { persistent_name: "persistent_kernel_name" }
            kernel_name: "kernel_name"
            arity: 42
            kernel_args_packing_spec {
              kernel_arguments {
                relocations { kind: KIND_BITS64_ABSOLUTE argument_index: 0 }
              }
              kernel_arguments { data: "\x34\x12\x00\x00" }
            }
          }
          block_dims { coordinates { x: 1 y: 1 z: 1 } }
          thread_dims { coordinates { x: 1 y: 1 z: 1 } }
          cluster_dim { coordinates { x: 1 y: 1 z: 1 } }
        }
      }
    }
  )pb");

  void* const kCudaSymbol = InventPointerToCudaKernel(0x1234567890);

  stream_executor::DeviceDescription device_description;
  device_description.set_gpu_compute_capability(
      se::GpuComputeCapability{se::CudaComputeCapability::Hopper()});

  int symbol_resolver_invocations = 0;
  const auto symbol_resolver = [&](absl::string_view name) {
    EXPECT_EQ(name, "persistent_kernel_name");
    ++symbol_resolver_invocations;
    return kCudaSymbol;
  };

  DebugOptions debug_options = GetDebugOptionsFromFlags();
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GpuExecutable> executable,
      GpuExecutable::FromProto(proto, device_description, "TEST_PLATFORM",
                               debug_options, symbol_resolver));

  const CustomKernelThunk* custom_kernel_thunk =
      dynamic_cast<const CustomKernelThunk*>(
          executable->thunk_executor().thunks().front().get());
  ASSERT_NE(custom_kernel_thunk, nullptr);
  EXPECT_THAT(
      custom_kernel_thunk->custom_kernel().kernel_spec().in_process_symbol(),
      Optional(Field(&stream_executor::InProcessSymbol::symbol, kCudaSymbol)));
  EXPECT_EQ(symbol_resolver_invocations, 1);
}

TEST_F(GpuExecutableTest, ToProtoReturnsUnchangedThunkGraph) {
  DebugOptions debug_options = GetDebugOptionsFromFlags();
  debug_options.set_xla_gpu_graph_min_graph_size(1);
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::FUSION);

  auto create_executable = [&]() {
    ThunkSequence thunk_sequence;
    thunk_sequence.Emplace<KernelThunk>(
        ThunkInfoWithId(1),
        /*kernel_name=*/"test_kernel_0",
        /*kernel_arguments=*/emitters::KernelArguments({}),
        /*launch_dimensions=*/LaunchDimensions(),
        /*cluster_dim=*/std::nullopt,
        /*shmem_bytes=*/0,
        /*tma_metadata=*/se::gpu::TmaMetadata(),
        /*devices_per_host=*/1);
    thunk_sequence.Emplace<KernelThunk>(
        ThunkInfoWithId(2),
        /*kernel_name=*/"test_kernel_1",
        /*kernel_arguments=*/emitters::KernelArguments({}),
        /*launch_dimensions=*/LaunchDimensions(),
        /*cluster_dim=*/std::nullopt,
        /*shmem_bytes=*/0,
        /*tma_metadata=*/se::gpu::TmaMetadata(),
        /*devices_per_host=*/1);
    thunk_sequence.Emplace<KernelThunk>(
        ThunkInfoWithId(3),
        /*kernel_name=*/"test_kernel_2",
        /*kernel_arguments=*/emitters::KernelArguments({}),
        /*launch_dimensions=*/LaunchDimensions(),
        /*cluster_dim=*/std::nullopt,
        /*shmem_bytes=*/0,
        /*tma_metadata=*/se::gpu::TmaMetadata(),
        /*devices_per_host=*/1);
    thunk_sequence.Emplace<KernelThunk>(
        ThunkInfoWithId(4),
        /*kernel_name=*/"test_kernel_3",
        /*kernel_arguments=*/emitters::KernelArguments({}),
        /*launch_dimensions=*/LaunchDimensions(),
        /*cluster_dim=*/std::nullopt,
        /*shmem_bytes=*/0,
        /*tma_metadata=*/se::gpu::TmaMetadata(),
        /*devices_per_host=*/1);
    thunk_sequence.Emplace<KernelThunk>(
        ThunkInfoWithId(5),
        /*kernel_name=*/"test_kernel_4",
        /*kernel_arguments=*/emitters::KernelArguments({}),
        /*launch_dimensions=*/LaunchDimensions(),
        /*cluster_dim=*/std::nullopt,
        /*shmem_bytes=*/0,
        /*tma_metadata=*/se::gpu::TmaMetadata(),
        /*devices_per_host=*/1);

    GpuExecutable::Params params;
    params.executable =
        std::make_unique<ThunkExecutor>(std::move(thunk_sequence));
    params.debug_options = debug_options;

    params.module_name = "test_module";
    // Command buffers require CUDA 12.3.
    se::DeviceDescription device_description;
    device_description.set_gpu_compute_capability(
        se::GpuComputeCapability{se::CudaComputeCapability::Volta()});
    device_description.set_driver_version({12, 3, 0});
    device_description.set_runtime_version({12, 3, 0});
    params.device_description = device_description;
    SetDummyBufferAssignment(params);
    return GpuExecutable::Create(std::move(params));
  };
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GpuExecutable> executable,
                       create_executable());

  // We expect our 5 kernel launches got wrapped in a command buffer thunk.
  // If this assertion fails, you might need to either adjust the thunk graph or
  // the debug options such that we do some kind of thunk graph transformation
  // that we can test for.
  ASSERT_THAT(executable->thunk_executor().thunks(), SizeIs(1));

  // The proto should be a straight dump of the thunk graph, without any
  // transformation.
  ASSERT_OK_AND_ASSIGN(GpuExecutableProto proto, executable->ToProto());
  EXPECT_THAT(proto.thunks(), SizeIs(5));
}

TEST_F(GpuExecutableTest, FromProtoRegistersHloModuleWithDebugInfoManager) {
  const auto proto = ParseTextProtoOrDie<GpuExecutableProto>(R"pb(
    module_name: "test_module"
    buffer_assignment {}
    buffer_allocations_debug_summary: "dummy"
    gpu_compute_capability: {
      cuda_compute_capability: { major: 9 minor: 0 feature_extension: NONE }
    }
    hlo_module_with_config {
      config {
        entry_computation_layout {
          parameters { element_type: F32 dimensions: 1 }
          parameter_names: "parameter"
        }
      }
      hlo_module {
        name: "test_module"
        entry_computation_name: "test_computation"
        host_program_shape {
          parameters { element_type: F32 dimensions: 1 }
          parameter_names: "parameter"
        }
        computations {
          name: "test_computation"
          instructions {
            name: "test_instruction"
            opcode: "parameter"
            shape: { element_type: F32 dimensions: 1 }
          }
          instructions {
            name: "constant"
            id: 1
            opcode: "constant"
            shape: {
              element_type: U8
              layout { format: DENSE minor_to_major: 0 }
              dimensions: 4
            }
            literal: {
              shape: {
                element_type: U8
                layout { format: DENSE minor_to_major: 0 }
                dimensions: 4
              }
              u8s: "data"
            }
          }
        }
      }
    }
    buffer_allocations { values { index: 0 size: 4 } }
    constants {
      symbol_name: "buffer_for_constant"
      content { data: "data" }
      allocation_index: 0
    }
    thunks {
      thunk_info { thunk_id: 2 }
      host_send_done_thunk { channel_id: 123 async_events_unique_id: 1 }
    }
  )pb");

  stream_executor::DeviceDescription device_description;
  device_description.set_gpu_compute_capability(
      se::GpuComputeCapability{se::CudaComputeCapability::Hopper()});

  DebugOptions debug_options = GetDebugOptionsFromFlags();
  debug_options.set_xla_gpu_executable_embed_debug_info(true);
  debug_options.set_xla_gpu_experimental_nccl_window_symmetric(true);

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GpuExecutable> executable,
      GpuExecutable::FromProto(proto, device_description, "TEST_PLATFORM",
                               debug_options));
  ASSERT_TRUE(executable->has_module());
  EXPECT_EQ(executable->constants().size(), 1);
  EXPECT_TRUE(executable->module_config()
                  .debug_options()
                  .xla_gpu_experimental_nccl_window_symmetric());

  EXPECT_TRUE(XlaDebugInfoManager::Get()->TracksModule(
      executable->module().unique_id()));

  debug_options.set_xla_gpu_executable_embed_debug_info(false);
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GpuExecutable> executable_without_debug_info_manager,
      GpuExecutable::FromProto(proto, device_description, "TEST_PLATFORM",
                               debug_options));
  ASSERT_TRUE(executable_without_debug_info_manager->has_module());
  EXPECT_FALSE(XlaDebugInfoManager::Get()->TracksModule(
      executable_without_debug_info_manager->module().unique_id()));
}

TEST_F(GpuExecutableTest, ConstantContentOnlySerializedInHloModule) {
  const auto proto = ParseTextProtoOrDie<GpuExecutableProto>(R"pb(
    module_name: "test_module"
    buffer_assignment {}
    buffer_allocations_debug_summary: "dummy"
    gpu_compute_capability: {
      cuda_compute_capability: { major: 9 minor: 0 feature_extension: NONE }
    }
    hlo_module_with_config {
      config {
        entry_computation_layout {
          parameters { element_type: F32 dimensions: 1 }
          parameter_names: "parameter"
        }
      }
      hlo_module {
        name: "test_module"
        entry_computation_name: "test_computation"
        host_program_shape {
          parameters { element_type: F32 dimensions: 1 }
          parameter_names: "parameter"
        }
        computations {
          name: "test_computation"
          instructions {
            name: "test_instruction"
            opcode: "parameter"
            shape: { element_type: F32 dimensions: 1 }
          }
          instructions {
            name: "constant"
            id: 1
            opcode: "constant"
            shape: {
              element_type: U8
              layout { format: DENSE minor_to_major: 0 }
              dimensions: 4
            }
            literal: {
              shape: {
                element_type: U8
                layout { format: DENSE minor_to_major: 0 }
                dimensions: 4
              }
              u8s: "data"
            }
          }
        }
      }
    }
    buffer_allocations { values { index: 0 size: 4 } }
    constants { symbol_name: "buffer_for_constant" allocation_index: 0 }
    thunks {
      thunk_info { thunk_id: 2 }
      host_send_done_thunk { channel_id: 123 async_events_unique_id: 1 }
    }
    gpu_topology {
      num_partitions: 1
      num_hosts_per_partition: 1
      num_devices_per_host: 1
    }
  )pb");

  stream_executor::DeviceDescription device_description;
  device_description.set_gpu_compute_capability(
      se::GpuComputeCapability{se::CudaComputeCapability::Hopper()});

  DebugOptions debug_options = GetDebugOptionsFromFlags();
  debug_options.set_xla_gpu_executable_embed_debug_info(true);

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<GpuExecutable> executable,
      GpuExecutable::FromProto(proto, device_description, "TEST_PLATFORM",
                               debug_options));
  ASSERT_TRUE(executable->has_module());
  EXPECT_EQ(executable->constants().size(), 1);

  EXPECT_EQ(ToString(executable->constants()[0].content.span()), "data");

  ASSERT_OK_AND_ASSIGN(auto serialized, executable->ToProto());
  EXPECT_EQ(serialized.constants().size(), 1);
  EXPECT_THAT(serialized.constants()[0], EqualsProto(proto.constants()[0]));
}

TEST_F(GpuExecutableTest, ExecutableAbiVersion) {
  stream_executor::ExecutableAbiVersionProto executable_abi_version_proto;
  executable_abi_version_proto.set_platform_name("TEST_PLATFORM");
  executable_abi_version_proto.mutable_cuda_platform_version()
      ->set_cuda_toolkit_version("9.9.9");

  GpuExecutable::Params params;
  ASSERT_OK_AND_ASSIGN(params.executable_abi_version,
                       stream_executor::ExecutableAbiVersion::FromProto(
                           executable_abi_version_proto));
  params.executable = std::make_unique<ThunkExecutor>(ThunkSequence{});
  SetDummyBufferAssignment(params);

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GpuExecutable> executable,
                       GpuExecutable::Create(std::move(params)));
  ASSERT_OK_AND_ASSIGN(
      stream_executor::ExecutableAbiVersion executable_abi_version,
      executable->GetExecutableAbiVersion());
  EXPECT_THAT(executable_abi_version.proto(),
              EqualsProto(executable_abi_version_proto));

  ASSERT_OK_AND_ASSIGN(GpuExecutableProto proto, executable->ToProto());
  EXPECT_THAT(proto.executable_abi_version(),
              EqualsProto(executable_abi_version_proto));
}

class FakeDeviceAddressAllocator : public se::DeviceAddressAllocator {
 public:
  explicit FakeDeviceAddressAllocator(const se::Platform* platform)
      : se::DeviceAddressAllocator(platform) {}
  absl::StatusOr<se::ScopedDeviceAddress<uint8_t>> Allocate(
      int device_ordinal, uint64_t size, bool retry_on_failure,
      int64_t memory_space) override {
    return se::ScopedDeviceAddress<uint8_t>();
  }
  absl::Status Deallocate(int device_ordinal,
                          se::DeviceAddressBase mem) override {
    return absl::OkStatus();
  }
  bool AllowsAsynchronousDeallocation() const override { return true; }
  absl::StatusOr<se::Stream*> GetStream(int device_ordinal) override {
    return nullptr;
  }
};

// Mimics Exec 1's CommandBufferThunk::ExecuteOnStream:
// - Device 0 (slow) evicts its command buffer cache and calls BeginCapture
//   (acquiring `driver_mutex`) before submitting the collective to the stream.
// - Device 1 (fast) hits its cache and submits the collective immediately.
class Exec1CollectiveThunk : public Thunk {
 public:
  Exec1CollectiveThunk(ThunkInfo thunk_info, absl::Mutex* driver_mutex,
                       absl::Notification* d1_finished_exec1_thunk,
                       absl::Notification* d1_entered_exec2,
                       absl::BlockingCounter* exec1_collective_submitted,
                       absl::Notification* exec1_collective_done)
      : Thunk(Kind::kCommandBuffer, std::move(thunk_info)),
        driver_mutex_(driver_mutex),
        d1_finished_exec1_thunk_(d1_finished_exec1_thunk),
        d1_entered_exec2_(d1_entered_exec2),
        exec1_collective_submitted_(exec1_collective_submitted),
        exec1_collective_done_(exec1_collective_done) {}

  absl::Status Prepare(const PrepareParams& params) override {
    return params.collective_clique_requests->RequestClique(
        GpuCliqueKey({GlobalDeviceId(0), GlobalDeviceId(1)},
                     /*num_local_participants=*/2),
        /*device_groups=*/{{GlobalDeviceId(0), GlobalDeviceId(1)}});
  }

  absl::Status ExecuteOnStream(const ExecuteParams& params) override {
    if (params.stream->parent()->device_ordinal() == 0) {
      d1_finished_exec1_thunk_->WaitForNotification();
      // Give device 1 a window to race into exec2 if exec1 does not
      // rendezvous after ExecuteOnStream.
      d1_entered_exec2_->WaitForNotificationWithTimeout(absl::Milliseconds(50));
      // Mimic cudaStreamBeginCapture acquiring libcuda's global lock before
      // submitting the command buffer containing the collective.
      absl::MutexLock lock(*driver_mutex_);
      if (exec1_collective_submitted_->DecrementCount()) {
        exec1_collective_done_->Notify();
      }
    } else {
      if (exec1_collective_submitted_->DecrementCount()) {
        exec1_collective_done_->Notify();
      }
      d1_finished_exec1_thunk_->Notify();
    }
    return absl::OkStatus();
  }

  BufferUses buffer_uses() const override { return {}; }

  absl::StatusOr<ThunkProto> ToProto() const override { return ThunkProto(); }

 private:
  absl::Mutex* driver_mutex_;
  absl::Notification* d1_finished_exec1_thunk_;
  absl::Notification* d1_entered_exec2_;
  absl::BlockingCounter* exec1_collective_submitted_;
  absl::Notification* exec1_collective_done_;
};

TEST_F(GpuExecutableTest,
       RendezvousAfterExecutionPreventsDriverLockDeadlockAcrossExecutables) {
  ASSERT_OK_AND_ASSIGN(
      se::Platform * platform,
      se::PlatformManager::PlatformWithId(stream_executor_platform_id()));

  constexpr int kNumDevices = 2;
  absl::Mutex driver_mutex;
  absl::Notification d1_finished_exec1_thunk;
  absl::Notification d1_entered_exec2;
  absl::BlockingCounter exec1_collective_submitted(kNumDevices);
  absl::Notification exec1_collective_done;
  std::atomic<int> d1_activate_calls{0};

  se::MockStreamExecutor executors[kNumDevices];
  se::MockStream streams[kNumDevices];
  const se::GpuComputeCapability& compute_capability =
      device_description().gpu_compute_capability();

  ThunkSequence thunks1;
  thunks1.push_back(std::make_unique<Exec1CollectiveThunk>(
      ThunkInfoWithId(1), &driver_mutex, &d1_finished_exec1_thunk,
      &d1_entered_exec2, &exec1_collective_submitted, &exec1_collective_done));

  GpuExecutable::Params params1;
  params1.module_name = "exec1";
  params1.executable = std::make_unique<ThunkExecutor>(std::move(thunks1));
  params1.device_description = device_description();
  params1.enable_debug_info_manager = false;
  SetDummyBufferAssignment(params1);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GpuExecutable> exec1,
                       GpuExecutable::Create(std::move(params1)));

  // exec2 has a non-empty binary so ResolveConstantGlobals calls
  // executor->LoadModule.
  GpuExecutable::Params params2;
  params2.module_name = "exec2";
  params2.binary = {1, 2, 3, 4};
  params2.executable = std::make_unique<ThunkExecutor>(ThunkSequence{});
  params2.device_description = device_description();
  params2.enable_debug_info_manager = false;
  SetDummyBufferAssignment(params2);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GpuExecutable> exec2,
                       GpuExecutable::Create(std::move(params2)));

  DeviceAssignment device_assn(/*replica_count=*/kNumDevices,
                               /*computation_count=*/1);
  device_assn.FillIota(0);
  LoopbackCollectives loopback_collectives;
  GpuExecutableRunOptions gpu_opts;
  gpu_opts.set_collectives(&loopback_collectives);
  FakeDeviceAddressAllocator allocator(platform);

  tsl::thread::ThreadPool pool(tsl::Env::Default(), "driver_lock_deadlock_test",
                               kNumDevices);
  absl::BlockingCounter done(kNumDevices);
  for (int d = 0; d < kNumDevices; ++d) {
    EXPECT_CALL(executors[d], GetPlatform()).WillRepeatedly(Return(platform));
    EXPECT_CALL(executors[d], device_ordinal()).WillRepeatedly(Return(d));
    EXPECT_CALL(executors[d], SynchronizeAllActivity())
        .WillRepeatedly(Return(true));
    EXPECT_CALL(executors[d], EnablePeerAccessTo(::testing::_))
        .WillRepeatedly(Return(absl::OkStatus()));
    EXPECT_CALL(executors[d], Activate()).WillRepeatedly([&, d]() {
      // Device 1's second Activate() call is its entry into exec2.
      if (d == 1 && d1_activate_calls.fetch_add(1) == 1) {
        d1_entered_exec2.Notify();
      }
      return nullptr;
    });
    // Mimic cuModuleLoadData in exec2's ResolveConstantGlobals holding
    // `driver_mutex` while waiting for in-flight GPU work (exec1's collective)
    // to finish on the device.
    EXPECT_CALL(executors[d], LoadModule(::testing::_))
        .WillRepeatedly([&](const se::MultiModuleLoaderSpec&) {
          absl::MutexLock lock(driver_mutex);
          exec1_collective_done.WaitForNotification();
          return se::ModuleHandle(reinterpret_cast<void*>(0x1));
        });
    EXPECT_CALL(executors[d], UnloadModule(::testing::_))
        .WillRepeatedly(Return(true));
    EXPECT_CALL(streams[d], parent()).WillRepeatedly(Return(&executors[d]));
    if (const se::CudaComputeCapability* cuda_cc =
            compute_capability.cuda_compute_capability()) {
      EXPECT_CALL(streams[d], GetCudaComputeCapability())
          .WillRepeatedly(Return(*cuda_cc));
    }
    if (const se::RocmComputeCapability* rocm_cc =
            compute_capability.rocm_compute_capability()) {
      EXPECT_CALL(streams[d], GetRocmComputeCapability())
          .WillRepeatedly(Return(*rocm_cc));
    }

    pool.Schedule([&, d]() {
      auto run = [&](GpuExecutable& exec, int run_id) {
        ExecutableRunOptions exec_run_options;
        exec_run_options.set_stream(&streams[d]);
        exec_run_options.set_allocator(&allocator);
        exec_run_options.set_device_assignment(&device_assn);
        exec_run_options.set_gpu_executable_run_options(&gpu_opts);
        exec_run_options.set_run_id(RunId(run_id));
        ServiceExecutableRunOptions run_opts(exec_run_options);
        EXPECT_OK(exec.ExecuteAsyncOnStream(&run_opts,
                                            std::vector<ExecutionInput>{}));
      };

      run(*exec1, 100);
      run(*exec2, 101);
      done.DecrementCount();
    });
  }
  done.Wait();
}

int64_t BufferSizeBytes(const BufferValue& buffer) {
  return ShapeUtil::ByteSizeOf(buffer.shape(), sizeof(void*));
}

}  // namespace
}  // namespace xla::gpu
