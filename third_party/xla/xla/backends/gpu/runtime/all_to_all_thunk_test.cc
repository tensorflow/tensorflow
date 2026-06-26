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

#include "xla/backends/gpu/runtime/all_to_all_thunk.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/btree_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/types/span.h"
#include "absl/status/status_macros.h"
#include "xla/backends/gpu/collectives/cancellation_token.h"
#include "xla/backends/gpu/collectives/gpu_clique.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_cliques.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/backends/gpu/runtime/collective_cliques.h"
#include "xla/backends/gpu/runtime/collective_execution.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/command_state.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/future.h"
#include "xla/runtime/device_id.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/computation_placer.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/platform_util.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"
#include "xla/tsl/util/proto/parse_text_proto.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

using ::tsl::proto_testing::EqualsProto;

static se::StreamExecutor* GpuExecutor() {
  auto name =
      absl::AsciiStrToUpper(PlatformUtil::CanonicalPlatformName("gpu").value());
  auto* platform = se::PlatformManager::PlatformWithName(name).value();
  return platform->ExecutorForDevice(0).value();
}

// Child command nodes (CreateChildCommand / UpdateChildCommand) require
// CUDA 12.9+ driver and toolkit.
static bool IsAtLeastCuda12900(const se::StreamExecutor* executor) {
  const auto& desc = executor->GetDeviceDescription();
  const auto* cuda_cc = desc.gpu_compute_capability().cuda_compute_capability();
  if (cuda_cc == nullptr) {
    return false;
  }
  return std::min(desc.driver_version(), desc.compile_time_toolkit_version()) >=
         se::SemanticVersion(12, 9, 0);
}

class FakeGpuCommunicator : public GpuCommunicator {
 public:
  absl::Status LaunchAllReduce(se::DeviceAddressBase, se::DeviceAddressBase,
                               PrimitiveType, size_t, ReductionKind,
                               const Executor&) override {
    return absl::UnimplementedError("unused");
  }

  absl::Status LaunchBroadcast(se::DeviceAddressBase, se::DeviceAddressBase,
                               PrimitiveType, size_t, RankId,
                               const Executor&) override {
    return absl::UnimplementedError("unused");
  }

  absl::Status LaunchReduceScatter(se::DeviceAddressBase, se::DeviceAddressBase,
                                   PrimitiveType, size_t, ReductionKind,
                                   const Executor&) override {
    return absl::UnimplementedError("unused");
  }

  absl::Status LaunchAllGather(se::DeviceAddressBase, se::DeviceAddressBase,
                               PrimitiveType, size_t,
                               const Executor&) override {
    return absl::UnimplementedError("unused");
  }

  absl::Status LaunchAllToAll(absl::InlinedVector<se::DeviceAddressBase, 4>,
                              absl::InlinedVector<se::DeviceAddressBase, 4>,
                              PrimitiveType, size_t, const Executor&) override {
    return absl::UnimplementedError("unused");
  }

  absl::Status LaunchCollectivePermute(se::DeviceAddressBase,
                                       se::DeviceAddressBase, PrimitiveType,
                                       size_t, std::optional<RankId>,
                                       absl::Span<const RankId>,
                                       const Executor&) override {
    return absl::UnimplementedError("unused");
  }

  absl::Status LaunchSend(se::DeviceAddressBase, PrimitiveType, size_t, RankId,
                          const Executor&) override {
    return absl::UnimplementedError("unused");
  }

  absl::Status LaunchRecv(se::DeviceAddressBase, PrimitiveType, size_t, RankId,
                          const Executor&) override {
    return absl::UnimplementedError("unused");
  }

  Future<> AllReduce(se::DeviceAddressBase, se::DeviceAddressBase,
                     PrimitiveType, size_t, ReductionKind,
                     const Executor&) override {
    return absl::UnimplementedError("unused");
  }

  Future<> Broadcast(se::DeviceAddressBase, se::DeviceAddressBase,
                     PrimitiveType, size_t, RankId, const Executor&) override {
    return absl::UnimplementedError("unused");
  }

  Future<> ReduceScatter(se::DeviceAddressBase, se::DeviceAddressBase,
                         PrimitiveType, size_t, ReductionKind,
                         const Executor&) override {
    return absl::UnimplementedError("unused");
  }

  Future<> AllGather(se::DeviceAddressBase, se::DeviceAddressBase,
                     PrimitiveType, size_t, const Executor&) override {
    return absl::UnimplementedError("unused");
  }

  Future<> CollectivePermute(se::DeviceAddressBase, se::DeviceAddressBase,
                             PrimitiveType, size_t, std::optional<RankId>,
                             absl::Span<const RankId>,
                             const Executor&) override {
    return absl::UnimplementedError("unused");
  }

  Future<> AllToAll(absl::InlinedVector<se::DeviceAddressBase, 4>,
                    absl::InlinedVector<se::DeviceAddressBase, 4>,
                    PrimitiveType, size_t, const Executor&) override {
    return absl::UnimplementedError("unused");
  }

  Future<> Send(se::DeviceAddressBase, PrimitiveType, size_t, RankId,
                const Executor&) override {
    return absl::UnimplementedError("unused");
  }

  Future<> Recv(se::DeviceAddressBase, PrimitiveType, size_t, RankId,
                const Executor&) override {
    return absl::UnimplementedError("unused");
  }

  absl::StatusOr<size_t> NumRanks() const override { return size_t{1}; }
  absl::StatusOr<size_t> CurrentRank() override { return size_t{0}; }
  std::string ToString() const override { return "fake gpu communicator"; }
};

// Test-only subclass that bypasses NCCL in RunCollective only. The command
// buffer tests still exercise the production CollectiveThunk::Record path.
class MemzeroAllToAllThunk : public AllToAllThunk {
 public:
  MemzeroAllToAllThunk(Thunk::ThunkInfo thunk_info, AllToAllConfig config,
                       std::vector<CollectiveThunk::Buffer> buffers)
      : AllToAllThunk(std::move(thunk_info), config, std::move(buffers),
                      /*p2p_memcpy_enabled=*/false) {}

 protected:
  absl::Status RunCollective(const ExecuteParams& params, const GpuCliqueKey&,
                             se::Stream& stream, Communicator&) override {
    const BufferAllocation::Slice& dst_slice =
        buffers()[0].destination_buffer.slice;
    se::DeviceAddressBase dst =
        params.buffer_allocations->GetDeviceAddress(dst_slice);
    return stream.MemZero(&dst, dst_slice.size());
  }
};

struct CollectiveTestState {
  DeviceAssignment device_assignment;
  ServiceExecutableRunOptions run_options;
  std::optional<CollectiveParams> collective_params;
  std::shared_ptr<LockableGpuClique> clique;
  CollectiveCliques collective_cliques;
};

TEST(CollectiveThunkTest, ProtoRoundTrip) {
  ThunkProto proto = tsl::proto_testing::ParseTextProtoOrDie<ThunkProto>(
      R"pb(
        thunk_info { profile_annotation: "partition_id_profile_annotation" }
        all_to_all_thunk {
          collective_config {}
          has_split_dimension: false
          p2p_memcpy_enabled: true
        }
      )pb");

  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = proto.thunk_info().profile_annotation();

  std::vector<BufferAllocation> buffer_allocations = {
      BufferAllocation(/*index=*/0, /*size=*/4, /*color=*/0)};

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<AllToAllThunk> thunk,
      AllToAllThunk::FromProto(thunk_info, proto.all_to_all_thunk(),
                               buffer_allocations));

  ASSERT_OK_AND_ASSIGN(ThunkProto round_trip_proto, thunk->ToProto());

  EXPECT_THAT(round_trip_proto, EqualsProto(proto));
}

// Builds a MemzeroAllToAllThunk with one F32[length] src->dst buffer pair.
static MemzeroAllToAllThunk MakeThunk(const BufferAllocation& alloc_src,
                                      const BufferAllocation& alloc_dst,
                                      int64_t length) {
  int64_t byte_length = sizeof(float) * length;
  ShapedSlice src_slice{BufferAllocation::Slice(&alloc_src, 0, byte_length),
                        ShapeUtil::MakeShape(F32, {length})};
  ShapedSlice dst_slice{BufferAllocation::Slice(&alloc_dst, 0, byte_length),
                        ShapeUtil::MakeShape(F32, {length})};
  CollectiveThunk::Buffer buffer{.element_count = length,
                                 .source_buffer = src_slice,
                                 .destination_buffer = dst_slice,
                                 .source_memory_space = 0,
                                 .destination_memory_space = 0};

  AllToAllConfig config;
  config.config.operand_element_type = {F32};
  config.has_split_dimension = false;

  return MemzeroAllToAllThunk(Thunk::ThunkInfo(), config, {buffer});
}

static absl::Status InitCollectiveTestState(se::Stream* stream,
                                            const AllToAllThunk& thunk,
                                            CollectiveTestState* state) {
  state->device_assignment = DeviceAssignment(/*replica_count=*/1,
                                              /*computation_count=*/1);
  state->device_assignment(0, 0) = 0;
  state->run_options.mutable_run_options()->set_stream(stream);
  state->run_options.mutable_run_options()->set_device_assignment(
      &state->device_assignment);

  ASSIGN_OR_RETURN(
      CollectiveParams collective_params,
      CollectiveParams::Create(state->run_options,
                               /*async_streams=*/{}, LocalDeviceId(0)));
  ASSIGN_OR_RETURN(
      GpuCliqueKey clique_key,
      GetGpuCliqueKey(collective_params, thunk.config().replica_groups,
                      thunk.config().group_mode, thunk.communication_id()));

  absl::btree_map<RankId, std::unique_ptr<Communicator>> communicators;
  communicators.emplace(RankId(0), std::make_unique<FakeGpuCommunicator>());
  state->clique = std::make_shared<LockableGpuClique>(
      clique_key, std::nullopt, std::move(communicators),
      /*peer_access_enabled=*/true, std::make_shared<CancellationToken>());

  AcquiredCliquesMap cliques_map;
  cliques_map.emplace(clique_key, std::make_shared<LockableGpuClique::Lock>(
                                      state->clique->Acquire()));
  state->collective_cliques = CollectiveCliques(std::move(cliques_map));
  state->collective_params.emplace(std::move(collective_params));
  return absl::OkStatus();
}

static absl::Status FillDeviceBuffer(se::Stream& stream,
                                     se::DeviceAddressBase buffer,
                                     int64_t length, float value) {
  std::vector<float> data(length, value);
  RETURN_IF_ERROR(stream.Memcpy(&buffer, data.data(), sizeof(float) * length));
  return stream.BlockHostUntilDone();
}

static absl::StatusOr<std::vector<float>> ReadDeviceBuffer(
    se::Stream& stream, se::DeviceAddressBase buffer, int64_t length) {
  std::vector<float> data(length);
  RETURN_IF_ERROR(stream.Memcpy(data.data(), buffer, sizeof(float) * length));
  RETURN_IF_ERROR(stream.BlockHostUntilDone());
  return data;
}

// Records AllToAllThunk into a primary command buffer (create phase) and
// verifies that a non-null command node is returned.
TEST(AllToAllThunkTest, RecordCommandBufferCreate) {
  se::StreamExecutor* executor = GpuExecutor();
  if (!IsAtLeastCuda12900(executor)) {
    GTEST_SKIP() << "Child command nodes require CUDA 12.9+";
  }

  ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  int64_t length = 4;
  int64_t byte_length = sizeof(float) * length;

  se::DeviceAddress<float> src = executor->AllocateArray<float>(length, 0);
  se::DeviceAddress<float> dst = executor->AllocateArray<float>(length, 0);

  BufferAllocation alloc_src(/*index=*/0, byte_length, /*color=*/0);
  BufferAllocation alloc_dst(/*index=*/1, byte_length, /*color=*/0);

  MemzeroAllToAllThunk thunk = MakeThunk(alloc_src, alloc_dst, length);
  CollectiveTestState collective_state;
  ASSERT_OK(InitCollectiveTestState(stream.get(), thunk, &collective_state));

  se::StreamExecutorAddressAllocator allocator(executor);
  BufferAllocations allocations({src, dst}, 0, &allocator);

  ASSERT_OK(FillDeviceBuffer(*stream, dst, length, 1.0f));
  Thunk::ExecuteParams execute_params = Thunk::ExecuteParams::Create(
      collective_state.run_options, allocations, stream.get(),
      /*command_buffer_trace_stream=*/stream.get(),
      &*collective_state.collective_params,
      &collective_state.collective_cliques,
      /*collective_memory=*/nullptr);

  CommandStateManager state;
  Command::RecordParams record_params = {state};

  ASSERT_OK_AND_ASSIGN(
      auto command_buffer,
      executor->CreateCommandBuffer(se::CommandBuffer::Mode::kPrimary));
  ASSERT_OK_AND_ASSIGN(const se::CommandBuffer::Command* cmd,
                       thunk.Record(execute_params, record_params,
                                    Command::RecordCreate{/*dependencies=*/{}},
                                    command_buffer.get()));
  EXPECT_NE(cmd, nullptr);

  ASSERT_OK(command_buffer->Finalize());
  ASSERT_OK(command_buffer->Submit(stream.get()));
  ASSERT_OK(stream->BlockHostUntilDone());

  ASSERT_OK_AND_ASSIGN(std::vector<float> output,
                       ReadDeviceBuffer(*stream, dst, length));
  EXPECT_THAT(output, ::testing::Each(0.0f));
}

// Records AllToAllThunk twice into the same command buffer: first as a create,
// then as an update with different buffer allocations. Verifies that the same
// command node pointer is returned on update.
TEST(AllToAllThunkTest, RecordCommandBufferUpdate) {
  se::StreamExecutor* executor = GpuExecutor();
  if (!IsAtLeastCuda12900(executor)) {
    GTEST_SKIP() << "Child command nodes require CUDA 12.9+";
  }

  ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  int64_t length = 4;
  int64_t byte_length = sizeof(float) * length;

  se::DeviceAddress<float> src1 = executor->AllocateArray<float>(length, 0);
  se::DeviceAddress<float> dst1 = executor->AllocateArray<float>(length, 0);

  se::DeviceAddress<float> src2 = executor->AllocateArray<float>(length, 0);
  se::DeviceAddress<float> dst2 = executor->AllocateArray<float>(length, 0);

  BufferAllocation alloc_src(/*index=*/0, byte_length, /*color=*/0);
  BufferAllocation alloc_dst(/*index=*/1, byte_length, /*color=*/0);

  MemzeroAllToAllThunk thunk = MakeThunk(alloc_src, alloc_dst, length);
  CollectiveTestState collective_state;
  ASSERT_OK(InitCollectiveTestState(stream.get(), thunk, &collective_state));

  se::StreamExecutorAddressAllocator allocator(executor);

  ASSERT_OK(FillDeviceBuffer(*stream, dst1, length, 1.0f));
  BufferAllocations allocations1({src1, dst1}, 0, &allocator);
  Thunk::ExecuteParams params1 = Thunk::ExecuteParams::Create(
      collective_state.run_options, allocations1, stream.get(),
      /*command_buffer_trace_stream=*/stream.get(),
      &*collective_state.collective_params,
      &collective_state.collective_cliques,
      /*collective_memory=*/nullptr);

  CommandStateManager state;
  Command::RecordParams record_params = {state};

  ASSERT_OK_AND_ASSIGN(
      auto command_buffer,
      executor->CreateCommandBuffer(se::CommandBuffer::Mode::kPrimary));
  ASSERT_OK_AND_ASSIGN(const se::CommandBuffer::Command* cmd,
                       thunk.Record(params1, record_params,
                                    Command::RecordCreate{/*dependencies=*/{}},
                                    command_buffer.get()));
  ASSERT_NE(cmd, nullptr);

  ASSERT_OK(command_buffer->Finalize());
  ASSERT_OK(command_buffer->Submit(stream.get()));
  ASSERT_OK(stream->BlockHostUntilDone());
  ASSERT_OK_AND_ASSIGN(std::vector<float> output1,
                       ReadDeviceBuffer(*stream, dst1, length));
  EXPECT_THAT(output1, ::testing::Each(0.0f));

  ASSERT_OK(FillDeviceBuffer(*stream, dst2, length, 2.0f));
  BufferAllocations allocations2({src2, dst2}, 0, &allocator);
  Thunk::ExecuteParams params2 = Thunk::ExecuteParams::Create(
      collective_state.run_options, allocations2, stream.get(),
      /*command_buffer_trace_stream=*/stream.get(),
      &*collective_state.collective_params,
      &collective_state.collective_cliques,
      /*collective_memory=*/nullptr);

  std::vector<BufferAllocation::Index> updated_allocs = {0, 1};
  Command::RecordParams record_params2 = {state, std::move(updated_allocs)};

  ASSERT_OK(command_buffer->Update());
  ASSERT_OK_AND_ASSIGN(
      const se::CommandBuffer::Command* updated_cmd,
      thunk.Record(params2, record_params2, Command::RecordUpdate{cmd},
                   command_buffer.get()));
  EXPECT_EQ(updated_cmd, cmd);

  ASSERT_OK(command_buffer->Finalize());
  ASSERT_OK(command_buffer->Submit(stream.get()));
  ASSERT_OK(stream->BlockHostUntilDone());
  ASSERT_OK_AND_ASSIGN(std::vector<float> output2,
                       ReadDeviceBuffer(*stream, dst2, length));
  EXPECT_THAT(output2, ::testing::Each(0.0f));
}

}  // namespace
}  // namespace xla::gpu
