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

#include "xla/backends/gpu/runtime/async_thunk.h"

#include <algorithm>
#include <array>
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
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/command_buffer_requests.h"
#include "xla/backends/gpu/runtime/command_state.h"
#include "xla/backends/gpu/runtime/device_to_device_copy_thunk.h"
#include "xla/backends/gpu/runtime/execution_stream_id.h"
#include "xla/backends/gpu/runtime/memset_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_executor.h"
#include "xla/backends/gpu/runtime/thunk_id.h"
#include "xla/debug_options_flags.h"
#include "xla/executable_run_options.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/platform_util.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_buffer.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"

namespace xla::gpu {
namespace {

class RecordingThunk : public Thunk {
 public:
  explicit RecordingThunk(se::Stream** executed_stream)
      : Thunk(kKernel, ThunkInfo()), executed_stream_(executed_stream) {}

  absl::Status ExecuteOnStream(const ExecuteParams& params) override {
    *executed_stream_ = params.stream;
    return absl::OkStatus();
  }

  BufferUses buffer_uses() const override { return {}; }

  absl::StatusOr<ThunkProto> ToProto() const override {
    return absl::UnimplementedError(
        "RecordingThunk::ToProto is not implemented");
  }

 private:
  se::Stream** executed_stream_;
};

static absl::StatusOr<se::StreamExecutor*> CreateExecutor() {
  ABSL_ASSIGN_OR_RETURN(std::string platform_name,
                   xla::PlatformUtil::CanonicalPlatformName("gpu"));
  ABSL_ASSIGN_OR_RETURN(se::Platform * platform,
                   se::PlatformManager::PlatformWithName(platform_name));
  return platform->ExecutorForDevice(0);
}

// Test that 4 async start thunks each memset a quarter of a buffer on separate
// streams, and 4 async done thunks synchronize back to the main stream.
TEST(AsyncThunkTest, ConcurrentMemsets) {
  ASSERT_OK_AND_ASSIGN(auto executor, CreateExecutor());
  ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  constexpr int32_t kNumChunks = 4;
  constexpr int32_t kChunkLength = 256;
  constexpr int32_t kTotalLength = kNumChunks * kChunkLength;
  constexpr int64_t kChunkBytes = kChunkLength * sizeof(int32_t);
  constexpr int64_t kTotalBytes = kTotalLength * sizeof(int32_t);

  // Create 4 async streams, one per chunk.
  std::vector<std::unique_ptr<se::Stream>> async_streams;
  std::vector<se::Stream*> additional_streams;
  for (int i = 0; i < kNumChunks; ++i) {
    ASSERT_OK_AND_ASSIGN(async_streams.emplace_back(),
                         executor->CreateStream());
    additional_streams.push_back(async_streams.back().get());
  }

  // Allocate device buffer and zero it.
  se::StreamExecutorAddressAllocator allocator(executor);
  ASSERT_OK_AND_ASSIGN(
      auto buf, allocator.Allocate(executor->device_ordinal(), kTotalBytes));
  ASSERT_OK(stream->MemZero(buf.ptr(), kTotalBytes));

  // Single buffer allocation covering the entire buffer.
  BufferAllocation alloc(/*index=*/0, kTotalBytes, /*color=*/0);

  // Build a thunk sequence: 4 async starts followed by 4 async dones. Each
  // async start contains a nested Memset32BitValueThunk that writes a distinct
  // value to a different quarter of the buffer.
  ThunkSequence thunks;

  for (int i = 0; i < kNumChunks; ++i) {
    BufferAllocation::Slice slice(&alloc, i * kChunkBytes, kChunkBytes);
    uint32_t value = static_cast<uint32_t>(100 + i);

    ThunkSequence nested = ThunkSequence::Of<Memset32BitValueThunk>(
        Thunk::ThunkInfo(), value, slice);

    Thunk::ThunkInfo start_info;
    start_info.profile_annotation = absl::StrCat("start#", i);
    start_info.thunk_id = ThunkId(i + 1);

    thunks.Emplace<AsyncStartThunk>(std::move(start_info),
                                    ComputationStreamId(i), std::move(nested),
                                    /*devices_per_host=*/1);
  }

  for (int i = 0; i < kNumChunks; ++i) {
    auto* start = static_cast<AsyncStartThunk*>(thunks[i].get());
    thunks.Emplace<AsyncDoneThunk>(Thunk::ThunkInfo(),
                                   start->async_execution());
  }

  // Build a ThunkExecutor to execute all thunks.
  ThunkExecutor thunk_executor(std::move(thunks));

  // Set up execute params.
  BufferAllocations allocations({buf.cref()}, 0, &allocator);
  ServiceExecutableRunOptions run_options;
  Thunk::ExecutionScopedState state;

  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      run_options, allocations, stream.get(), stream.get(),
      /*collective_params=*/nullptr, /*collective_cliques=*/nullptr,
      /*collective_memory=*/nullptr, additional_streams, &state);

  // Initialize and execute all thunks via ThunkExecutor.
  CommandBufferRequests command_buffer_requests;
  Thunk::InitializeParams init_params(&command_buffer_requests);
  init_params.executor = executor;
  init_params.execution_scoped_state = &state;
  ASSERT_OK(thunk_executor.Initialize(init_params));
  ASSERT_OK(thunk_executor.ExecuteOnStream(params));

  // Wait for everything to complete.
  ASSERT_OK(stream->BlockHostUntilDone());

  // Copy result back to host and verify.
  std::vector<int32_t> result(kTotalLength, 0);
  ASSERT_OK(stream->Memcpy(result.data(), buf.cref(), kTotalBytes));

  for (int i = 0; i < kNumChunks; ++i) {
    int32_t expected = 100 + i;
    for (int j = 0; j < kChunkLength; ++j) {
      ASSERT_EQ(result[i * kChunkLength + j], expected)
          << "Mismatch at chunk " << i << " element " << j;
    }
  }
}

// Test that AsyncStartThunk dispatches to device_to_host_stream when
// execution_stream_id is kMemcpyD2HStreamId, and to host_to_device_stream
// when it is kMemcpyH2DStreamId.  We do this by running a no-op thunk on each
// memcpy stream and confirming execution completes without error.
TEST(AsyncThunkTest, MemcpyStreamDispatch) {
  ASSERT_OK_AND_ASSIGN(auto executor, CreateExecutor());
  ASSERT_OK_AND_ASSIGN(auto main_stream, executor->CreateStream());
  ASSERT_OK_AND_ASSIGN(auto d2h_stream, executor->CreateStream());
  ASSERT_OK_AND_ASSIGN(auto h2d_stream, executor->CreateStream());

  Thunk::ThunkInfo d2h_info;
  d2h_info.thunk_id = ThunkId(10);
  d2h_info.profile_annotation = "d2h_start";

  Thunk::ThunkInfo h2d_info;
  h2d_info.thunk_id = ThunkId(11);
  h2d_info.profile_annotation = "h2d_start";

  ThunkSequence thunks;
  thunks.Emplace<AsyncStartThunk>(d2h_info, kMemcpyD2HStreamId, ThunkSequence{},
                                  /*devices_per_host=*/1);
  auto* d2h_start = static_cast<AsyncStartThunk*>(thunks[0].get());
  thunks.Emplace<AsyncDoneThunk>(Thunk::ThunkInfo(),
                                 d2h_start->async_execution());

  thunks.Emplace<AsyncStartThunk>(h2d_info, kMemcpyH2DStreamId, ThunkSequence{},
                                  /*devices_per_host=*/1);
  auto* h2d_start = static_cast<AsyncStartThunk*>(thunks[2].get());
  thunks.Emplace<AsyncDoneThunk>(Thunk::ThunkInfo(),
                                 h2d_start->async_execution());

  ThunkExecutor thunk_executor(std::move(thunks));

  se::StreamExecutorAddressAllocator allocator(executor);
  BufferAllocations allocations({}, 0, &allocator);
  ServiceExecutableRunOptions run_options;
  Thunk::ExecutionScopedState state;

  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      run_options, allocations, main_stream.get(), main_stream.get(),
      /*collective_params=*/nullptr, /*collective_cliques=*/nullptr,
      /*collective_memory=*/nullptr, /*additional_compute_streams=*/{}, &state);

  params.device_to_host_stream = d2h_stream.get();
  params.host_to_device_stream = h2d_stream.get();

  CommandBufferRequests command_buffer_requests;
  Thunk::InitializeParams init_params(&command_buffer_requests);
  init_params.executor = executor;
  init_params.execution_scoped_state = &state;
  ASSERT_OK(thunk_executor.Initialize(init_params));
  ASSERT_OK(thunk_executor.ExecuteOnStream(params));
  ASSERT_OK(main_stream->BlockHostUntilDone());
}

TEST(AsyncThunkTest, MemcpyStreamNullReturnsError) {
  ASSERT_OK_AND_ASSIGN(auto executor, CreateExecutor());
  ASSERT_OK_AND_ASSIGN(auto main_stream, executor->CreateStream());

  AsyncStartThunk d2h_start(Thunk::ThunkInfo(), kMemcpyD2HStreamId,
                            ThunkSequence{}, /*devices_per_host=*/1);
  AsyncStartThunk h2d_start(Thunk::ThunkInfo(), kMemcpyH2DStreamId,
                            ThunkSequence{}, /*devices_per_host=*/1);

  se::StreamExecutorAddressAllocator allocator(executor);
  BufferAllocations allocations({}, 0, &allocator);
  ServiceExecutableRunOptions run_options;
  Thunk::ExecutionScopedState state;

  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      run_options, allocations, main_stream.get(), main_stream.get(),
      /*collective_params=*/nullptr, /*collective_cliques=*/nullptr,
      /*collective_memory=*/nullptr, /*additional_compute_streams=*/{}, &state);
  ASSERT_EQ(params.device_to_host_stream, nullptr);
  ASSERT_EQ(params.host_to_device_stream, nullptr);

  CommandBufferRequests command_buffer_requests;
  Thunk::InitializeParams init_params(&command_buffer_requests);
  init_params.executor = executor;
  init_params.execution_scoped_state = &state;
  ASSERT_OK(d2h_start.Initialize(init_params));
  ASSERT_OK(h2d_start.Initialize(init_params));

  EXPECT_FALSE(d2h_start.ExecuteOnStream(params).ok());
  EXPECT_FALSE(h2d_start.ExecuteOnStream(params).ok());
}

TEST(AsyncThunkTest, GpuExecutableFallsBackToMainStreamForMemcpyStreams) {
  ASSERT_OK_AND_ASSIGN(auto executor, CreateExecutor());
  ASSERT_OK_AND_ASSIGN(auto main_stream, executor->CreateStream());

  ThunkSequence thunks;
  thunks.Emplace<AsyncStartThunk>(Thunk::ThunkInfo(), kMemcpyD2HStreamId,
                                  ThunkSequence{}, /*devices_per_host=*/1);
  auto* d2h_start = static_cast<AsyncStartThunk*>(thunks[0].get());
  thunks.Emplace<AsyncDoneThunk>(Thunk::ThunkInfo(),
                                 d2h_start->async_execution());
  thunks.Emplace<AsyncStartThunk>(Thunk::ThunkInfo(), kMemcpyH2DStreamId,
                                  ThunkSequence{}, /*devices_per_host=*/1);
  auto* h2d_start = static_cast<AsyncStartThunk*>(thunks[2].get());
  thunks.Emplace<AsyncDoneThunk>(Thunk::ThunkInfo(),
                                 h2d_start->async_execution());

  GpuExecutable::Params params;
  params.executable = std::make_unique<ThunkExecutor>(std::move(thunks));
  params.debug_options = GetDebugOptionsFromFlags();
  params.buffer_assignment_proto = BufferAssignmentProto();
  params.buffer_allocations_debug_summary = "dummy";
  params.module_name = "memcpy_stream_fallback_test";
  params.device_description = executor->GetDeviceDescription();
  params.enable_debug_info_manager = false;

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<GpuExecutable> executable,
                       GpuExecutable::Create(std::move(params)));

  se::StreamExecutorAddressAllocator allocator(executor);
  ExecutableRunOptions run_options;
  run_options.set_stream(main_stream.get());
  run_options.set_allocator(&allocator);
  run_options.set_device_ordinal(executor->device_ordinal());
  // No stream_borrower is installed, so GpuExecutable must fall back to
  // main_stream for device_to_host_stream/host_to_device_stream.
  ServiceExecutableRunOptions service_run_options(run_options);

  ASSERT_OK(executable
                ->ExecuteAsyncOnStream(&service_run_options,
                                       absl::Span<const ShapedBuffer* const>{})
                .status());
  ASSERT_OK(main_stream->BlockHostUntilDone());
}

// Test that AsyncDoneThunk::Record() creates an empty command buffer node and
// is a no-op on update.
TEST(AsyncThunkTest, AsyncDoneRecordCommandBuffer) {
  ASSERT_OK_AND_ASSIGN(auto executor, CreateExecutor());
  ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  // Create a paired start/done to obtain a valid AsyncExecution.
  AsyncStartThunk start(Thunk::ThunkInfo(), ComputationStreamId(0),
                        ThunkSequence(), /*devices_per_host=*/1);
  AsyncDoneThunk done(Thunk::ThunkInfo(), start.async_execution());

  // Set up minimal execute params (not used by AsyncDoneThunk::Record()).
  se::StreamExecutorAddressAllocator allocator(executor);
  BufferAllocations allocations({}, 0, &allocator);
  ServiceExecutableRunOptions run_options;
  Thunk::ExecutionScopedState thunk_state;
  Thunk::ExecuteParams execute_params = Thunk::ExecuteParams::Create(
      run_options, allocations, stream.get(), stream.get(),
      /*collective_params=*/nullptr, /*collective_cliques=*/nullptr,
      /*collective_memory=*/nullptr, /*additional_compute_streams=*/{},
      &thunk_state);

  ASSERT_OK_AND_ASSIGN(
      auto command_buffer,
      executor->CreateCommandBuffer(se::CommandBuffer::Mode::kPrimary));

  CommandStateManager cmd_state;
  Command::RecordParams record_params = {cmd_state};

  // RecordCreate: should produce a non-null empty command node.
  ASSERT_OK_AND_ASSIGN(const se::CommandBuffer::Command* cmd,
                       done.Record(execute_params, record_params,
                                   Command::RecordCreate{/*.dependencies=*/{}},
                                   command_buffer.get()));
  ASSERT_NE(cmd, nullptr);

  // RecordUpdate: empty command is not updatable; same pointer returned.
  ASSERT_OK_AND_ASSIGN(const se::CommandBuffer::Command* updated_cmd,
                       done.Record(execute_params, record_params,
                                   Command::RecordUpdate{/*.command=*/cmd},
                                   command_buffer.get()));
  EXPECT_EQ(updated_cmd, cmd);
}

TEST(AsyncThunkTest, LocalityDomainStreamSelection) {
  ASSERT_OK_AND_ASSIGN(auto executor, CreateExecutor());
  ASSERT_OK_AND_ASSIGN(auto main_stream, executor->CreateStream());
  ASSERT_OK_AND_ASSIGN(auto locality_stream, executor->CreateStream());

  Thunk::ThunkInfo info;
  info.thunk_id = ThunkId(1);
  se::Stream* executed_stream = nullptr;
  ThunkSequence nested = ThunkSequence::Of<RecordingThunk>(&executed_stream);
  AsyncStartThunk start(std::move(info), LocalityDomainStreamId(3),
                        std::move(nested), /*devices_per_host=*/1);

  Thunk::ExecutionScopedState state;
  CommandBufferRequests command_buffer_requests;
  Thunk::InitializeParams init_params(&command_buffer_requests);
  init_params.executor = executor;
  init_params.execution_scoped_state = &state;
  ASSERT_OK(start.Initialize(init_params));

  se::StreamExecutorAddressAllocator allocator(executor);
  BufferAllocations allocations({}, 0, &allocator);
  ServiceExecutableRunOptions run_options;
  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      run_options, allocations, main_stream.get(), main_stream.get(),
      /*collective_params=*/nullptr, /*collective_cliques=*/nullptr,
      /*collective_memory=*/nullptr, /*additional_compute_streams=*/{}, &state,
      std::nullopt, {{LocalityDomainStreamId(3), locality_stream.get()}});

  ASSERT_OK(start.ExecuteOnStream(params));
  EXPECT_EQ(executed_stream, locality_stream.get());
}

TEST(AsyncThunkTest,
     LocalityDomainsExecuteIndependentDefaultMemoryReadsAndSynchronize) {
  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor, CreateExecutor());
  const se::DeviceDescription& device = executor->GetDeviceDescription();
  if (std::min(device.driver_version(), device.compile_time_toolkit_version()) <
      se::SemanticVersion{13, 4, 0}) {
    GTEST_SKIP() << "Requires CUDA toolkit and driver 13.4 or newer";
  }
  if (device.locality_domain_count() < 2) {
    GTEST_SKIP() << "Requires at least two usable locality domains";
  }
  absl::StatusOr<std::unique_ptr<se::CommandBuffer>> command_buffer =
      executor->CreateCommandBuffer(se::CommandBuffer::Mode::kPrimary);
  if (!command_buffer.ok()) {
    GTEST_SKIP() << "Command buffers unavailable: " << command_buffer.status();
  }

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Stream> primary,
                       executor->CreateStream());
  std::array<std::unique_ptr<se::Stream>, 2> locality_streams;
  for (int domain = 0; domain < static_cast<int>(locality_streams.size());
       ++domain) {
    absl::StatusOr<std::unique_ptr<se::Stream>> stream =
        executor->CreateStreamInLocalityDomain(domain);
    if (!stream.ok()) {
      GTEST_SKIP() << "Locality domain " << domain
                   << " is unusable: " << stream.status();
    }
    locality_streams[domain] = std::move(*stream);
  }

  constexpr int64_t kLength = 4;
  constexpr int64_t kBytes = kLength * sizeof(int32_t);
  se::StreamExecutorAddressAllocator allocator(executor);
  ASSERT_OK_AND_ASSIGN(auto source,
                       allocator.Allocate(executor->device_ordinal(), kBytes));
  ASSERT_OK_AND_ASSIGN(auto output0,
                       allocator.Allocate(executor->device_ordinal(), kBytes));
  ASSERT_OK_AND_ASSIGN(auto output1,
                       allocator.Allocate(executor->device_ordinal(), kBytes));

  ASSERT_OK(primary->Memset32(source.ptr(), 42, kBytes));
  ASSERT_OK(primary->MemZero(output0.ptr(), kBytes));
  ASSERT_OK(primary->MemZero(output1.ptr(), kBytes));

  BufferAllocation source_alloc(/*index=*/0, kBytes, /*color=*/0);
  BufferAllocation output0_alloc(/*index=*/1, kBytes, /*color=*/0);
  BufferAllocation output1_alloc(/*index=*/2, kBytes, /*color=*/0);
  Shape shape = ShapeUtil::MakeShape(S32, {kLength});
  ShapedSlice source_slice{
      BufferAllocation::Slice(&source_alloc, /*offset=*/0, kBytes), shape};
  std::array<ShapedSlice, 2> output_slices = {
      ShapedSlice{BufferAllocation::Slice(&output0_alloc, /*offset=*/0, kBytes),
                  shape},
      ShapedSlice{BufferAllocation::Slice(&output1_alloc, /*offset=*/0, kBytes),
                  shape}};

  std::array<se::Stream*, 2> executed_streams = {nullptr, nullptr};
  ThunkSequence thunks;
  std::array<std::shared_ptr<AsyncExecution>, 2> async_executions;
  for (int domain = 0; domain < static_cast<int>(locality_streams.size());
       ++domain) {
    ThunkSequence nested;
    nested.Emplace<RecordingThunk>(&executed_streams[domain]);
    nested.Emplace<DeviceToDeviceCopyThunk>(Thunk::ThunkInfo(), source_slice,
                                            output_slices[domain], kBytes);
    Thunk::ThunkInfo start_info;
    start_info.thunk_id = ThunkId(domain + 1);
    auto start = std::make_unique<AsyncStartThunk>(
        std::move(start_info), LocalityDomainStreamId(domain),
        std::move(nested), /*devices_per_host=*/1);
    async_executions[domain] = start->async_execution();
    thunks.push_back(std::move(start));
  }
  for (const std::shared_ptr<AsyncExecution>& execution : async_executions) {
    thunks.Emplace<AsyncDoneThunk>(Thunk::ThunkInfo(), execution);
  }
  ThunkExecutor thunk_executor(std::move(thunks));

  BufferAllocations allocations({source.cref(), output0.cref(), output1.cref()},
                                executor->device_ordinal(), &allocator);
  ServiceExecutableRunOptions run_options;
  Thunk::ExecutionScopedState state;
  CommandBufferRequests command_buffer_requests;
  Thunk::InitializeParams initialize_params(&command_buffer_requests);
  initialize_params.executor = executor;
  initialize_params.execution_scoped_state = &state;
  ASSERT_OK(thunk_executor.Initialize(initialize_params));

  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      run_options, allocations, primary.get(), primary.get(), nullptr, nullptr,
      nullptr, /*additional_compute_streams=*/{}, &state, std::nullopt,
      {{LocalityDomainStreamId(0), locality_streams[0].get()},
       {LocalityDomainStreamId(1), locality_streams[1].get()}});
  ASSERT_OK(thunk_executor.ExecuteOnStream(params));

  std::array<int32_t, kLength> result0;
  std::array<int32_t, kLength> result1;
  ASSERT_OK(primary->Memcpy(result0.data(), output0.cref(), kBytes));
  ASSERT_OK(primary->Memcpy(result1.data(), output1.cref(), kBytes));
  ASSERT_OK(primary->BlockHostUntilDone());

  EXPECT_EQ(executed_streams[0], locality_streams[0].get());
  EXPECT_EQ(executed_streams[1], locality_streams[1].get());
  EXPECT_EQ(result0, (std::array<int32_t, kLength>{42, 42, 42, 42}));
  EXPECT_EQ(result1, (std::array<int32_t, kLength>{42, 42, 42, 42}));
}

}  // namespace
}  // namespace xla::gpu
