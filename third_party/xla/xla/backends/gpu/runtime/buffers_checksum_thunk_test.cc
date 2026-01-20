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

#include "xla/backends/gpu/runtime/buffers_checksum_thunk.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/runtime/buffer_debug_log_entry_metadata_store.h"
#include "xla/backends/gpu/runtime/buffer_debug_log_structs.h"
#include "xla/backends/gpu/runtime/collective_clique_requests.h"
#include "xla/backends/gpu/runtime/collective_multimem_registry.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_id.h"
#include "xla/runtime/device_id.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/buffer_debug_log.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor_memory_allocator.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

namespace se = stream_executor;

using ::stream_executor::gpu::BufferDebugLog;
using Metadata = BufferDebugLogEntryMetadataStore::Metadata;

using ::testing::AllOf;
using ::testing::Field;
using ::testing::UnorderedElementsAre;

MATCHER_P2(IsEntryWithMetadata, store, metadata, "") {
  std::optional<Metadata> actual_metadata =
      store->GetEntryMetadata(arg.entry_id);
  if (!actual_metadata.has_value()) {
    *result_listener << "metadata not found for entry_id "
                     << arg.entry_id.value();
    return false;
  }

  return ExplainMatchResult(
      AllOf(Field(&Metadata::thunk_id, metadata.thunk_id),
            Field(&Metadata::buffer_idx, metadata.buffer_idx),
            Field(&Metadata::execution_id, metadata.execution_id),
            Field(&Metadata::is_input, metadata.is_input)),
      *actual_metadata, result_listener);
}

class FakeThunk : public Thunk {
 public:
  explicit FakeThunk(ThunkInfo info, BufferUses buffer_uses)
      : Thunk(Thunk::Kind::kGemm, std::move(info)),
        buffer_uses_(std::move(buffer_uses)) {}

  absl::Status ExecuteOnStream(const ExecuteParams& params) override {
    return absl::OkStatus();
  }

  BufferUses buffer_uses() const override { return buffer_uses_; }

 private:
  BufferUses buffer_uses_;
};

class BuffersDebugChecksumThunkTest : public ::testing::Test {
 protected:
  void SetUp() override {
    TF_ASSERT_OK_AND_ASSIGN(platform_,
                            se::PlatformManager::PlatformWithName("CUDA"));
    TF_ASSERT_OK_AND_ASSIGN(executor_, platform_->ExecutorForDevice(0));
    TF_ASSERT_OK_AND_ASSIGN(stream_, executor_->CreateStream(std::nullopt));
    allocator_ =
        std::make_unique<stream_executor::StreamExecutorAddressAllocator>(
            stream_->parent());

    if (!executor_->GetDeviceDescription()
             .cuda_compute_capability()
             .IsAtLeastPascal()) {
      GTEST_SKIP()
          << "buffer checksumming is not supported on CUDA architectures "
             "older than Pascal due to missing atomic fetch_add with "
             "system scope";
    }
  }

  se::Platform* platform_;
  se::StreamExecutor* executor_;
  std::unique_ptr<se::Stream> stream_;
  std::unique_ptr<stream_executor::StreamExecutorAddressAllocator> allocator_;
};

TEST_F(BuffersDebugChecksumThunkTest, CalculatesChecksums) {
  static constexpr size_t kLogSize =
      BufferDebugLog<BufferDebugLogEntry>::RequiredSizeForEntries(10);
  static constexpr size_t kInputSize = 1024;
  static constexpr size_t kInputCount = 2;
  static constexpr size_t kTotalDeviceMemoryBytes =
      kLogSize + kInputSize * kInputCount;
  // Setup memory allocations for the log and inputs
  BufferAllocation alloc(/*index=*/0,
                         /*size=*/kTotalDeviceMemoryBytes,
                         /*color=*/0);
  BufferAllocation::Slice log_slice(&alloc, /*offset=*/0, kLogSize);
  BufferAllocation::Slice inputs[kInputCount];
  for (int i = 0; i < kInputCount; ++i) {
    inputs[i] = BufferAllocation::Slice(
        &alloc, /*offset=*/kLogSize + i * kInputSize, kInputSize);
  }
  BufferAllocations allocations(
      {executor_->AllocateArray<uint8_t>(kTotalDeviceMemoryBytes)},
      executor_->device_ordinal(), allocator_.get());
  se::DeviceAddressBase log_mem = allocations.GetDeviceAddress(log_slice);
  se::DeviceAddressBase inputs0_mem = allocations.GetDeviceAddress(inputs[0]);
  se::DeviceAddressBase inputs1_mem = allocations.GetDeviceAddress(inputs[1]);
  // Initialize the log in device memory
  TF_ASSERT_OK_AND_ASSIGN(auto device_log,
                          BufferDebugLog<BufferDebugLogEntry>::CreateOnDevice(
                              *stream_, se::DeviceAddress<uint8_t>(log_mem)));
  // Fill inputs with some data
  std::vector<uint32_t> zeros(1024, 0);
  zeros[123] = 12341234;  // expected checksum for inputs_mem[0]
  TF_ASSERT_OK(stream_->Memcpy(&inputs0_mem, zeros.data(), zeros.size()));
  zeros[123] = 56785678;  // expected checksum for inputs_mem[1]
  TF_ASSERT_OK(stream_->Memcpy(&inputs1_mem, zeros.data(), zeros.size()));

  // Setup parameters for Initialize/Prepare/ExecuteOnStream
  Thunk::InitializeParams init_params;
  init_params.executor = executor_;
  init_params.stream = stream_.get();

  ServiceExecutableRunOptions run_options;
  run_options.mutable_run_options()->set_stream(stream_.get());
  ASSERT_OK_AND_ASSIGN(
      CollectiveParams collective_params,
      CollectiveParams::Create(run_options, /*async_streams=*/{},
                               LocalDeviceId(executor_->device_ordinal())));
  CollectiveCliqueRequests clique_requests;
  CollectiveMultimemRegistry multimem_registry(
      executor_, collective_params.global_device_id);
  Thunk::PrepareParams prepare_params{&collective_params, &clique_requests,
                                      &multimem_registry, executor_,
                                      &allocations};

  Thunk::ExecuteParams execute_params = Thunk::ExecuteParams::Create(
      ServiceExecutableRunOptions(), allocations, stream_.get(),
      /*command_buffer_trace_stream=*/stream_.get(),
      /*collective_params=*/nullptr, /*collective_cliques=*/nullptr);
  auto metadata_store = std::make_shared<BufferDebugLogEntryMetadataStore>();

  BuffersDebugChecksumThunk thunk(
      Thunk::ThunkInfo(), log_slice,
      /*checked_thunk_id=*/ThunkId(123),
      {{/*buffer_idx=*/0, inputs[0]}, {/*buffer_idx=*/1, inputs[1]}},
      /*runs_before_checked_thunk=*/true, metadata_store);
  TF_ASSERT_OK(thunk.Initialize(init_params));
  TF_ASSERT_OK(thunk.Prepare(prepare_params));
  TF_ASSERT_OK(thunk.ExecuteOnStream(execute_params));
  TF_ASSERT_OK_AND_ASSIGN(std::vector<BufferDebugLogEntry> entries,
                          device_log.ReadFromDevice(*stream_));

  // BuffersDebugChecksumThunk launches a kernel for each input buffer, they may
  // complete in any order.
  EXPECT_THAT(entries,
              UnorderedElementsAre(
                  AllOf(IsEntryWithMetadata(
                            metadata_store,
                            Metadata{
                                /*thunk_id=*/ThunkId(123),
                                /*buffer_idx=*/0,
                                /*execution_id=*/0,
                                /*is_input=*/true,
                                BufferDebugLogEntryProto::CHECK_TYPE_CHECKSUM,
                            }),
                        Field(&BufferDebugLogEntry::value, 12341234)),
                  AllOf(IsEntryWithMetadata(
                            metadata_store,
                            Metadata{
                                /*thunk_id=*/ThunkId(123),
                                /*buffer_idx=*/1,
                                /*execution_id=*/0,
                                /*is_input=*/true,
                                BufferDebugLogEntryProto::CHECK_TYPE_CHECKSUM,
                            }),
                        Field(&BufferDebugLogEntry::value, 56785678))));
}

TEST_F(BuffersDebugChecksumThunkTest,
       ExecutesCorrectKernelsForDifferentDevices) {
  // Loaded kernels are associated with a specific device represented by its
  // StreamExecutor. The same Thunk will be Initialized once for each device,
  // which will load the kernel onto that device. During ExecuteOnStream, the
  // correct kernel needs to be launched.
  if (platform_->VisibleDeviceCount() < 2) {
    GTEST_SKIP() << "need at least 2 devices for this test";
  }

  static constexpr size_t kLogSizeBytes = 1024;
  static constexpr size_t kInputSizeBytes = 1024;

  struct TestDevice {
    se::StreamExecutor* executor;
    std::unique_ptr<se::Stream> stream;
    std::unique_ptr<stream_executor::StreamExecutorAddressAllocator> allocator;
    BufferAllocations allocations;
  };
  auto setup_device = [this](int device_ordinal) -> absl::StatusOr<TestDevice> {
    TF_ASSIGN_OR_RETURN(se::StreamExecutor * executor,
                        platform_->ExecutorForDevice(device_ordinal));
    TF_ASSIGN_OR_RETURN(std::unique_ptr<se::Stream> stream,
                        executor->CreateStream());
    auto allocator =
        std::make_unique<stream_executor::StreamExecutorAddressAllocator>(
            executor);
    BufferAllocations allocations(
        {executor->AllocateArray<uint8_t>(kLogSizeBytes + kInputSizeBytes)},
        executor->device_ordinal(), allocator.get());

    return TestDevice{std::move(executor), std::move(stream),
                      std::move(allocator), std::move(allocations)};
  };
  TF_ASSERT_OK_AND_ASSIGN(TestDevice device0, setup_device(0));
  TF_ASSERT_OK_AND_ASSIGN(TestDevice device1, setup_device(1));
  BufferAllocation allocation(0, kLogSizeBytes + kInputSizeBytes, 0);
  BufferAllocation::Slice log_slice(&allocation, 0, kLogSizeBytes);
  BufferAllocation::Slice input_slice(&allocation, kLogSizeBytes,
                                      kInputSizeBytes);
  BuffersDebugChecksumThunk thunk(
      Thunk::ThunkInfo(), log_slice,
      /*checked_thunk_id=*/ThunkId(123), {{/*buffer_idx=*/0, input_slice}},
      /*runs_before_checked_thunk=*/true,
      std::make_shared<BufferDebugLogEntryMetadataStore>());

  // Initialize the Thunk on both devices and run the kernel. An attempt to run
  // a kernel on the wrong device will fail with CUDA_ERROR_INVALID_HANDLE. The
  // error may be reported from the next operation on the stream, so assert on
  // BlockHostUntilDone as well.
  TF_ASSERT_OK(
      thunk.Initialize(Thunk::InitializeParams{/*executor=*/device0.executor}));
  TF_ASSERT_OK(thunk.ExecuteOnStream(Thunk::ExecuteParams::Create(
      ServiceExecutableRunOptions(), device0.allocations, device0.stream.get(),
      /*command_buffer_trace_stream=*/device0.stream.get(),
      /*collective_params=*/nullptr,
      /*collective_cliques=*/nullptr)));
  TF_ASSERT_OK(device0.stream->BlockHostUntilDone());

  TF_ASSERT_OK(
      thunk.Initialize(Thunk::InitializeParams{/*executor=*/device1.executor}));
  TF_ASSERT_OK(thunk.ExecuteOnStream(Thunk::ExecuteParams::Create(
      ServiceExecutableRunOptions(), device1.allocations, device1.stream.get(),
      /*command_buffer_trace_stream=*/device1.stream.get(),
      /*collective_params=*/nullptr,
      /*collective_cliques=*/nullptr)));
  TF_ASSERT_OK(device1.stream->BlockHostUntilDone());
}

}  // namespace
}  // namespace xla::gpu
