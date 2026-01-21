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

#include "xla/backends/gpu/runtime/buffers_float_check_thunk.h"

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
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
#include "xla/stream_executor/gpu/buffer_debug_log.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"
#include "xla/stream_executor/stream_executor_memory_allocator.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/types.h"
#include "xla/util.h"

namespace xla::gpu {
namespace {

namespace se = stream_executor;

using Metadata = BufferDebugLogEntryMetadataStore::Metadata;

using ::stream_executor::gpu::BufferDebugLog;
using ::testing::AllOf;
using ::testing::Field;
using ::testing::IsEmpty;
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

MATCHER_P(NanCountIs, value, "nan_count") {
  return ExplainMatchResult(value, arg.nan_count, result_listener);
}

MATCHER_P(InfCountIs, value, "inf_count") {
  return ExplainMatchResult(value, arg.inf_count, result_listener);
}

MATCHER_P(ZeroCountIs, value, "zero_count") {
  return ExplainMatchResult(value, arg.zero_count, result_listener);
}

class BuffersDebugFloatCheckThunkTest : public ::testing::Test {
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
          << "buffer float checking is not supported on CUDA architectures "
             "older than Pascal due to missing atomic fetch_add with "
             "system scope";
    }
  }

  se::Platform* platform_;
  se::StreamExecutor* executor_;
  std::unique_ptr<se::Stream> stream_;
  std::unique_ptr<stream_executor::StreamExecutorAddressAllocator> allocator_;
};

TEST_F(BuffersDebugFloatCheckThunkTest, CalculatesNanCounts) {
  static constexpr size_t kLogSize =
      BufferDebugLog<BufferDebugFloatCheckEntry>::RequiredSizeForEntries(10);
  static constexpr size_t kTmpSizeElems = 1024;
  static constexpr size_t kTmpSizeBytes = kTmpSizeElems * sizeof(uint32_t);
  static constexpr size_t kInputElems = 1024;
  static constexpr size_t kInputSizeInBytes = kInputElems * sizeof(float);
  static constexpr size_t kTotalDeviceMemoryBytes =
      kLogSize + kTmpSizeBytes + kInputSizeInBytes * 2;
  // Setup memory allocations for the log and inputs
  BufferAllocation alloc(/*index=*/0,
                         /*size=*/kTotalDeviceMemoryBytes,
                         /*color=*/0);
  BufferAllocation::Slice log_slice(&alloc, /*offset=*/0, kLogSize);
  BufferAllocation::Slice tmp_slice(&alloc, /*offset=*/kLogSize, kTmpSizeBytes);
  int64_t input_offset = kLogSize + kTmpSizeBytes;

  BufferAllocation::Slice inputs[2];
  int64_t input_size_bf16 = kInputElems * sizeof(Eigen::bfloat16);
  inputs[0] = BufferAllocation::Slice(&alloc, input_offset, input_size_bf16,
                                      PrimitiveType::BF16);
  input_offset += input_size_bf16;

  inputs[1] = BufferAllocation::Slice(
      &alloc, input_offset, kInputElems * sizeof(float), PrimitiveType::F32);

  BufferAllocations allocations(
      {executor_->AllocateArray<uint8_t>(kTotalDeviceMemoryBytes)},
      executor_->device_ordinal(), allocator_.get());
  se::DeviceAddressBase log_mem = allocations.GetDeviceAddress(log_slice);
  se::DeviceAddressBase inputs0_mem = allocations.GetDeviceAddress(inputs[0]);
  se::DeviceAddressBase inputs1_mem = allocations.GetDeviceAddress(inputs[1]);
  // Initialize the log in device memory
  TF_ASSERT_OK_AND_ASSIGN(
      auto device_log,
      BufferDebugLog<BufferDebugFloatCheckEntry>::CreateOnDevice(
          *stream_, se::DeviceAddress<uint8_t>(log_mem)));
  // Fill inputs with some data
  {
    std::vector<Eigen::bfloat16> data(kInputElems, Eigen::bfloat16(0));
    data[123] = std::numeric_limits<Eigen::bfloat16>::quiet_NaN();
    TF_ASSERT_OK(stream_->Memcpy(&inputs0_mem, data.data(), kInputSizeInBytes));
  }
  {
    std::vector<float> data(kInputElems, 0);
    data[456] = std::numeric_limits<float>::quiet_NaN();
    data[789] = std::numeric_limits<float>::quiet_NaN();
    TF_ASSERT_OK(stream_->Memcpy(&inputs1_mem, data.data(), kInputSizeInBytes));
  }

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

  Thunk::ThunkInfo checked_thunk_info;
  checked_thunk_info.thunk_id = ThunkId(123);
  BuffersDebugFloatCheckThunk thunk(
      Thunk::ThunkInfo(), checked_thunk_info, log_slice, tmp_slice,
      {{/*buffer_idx=*/0, inputs[0]}, {/*buffer_idx=*/1, inputs[1]}},
      metadata_store);
  TF_ASSERT_OK(thunk.Initialize(init_params));
  TF_ASSERT_OK(thunk.Prepare(prepare_params));
  TF_ASSERT_OK(thunk.ExecuteOnStream(execute_params));
  TF_ASSERT_OK_AND_ASSIGN(std::vector<BufferDebugFloatCheckEntry> entries,
                          device_log.ReadFromDevice(*stream_));

  // BuffersDebugFloatCheckThunk launches a kernel for each input buffer, they
  // may complete in any order.
  EXPECT_THAT(entries,
              UnorderedElementsAre(
                  IsEntryWithMetadata(
                      metadata_store,
                      Metadata{
                          /*thunk_id=*/ThunkId(123),
                          /*buffer_idx=*/0,
                          /*execution_id=*/0,
                          /*is_input=*/false,
                          BufferDebugLogEntryProto::CHECK_TYPE_FLOAT_CHECKS,
                      }),
                  IsEntryWithMetadata(
                      metadata_store,
                      Metadata{
                          /*thunk_id=*/ThunkId(123),
                          /*buffer_idx=*/1,
                          /*execution_id=*/0,
                          /*is_input=*/false,
                          BufferDebugLogEntryProto::CHECK_TYPE_FLOAT_CHECKS,
                      })));
}

TEST_F(BuffersDebugFloatCheckThunkTest,
       ExecutesCorrectKernelsForDifferentDevices) {
  // Loaded kernels are associated with a specific device represented by its
  // StreamExecutor. The same Thunk will be Initialized once for each device,
  // which will load the kernel onto that device. During ExecuteOnStream, the
  // correct kernel needs to be launched.
  if (platform_->VisibleDeviceCount() < 2) {
    GTEST_SKIP() << "need at least 2 devices for this test";
  }

  static constexpr size_t kLogOffset = 0;
  static constexpr size_t kLogSizeBytes = 1024;
  static constexpr size_t kTmpOffset = kLogOffset + kLogSizeBytes;
  static constexpr size_t kTmpSizeBytes = 1024 * sizeof(uint32_t);
  static constexpr size_t kInputOffset = kTmpOffset + kTmpSizeBytes;
  static constexpr size_t kInputSizeBytes = 1024;
  static constexpr size_t kTotalDeviceMemory = kInputOffset + kInputSizeBytes;

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
        {executor->AllocateArray<uint8_t>(kTotalDeviceMemory)},
        executor->device_ordinal(), allocator.get());

    return TestDevice{std::move(executor), std::move(stream),
                      std::move(allocator), std::move(allocations)};
  };

  TF_ASSERT_OK_AND_ASSIGN(TestDevice device0, setup_device(0));
  TF_ASSERT_OK_AND_ASSIGN(TestDevice device1, setup_device(1));

  BufferAllocation allocation(/*index=*/0, kTotalDeviceMemory, /*color=*/0);
  BufferAllocation::Slice log_slice(&allocation, kLogOffset, kLogSizeBytes);
  BufferAllocation::Slice tmp_slice(&allocation, kTmpOffset, kTmpSizeBytes);
  BufferAllocation::Slice f32_slice(&allocation, kInputOffset, kInputSizeBytes,
                                    PrimitiveType::F32);
  BufferAllocation::Slice bf16_slice(&allocation, kInputOffset, kInputSizeBytes,
                                     PrimitiveType::BF16);
  Thunk::ThunkInfo checked_thunk_info;
  checked_thunk_info.thunk_id = ThunkId(123);
  BuffersDebugFloatCheckThunk thunk(
      Thunk::ThunkInfo(), checked_thunk_info, log_slice, tmp_slice,
      {{/*buffer_idx=*/0, f32_slice}, {/*buffer_idx=*/1, bf16_slice}},
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

TEST_F(BuffersDebugFloatCheckThunkTest,
       CalculatesNanCountsWithoutGoingOutOfBounds) {
  // Reproduces a bug where the size passed to the kernel was given as bytes
  // instead of elements, resulting in out-of-bounds accesses.
  //
  // Fill in buffers of 1024 elements with some data, including NaNs in second
  // halves of the buffers, then run the check thunk on the first half of the
  // buffers.
  //
  // The kernel should not access the second half of the buffers. If it does, it
  // will count the second half and give the wrong result.
  static constexpr size_t kLogSize =
      BufferDebugLog<BufferDebugFloatCheckEntry>::RequiredSizeForEntries(10);
  static constexpr size_t kTmpSizeElems = 1024;
  static constexpr size_t kTmpSizeBytes = kTmpSizeElems * sizeof(uint32_t);
  static constexpr size_t kInputElems = 1024;
  static constexpr size_t kInputSizeInBytes = kInputElems * sizeof(float);
  static constexpr size_t kTotalDeviceMemoryBytes =
      kLogSize + kTmpSizeBytes + kInputSizeInBytes * 2;
  // Setup memory allocations for the log and inputs
  BufferAllocation alloc(/*index=*/0,
                         /*size=*/kTotalDeviceMemoryBytes,
                         /*color=*/0);
  BufferAllocation::Slice log_slice(&alloc, /*offset=*/0, kLogSize);
  BufferAllocation::Slice tmp_slice(&alloc, /*offset=*/kLogSize, kTmpSizeBytes);
  int64_t input_offset = kLogSize + kTmpSizeBytes;

  BufferAllocation::Slice inputs[2];
  int64_t input_size_bf16 = kInputElems * sizeof(Eigen::bfloat16);
  inputs[0] = BufferAllocation::Slice(&alloc, input_offset, input_size_bf16,
                                      PrimitiveType::BF16);
  input_offset += input_size_bf16;

  inputs[1] = BufferAllocation::Slice(
      &alloc, input_offset, kInputElems * sizeof(float), PrimitiveType::F32);

  BufferAllocations allocations(
      {executor_->AllocateArray<uint8_t>(kTotalDeviceMemoryBytes)},
      executor_->device_ordinal(), allocator_.get());
  se::DeviceAddressBase log_mem = allocations.GetDeviceAddress(log_slice);
  se::DeviceAddressBase inputs0_mem = allocations.GetDeviceAddress(inputs[0]);
  se::DeviceAddressBase inputs1_mem = allocations.GetDeviceAddress(inputs[1]);
  // Initialize the log in device memory
  TF_ASSERT_OK_AND_ASSIGN(
      auto device_log,
      BufferDebugLog<BufferDebugFloatCheckEntry>::CreateOnDevice(
          *stream_, se::DeviceAddress<uint8_t>(log_mem)));
  // Fill inputs with some data
  {
    std::vector<Eigen::bfloat16> data(kInputElems, Eigen::bfloat16(1));
    std::fill_n(data.begin() + 123, 1,
                std::numeric_limits<Eigen::bfloat16>::quiet_NaN());
    std::fill_n(data.begin() + 234, 2,
                std::numeric_limits<Eigen::bfloat16>::infinity());
    std::fill_n(data.begin() + 345, 3, Eigen::bfloat16(0));
    // We're only running the kernel on the first 512 elements of the buffer, so
    // this is not supposed to be counted.
    std::fill(data.begin() + 512, data.end(),
              std::numeric_limits<Eigen::bfloat16>::quiet_NaN());
    TF_ASSERT_OK(stream_->Memcpy(&inputs0_mem, data.data(), kInputSizeInBytes));
    inputs[0] = BufferAllocation::Slice(&alloc, inputs[0].offset(),
                                        512 * sizeof(Eigen::bfloat16),
                                        PrimitiveType::BF16);
  }
  {
    std::vector<float> data(kInputElems, 1.f);
    std::fill_n(data.begin() + 123, 3, std::numeric_limits<float>::quiet_NaN());
    std::fill_n(data.begin() + 234, 2, std::numeric_limits<float>::infinity());
    std::fill_n(data.begin() + 345, 1, 0.f);
    // We're only running the kernel on the first 512 elements of the buffer, so
    // this is not supposed to be counted.
    std::fill(data.begin() + 512, data.end(),
              std::numeric_limits<float>::quiet_NaN());
    TF_ASSERT_OK(stream_->Memcpy(&inputs1_mem, data.data(), kInputSizeInBytes));
    inputs[1] = BufferAllocation::Slice(
        &alloc, inputs[1].offset(), 512 * sizeof(float), PrimitiveType::F32);
  }

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

  Thunk::ThunkInfo checked_thunk_info;
  checked_thunk_info.thunk_id = ThunkId(123);
  BuffersDebugFloatCheckThunk thunk(
      Thunk::ThunkInfo(), checked_thunk_info, log_slice, tmp_slice,
      {{/*buffer_idx=*/0, inputs[0]}, {/*buffer_idx=*/1, inputs[1]}},
      metadata_store);
  TF_ASSERT_OK(thunk.Initialize(init_params));
  TF_ASSERT_OK(thunk.Prepare(prepare_params));
  TF_ASSERT_OK(thunk.ExecuteOnStream(execute_params));
  TF_ASSERT_OK_AND_ASSIGN(std::vector<BufferDebugFloatCheckEntry> entries,
                          device_log.ReadFromDevice(*stream_));

  // BuffersDebugFloatCheckThunk launches a kernel for each input buffer, they
  // may complete in any order.
  EXPECT_THAT(
      entries,
      UnorderedElementsAre(
          AllOf(NanCountIs(1), InfCountIs(2), ZeroCountIs(3),
                IsEntryWithMetadata(
                    metadata_store,
                    Metadata{
                        /*thunk_id=*/ThunkId(123),
                        /*buffer_idx=*/0,
                        /*execution_id=*/0,
                        /*is_input=*/false,
                        BufferDebugLogEntryProto::CHECK_TYPE_FLOAT_CHECKS,
                    })),
          AllOf(NanCountIs(3), InfCountIs(2), ZeroCountIs(1),
                IsEntryWithMetadata(
                    metadata_store,
                    Metadata{
                        /*thunk_id=*/ThunkId(123),
                        /*buffer_idx=*/1,
                        /*execution_id=*/0,
                        /*is_input=*/false,
                        BufferDebugLogEntryProto::CHECK_TYPE_FLOAT_CHECKS,
                    }))));
}

TEST_F(BuffersDebugFloatCheckThunkTest, DoesNotAttemptLaunchingWithZeroBlocks) {
  // Reproduces a bug where the input buffer has 0 elements and the kernel
  // attempts to launch with BlockDim(0).
  static constexpr size_t kLogSize =
      BufferDebugLog<BufferDebugFloatCheckEntry>::RequiredSizeForEntries(10);
  static constexpr size_t kTmpSizeElems = 1024;
  static constexpr size_t kTmpSizeBytes = kTmpSizeElems * sizeof(uint32_t);
  static constexpr size_t kTotalDeviceMemoryBytes = kLogSize + kTmpSizeBytes;
  // Setup memory allocations for the log and inputs
  BufferAllocation alloc(/*index=*/0,
                         /*size=*/kTotalDeviceMemoryBytes,
                         /*color=*/0);
  BufferAllocation::Slice log_slice(&alloc, /*offset=*/0, kLogSize);
  BufferAllocation::Slice tmp_slice(&alloc, /*offset=*/kLogSize, kTmpSizeBytes);

  BufferAllocation::Slice inputs[2];
  inputs[0] = BufferAllocation::Slice(&alloc, 0, 0, PrimitiveType::BF16);
  inputs[1] = BufferAllocation::Slice(&alloc, 0, 0, PrimitiveType::F32);

  BufferAllocations allocations(
      {executor_->AllocateArray<uint8_t>(kTotalDeviceMemoryBytes)},
      executor_->device_ordinal(), allocator_.get());
  se::DeviceAddressBase log_mem = allocations.GetDeviceAddress(log_slice);
  // Initialize the log in device memory
  TF_ASSERT_OK_AND_ASSIGN(
      auto device_log,
      BufferDebugLog<BufferDebugFloatCheckEntry>::CreateOnDevice(
          *stream_, se::DeviceAddress<uint8_t>(log_mem)));

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

  Thunk::ThunkInfo checked_thunk_info;
  checked_thunk_info.thunk_id = ThunkId(123);
  BuffersDebugFloatCheckThunk thunk(
      Thunk::ThunkInfo(), checked_thunk_info, log_slice, tmp_slice,
      {{/*buffer_idx=*/0, inputs[0]}, {/*buffer_idx=*/1, inputs[1]}},
      metadata_store);
  TF_ASSERT_OK(thunk.Initialize(init_params));
  TF_ASSERT_OK(thunk.Prepare(prepare_params));
  // If the kernel is launched with BlockDim(0), then this will fail with
  // INVALID_ARGUMENT.
  TF_ASSERT_OK(thunk.ExecuteOnStream(execute_params));
  TF_ASSERT_OK_AND_ASSIGN(std::vector<BufferDebugFloatCheckEntry> entries,
                          device_log.ReadFromDevice(*stream_));

  // The zero-sized buffers should be skipped, so no entries should be written.
  EXPECT_THAT(entries, IsEmpty());
}

TEST_F(BuffersDebugFloatCheckThunkTest,
       DoesNotAccessOutOfBoundsMemoryWhenRoundingCausesABlockToDoNothing) {
  static constexpr size_t kLogSize =
      BufferDebugLog<BufferDebugFloatCheckEntry>::RequiredSizeForEntries(10);
  static constexpr size_t kTmpSizeElems = 1255;
  static constexpr size_t kTmpSizeBytes = kTmpSizeElems * sizeof(uint32_t);
  static constexpr size_t kInputElems = 1572864;
  static constexpr size_t kPaddedInputElems = kInputElems * 2;
  static constexpr size_t kPaddedInputSizeInBytes =
      kPaddedInputElems * sizeof(float);
  static constexpr size_t kTotalDeviceMemoryBytes =
      kLogSize + kTmpSizeBytes + kPaddedInputSizeInBytes * 2;

  // Repro test for a case where the particular combination of input size and
  // number of blocks makes "round up per-block input size to next 128b" makes
  // at least one block attempt to calculate a negative input size, overflowing
  // an unsigned integer.
  //
  // This happens when for any block:
  // RoundUpTo(input size / num_blocks, 128_bits) * block_idx > input_size
  //
  // Sanity check assert: assuming we launch as many blocks as temp space
  // allows, the last block will attempt to reach out of bounds when executing
  // this test.
  static_assert(
      xla::RoundUpTo(kInputElems / kTmpSizeElems, size_t{128 / CHAR_BIT}) *
          (kTmpSizeElems - 1) >
      kInputElems);

  // Setup memory allocations for the log and inputs
  BufferAllocation alloc(/*index=*/0,
                         /*size=*/kTotalDeviceMemoryBytes,
                         /*color=*/0);
  BufferAllocation::Slice log_slice(&alloc, /*offset=*/0, kLogSize);
  BufferAllocation::Slice tmp_slice(&alloc, /*offset=*/kLogSize, kTmpSizeBytes);
  int64_t input_offset = kLogSize + kTmpSizeBytes;

  BufferAllocation::Slice inputs[2];
  int64_t input_size_bf16 = kPaddedInputElems * sizeof(Eigen::bfloat16);
  inputs[0] = BufferAllocation::Slice(&alloc, input_offset, input_size_bf16,
                                      PrimitiveType::BF16);
  input_offset += input_size_bf16;

  inputs[1] = BufferAllocation::Slice(&alloc, input_offset,
                                      kPaddedInputElems * sizeof(float),
                                      PrimitiveType::F32);

  BufferAllocations allocations(
      {executor_->AllocateArray<uint8_t>(kTotalDeviceMemoryBytes)},
      executor_->device_ordinal(), allocator_.get());
  se::DeviceAddressBase log_mem = allocations.GetDeviceAddress(log_slice);
  se::DeviceAddressBase inputs0_mem = allocations.GetDeviceAddress(inputs[0]);
  se::DeviceAddressBase inputs1_mem = allocations.GetDeviceAddress(inputs[1]);
  // Initialize the log in device memory
  TF_ASSERT_OK_AND_ASSIGN(
      auto device_log,
      BufferDebugLog<BufferDebugFloatCheckEntry>::CreateOnDevice(
          *stream_, se::DeviceAddress<uint8_t>(log_mem)));
  // Fill inputs with some data
  {
    std::vector<Eigen::bfloat16> data(kPaddedInputElems, Eigen::bfloat16(1));
    std::fill_n(data.begin() + 123, 1,
                std::numeric_limits<Eigen::bfloat16>::quiet_NaN());
    std::fill_n(data.begin() + 234, 2,
                std::numeric_limits<Eigen::bfloat16>::infinity());
    std::fill_n(data.begin() + 345, 3, Eigen::bfloat16(0));
    // We're only running the kernel on the first 512 elements of the buffer, so
    // this is not supposed to be counted.
    std::fill(data.begin() + kInputElems, data.end(),
              std::numeric_limits<Eigen::bfloat16>::quiet_NaN());
    TF_ASSERT_OK(
        stream_->Memcpy(&inputs0_mem, data.data(), kPaddedInputSizeInBytes));
    inputs[0] = BufferAllocation::Slice(&alloc, inputs[0].offset(),
                                        kInputElems * sizeof(Eigen::bfloat16),
                                        PrimitiveType::BF16);
  }
  {
    std::vector<float> data(kPaddedInputElems, 1.f);
    std::fill_n(data.begin() + 123, 3, std::numeric_limits<float>::quiet_NaN());
    std::fill_n(data.begin() + 234, 2, std::numeric_limits<float>::infinity());
    std::fill_n(data.begin() + 345, 1, 0.f);
    // We're only running the kernel on the first 512 elements of the buffer, so
    // this is not supposed to be counted.
    std::fill(data.begin() + kInputElems, data.end(),
              std::numeric_limits<float>::quiet_NaN());
    TF_ASSERT_OK(
        stream_->Memcpy(&inputs1_mem, data.data(), kPaddedInputSizeInBytes));
    inputs[1] = BufferAllocation::Slice(&alloc, inputs[1].offset(),
                                        kInputElems * sizeof(float),
                                        PrimitiveType::F32);
  }

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

  Thunk::ThunkInfo checked_thunk_info;
  checked_thunk_info.thunk_id = ThunkId(123);
  BuffersDebugFloatCheckThunk thunk(
      Thunk::ThunkInfo(), checked_thunk_info, log_slice, tmp_slice,
      {{/*buffer_idx=*/0, inputs[0]}, {/*buffer_idx=*/1, inputs[1]}},
      metadata_store);
  TF_ASSERT_OK(thunk.Initialize(init_params));
  TF_ASSERT_OK(thunk.Prepare(prepare_params));
  TF_ASSERT_OK(thunk.ExecuteOnStream(execute_params));
  TF_ASSERT_OK_AND_ASSIGN(std::vector<BufferDebugFloatCheckEntry> entries,
                          device_log.ReadFromDevice(*stream_));

  // BuffersDebugFloatCheckThunk launches a kernel for each input buffer, they
  // may complete in any order.
  EXPECT_THAT(
      entries,
      UnorderedElementsAre(
          AllOf(NanCountIs(1), InfCountIs(2), ZeroCountIs(3),
                IsEntryWithMetadata(
                    metadata_store,
                    Metadata{
                        /*thunk_id=*/ThunkId(123),
                        /*buffer_idx=*/0,
                        /*execution_id=*/0,
                        /*is_input=*/false,
                        BufferDebugLogEntryProto::CHECK_TYPE_FLOAT_CHECKS,
                    })),
          AllOf(NanCountIs(3), InfCountIs(2), ZeroCountIs(1),
                IsEntryWithMetadata(
                    metadata_store,
                    Metadata{
                        /*thunk_id=*/ThunkId(123),
                        /*buffer_idx=*/1,
                        /*execution_id=*/0,
                        /*is_input=*/false,
                        BufferDebugLogEntryProto::CHECK_TYPE_FLOAT_CHECKS,
                    }))));
}

}  // namespace
}  // namespace xla::gpu
