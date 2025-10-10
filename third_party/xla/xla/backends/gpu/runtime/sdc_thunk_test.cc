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

#include "xla/backends/gpu/runtime/sdc_thunk.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/sdc_buffer_id.h"
#include "xla/backends/gpu/runtime/sdc_log_structs.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_id.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/resource_requests.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/stream_executor/cuda/sdc_log.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor_memory_allocator.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/testing/temporary_directory.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "tsl/platform/path.h"

namespace xla::gpu {
namespace {

namespace se = stream_executor;

using ::stream_executor::cuda::SdcLog;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAre;
using ::tsl::proto_testing::EqualsProto;

class SdcThunkTest : public ::testing::Test {
 protected:
  void SetUp() override {
    TF_ASSERT_OK_AND_ASSIGN(platform_,
                            se::PlatformManager::PlatformWithName("CUDA"));
    TF_ASSERT_OK_AND_ASSIGN(executor_, platform_->ExecutorForDevice(0));
    TF_ASSERT_OK_AND_ASSIGN(stream_, executor_->CreateStream(std::nullopt));
    allocator_ =
        std::make_unique<se::StreamExecutorMemoryAllocator>(stream_->parent());
  }

  se::Platform* platform_;
  se::StreamExecutor* executor_;
  std::unique_ptr<se::Stream> stream_;
  std::unique_ptr<se::StreamExecutorMemoryAllocator> allocator_;
};

TEST_F(SdcThunkTest, CalculatesChecksums) {
  static constexpr size_t kLogSize = SdcLog::RequiredSizeForEntries(10);
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
  se::DeviceMemoryBase log_mem = allocations.GetDeviceAddress(log_slice);
  se::DeviceMemoryBase inputs0_mem = allocations.GetDeviceAddress(inputs[0]);
  se::DeviceMemoryBase inputs1_mem = allocations.GetDeviceAddress(inputs[1]);
  // Initialize the log in device memory
  TF_ASSERT_OK_AND_ASSIGN(
      SdcLog device_log,
      SdcLog::CreateOnDevice(*stream_, se::DeviceMemory<uint8_t>(log_mem)));
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
  ResourceRequests resource_requests;
  auto execute_params = Thunk::ExecuteParams::Create(
      ServiceExecutableRunOptions(), allocations, stream_.get(),
      /*command_buffer_trace_stream=*/stream_.get(),
      /*collective_params=*/nullptr, /*collective_cliques=*/nullptr);

  SdcThunk thunk(Thunk::ThunkInfo(), log_slice,
                 {{SdcBufferId::Create(ThunkId(123), 4).value(), inputs[0]},
                  {SdcBufferId::Create(ThunkId(456), 8).value(), inputs[1]}});
  TF_ASSERT_OK(thunk.Initialize(init_params));
  TF_ASSERT_OK(thunk.Prepare(Thunk::PrepareParams{}, resource_requests));
  TF_ASSERT_OK(thunk.ExecuteOnStream(execute_params));
  TF_ASSERT_OK_AND_ASSIGN(std::vector<SdcLogEntry> entries,
                          device_log.ReadFromDevice(*stream_));

  // SdcThunk launches a kernel for each input buffer, they may complete in any
  // order.
  EXPECT_THAT(entries,
              UnorderedElementsAre(
                  SdcLogEntry{
                      /*entry_id=*/SdcBufferId::Create(ThunkId(123), 4).value(),
                      /*checksum=*/12341234,
                  },
                  SdcLogEntry{
                      /*entry_id=*/SdcBufferId::Create(ThunkId(456), 8).value(),
                      /*checksum=*/56785678,
                  }));
}

TEST_F(SdcThunkTest, SdcDumpLogThunkDumpsLog) {
  static constexpr size_t kLogSize = SdcLog::RequiredSizeForEntries(10);
  // Setup log in device memory
  BufferAllocation alloc(/*index=*/0, /*size=*/kLogSize, /*color=*/0);
  BufferAllocation::Slice log_slice(&alloc, /*offset=*/0, kLogSize);
  BufferAllocations allocations({executor_->AllocateArray<uint8_t>(kLogSize)},
                                executor_->device_ordinal(), allocator_.get());
  auto log_mem =
      se::DeviceMemory<uint8_t>(allocations.GetDeviceAddress(log_slice));
  const SdcLogHeader header = {/*write_idx=*/2,
                               /*capacity=*/10};
  const SdcLogEntry entries[] = {
      {/*entry_id=*/SdcBufferId::Create(ThunkId(123), 4).value(),
       /*checksum=*/12341234},
      {/*entry_id=*/SdcBufferId::Create(ThunkId(567), 8).value(),
       /*checksum=*/56785678},
  };
  std::vector<uint8_t> log_data(sizeof(header) + sizeof(entries));
  memcpy(log_data.data(), &header, sizeof(header));
  memcpy(log_data.data() + sizeof(header), entries, sizeof(entries));
  TF_ASSERT_OK(stream_->MemcpyH2D(absl::MakeConstSpan(log_data), &log_mem));
  TF_ASSERT_OK(stream_->BlockHostUntilDone());
  // Setup parameters for SdcDumpLogThunk constructor
  TF_ASSERT_OK_AND_ASSIGN(
      tsl::testing::TemporaryDirectory dump_folder,
      tsl::testing::TemporaryDirectory::CreateForCurrentTestcase());
  const HloModule hlo_module("test_module", HloModuleConfig());
  DebugOptions debug_options = DefaultDebugOptionsIgnoringFlags();
  debug_options.set_xla_dump_to(dump_folder.path());
  // Setup parameters for Initialize/Prepare/ExecuteOnStream
  Thunk::InitializeParams init_params;
  init_params.executor = executor_;
  init_params.stream = stream_.get();
  ResourceRequests resource_requests;
  auto execute_params = Thunk::ExecuteParams::Create(
      ServiceExecutableRunOptions(), allocations, stream_.get(),
      /*command_buffer_trace_stream=*/stream_.get(),
      /*collective_params=*/nullptr, /*collective_cliques=*/nullptr);

  SdcDumpLogThunk thunk(Thunk::ThunkInfo(), log_slice, hlo_module,
                        debug_options);
  TF_ASSERT_OK(thunk.Initialize(init_params));
  TF_ASSERT_OK(thunk.Prepare(Thunk::PrepareParams{}, resource_requests));
  TF_ASSERT_OK(thunk.ExecuteOnStream(execute_params));

  std::vector<std::string> matches;
  TF_ASSERT_OK(tsl::Env::Default()->GetMatchingPaths(
      tsl::io::JoinPath(dump_folder.path(), "*sdc_log*"), &matches));
  ASSERT_THAT(matches, SizeIs(1));

  SdcLogProto sdc_log_proto;
  TF_ASSERT_OK(tsl::ReadTextOrBinaryProto(tsl::Env::Default(), matches[0],
                                          &sdc_log_proto));
  EXPECT_THAT(sdc_log_proto, EqualsProto(R"pb(
                entries { thunk_id: 123 buffer_idx: 4 checksum: 12341234 }
                entries { thunk_id: 567 buffer_idx: 8 checksum: 56785678 }
              )pb"));
}

}  // namespace
}  // namespace xla::gpu
