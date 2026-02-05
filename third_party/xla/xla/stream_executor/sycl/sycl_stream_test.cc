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

#include "xla/stream_executor/sycl/sycl_stream.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/kernel_thunk.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/sycl/sycl_event.h"
#include "xla/stream_executor/sycl/sycl_executor.h"
#include "xla/stream_executor/sycl/sycl_platform_id.h"
#include "xla/stream_executor/typed_kernel_factory.h"
#include "xla/tests/llvm_irgen_test_base.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor::sycl {
namespace {

constexpr int kDefaultDeviceOrdinal = 0;

using ::testing::Each;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::HasSubstr;
using ::testing::UnorderedElementsAreArray;

class SyclStreamTest : public xla::LlvmIrGenTestBase {
 public:
  std::optional<SyclExecutor> executor_;

 private:
  void SetUp() override {
    TF_ASSERT_OK_AND_ASSIGN(
        Platform * platform,
        stream_executor::PlatformManager::PlatformWithId(kSyclPlatformId));
    executor_.emplace(platform, kDefaultDeviceOrdinal);
    ASSERT_THAT(executor_->Init(), absl_testing::IsOk());
  }
};

TEST_F(SyclStreamTest, CreateWithNonDefaultPriority) {
  // SYCL doesn't support stream priorities yet, so we expect creation to fail
  // if a non-default priority is requested.
  EXPECT_THAT(SyclStream::Create(&executor_.value(),
                                 /*enable_multiple_streams=*/false,
                                 /*priority=*/StreamPriority::Highest),
              absl_testing::StatusIs(absl::StatusCode::kUnimplemented));
}

TEST_F(SyclStreamTest, Memset32) {
  constexpr int kBufferNumElements = 42;
  DeviceAddress<uint32_t> device_buffer =
      executor_->AllocateArray<uint32_t>(kBufferNumElements, 0);

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<SyclStream> stream,
                          SyclStream::Create(&executor_.value(),
                                             /*enable_multiple_streams=*/false,
                                             /*priority=*/std::nullopt));

  constexpr uint64_t kBufferSizeBytes = kBufferNumElements * sizeof(uint32_t);

  // Should fail due to the invalid size parameter.
  EXPECT_THAT(
      stream->Memset32(&device_buffer, 0xDEADBEEF, kBufferSizeBytes + 1),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));

  // Should fail due to the non-4-byte-aligned pointer.
  DeviceAddressBase unaligned_device_memory =
      device_buffer.GetByteSlice(/*offset_bytes=*/1, /*size_bytes=*/0);
  EXPECT_THAT(stream->Memset32(&unaligned_device_memory, 0xDEADBEEF,
                               kBufferSizeBytes + 1),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));

  // Correct call. Should succeed.
  EXPECT_THAT(stream->Memset32(&device_buffer, 0xDEADBEEF, kBufferSizeBytes),
              absl_testing::IsOk());

  std::array<uint32_t, kBufferNumElements> host_buffer;
  EXPECT_THAT(
      stream->Memcpy(host_buffer.data(), device_buffer, kBufferSizeBytes),
      absl_testing::IsOk());

  EXPECT_THAT(stream->BlockHostUntilDone(), absl_testing::IsOk());
  EXPECT_THAT(host_buffer, Each(0xDEADBEEF));
}

TEST_F(SyclStreamTest, MemZero) {
  constexpr int kBufferNumElements = 42;
  DeviceAddress<uint32_t> device_buffer =
      executor_->AllocateArray<uint32_t>(kBufferNumElements, 0);

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<SyclStream> stream,
                          SyclStream::Create(&executor_.value(),
                                             /*enable_multiple_streams=*/false,
                                             /*priority=*/std::nullopt));

  constexpr uint64_t kBufferSizeBytes = kBufferNumElements * sizeof(uint32_t);

  EXPECT_THAT(stream->Memset32(&device_buffer, 0xDEADBEEF, kBufferSizeBytes),
              absl_testing::IsOk());

  // We overwrite half the device_buffer with zeros.
  EXPECT_THAT(stream->MemZero(&device_buffer,
                              kBufferNumElements / 2 * sizeof(uint32_t)),
              absl_testing::IsOk());

  std::array<uint32_t, kBufferNumElements> host_buffer;
  EXPECT_THAT(
      stream->Memcpy(host_buffer.data(), device_buffer, kBufferSizeBytes),
      absl_testing::IsOk());

  EXPECT_THAT(stream->BlockHostUntilDone(), absl_testing::IsOk());
  // We expect the first half of the host_buffer to be zeros.
  EXPECT_THAT(
      absl::MakeConstSpan(host_buffer).subspan(0, kBufferNumElements / 2),
      Each(0x0));

  // And it shouldn't have touched the second half.
  EXPECT_THAT(absl::MakeConstSpan(host_buffer).subspan(kBufferNumElements / 2),
              Each(0xDEADBEEF));
}

TEST_F(SyclStreamTest, MemcpyHostToDeviceAndBack) {
  constexpr int kBufferNumElements = 42;
  DeviceAddress<uint32_t> device_buffer =
      executor_->AllocateArray<uint32_t>(kBufferNumElements, 0);

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<SyclStream> stream,
                          SyclStream::Create(&executor_.value(),
                                             /*enable_multiple_streams=*/false,
                                             /*priority=*/std::nullopt));

  constexpr uint64_t kBufferSizeBytes = kBufferNumElements * sizeof(uint32_t);

  std::array<uint32_t, kBufferNumElements> src_buffer;
  std::generate(src_buffer.begin(), src_buffer.end(),
                [i = 0]() mutable { return i++; });

  EXPECT_THAT(
      stream->Memcpy(&device_buffer, src_buffer.data(), kBufferSizeBytes),
      absl_testing::IsOk());

  std::array<uint32_t, kBufferNumElements> host_buffer;
  EXPECT_THAT(
      stream->Memcpy(host_buffer.data(), device_buffer, kBufferSizeBytes),
      absl_testing::IsOk());

  EXPECT_THAT(stream->BlockHostUntilDone(), absl_testing::IsOk());
  EXPECT_THAT(host_buffer, ElementsAreArray(src_buffer));
}

TEST_F(SyclStreamTest, MemcpyDeviceToDevice) {
  constexpr int kBufferNumElements = 42;
  DeviceAddress<uint32_t> device_buffer1 =
      executor_->AllocateArray<uint32_t>(kBufferNumElements, 0);
  DeviceAddress<uint32_t> device_buffer2 =
      executor_->AllocateArray<uint32_t>(kBufferNumElements, 0);

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<SyclStream> stream,
                          SyclStream::Create(&executor_.value(),
                                             /*enable_multiple_streams=*/false,
                                             /*priority=*/std::nullopt));

  constexpr uint64_t kBufferSizeBytes = kBufferNumElements * sizeof(uint32_t);

  EXPECT_THAT(stream->Memset32(&device_buffer1, 0xDEADBEEF, kBufferSizeBytes),
              absl_testing::IsOk());

  EXPECT_THAT(stream->Memcpy(&device_buffer2, device_buffer1, kBufferSizeBytes),
              absl_testing::IsOk());

  std::array<uint32_t, kBufferNumElements> host_buffer;
  EXPECT_THAT(
      stream->Memcpy(host_buffer.data(), device_buffer2, kBufferSizeBytes),
      absl_testing::IsOk());

  EXPECT_THAT(stream->BlockHostUntilDone(), absl_testing::IsOk());
  EXPECT_THAT(host_buffer, Each(0xDEADBEEF));
}

TEST_F(SyclStreamTest, DoHostCallbackAndBlockHostUntilDone) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<SyclStream> stream,
                          SyclStream::Create(&executor_.value(),
                                             /*enable_multiple_streams=*/false,
                                             /*priority=*/std::nullopt));

  bool callback_called = false;
  EXPECT_THAT(
      stream->DoHostCallback([&callback_called]() { callback_called = true; }),
      absl_testing::IsOk());

  EXPECT_THAT(stream->BlockHostUntilDone(), absl_testing::IsOk());
  EXPECT_TRUE(callback_called);
}

TEST_F(SyclStreamTest, LaunchKernel) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<SyclStream> stream,
                          SyclStream::Create(&executor_.value(),
                                             /*enable_multiple_streams=*/false,
                                             /*priority=*/std::nullopt));

  using AddKernel =
      TypedKernelFactory<DeviceAddress<int32_t>, DeviceAddress<int32_t>,
                         DeviceAddress<int32_t>>;

  absl::string_view hlo_ir = R"(
    ENTRY e {
      p0 = u32[4] parameter(0)
      p1 = u32[4] parameter(1)
      ROOT res = u32[4] add(p0, p1)
    })";

  xla::HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::HloModule> hlo_module,
                          xla::ParseAndReturnUnverifiedModule(hlo_ir, config));

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::Executable> exec,
      CompileToExecutable(std::move(hlo_module),
                          /*run_optimization_passes=*/true));

  auto* gpu_exec = static_cast<xla::gpu::GpuExecutable*>(exec.get());
  ASSERT_NE(gpu_exec, nullptr);

  const xla::gpu::SequentialThunk& seq_thunk = gpu_exec->GetThunk();
  EXPECT_EQ(seq_thunk.thunks().size(), 1);

  const xla::gpu::Thunk* thunk = seq_thunk.thunks().at(0).get();
  ASSERT_NE(thunk, nullptr);
  EXPECT_EQ(thunk->kind(), xla::gpu::Thunk::Kind::kKernel);

  const auto* kernel_thunk = dynamic_cast<const xla::gpu::KernelThunk*>(thunk);
  ASSERT_NE(kernel_thunk, nullptr);

  std::string kernel_name = kernel_thunk->kernel_name();

  std::vector<uint8_t> spirv_binary(gpu_exec->binary());

  KernelLoaderSpec spec = KernelLoaderSpec::CreateCudaCubinInMemorySpec(
      spirv_binary, kernel_name, 3);

  TF_ASSERT_OK_AND_ASSIGN(auto add,
                          AddKernel::Create(&executor_.value(), spec));

  constexpr int64_t kLength = 4;
  constexpr int64_t kByteLength = sizeof(int32_t) * kLength;

  // Prepare arguments: a=3, b=2, c=0
  DeviceAddress<int32_t> a = executor_->AllocateArray<int32_t>(kLength, 0);
  DeviceAddress<int32_t> b = executor_->AllocateArray<int32_t>(kLength, 0);
  DeviceAddress<int32_t> c = executor_->AllocateArray<int32_t>(kLength, 0);

  EXPECT_THAT(stream->Memset32(&a, 3, kByteLength), absl_testing::IsOk());
  EXPECT_THAT(stream->Memset32(&b, 2, kByteLength), absl_testing::IsOk());
  EXPECT_THAT(stream->MemZero(&c, kByteLength), absl_testing::IsOk());
  EXPECT_THAT(add.Launch(ThreadDim(kLength), BlockDim(), stream.get(), a, b, c),
              absl_testing::IsOk());

  EXPECT_THAT(stream->BlockHostUntilDone(), absl_testing::IsOk());

  std::array<int32_t, kLength> host_buffer;
  EXPECT_THAT(stream->Memcpy(host_buffer.data(), c, kByteLength),
              absl_testing::IsOk());
  EXPECT_THAT(host_buffer, Each(5));
}

TEST_F(SyclStreamTest, SetName) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<SyclStream> stream,
                          SyclStream::Create(&executor_.value(),
                                             /*enable_multiple_streams=*/false,
                                             /*priority=*/std::nullopt));

  constexpr absl::string_view kStreamName = "Test stream";
  stream->SetName(std::string(kStreamName));
  EXPECT_EQ(stream->GetName(), kStreamName);
}

TEST_F(SyclStreamTest, WaitForEvent) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<SyclStream> stream,
                          SyclStream::Create(&executor_.value(),
                                             /*enable_multiple_streams=*/false,
                                             /*priority=*/std::nullopt));

  TF_ASSERT_OK_AND_ASSIGN(SyclEvent event,
                          SyclEvent::Create(&executor_.value()));

  EXPECT_THAT(stream->WaitFor(&event), absl_testing::IsOk());

  bool callback_called = false;
  EXPECT_THAT(
      stream->DoHostCallback([&callback_called]() { callback_called = true; }),
      absl_testing::IsOk());

  EXPECT_THAT(stream->RecordEvent(&event), absl_testing::IsOk());
  EXPECT_THAT(stream->BlockHostUntilDone(), absl_testing::IsOk());
  EXPECT_TRUE(callback_called);
}

// SYCL does not guarantee any specific ordering of host callbacks
// across different streams, even when using event or stream waits.
// Therefore, the WaitForOtherStream test (which tests WaitFor(Stream* other))
// is omitted for SYCL.

TEST_F(SyclStreamTest, MultipleStreams) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<SyclStream> stream1,
                          SyclStream::Create(&executor_.value(),
                                             /*enable_multiple_streams=*/true,
                                             /*priority=*/std::nullopt));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<SyclStream> stream2,
                          SyclStream::Create(&executor_.value(),
                                             /*enable_multiple_streams=*/true,
                                             /*priority=*/std::nullopt));

  std::vector<int> host_buffer;
  EXPECT_THAT(
      stream1->DoHostCallback([&host_buffer]() { host_buffer.push_back(1); }),
      absl_testing::IsOk());
  EXPECT_THAT(
      stream2->DoHostCallback([&host_buffer]() { host_buffer.push_back(2); }),
      absl_testing::IsOk());

  EXPECT_THAT(stream1->BlockHostUntilDone(), absl_testing::IsOk());
  EXPECT_THAT(stream2->BlockHostUntilDone(), absl_testing::IsOk());

  std::vector<int> expected = {1, 2};

  // Callbacks may run concurrently or in any order since the streams are
  // independent.
  EXPECT_THAT(host_buffer, UnorderedElementsAreArray(expected));
}

}  // namespace
}  // namespace stream_executor::sycl
