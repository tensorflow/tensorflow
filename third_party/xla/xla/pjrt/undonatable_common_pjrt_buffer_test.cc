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

#include "xla/pjrt/undonatable_common_pjrt_buffer.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/casts.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/future.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/pjrt/common_pjrt_client.h"
#include "xla/pjrt/device_event.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_client_options.h"
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/future.h"

namespace xla {
namespace {

using ::absl_testing::StatusIs;
using ::testing::Eq;

// Test fixture holding common lifecycle objects.
class UndonatableCommonPjRtBufferTest : public ::testing::Test {
 protected:
  void SetUp() override {
    shape_ = std::make_shared<Shape>(ShapeUtil::MakeShape(F32, {2, 2}));
    ASSERT_OK_AND_ASSIGN(client_, GetXlaPjrtCpuClient(CpuClientOptions{}));
    ASSERT_EQ(client_->addressable_devices().size(), 1);
    device_ = client_->addressable_devices().front();
    ASSERT_OK_AND_ASSIGN(memory_space_, device_->default_memory_space());
    std::vector<float> data(4, 0.0f);
    ASSERT_OK_AND_ASSIGN(
        src_buffer_,
        client_->BufferFromHostBuffer(
            data.data(), F32, {2, 2}, /*byte_strides=*/std::nullopt,
            /*host_buffer_semantics=*/
            PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
            /*on_done_with_host_buffer=*/nullptr, memory_space_,
            /*device_layout=*/nullptr));

    ASSERT_OK_AND_ASSIGN(
        raw_buffer_, PjRtRawBuffer::CreateRawAliasOfBuffer(src_buffer_.get()));
  }

  std::shared_ptr<const Shape> shape_;
  std::unique_ptr<PjRtClient> client_;
  PjRtDevice* device_;
  PjRtMemorySpace* memory_space_;
  std::unique_ptr<PjRtBuffer> src_buffer_;
  PjRtRawBufferRef raw_buffer_;
};

// 1. Verify basic construction, acquisition, and identification.
TEST_F(UndonatableCommonPjRtBufferTest, LifetimeAndAcquisition) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtBuffer> buffer,
      UndonatableCommonPjRtBuffer::Create(std::move(src_buffer_)));
  auto* undonatable_buffer =
      dynamic_cast<UndonatableCommonPjRtBuffer*>(buffer.get());
  ASSERT_NE(undonatable_buffer, nullptr);

  EXPECT_FALSE(buffer->IsDeleted());
  EXPECT_THAT(buffer->memory_space(), Eq(memory_space_));

  // Fast hold-free retrieval should yield the exact same underlying buffer.
  PjRtRawBufferRef retrieved_raw = undonatable_buffer->AcquireRawBufferRef();
  EXPECT_THAT(retrieved_raw.get(), Eq(raw_buffer_.get()));
}

// 2. Verify Metadata Accessors.
TEST_F(UndonatableCommonPjRtBufferTest, MetadataAccessors) {
  bool expected_is_on_cpu = src_buffer_->IsOnCpu();
  ASSERT_OK_AND_ASSIGN(size_t expected_size,
                       src_buffer_->GetOnDeviceSizeInBytes());

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtBuffer> buffer,
      UndonatableCommonPjRtBuffer::Create(std::move(src_buffer_)));

  EXPECT_THAT(buffer->device(), Eq(device_));
  EXPECT_THAT(buffer->client(), Eq(client_.get()));
  EXPECT_EQ(buffer->IsOnCpu(), expected_is_on_cpu);

  ASSERT_OK_AND_ASSIGN(size_t size, buffer->GetOnDeviceSizeInBytes());
  EXPECT_EQ(size, expected_size);
}

// 3. Verify RawBuffer factory registry.
TEST_F(UndonatableCommonPjRtBufferTest, RawBufferFactoryRegistryWorks) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtBuffer> buffer,
      UndonatableCommonPjRtBuffer::Create(std::move(src_buffer_)));

  ASSERT_OK_AND_ASSIGN(PjRtRawBufferRef retrieved_raw,
                       PjRtRawBuffer::CreateRawAliasOfBuffer(buffer.get()));
  EXPECT_THAT(retrieved_raw.get(), Eq(raw_buffer_.get()));
}

// 4. Verify Constructor protections (safety against being born deleted).
TEST_F(UndonatableCommonPjRtBufferTest, DiesOnNullRawBuffer) {
  EXPECT_DEATH(UndonatableCommonPjRtBuffer(
                   shape_, PjRtRawBufferRef(),
                   absl::InlinedVector<PjRtDeviceEventRef, 2>(), memory_space_),
               "raw_buffer cannot be null");
}

TEST_F(UndonatableCommonPjRtBufferTest, CreateReturnsErrorOnNullBuffer) {
  EXPECT_THAT(UndonatableCommonPjRtBuffer::Create(nullptr),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

// 5. Verify calling Delete() on UndonatableCommonPjRtBuffer is a no-op:
// IsDeleted() remains false and raw buffer acquisition continues to succeed.
TEST_F(UndonatableCommonPjRtBufferTest, DeleteIsNoOpAndBufferRemainsValid) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtBuffer> buffer,
      UndonatableCommonPjRtBuffer::Create(std::move(src_buffer_)));

  buffer->Delete();
  EXPECT_FALSE(buffer->IsDeleted());

  // Retrieving raw should continue to return the valid underlying buffer.
  auto* undonatable_buffer =
      dynamic_cast<UndonatableCommonPjRtBuffer*>(buffer.get());
  ASSERT_NE(undonatable_buffer, nullptr);
  PjRtRawBufferRef retrieved_raw = undonatable_buffer->AcquireRawBufferRef();
  EXPECT_THAT(retrieved_raw.get(), Eq(raw_buffer_.get()));

  // Critical constraint: Memory space metadata must still be safe/retrievable
  // locklessly.
  EXPECT_THAT(buffer->memory_space(), Eq(memory_space_));
}

// 6. Verify rejection of donations (Core Guarantee Contract).
TEST_F(UndonatableCommonPjRtBufferTest, RejectsDonation) {
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtBuffer> buffer,
      UndonatableCommonPjRtBuffer::Create(std::move(src_buffer_)));

  EXPECT_THAT(buffer->DonateWithControlDependency({}),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

// 7. Verify GetReadyFuture behaviors.
TEST_F(UndonatableCommonPjRtBufferTest, GetReadyFuture_CachesFuture) {
  auto [promise, future] = tsl::MakePromise<void>();
  PjRtDeviceEventRef event =
      PjRtDeviceEventPtr::FromAsyncValue(future.async_value()).CopyRef();
  absl::InlinedVector<PjRtDeviceEventRef, 2> events;
  events.push_back(std::move(event));

  auto buffer = std::make_unique<UndonatableCommonPjRtBuffer>(
      shape_, raw_buffer_, std::move(events), memory_space_);

  Future<> fut1 = buffer->GetReadyFuture();
  Future<> fut2 = buffer->GetReadyFuture();

  // Verify both futures reference the exact same underlying async value.
  EXPECT_EQ(fut1.async_value(), fut2.async_value());

  EXPECT_FALSE(fut1.IsReady());
  EXPECT_FALSE(fut2.IsReady());

  // Resolve the underlying event.
  promise.Set();

  // Both should resolve, indicating they share the same cached state.
  EXPECT_TRUE(fut1.IsReady());
  EXPECT_TRUE(fut1.Await().ok());

  EXPECT_TRUE(fut2.IsReady());
  EXPECT_TRUE(fut2.Await().ok());
}

TEST_F(UndonatableCommonPjRtBufferTest, GetReadyFuture_PropagatesErrors) {
  auto future = tsl::MakeUnconstructedAsyncValueRef<int>();
  PjRtDeviceEventRef event =
      PjRtDeviceEventPtr::FromAsyncValue(future.GetAsyncValue()).CopyRef();
  absl::InlinedVector<PjRtDeviceEventRef, 2> events;
  events.push_back(std::move(event));

  auto buffer = std::make_unique<UndonatableCommonPjRtBuffer>(
      shape_, raw_buffer_, std::move(events), memory_space_);

  Future<> ready_fut = buffer->GetReadyFuture();
  EXPECT_FALSE(ready_fut.IsReady());

  // Resolve with an error!
  future.SetError(absl::InternalError("fake error"));

  absl::Status status = ready_fut.Await();
  EXPECT_TRUE(ready_fut.IsReady());
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInternal, "fake error"));
}

TEST_F(UndonatableCommonPjRtBufferTest, ExecutesSuccessfullyWithoutDonation) {
  XlaBuilder builder("identity");
  auto param = Parameter(&builder, 0, *shape_, "param");
  ASSERT_OK_AND_ASSIGN(auto comp, builder.Build(param));

  ASSERT_OK_AND_ASSIGN(auto executable,
                       client_->CompileAndLoad(comp, CompileOptions{}));

  auto* common_pjrt_client = absl::down_cast<CommonPjRtClient*>(client_.get());
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtBuffer> buffer,
      common_pjrt_client->MakeUndonatable(std::move(src_buffer_)));

  ASSERT_OK_AND_ASSIGN(auto results,
                       executable->Execute({{buffer.get()}}, ExecuteOptions{}));

  ASSERT_EQ(results.size(), 1);     // One replica
  ASSERT_EQ(results[0].size(), 1);  // One output buffer
  PjRtBuffer* output_buffer = results[0][0].get();
  EXPECT_NE(output_buffer, nullptr);
}

TEST_F(UndonatableCommonPjRtBufferTest, ExecuteFailsIfDonationRequested) {
  XlaBuilder builder("identity_aliased");
  auto param = Parameter(&builder, 0, *shape_, "param");
  builder.SetUpAlias({/*output_index=*/}, /*param_number=*/0,
                     /*param_index=*/{});
  ASSERT_OK_AND_ASSIGN(auto comp, builder.Build(param));

  ASSERT_OK_AND_ASSIGN(auto executable,
                       client_->CompileAndLoad(comp, CompileOptions{}));

  auto* common_pjrt_client = absl::down_cast<CommonPjRtClient*>(client_.get());
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtBuffer> buffer,
      common_pjrt_client->MakeUndonatable(std::move(src_buffer_)));

  auto result_or = executable->Execute({{buffer.get()}}, ExecuteOptions{});

  EXPECT_THAT(
      result_or.status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               ::testing::HasSubstr(
                   "Donation requested on UndonatableCommonPjRtBuffer")));
}

TEST_F(UndonatableCommonPjRtBufferTest,
       ExecuteFailsIfDefinitionEventIsInError) {
  auto future = tsl::MakeUnconstructedAsyncValueRef<int>();
  future.SetError(absl::InternalError("Simulated upstream error"));

  absl::InlinedVector<PjRtDeviceEventRef, 2> events;
  events.push_back(
      PjRtDeviceEventPtr::FromAsyncValue(future.GetAsyncValue()).CopyRef());

  auto buffer = std::make_unique<UndonatableCommonPjRtBuffer>(
      shape_, raw_buffer_, std::move(events), memory_space_);

  XlaBuilder builder("identity");
  auto param = Parameter(&builder, 0, *shape_, "param");
  ASSERT_OK_AND_ASSIGN(auto comp, builder.Build(param));
  ASSERT_OK_AND_ASSIGN(auto executable,
                       client_->CompileAndLoad(comp, CompileOptions{}));

  ASSERT_OK_AND_ASSIGN(auto results,
                       executable->Execute({{buffer.get()}}, ExecuteOptions{}));
  ASSERT_EQ(results.size(), 1);     // One replica
  ASSERT_EQ(results[0].size(), 1);  // One output buffer
  PjRtBuffer* output_buffer = results[0][0].get();
  EXPECT_THAT(output_buffer->GetReadyFuture().Await(),
              StatusIs(absl::StatusCode::kInternal,
                       ::testing::HasSubstr("Simulated upstream error")));
}

}  // namespace
}  // namespace xla
