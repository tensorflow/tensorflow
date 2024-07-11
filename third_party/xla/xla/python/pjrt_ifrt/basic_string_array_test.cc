/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/python/pjrt_ifrt/basic_string_array.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/notification.h"
#include "absl/types/span.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/future.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/test_util.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/env.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace ifrt {
namespace {

using ::tsl::testing::StatusIs;

// Makes a simple single device sharded string array by means of
// `BasicStringArray::Create` factory method.
absl::StatusOr<tsl::RCReference<BasicStringArray>> CreateTestArray(
    Client* client, Future<BasicStringArray::Buffers> buffers,
    BasicStringArray::OnDoneWithBuffer on_done_with_buffer) {
  Shape shape({1});
  Device* device = client->addressable_devices().at(0);
  std::shared_ptr<const Sharding> sharding =
      SingleDeviceSharding::Create(device, MemoryKind());

  return BasicStringArray::Create(client, shape, sharding, std::move(buffers),
                                  std::move(on_done_with_buffer));
}

TEST(BasicStringArrayTest, CreateSuccess) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  BasicStringArray::Buffers buffers;
  buffers.push_back({"abc", "def"});

  // This test implicitly tests that the on_done_with_buffer can be a nullptr,
  // and that the destruction of the BasicStringArray object completes
  // successfully (even when the callback is a nullptr).
  TF_EXPECT_OK(CreateTestArray(client.get(),
                               Future<BasicStringArray::Buffers>(buffers),
                               /*on_done_with_buffer=*/nullptr));
}

TEST(BasicStringArrayTest, CreateFailure) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  // Create fails if with invalid future.
  EXPECT_THAT(CreateTestArray(client.get(), Future<BasicStringArray::Buffers>(),
                              /*on_done_with_buffer=*/nullptr),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(BasicStringArrayTest, Destruction) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  BasicStringArray::Buffers buffers;
  buffers.push_back({"abc", "def"});

  absl::Notification on_done_with_buffer_called;
  BasicStringArray::OnDoneWithBuffer on_done_with_buffer =
      [&on_done_with_buffer_called]() { on_done_with_buffer_called.Notify(); };

  auto array_creation_status_promise = PjRtFuture<>::CreatePromise();

  tsl::Env::Default()->SchedClosure(([&]() {
    auto array = CreateTestArray(client.get(),
                                 Future<BasicStringArray::Buffers>(buffers),
                                 std::move(on_done_with_buffer));

    array_creation_status_promise.Set(array.status());
    // `array` goes out of scope and gets destroyed.
  }));

  // Make sure that the array has been created successfully.
  TF_ASSERT_OK(Future<>(array_creation_status_promise).Await());

  // Destruction must release the buffer. That is, the `on_done_with_buffer`
  // callback must be called.
  on_done_with_buffer_called.WaitForNotification();
}

TEST(BasicStringArrayTest, GetReadyFutureSuccess) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  // Make a BasicStringArray with a future that is not ready.
  auto promise = Future<BasicStringArray::Buffers>::CreatePromise();
  auto buffers_future = Future<BasicStringArray::Buffers>(promise);
  TF_ASSERT_OK_AND_ASSIGN(auto array,
                          CreateTestArray(client.get(), buffers_future,
                                          /*on_done_with_buffer=*/nullptr));

  // Array should not be ready since the buffers future is not ready.
  auto ready_future = array->GetReadyFuture();
  EXPECT_FALSE(ready_future.IsKnownReady());

  // Make the buffers future ready asynchronously.
  BasicStringArray::Buffers buffers;
  buffers.push_back({"abc", "def"});
  tsl::Env::Default()->SchedClosure([&]() { promise.Set(buffers); });
  TF_EXPECT_OK(ready_future.Await());
}

TEST(BasicStringArrayTest, GetReadyFutureFailure) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  // Make a BasicStringArray with a future that is not ready.
  auto promise = Future<BasicStringArray::Buffers>::CreatePromise();
  auto buffers_future = Future<BasicStringArray::Buffers>(promise);
  TF_ASSERT_OK_AND_ASSIGN(auto array,
                          CreateTestArray(client.get(), buffers_future,
                                          /*on_done_with_buffer=*/nullptr));

  // Array should not be ready since the buffers future is not ready.
  auto ready_future = array->GetReadyFuture();
  EXPECT_FALSE(ready_future.IsKnownReady());

  // Make the buffers future ready with an error asynchronously
  tsl::Env::Default()->SchedClosure(
      [&]() { promise.Set(absl::InternalError("injected error")); });

  EXPECT_THAT(ready_future.Await(), StatusIs(absl::StatusCode::kInternal));
}

TEST(BasicStringArrayTest, Delete) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  BasicStringArray::Buffers buffers;
  buffers.push_back({"abc", "def"});
  absl::Notification on_done_with_buffer_called;
  BasicStringArray::OnDoneWithBuffer on_done_with_buffer =
      [&on_done_with_buffer_called]() { on_done_with_buffer_called.Notify(); };

  TF_ASSERT_OK_AND_ASSIGN(
      auto array,
      CreateTestArray(client.get(), Future<BasicStringArray::Buffers>(buffers),
                      std::move(on_done_with_buffer)));

  tsl::Env::Default()->SchedClosure([&]() { array->Delete(); });

  // Delete must have released the buffer by calling `on_done_with_buffer`.
  on_done_with_buffer_called.WaitForNotification();

  // IsDeleted should return true.
  EXPECT_TRUE(array->IsDeleted());
}

TEST(BasicStringArrayTest, MakeArrayFromHostBufferSuccess) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  Shape shape({1});
  Device* device = client->addressable_devices().at(0);
  std::shared_ptr<const Sharding> sharding =
      SingleDeviceSharding::Create(device, MemoryKind());

  auto string_views = std::make_shared<std::vector<absl::string_view>>();
  string_views->push_back("abc");
  string_views->push_back("def");
  const void* data = string_views->data();
  auto on_done_with_host_buffer = [string_views = std::move(string_views)]() {};

  TF_ASSERT_OK(client->MakeArrayFromHostBuffer(
      data, DType(DType::kString), shape,
      /*byte_strides=*/std::nullopt, std::move(sharding),
      Client::HostBufferSemantics::kImmutableOnlyDuringCall,
      std::move(on_done_with_host_buffer)));
}

TEST(BasicStringArrayTest, MakeArrayFromHostBufferErrorHandling) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  Shape shape({1});
  Device* device = client->addressable_devices().at(0);
  std::shared_ptr<const Sharding> single_device_sharding =
      SingleDeviceSharding::Create(device, MemoryKind());
  auto string_views = std::make_shared<std::vector<absl::string_view>>();
  string_views->push_back("abc");
  string_views->push_back("def");
  const void* data = string_views->data();
  auto on_done_with_host_buffer = [string_views = std::move(string_views)]() {};

  // MakeArrayFromHostBuffer should check and fail if `byte_strides` in not
  // nullopt.
  EXPECT_THAT(
      client->MakeArrayFromHostBuffer(
          data, DType(DType::kString), shape,
          /*byte_strides=*/std::optional<absl::Span<const int64_t>>({8}),
          single_device_sharding,
          Client::HostBufferSemantics::kImmutableOnlyDuringCall,
          on_done_with_host_buffer),
      StatusIs(absl::StatusCode::kInvalidArgument));

  // MakeArrayFromHostBuffer should check and fail if the sharding is not a
  // SingleDeviceSharding.
  std::shared_ptr<const Sharding> opaque_sharding =
      OpaqueSharding::Create(DeviceList({device}), MemoryKind());
  EXPECT_THAT(client->MakeArrayFromHostBuffer(
                  data, DType(DType::kString), shape,
                  /*byte_strides=*/std::nullopt, opaque_sharding,
                  Client::HostBufferSemantics::kImmutableOnlyDuringCall,
                  on_done_with_host_buffer),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // MakeArrayFromHostBuffer should check and fail if the requested
  // HostBufferSemantics is not supported.
  for (Client::HostBufferSemantics host_buffer_semantics :
       {Client::HostBufferSemantics::kImmutableUntilTransferCompletes,
        Client::HostBufferSemantics::kImmutableZeroCopy,
        Client::HostBufferSemantics::kMutableZeroCopy}) {
    SCOPED_TRACE(
        absl::StrCat("host_buffer_semantics: ", host_buffer_semantics));
    EXPECT_THAT(client->MakeArrayFromHostBuffer(
                    data, DType(DType::kString), shape,
                    /*byte_strides=*/std::nullopt, single_device_sharding,
                    host_buffer_semantics, on_done_with_host_buffer),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
