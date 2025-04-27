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
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/notification.h"
#include "absl/types/span.h"
#include "llvm/Support/Casting.h"
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
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace ifrt {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::HasSubstr;
using ::tsl::testing::StatusIs;

// ////////////////////////////////////////////////////////////////////////////
//
// Common utility functions.
//

// Makes a simple single device sharded `BasicStringArray` from the
// user-supplied buffers and on_done_with_buffer callback by means of the
// factory method: `BasicStringArray::Create`. Uses the first device from the
// `client->addressable_devices()`.
absl::StatusOr<tsl::RCReference<Array>> CreateTestArray(
    Client* client, Future<BasicStringArray::Buffers> buffers,
    BasicStringArray::OnDoneWithBuffer on_done_with_buffer) {
  Shape shape({1});
  Device* device = client->addressable_devices().at(0);
  std::shared_ptr<const Sharding> sharding =
      SingleDeviceSharding::Create(device, MemoryKind());

  return BasicStringArray::Create(client, shape, sharding, std::move(buffers),
                                  std::move(on_done_with_buffer));
}

// Makes a single-sharded `BasicStringArray::Buffers` and its associated
// `BasicStringArray::OnDoneWithBuffer` from the given span of strings.
std::pair<BasicStringArray::Buffers, BasicStringArray::OnDoneWithBuffer>
MakeBuffersAndOnDoneWithBuffer(
    absl::Span<const absl::string_view> input_strings) {
  BasicStringArray::Buffers buffers;
  auto strings = std::make_shared<std::vector<absl::Cord>>();
  strings->reserve(input_strings.size());
  for (const auto input_str : input_strings) {
    strings->push_back(absl::Cord(input_str));
  }
  buffers.push_back(*strings);

  BasicStringArray::OnDoneWithBuffer on_done_with_buffer =
      [strings = std::move(strings)]() {};

  return std::make_pair(std::move(buffers), std::move(on_done_with_buffer));
}

// Makes a simple single device sharded `BasicStringArray` that is not ready at
// the time of creation. Returns a promise that can be set to make the array
// ready. If the callers set this promise with buffers (i.e., not an error),
// then they must ensure that the underlying strings live until the
// `on-host-buffer-done` callback they provided is run.
absl::StatusOr<std::pair<tsl::RCReference<BasicStringArray>,
                         Promise<BasicStringArray::Buffers>>>
CreateNonReadyTestArray(
    Client* client, Device* const device,
    BasicStringArray::OnDoneWithBuffer on_done_with_buffer) {
  auto buffers_promise = Future<BasicStringArray::Buffers>::CreatePromise();
  auto buffers_future = Future<BasicStringArray::Buffers>(buffers_promise);
  Shape shape({1});
  std::shared_ptr<const Sharding> sharding =
      SingleDeviceSharding::Create(device, MemoryKind());

  TF_ASSIGN_OR_RETURN(auto array,
                      BasicStringArray::Create(client, shape, sharding,
                                               std::move(buffers_future),
                                               std::move(on_done_with_buffer)));

  return std::make_pair(std::move(array), std::move(buffers_promise));
}

/////////////////////////////////////////////////////////////////////////////
//
// Tests related to BasicStringArray.
//

TEST(BasicStringArrayTest, CreateSuccess) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  BasicStringArray::Buffers buffers;
  buffers.push_back({absl::Cord("abc"), absl::Cord("def")});

  // This test implicitly tests that the on_done_with_buffer can be a nullptr,
  // and that the destruction of the BasicStringArray object completes
  // successfully (even when the callback is a nullptr).
  TF_EXPECT_OK(CreateTestArray(client.get(),
                               Future<BasicStringArray::Buffers>(buffers),
                               /*on_done_with_buffer=*/nullptr));
}

TEST(BasicStringArrayTest, CreateFailureWithInvalidFuture) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  // Create fails if with invalid future.
  EXPECT_THAT(CreateTestArray(client.get(), Future<BasicStringArray::Buffers>(),
                              /*on_done_with_buffer=*/nullptr),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(BasicStringArrayTest, Destruction) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  BasicStringArray::Buffers buffers;
  buffers.push_back({absl::Cord("abc"), absl::Cord("def")});

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

TEST(BasicStringArrayTest, InvalidBuffersAreHandledCorrectly) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  auto devices = client->addressable_devices();
  ASSERT_GE(devices.size(), 1);

  // Make a BasicStringArray::Buffer with two shards.
  auto shard0_data = std::make_shared<std::vector<absl::Cord>>();
  shard0_data->push_back(absl::Cord("abc"));
  auto shard1_data = std::make_shared<std::vector<absl::Cord>>();
  shard1_data->push_back(absl::Cord("def"));
  BasicStringArray::Buffers buffers;
  buffers.push_back(*shard0_data);
  buffers.push_back(*shard1_data);

  auto on_done_with_buffer = [shard0_data = std::move(shard0_data),
                              shard1_data = std::move(shard1_data)]() {};

  // Make a single device array that is not ready at the time of creation.
  TF_ASSERT_OK_AND_ASSIGN(
      auto ret, CreateNonReadyTestArray(client.get(), devices[0],
                                        std::move(on_done_with_buffer)));
  auto array = ret.first;
  auto promise = ret.second;
  auto basic_string_array = llvm::dyn_cast<BasicStringArray>(array.get());

  // Buffers with two shards and a single-device array are inconsistent.
  tsl::Env::Default()->SchedClosure([&]() { promise.Set(buffers); });

  EXPECT_THAT(basic_string_array->GetReadyFuture().Await(),
              StatusIs(absl::StatusCode::kFailedPrecondition));

  EXPECT_THAT(basic_string_array->buffers().Await(),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(BasicStringArrayTest, Delete) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  BasicStringArray::Buffers buffers;
  buffers.push_back({absl::Cord("abc"), absl::Cord("def")});
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

TEST(GetReadyFutureTest, SuccessCase) {
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
  buffers.push_back({absl::Cord("abc"), absl::Cord("def")});
  tsl::Env::Default()->SchedClosure([&]() { promise.Set(buffers); });
  TF_EXPECT_OK(ready_future.Await());
}

TEST(GetReadyFutureTest, FailureCases) {
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

TEST(MakeArrayFromHostBufferTest, SuccessCase) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  Shape shape({1});
  Device* device = client->addressable_devices().at(0);
  std::shared_ptr<const Sharding> sharding =
      SingleDeviceSharding::Create(device, MemoryKind());

  auto strings = std::make_shared<std::vector<absl::Cord>>();
  strings->push_back(absl::Cord("abc"));
  strings->push_back(absl::Cord("def"));
  const void* data = strings->data();
  auto on_done_with_host_buffer = [strings = std::move(strings)]() {};

  TF_ASSERT_OK(client->MakeArrayFromHostBuffer(
      data, DType(DType::kString), shape,
      /*byte_strides=*/std::nullopt, std::move(sharding),
      Client::HostBufferSemantics::kImmutableOnlyDuringCall,
      std::move(on_done_with_host_buffer)));
}

TEST(MakeArrayFromHostBufferTest, FailureCases) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  Shape shape({1});
  Device* device = client->addressable_devices().at(0);
  std::shared_ptr<const Sharding> single_device_sharding =
      SingleDeviceSharding::Create(device, MemoryKind());
  auto strings = std::make_shared<std::vector<absl::Cord>>();
  strings->push_back(absl::Cord("abc"));
  strings->push_back(absl::Cord("def"));
  const void* data = strings->data();
  auto on_done_with_host_buffer = [strings = std::move(strings)]() {};

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
      OpaqueSharding::Create(client->MakeDeviceList({device}), MemoryKind());
  EXPECT_THAT(client->MakeArrayFromHostBuffer(
                  data, DType(DType::kString), shape,
                  /*byte_strides=*/std::nullopt, opaque_sharding,
                  Client::HostBufferSemantics::kImmutableOnlyDuringCall,
                  on_done_with_host_buffer),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // MakeArrayFromHostBuffer should check and fail if the requested
  // HostBufferSemantics is not supported.
  for (Client::HostBufferSemantics host_buffer_semantics :
       {Client::HostBufferSemantics::kImmutableZeroCopy,
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

// Makes a single device sharded string ifrt::Array. Makes the necessary host
// string buffers.
absl::StatusOr<tsl::RCReference<Array>> MakeSingleDeviceStringTestArray(
    absl::Span<const std::string> contents, Client* client,
    Device* const device) {
  Shape shape(absl::MakeConstSpan({static_cast<int64_t>(contents.size())}));
  std::shared_ptr<const Sharding> sharding =
      SingleDeviceSharding::Create(device, MemoryKind());

  auto strings = std::make_shared<std::vector<absl::Cord>>();
  for (const auto& content : contents) {
    strings->push_back(absl::Cord(content));
  }
  const void* data = strings->data();
  auto on_done_with_host_buffer = [strings = std::move(strings)]() {};

  return client->MakeArrayFromHostBuffer(
      data, DType(DType::kString), shape,
      /*byte_strides=*/std::nullopt, std::move(sharding),
      Client::HostBufferSemantics::kImmutableOnlyDuringCall,
      std::move(on_done_with_host_buffer));
}

// Makes a single device sharded test array containing floats on the given
// Device.
absl::StatusOr<tsl::RCReference<Array>> MakeSingleDeviceFloatTestArray(
    Client* client, Device* const device) {
  DType dtype(DType::kF32);
  Shape shape({2, 3});
  auto data = std::make_unique<std::vector<float>>(6);
  std::iota(data->begin(), data->end(), 0);
  std::shared_ptr<const Sharding> sharding =
      SingleDeviceSharding::Create(device, MemoryKind());

  return client->MakeArrayFromHostBuffer(
      data->data(), dtype, shape,
      /*byte_strides=*/std::nullopt, sharding,
      Client::HostBufferSemantics::kImmutableOnlyDuringCall,
      /*on_done_with_host_buffer=*/nullptr);
}

// Makes a sharded string array with two shards. Uses the first two strings from
// the input `data`, one per shard.
absl::StatusOr<tsl::RCReference<Array>> MakeShardedStringTestArray(
    Client* client, absl::Span<const std::string> data,
    bool is_fully_replicated) {
  if (data.size() < 2) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Input data has too few strings. Need at least 2. got: ", data.size()));
  }
  auto devices = client->addressable_devices();
  if (devices.size() < 2) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Test client has too few devices. Need 2, got:", devices.size()));
  }

  std::shared_ptr<const Sharding> sharding = ConcreteEvenSharding::Create(
      client->MakeDeviceList({devices[0], devices[1]}), MemoryKind(),
      Shape({2, 1}), Shape({1}), is_fully_replicated);

  std::vector<tsl::RCReference<Array>> arrays;
  for (int i = 0; i < 2; ++i) {
    TF_ASSIGN_OR_RETURN(auto array, MakeSingleDeviceStringTestArray(
                                        {data[i]}, client, devices[i]));
    arrays.push_back(std::move(array));
  }

  return client->AssembleArrayFromSingleDeviceArrays(
      Shape({2, 1}), std::move(sharding), absl::MakeSpan(arrays),
      ArrayCopySemantics::kAlwaysCopy);
}

TEST(AssembleArrayFromSingleDeviceArraysTest,
     SuccessWithReadySingleDeviceArrays) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  // Make a BasicStringArray with two underlying basic string arrays.
  const std::vector<std::string> per_shard_contents({"shard 0", "shard 1"});
  TF_ASSERT_OK_AND_ASSIGN(
      auto array, MakeShardedStringTestArray(client.get(), per_shard_contents,
                                             /*is_fully_replicated=*/false));
  auto basic_string_array = llvm::dyn_cast<BasicStringArray>(array.get());
  ASSERT_NE(basic_string_array, nullptr);
  TF_ASSERT_OK_AND_ASSIGN(auto buffers, basic_string_array->buffers().Await());
  EXPECT_EQ(buffers.size(), 2);

  for (int i = 0; i < buffers.size(); ++i) {
    SCOPED_TRACE(absl::StrCat("buffer #", i));
    auto buffer = buffers[i];
    EXPECT_THAT(buffer, ElementsAre(per_shard_contents[i]));
  }
}

TEST(AssembleArrayFromSingleDeviceArraysTest, FailsWithNonStringArrays) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  auto devices = client->addressable_devices();
  ASSERT_GE(devices.size(), 2);
  std::shared_ptr<const Sharding> opaque_sharding = OpaqueSharding::Create(
      client->MakeDeviceList({devices[0], devices[1]}), MemoryKind());

  std::vector<tsl::RCReference<Array>> arrays(2);
  TF_ASSERT_OK_AND_ASSIGN(
      arrays[0], MakeSingleDeviceFloatTestArray(client.get(), devices[0]));
  TF_ASSERT_OK_AND_ASSIGN(
      arrays[1], MakeSingleDeviceStringTestArray({"string_array_contents"},
                                                 client.get(), devices[1]));

  EXPECT_THAT(client->AssembleArrayFromSingleDeviceArrays(
                  Shape({2}), std::move(opaque_sharding),
                  absl::MakeSpan(arrays), ArrayCopySemantics::kAlwaysCopy),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(AssembleArrayFromSingleDeviceArraysTest,
     FailsWithNonSingleDeviceStringArrays) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  auto devices = client->addressable_devices();
  ASSERT_GE(devices.size(), 2);
  std::shared_ptr<const Sharding> opaque_sharding = OpaqueSharding::Create(
      client->MakeDeviceList({devices[0], devices[1]}), MemoryKind());

  std::vector<tsl::RCReference<Array>> arrays(2);
  const std::vector<std::string> per_shard_contents({"abc", "def"});
  TF_ASSERT_OK_AND_ASSIGN(
      arrays[0], MakeShardedStringTestArray(client.get(), per_shard_contents,
                                            /*is_fully_replicated=*/false));
  TF_ASSERT_OK_AND_ASSIGN(
      arrays[1], MakeSingleDeviceStringTestArray({"string_array_contents"},
                                                 client.get(), devices[1]));

  EXPECT_THAT(client->AssembleArrayFromSingleDeviceArrays(
                  Shape({2}), std::move(opaque_sharding),
                  absl::MakeSpan(arrays), ArrayCopySemantics::kAlwaysCopy),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(AssembleArrayFromSingleDeviceArraysTest,
     FromNonReadySingleDeviceArraysSuccess) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  auto devices = client->addressable_devices();
  ASSERT_GE(devices.size(), 2);
  std::shared_ptr<const Sharding> opaque_sharding = OpaqueSharding::Create(
      client->MakeDeviceList({devices[0], devices[1]}), MemoryKind());

  // Make two non-ready single device sharded arrays.
  std::vector<tsl::RCReference<Array>> arrays;
  std::vector<Promise<BasicStringArray::Buffers>> promises;
  arrays.reserve(2);
  auto buf_and_on_done_with_buffer = MakeBuffersAndOnDoneWithBuffer({"abc"});
  auto buffers0 = buf_and_on_done_with_buffer.first;
  auto on_done_with_buffer0 = buf_and_on_done_with_buffer.second;
  TF_ASSERT_OK_AND_ASSIGN(
      auto ret, CreateNonReadyTestArray(client.get(), devices[0],
                                        std::move(on_done_with_buffer0)));
  arrays.push_back(std::move(ret.first));
  promises.push_back(std::move(ret.second));

  buf_and_on_done_with_buffer = MakeBuffersAndOnDoneWithBuffer({"def"});
  auto buffers1 = buf_and_on_done_with_buffer.first;
  auto on_done_with_buffer1 = buf_and_on_done_with_buffer.second;
  TF_ASSERT_OK_AND_ASSIGN(
      ret, CreateNonReadyTestArray(client.get(), devices[1],
                                   std::move(on_done_with_buffer1)));
  arrays.push_back(std::move(ret.first));
  promises.push_back(std::move(ret.second));

  TF_ASSERT_OK_AND_ASSIGN(
      auto array, client->AssembleArrayFromSingleDeviceArrays(
                      Shape({1}), std::move(opaque_sharding),
                      absl::MakeSpan(arrays), ArrayCopySemantics::kAlwaysCopy));

  tsl::Env::Default()->SchedClosure(([&]() mutable {
    promises[0].Set(buffers0);
    promises[1].Set(buffers1);
  }));

  auto basic_string_array = llvm::dyn_cast<BasicStringArray>(array.get());
  ASSERT_NE(basic_string_array, nullptr);

  auto buffers_future = basic_string_array->buffers();
  TF_ASSERT_OK_AND_ASSIGN(auto buffers, buffers_future.Await());
  ASSERT_EQ(buffers.size(), 2);
  EXPECT_THAT(buffers[0], ElementsAre("abc"));
  EXPECT_THAT(buffers[1], ElementsAre("def"));
}

TEST(AssembleArrayFromSingleDeviceArraysTest,
     FromNonReadySingleDeviceArraysFailure) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  auto devices = client->addressable_devices();
  ASSERT_GE(devices.size(), 2);
  std::shared_ptr<const Sharding> opaque_sharding = OpaqueSharding::Create(
      client->MakeDeviceList({devices[0], devices[1]}), MemoryKind());

  // Make two non-ready single device sharded arrays.
  std::vector<tsl::RCReference<Array>> arrays;
  std::vector<Promise<BasicStringArray::Buffers>> promises;
  arrays.reserve(2);

  TF_ASSERT_OK_AND_ASSIGN(
      auto ret, CreateNonReadyTestArray(client.get(), devices[0],
                                        /*on_done_with_buffer=*/nullptr));
  arrays.push_back(std::move(ret.first));
  promises.push_back(std::move(ret.second));

  TF_ASSERT_OK_AND_ASSIGN(
      ret, CreateNonReadyTestArray(client.get(), devices[1],
                                   /*on_done_with_buffer=*/nullptr));
  arrays.push_back(std::move(ret.first));
  promises.push_back(std::move(ret.second));

  // Make a sharded BasicStringArray out of the single device arrays.
  TF_ASSERT_OK_AND_ASSIGN(
      auto array, client->AssembleArrayFromSingleDeviceArrays(
                      Shape({1}), std::move(opaque_sharding),
                      absl::MakeSpan(arrays), ArrayCopySemantics::kAlwaysCopy));

  // Make the single device arrays become ready with an error.
  absl::Notification done_readying_single_device_arrays;
  tsl::Env::Default()->SchedClosure(([&]() mutable {
    promises[0].Set(absl::InternalError("injected from the test"));
    promises[1].Set(absl::InternalError("injected from the test"));
    done_readying_single_device_arrays.Notify();
  }));

  auto basic_string_array = llvm::dyn_cast<BasicStringArray>(array.get());
  ASSERT_NE(basic_string_array, nullptr);

  auto buffers_future = basic_string_array->buffers();
  EXPECT_THAT(buffers_future.Await(),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("injected from the test")));

  // Make sure to wait for the Closure to complete its work and set both
  // promises before returning from the test. The consequent destruction of the
  // promises can race with the Closure.
  done_readying_single_device_arrays.WaitForNotification();
}

TEST(DisassembleArrayIntoSingleDeviceArrays,
     SingleDeviceArrayDisassembleSuccess) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  auto [buffers, on_done_with_buffer] = MakeBuffersAndOnDoneWithBuffer({"abc"});

  TF_ASSERT_OK_AND_ASSIGN(
      auto array,
      CreateTestArray(client.get(), Future<BasicStringArray::Buffers>(buffers),
                      std::move(on_done_with_buffer)));

  TF_ASSERT_OK_AND_ASSIGN(auto disassembled_arrays,
                          array->DisassembleIntoSingleDeviceArrays(
                              ArrayCopySemantics::kAlwaysCopy,
                              SingleDeviceShardSemantics::kAddressableShards));

  ASSERT_EQ(disassembled_arrays.size(), 1);
  auto basic_string_array =
      llvm::dyn_cast<BasicStringArray>(disassembled_arrays[0].get());

  TF_ASSERT_OK_AND_ASSIGN(auto new_buffers,
                          basic_string_array->buffers().Await());
  ASSERT_EQ(new_buffers.size(), 1);
  EXPECT_THAT(new_buffers[0], ElementsAre("abc"));
}

TEST(DisassembleArrayIntoSingleDeviceArrays, ShardedArrayDisassembleSuccess) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  const std::vector<std::string> per_shard_contents({"abc", "def"});
  TF_ASSERT_OK_AND_ASSIGN(
      auto array, MakeShardedStringTestArray(client.get(), per_shard_contents,
                                             /*is_fully_replicated=*/false));

  TF_ASSERT_OK_AND_ASSIGN(auto disassembled_arrays,
                          array->DisassembleIntoSingleDeviceArrays(
                              ArrayCopySemantics::kAlwaysCopy,
                              SingleDeviceShardSemantics::kAddressableShards));

  ASSERT_EQ(disassembled_arrays.size(), 2);

  for (int i = 0; i < disassembled_arrays.size(); ++i) {
    SCOPED_TRACE(absl::StrCat("dissembled array: ", i));
    auto basic_string_array =
        llvm::dyn_cast<BasicStringArray>(disassembled_arrays[i].get());
    TF_ASSERT_OK_AND_ASSIGN(auto buffer, basic_string_array->buffers().Await());
    ASSERT_EQ(buffer.size(), 1);
    EXPECT_THAT(buffer[0], ElementsAre(per_shard_contents[i]));
  }
}

TEST(DisassembleArrayIntoSingleDeviceArrays, FailsIfTheArrayHasBeenDeleted) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  auto [buffers, on_done_with_buffer] = MakeBuffersAndOnDoneWithBuffer({"abc"});

  TF_ASSERT_OK_AND_ASSIGN(
      auto array,
      CreateTestArray(client.get(), Future<BasicStringArray::Buffers>(buffers),
                      std::move(on_done_with_buffer)));

  array->Delete();

  EXPECT_THAT(array->DisassembleIntoSingleDeviceArrays(
                  ArrayCopySemantics::kAlwaysCopy,
                  SingleDeviceShardSemantics::kAddressableShards),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(CopyTest, SuccessSingleDeviceShardedArray) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  auto devices = client->addressable_devices();
  ASSERT_GE(devices.size(), 2);

  auto [buffers, on_done_with_buffer] = MakeBuffersAndOnDoneWithBuffer({"abc"});
  std::vector<tsl::RCReference<Array>> arrays;
  TF_ASSERT_OK_AND_ASSIGN(
      arrays.emplace_back(),
      CreateTestArray(client.get(), Future<BasicStringArray::Buffers>(buffers),
                      std::move(on_done_with_buffer)));

  // CreateTestArray above would place the array on the first device. Use the
  // second one for the new array.
  TF_ASSERT_OK_AND_ASSIGN(
      auto new_arrays,
      client->CopyArrays(absl::MakeSpan(arrays),
                         client->MakeDeviceList({devices[1]}), MemoryKind(),
                         ArrayCopySemantics::kAlwaysCopy));

  auto new_basic_string_array =
      llvm::dyn_cast<BasicStringArray>(new_arrays[0].get());
  TF_ASSERT_OK_AND_ASSIGN(auto new_buffers,
                          new_basic_string_array->buffers().Await());
  ASSERT_EQ(new_buffers.size(), 1);
  EXPECT_THAT(new_buffers[0], ElementsAre("abc"));
}

TEST(CopyTest, SuccessMultiDeviceShardedArray) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  auto devices = client->addressable_devices();
  ASSERT_GE(devices.size(), 4);

  const std::vector<std::string> per_shard_contents({"shard 0", "shard 1"});
  std::vector<tsl::RCReference<Array>> arrays;
  TF_ASSERT_OK_AND_ASSIGN(
      arrays.emplace_back(),
      MakeShardedStringTestArray(client.get(), per_shard_contents,
                                 /*is_fully_replicated=*/false));

  TF_ASSERT_OK_AND_ASSIGN(
      auto new_arrays,
      client->CopyArrays(absl::MakeSpan(arrays),
                         client->MakeDeviceList({devices[2], devices[3]}),
                         MemoryKind(), ArrayCopySemantics::kAlwaysCopy));

  auto new_basic_string_array =
      llvm::dyn_cast<BasicStringArray>(new_arrays[0].get());
  TF_ASSERT_OK_AND_ASSIGN(auto new_buffers,
                          new_basic_string_array->buffers().Await());
  ASSERT_EQ(new_buffers.size(), 2);
  EXPECT_THAT(new_buffers[0], ElementsAre("shard 0"));
  EXPECT_THAT(new_buffers[1], ElementsAre("shard 1"));
}

TEST(CopyTest, FailsAfterDeletion) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  auto devices = client->addressable_devices();
  ASSERT_GE(devices.size(), 2);

  auto [buffers, on_done_with_buffer] = MakeBuffersAndOnDoneWithBuffer({"abc"});
  std::vector<tsl::RCReference<Array>> arrays;
  TF_ASSERT_OK_AND_ASSIGN(
      arrays.emplace_back(),
      CreateTestArray(client.get(), Future<BasicStringArray::Buffers>(buffers),
                      std::move(on_done_with_buffer)));

  arrays[0]->Delete();

  EXPECT_THAT(client->CopyArrays(absl::MakeSpan(arrays),
                                 client->MakeDeviceList({devices[1]}),
                                 MemoryKind(), ArrayCopySemantics::kAlwaysCopy),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(CopyTest, FailsWithDifferentNumbersDevices) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  auto devices = client->addressable_devices();
  ASSERT_GE(devices.size(), 2);

  auto [buffers, on_done_with_buffer] = MakeBuffersAndOnDoneWithBuffer({"abc"});
  std::vector<tsl::RCReference<Array>> arrays;
  TF_ASSERT_OK_AND_ASSIGN(
      arrays.emplace_back(),
      CreateTestArray(client.get(), Future<BasicStringArray::Buffers>(buffers),
                      std::move(on_done_with_buffer)));

  EXPECT_THAT(
      client->CopyArrays(absl::MakeSpan(arrays),
                         client->MakeDeviceList({devices[0], devices[1]}),
                         MemoryKind(), ArrayCopySemantics::kAlwaysCopy),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(CopyTest, NonReadySourceArraySuccessfullyBecomesReadyAfterCopy) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  auto devices = client->addressable_devices();
  ASSERT_GE(devices.size(), 2);

  auto buf_and_on_done_with_buffer = MakeBuffersAndOnDoneWithBuffer({"abc"});
  auto buffers = buf_and_on_done_with_buffer.first;
  auto on_done_with_buffer = buf_and_on_done_with_buffer.second;
  TF_ASSERT_OK_AND_ASSIGN(
      auto ret, CreateNonReadyTestArray(client.get(), devices[0],
                                        std::move(on_done_with_buffer)));
  std::vector<tsl::RCReference<Array>> arrays;
  arrays.push_back(std::move(ret.first));
  auto promise = std::move(ret.second);

  TF_ASSERT_OK(client->CopyArrays(
      absl::MakeSpan(arrays), client->MakeDeviceList({devices[1]}),
      MemoryKind(), ArrayCopySemantics::kAlwaysCopy));

  absl::Notification done_readying_single_device_arrays;
  tsl::Env::Default()->SchedClosure(([&]() mutable {
    promise.Set(std::move(buffers));
    done_readying_single_device_arrays.Notify();
  }));

  auto basic_string_array = llvm::dyn_cast<BasicStringArray>(arrays[0].get());
  ASSERT_NE(basic_string_array, nullptr);

  TF_ASSERT_OK_AND_ASSIGN(auto new_buffers,
                          basic_string_array->buffers().Await());
  ASSERT_EQ(new_buffers.size(), 1);
  EXPECT_THAT(new_buffers[0], ElementsAre("abc"));

  // Make sure to wait for the Closure to complete its work and set both
  // promises before returning from the test. The consequent destruction of the
  // promises can race with the Closure.
  done_readying_single_device_arrays.WaitForNotification();
}

TEST(CopyTest, NonReadySourceArrayFailsToBecomeReadyAfterCopy) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  auto devices = client->addressable_devices();
  ASSERT_GE(devices.size(), 2);

  auto buf_and_on_done_with_buffer = MakeBuffersAndOnDoneWithBuffer({"abc"});
  auto on_done_with_buffer = buf_and_on_done_with_buffer.second;
  TF_ASSERT_OK_AND_ASSIGN(
      auto ret, CreateNonReadyTestArray(client.get(), devices[0],
                                        std::move(on_done_with_buffer)));
  std::vector<tsl::RCReference<Array>> arrays;
  arrays.push_back(std::move(ret.first));
  auto promise = std::move(ret.second);

  TF_ASSERT_OK(client->CopyArrays(
      absl::MakeSpan(arrays), client->MakeDeviceList({devices[1]}),
      MemoryKind(), ArrayCopySemantics::kAlwaysCopy));

  absl::Notification done_readying_single_device_arrays;
  tsl::Env::Default()->SchedClosure(([&]() mutable {
    promise.Set(absl::InternalError("injected from the test"));
    done_readying_single_device_arrays.Notify();
  }));

  auto basic_string_array = llvm::dyn_cast<BasicStringArray>(arrays[0].get());
  ASSERT_NE(basic_string_array, nullptr);

  auto buffers_future = basic_string_array->buffers();
  EXPECT_THAT(buffers_future.Await(),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("injected from the test")));

  // Make sure to wait for the Closure to complete its work and set both
  // promises before returning from the test. The consequent destruction of the
  // promises can race with the Closure.
  done_readying_single_device_arrays.WaitForNotification();
}

TEST(FullyReplicatedShardTest, SuccessSingleDeviceShardedArray) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  constexpr char kContents[] = "abc";
  auto [buffers, on_done_with_buffer] =
      MakeBuffersAndOnDoneWithBuffer({kContents});
  TF_ASSERT_OK_AND_ASSIGN(
      auto array,
      CreateTestArray(client.get(), Future<BasicStringArray::Buffers>(buffers),
                      std::move(on_done_with_buffer)));

  TF_ASSERT_OK_AND_ASSIGN(
      auto relicated_shard,
      array->FullyReplicatedShard(ArrayCopySemantics::kAlwaysCopy));

  auto replicated_basic_string_array =
      llvm::dyn_cast<BasicStringArray>(relicated_shard.get());
  TF_ASSERT_OK_AND_ASSIGN(auto replicated_buffers,
                          replicated_basic_string_array->buffers().Await());
  ASSERT_EQ(replicated_buffers.size(), 1);
  EXPECT_THAT(replicated_buffers[0], ElementsAre(kContents));
}

TEST(FullyReplicatedShardTest, SuccessMultiDeviceShardedArray) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  constexpr char kReplicatedContents[] = "abc";
  const std::vector<std::string> per_shard_contents(
      {kReplicatedContents, kReplicatedContents});
  TF_ASSERT_OK_AND_ASSIGN(
      auto array, MakeShardedStringTestArray(client.get(), per_shard_contents,
                                             /*is_fully_replicated=*/true));

  TF_ASSERT_OK_AND_ASSIGN(
      auto replicated_shard,
      array->FullyReplicatedShard(ArrayCopySemantics::kAlwaysCopy));

  auto replicated_basic_string_array =
      llvm::dyn_cast<BasicStringArray>(replicated_shard.get());
  TF_ASSERT_OK_AND_ASSIGN(auto replicated_buffers,
                          replicated_basic_string_array->buffers().Await());
  ASSERT_EQ(replicated_buffers.size(), 1);
  EXPECT_THAT(replicated_buffers[0], ElementsAre(kReplicatedContents));
}

TEST(FullyReplicatedShardTest, FailsAfterDeletion) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  constexpr char kContents[] = "abc";
  auto [buffers, on_done_with_buffer] =
      MakeBuffersAndOnDoneWithBuffer({kContents});
  TF_ASSERT_OK_AND_ASSIGN(
      auto array,
      CreateTestArray(client.get(), Future<BasicStringArray::Buffers>(buffers),
                      std::move(on_done_with_buffer)));

  array->Delete();

  EXPECT_THAT(array->FullyReplicatedShard(ArrayCopySemantics::kAlwaysCopy),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(LayoutTest, Success) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  constexpr char kContents[] = "abc";
  auto [buffers, on_done_with_buffer] =
      MakeBuffersAndOnDoneWithBuffer({kContents});
  TF_ASSERT_OK_AND_ASSIGN(
      auto array,
      CreateTestArray(client.get(),
                      Future<BasicStringArray::Buffers>(std::move(buffers)),
                      std::move(on_done_with_buffer)));
}

TEST(LayoutTest, FailsAfterDeletion) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  constexpr char kContents[] = "abc";
  auto [buffers, on_done_with_buffer] =
      MakeBuffersAndOnDoneWithBuffer({kContents});
  TF_ASSERT_OK_AND_ASSIGN(
      auto array,
      CreateTestArray(client.get(), Future<BasicStringArray::Buffers>(buffers),
                      std::move(on_done_with_buffer)));

  array->Delete();
}

/////////////////////////////////////////////////////////////////////////////
//
// Tests related to CopyToHostBuffer
//

TEST(CopyToHostBufferTest, Success) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  auto devices = client->addressable_devices();
  ASSERT_GE(devices.size(), 1);
  std::vector<std::string> input_data = {"abc", "def"};
  TF_ASSERT_OK_AND_ASSIGN(
      auto array,
      MakeSingleDeviceStringTestArray(input_data, client.get(), devices[0]));

  auto data_read = std::make_unique<std::vector<absl::Cord>>(input_data.size());
  TF_ASSERT_OK(array
                   ->CopyToHostBuffer(data_read->data(),
                                      /*byte_strides=*/std::nullopt,
                                      ArrayCopySemantics::kAlwaysCopy)
                   .Await());
  EXPECT_THAT(*data_read, ElementsAreArray(input_data));
}

TEST(CopyToHostBufferTest, FailsAfterDeletion) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  auto devices = client->addressable_devices();
  ASSERT_GE(devices.size(), 1);
  std::vector<std::string> input_data = {"abc", "def"};
  TF_ASSERT_OK_AND_ASSIGN(
      auto array,
      MakeSingleDeviceStringTestArray(input_data, client.get(), devices[0]));

  TF_ASSERT_OK(array->Delete().Await());

  auto data_read = std::make_unique<std::vector<absl::Cord>>(input_data.size());
  EXPECT_THAT(array
                  ->CopyToHostBuffer(data_read->data(),
                                     /*byte_strides=*/std::nullopt,
                                     ArrayCopySemantics::kAlwaysCopy)
                  .Await(),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST(CopyToHostBufferTest, FailsWithMultiDeviceShardedArray) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  auto devices = client->addressable_devices();
  ASSERT_GE(devices.size(), 2);
  std::vector<std::string> per_shard_data = {"shard-0", "shard-1"};
  TF_ASSERT_OK_AND_ASSIGN(
      auto array, MakeShardedStringTestArray(client.get(), per_shard_data,
                                             /*is_fully_replicated=*/false));

  auto data_read =
      std::make_unique<std::vector<absl::Cord>>(per_shard_data.size());
  EXPECT_THAT(array
                  ->CopyToHostBuffer(data_read->data(),
                                     /*byte_strides=*/std::nullopt,
                                     ArrayCopySemantics::kAlwaysCopy)
                  .Await(),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(CopytoHostBufferTest,
     WorksWithNonReadySourceArrayThatSuccessfullyBecomesReadyAfterCreation) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  auto devices = client->addressable_devices();
  ASSERT_GE(devices.size(), 1);
  auto buf_and_on_done_with_buffer = MakeBuffersAndOnDoneWithBuffer({"abc"});
  auto buffers = buf_and_on_done_with_buffer.first;
  auto on_done_with_buffer = buf_and_on_done_with_buffer.second;
  TF_ASSERT_OK_AND_ASSIGN(
      auto ret, CreateNonReadyTestArray(client.get(), devices[0],
                                        std::move(on_done_with_buffer)));
  auto array = ret.first;
  auto promise = std::move(ret.second);

  auto data_read = std::make_unique<std::vector<absl::Cord>>(1);
  auto copy_completion_future =
      array->CopyToHostBuffer(data_read->data(), /*byte_strides=*/std::nullopt,
                              ArrayCopySemantics::kAlwaysCopy);

  absl::Notification done_readying_single_device_arrays;
  tsl::Env::Default()->SchedClosure(([&]() mutable {
    promise.Set(std::move(buffers));
    done_readying_single_device_arrays.Notify();
  }));

  done_readying_single_device_arrays.WaitForNotification();

  TF_ASSERT_OK(copy_completion_future.Await());
  EXPECT_THAT(*data_read, ElementsAre("abc"));
}

TEST(CopytoHostBufferTest,
     WorksWithNonReadySourceArrayThatFailsToBecomeReadyAfterCreation) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  auto devices = client->addressable_devices();
  ASSERT_GE(devices.size(), 1);
  TF_ASSERT_OK_AND_ASSIGN(
      auto ret, CreateNonReadyTestArray(client.get(), devices[0],
                                        /*on_done_with_buffer=*/[]() {}));
  auto array = ret.first;
  auto promise = std::move(ret.second);

  auto data_read = std::make_unique<std::vector<absl::Cord>>(1);
  auto copy_completion_future =
      array->CopyToHostBuffer(data_read->data(), /*byte_strides=*/std::nullopt,
                              ArrayCopySemantics::kAlwaysCopy);

  absl::Notification done_readying_single_device_arrays;
  tsl::Env::Default()->SchedClosure(([&]() mutable {
    promise.Set(absl::InternalError("injected from the test"));
    done_readying_single_device_arrays.Notify();
  }));

  done_readying_single_device_arrays.WaitForNotification();

  EXPECT_THAT(copy_completion_future.Await(),
              StatusIs(absl::StatusCode::kInternal));
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
