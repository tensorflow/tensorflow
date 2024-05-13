// Copyright 2023 The OpenXLA Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/thread_annotations.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/pjrt/cpu/cpu_client.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/future.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/mock.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt_proxy/client/client.h"
#include "xla/python/ifrt_proxy/client/registry.h"
#include "xla/python/ifrt_proxy/server/grpc_server.h"
#include "xla/python/pjrt_ifrt/pjrt_client.h"
#include "xla/status.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "tsl/platform/env.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"
#include "tsl/platform/threadpool.h"

namespace xla {
namespace ifrt {
namespace proxy {
namespace {

using ::tsl::testing::IsOk;
using ::tsl::testing::StatusIs;

constexpr absl::StatusCode kInternal = absl::StatusCode::kInternal;

constexpr absl::Duration kSomeTime = absl::Seconds(1);

class MockArrayTest : public testing::Test {
 public:
  void SetUp() override {
    std::string address =
        absl::StrCat("localhost:", tsl::testing::PickUnusedPortOrDie());
    TF_ASSERT_OK_AND_ASSIGN(
        server_, GrpcServer::CreateFromIfrtClientFactory(
                     address, [this] { return CreateMockBackend(); }));
    TF_ASSERT_OK_AND_ASSIGN(client_,
                            CreateClient(absl::StrCat("grpc://", address)));
  }

  struct ArrayPair {
    // IFRT array exposed to the proxy's user. Not a mock.
    tsl::RCReference<xla::ifrt::Array> proxy_client_array;
    // IFRT array owned by the proxy server whose behavior should be
    // reflected by proxy_client_array. Mock but delegated.
    tsl::RCReference<MockArray> backend_array;
  };

  absl::StatusOr<ArrayPair> NewArray() {
    DType dtype(DType::kF32);
    Shape shape({2, 3});
    auto data = std::make_unique<std::vector<float>>(6);
    std::iota(data->begin(), data->end(), 0);
    xla::ifrt::Device* device = client_->addressable_devices().at(0);
    std::shared_ptr<const Sharding> sharding =
        SingleDeviceSharding::Create(device, MemoryKind());

    TF_ASSIGN_OR_RETURN(
        auto client_arr,
        client_->MakeArrayFromHostBuffer(
            data->data(), dtype, shape,
            /*byte_strides=*/std::nullopt, sharding,
            Client::HostBufferSemantics::kImmutableOnlyDuringCall,
            /*on_done_with_host_buffer=*/nullptr));

    // When the above `MakeArrayFromHostBuffer` results in the server issuing a
    // `MakeArrayFromHostBuffer()` to the underlying mock backend, the mock
    // backend enqueues the returned mock array onto `mock_arrays_` (this code
    // is in `CreateMockBackend()`).
    absl::MutexLock l(&mu_);
    CHECK_EQ(mock_arrays_.size(), 1);
    auto mock = mock_arrays_.back();
    mock_arrays_.pop_back();
    return ArrayPair{client_arr, mock};
  }

  std::unique_ptr<GrpcServer> server_;
  std::unique_ptr<xla::ifrt::Client> client_;

 private:
  absl::StatusOr<std::unique_ptr<xla::ifrt::Client>> CreateMockBackend() {
    // TODO(b/292339723): Use reference backend as the delegate while mocking.
    CpuClientOptions options;
    options.asynchronous = true;
    options.cpu_device_count = 2;
    TF_ASSIGN_OR_RETURN(auto tfrt_cpu_client, xla::GetTfrtCpuClient(options));
    auto mock_backend = std::make_unique<MockClient>(
        /*delegate=*/xla::ifrt::PjRtClient::Create(std::move(tfrt_cpu_client)));

    ON_CALL(*mock_backend, MakeArrayFromHostBuffer)
        .WillByDefault(
            [this, mock_backend = mock_backend.get()](
                const void* data, DType dtype, Shape shape,
                std::optional<absl::Span<const int64_t>> byte_strides,
                std::shared_ptr<const Sharding> sharding,
                Client::HostBufferSemantics semantics,
                std::function<void()> on_done_with_host_buffer)
                -> absl::StatusOr<tsl::RCReference<xla::ifrt::Array>> {
              TF_ASSIGN_OR_RETURN(
                  auto delegated,
                  mock_backend->delegated()->MakeArrayFromHostBuffer(
                      data, dtype, shape, byte_strides, sharding, semantics,
                      on_done_with_host_buffer));
              auto result = tsl::MakeRef<MockArray>(delegated);

              absl::MutexLock l(&mu_);
              mock_arrays_.push_back(result);
              return result;
            });

    return mock_backend;
  }

  absl::Mutex mu_;
  std::vector<tsl::RCReference<MockArray>> mock_arrays_ ABSL_GUARDED_BY(mu_);
};

TEST_F(MockArrayTest, ReadyFutureWaitsUntilReady) {
  TF_ASSERT_OK_AND_ASSIGN(ArrayPair arr, NewArray());

  absl::Notification wait_ready;

  EXPECT_CALL(*arr.backend_array, GetReadyFuture).WillOnce([&] {
    wait_ready.WaitForNotification();
    return arr.backend_array->delegated()->GetReadyFuture();
  });

  auto ready = arr.proxy_client_array->GetReadyFuture();

  absl::SleepFor(kSomeTime);
  EXPECT_FALSE(ready.IsReady());

  wait_ready.Notify();
  EXPECT_THAT(ready.Await(), IsOk());
}

TEST_F(MockArrayTest, ReadyFuturePropagatesError) {
  TF_ASSERT_OK_AND_ASSIGN(ArrayPair arr, NewArray());

  EXPECT_CALL(*arr.backend_array, GetReadyFuture).WillOnce([&] {
    return Future<>(absl::InternalError("testing"));
  });

  EXPECT_THAT(arr.proxy_client_array->GetReadyFuture().Await(),
              StatusIs(kInternal));
}

TEST_F(MockArrayTest, DeletionFutureWaitsUntilDeleted) {
  TF_ASSERT_OK_AND_ASSIGN(ArrayPair arr, NewArray());

  tsl::thread::ThreadPool threads(tsl::Env::Default(), "t", /*num_threads=*/1);
  absl::Notification wait_ready;

  EXPECT_CALL(*arr.backend_array, Delete).WillOnce([&] {
    // TODO(b/266635130): Write a version of this testcase where the Delete()
    // call of the MockArray blocks on `wait_ready`, instead of the Future it
    // returns being blocked on `wait_ready`. That version of the testcase does
    // not currently work since both the client and the server synchronously
    // block until the MockArray's Delete() returns.
    auto promise = Future<>::CreatePromise();
    threads.Schedule([&, promise]() mutable {
      wait_ready.WaitForNotification();
      promise.Set(arr.backend_array->delegated()->Delete().Await());
    });
    return Future<>(promise);
  });

  EXPECT_FALSE(arr.proxy_client_array->IsDeleted());
  auto deleted_future = arr.proxy_client_array->Delete();

  absl::SleepFor(kSomeTime);
  EXPECT_FALSE(deleted_future.IsReady());
  EXPECT_FALSE(arr.proxy_client_array->IsDeleted());

  wait_ready.Notify();
  EXPECT_THAT(deleted_future.Await(), IsOk());
  EXPECT_TRUE(arr.proxy_client_array->IsDeleted());
}

TEST_F(MockArrayTest, DeletionPropagatesError) {
  TF_ASSERT_OK_AND_ASSIGN(ArrayPair arr, NewArray());

  EXPECT_CALL(*arr.backend_array, Delete).WillOnce([&] {
    return Future<>(absl::InternalError("testing"));
  });

  EXPECT_FALSE(arr.proxy_client_array->IsDeleted());
  EXPECT_THAT(arr.proxy_client_array->Delete().Await(), StatusIs(kInternal));
}

TEST_F(MockArrayTest, CopyToHostFutureWaitsUntilCopied) {
  TF_ASSERT_OK_AND_ASSIGN(ArrayPair arr, NewArray());

  absl::Notification wait_ready;

  EXPECT_CALL(*arr.backend_array, CopyToHostBuffer)
      .WillOnce([&](auto data, auto byte_strides, auto semantics) {
        wait_ready.WaitForNotification();
        return arr.backend_array->delegated()->CopyToHostBuffer(
            data, byte_strides, semantics);
      });

  char data[1000];
  auto copied = arr.proxy_client_array->CopyToHostBuffer(
      data, /*byte_strides=*/std::nullopt, ArrayCopySemantics::kAlwaysCopy);

  absl::SleepFor(kSomeTime);
  EXPECT_FALSE(copied.IsReady());

  wait_ready.Notify();
  EXPECT_THAT(copied.Await(), IsOk());
}

TEST_F(MockArrayTest, CopyToHostFuturePropagatesError) {
  TF_ASSERT_OK_AND_ASSIGN(ArrayPair arr, NewArray());

  absl::Notification wait_ready;

  EXPECT_CALL(*arr.backend_array, CopyToHostBuffer).WillOnce([&] {
    return Future<>(absl::InternalError("testing"));
  });

  char data[1000];
  auto copied = arr.proxy_client_array->CopyToHostBuffer(
      data, /*byte_strides=*/std::nullopt, ArrayCopySemantics::kAlwaysCopy);

  EXPECT_THAT(copied.Await(), StatusIs(kInternal));
}

}  // namespace
}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
