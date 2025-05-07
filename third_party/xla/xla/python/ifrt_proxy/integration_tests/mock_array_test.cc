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
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_client_options.h"
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/attribute_map.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/future.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/mock.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/user_context.h"
#include "xla/python/ifrt/value.h"
#include "xla/python/ifrt_proxy/client/client.h"
#include "xla/python/ifrt_proxy/client/registry.h"
#include "xla/python/ifrt_proxy/server/grpc_server.h"
#include "xla/python/pjrt_ifrt/pjrt_client.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

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
                     address, [this](AttributeMap initialization_data) {
                       return CreateMockBackend();
                     }));
    TF_ASSERT_OK_AND_ASSIGN(client_,
                            CreateClient(absl::StrCat("grpc://", address)));
  }

  absl::StatusOr<xla::ifrt::ArrayRef> NewArray() {
    DType dtype(DType::kF32);
    Shape shape({2, 3});
    auto data = std::make_unique<std::vector<float>>(6);
    std::iota(data->begin(), data->end(), 0);
    xla::ifrt::Device* device = client_->addressable_devices().at(0);
    ShardingRef sharding = SingleDeviceSharding::Create(device, MemoryKind());

    TF_ASSIGN_OR_RETURN(
        auto client_arr,
        client_->MakeArrayFromHostBuffer(
            data->data(), dtype, shape,
            /*byte_strides=*/std::nullopt, sharding,
            Client::HostBufferSemantics::kImmutableOnlyDuringCall,
            /*on_done_with_host_buffer=*/nullptr));

    return client_arr;
  }

  std::unique_ptr<GrpcServer> server_;
  std::unique_ptr<xla::ifrt::Client> client_;

 protected:
  absl::StatusOr<std::unique_ptr<xla::ifrt::Client>> CreateMockBackend() {
    // TODO(b/292339723): Use reference backend as the delegate while mocking.
    xla::CpuClientOptions options;
    options.asynchronous = true;
    options.cpu_device_count = 2;
    TF_ASSIGN_OR_RETURN(auto pjrt_cpu_client,
                        xla::GetXlaPjrtCpuClient(std::move(options)));
    auto mock_backend = std::make_unique<MockClient>(
        /*delegate=*/xla::ifrt::PjRtClient::Create(std::move(pjrt_cpu_client)));

    ON_CALL(*mock_backend, MakeArrayFromHostBuffer)
        .WillByDefault(
            [this, mock_backend = mock_backend.get()](
                const void* data, DType dtype, Shape shape,
                std::optional<absl::Span<const int64_t>> byte_strides,
                ShardingRef sharding, Client::HostBufferSemantics semantics,
                std::function<void()> on_done_with_host_buffer,
                tsl::RCReference<UserContext> user_context)
                -> absl::StatusOr<xla::ifrt::ArrayRef> {
              TF_ASSIGN_OR_RETURN(
                  auto delegated,
                  mock_backend->delegated()->MakeArrayFromHostBuffer(
                      data, dtype, shape, byte_strides, sharding, semantics,
                      on_done_with_host_buffer));
              auto result = tsl::MakeRef<MockArray>(delegated);
              ON_CALL(*result, GetReadyFuture)
                  .WillByDefault([this, delegated]() {
                    absl::MutexLock l(&mu_);
                    if (get_ready_hook_) {
                      absl::Status s = get_ready_hook_();
                      if (!s.ok()) return Future<>(s);
                    }
                    return delegated->GetReadyFuture();
                  });
              ON_CALL(*result, CopyToHostBuffer)
                  .WillByDefault([this, delegated](auto data, auto byte_strides,
                                                   auto semantics) {
                    absl::MutexLock l(&mu_);
                    if (copy_host_hook_) {
                      absl::Status s = copy_host_hook_();
                      if (!s.ok()) return Future<>(s);
                    }
                    return delegated->CopyToHostBuffer(data, byte_strides,
                                                       semantics);
                  });
              return result;
            });

    ON_CALL(*mock_backend, GetReadyFuture)
        .WillByDefault([](absl::Span<const tsl::RCReference<Value>> values) {
          std::vector<Future<>> futures;
          futures.reserve(values.size());
          for (const auto& value : values) {
            futures.push_back(value->GetReadyFuture());
          }
          return JoinFutures(futures);
        });

    return mock_backend;
  }

  absl::Mutex mu_;
  absl::AnyInvocable<absl::Status()> get_ready_hook_ ABSL_GUARDED_BY(mu_);
  absl::AnyInvocable<absl::Status()> copy_host_hook_ ABSL_GUARDED_BY(mu_);
};

TEST_F(MockArrayTest, ReadyFutureWaitsUntilReady) {
  TF_ASSERT_OK_AND_ASSIGN(auto arr, NewArray());

  absl::Notification wait_ready;

  {
    absl::MutexLock l(&mu_);
    get_ready_hook_ = [&]() {
      wait_ready.WaitForNotification();
      return absl::OkStatus();
    };
  }

  auto ready = arr->GetReadyFuture();

  absl::SleepFor(kSomeTime);
  EXPECT_FALSE(ready.IsReady());

  wait_ready.Notify();
  EXPECT_THAT(ready.Await(), IsOk());
}

TEST_F(MockArrayTest, ReadyFuturePropagatesError) {
  TF_ASSERT_OK_AND_ASSIGN(auto arr, NewArray());

  absl::Notification wait_ready;

  {
    absl::MutexLock l(&mu_);
    get_ready_hook_ = [&]() { return absl::InternalError("testing"); };
  }

  EXPECT_THAT(arr->GetReadyFuture().Await(), StatusIs(kInternal));
}

TEST_F(MockArrayTest, CopyToHostFutureWaitsUntilCopied) {
  TF_ASSERT_OK_AND_ASSIGN(auto arr, NewArray());

  absl::Notification wait_ready;

  {
    absl::MutexLock l(&mu_);
    copy_host_hook_ = [&]() {
      wait_ready.WaitForNotification();
      return absl::OkStatus();
    };
  }

  char data[1000];
  auto copied = arr->CopyToHostBuffer(data, /*byte_strides=*/std::nullopt,
                                      ArrayCopySemantics::kAlwaysCopy);

  absl::SleepFor(kSomeTime);
  EXPECT_FALSE(copied.IsReady());

  wait_ready.Notify();
  EXPECT_THAT(copied.Await(), IsOk());
}

TEST_F(MockArrayTest, CopyToHostFuturePropagatesError) {
  TF_ASSERT_OK_AND_ASSIGN(auto arr, NewArray());

  absl::Notification wait_ready;

  {
    absl::MutexLock l(&mu_);
    copy_host_hook_ = [&]() { return absl::InternalError("testing"); };
  }

  char data[1000];
  auto copied = arr->CopyToHostBuffer(data, /*byte_strides=*/std::nullopt,
                                      ArrayCopySemantics::kAlwaysCopy);

  EXPECT_THAT(copied.Await(), StatusIs(kInternal));
}

}  // namespace
}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
