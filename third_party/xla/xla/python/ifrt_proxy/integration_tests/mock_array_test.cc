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
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/no_destructor.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/functional/bind_front.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/future.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_client_options.h"
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/attribute_map.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/mock.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/value.h"
#include "xla/python/ifrt_proxy/client/client.h"
#include "xla/python/ifrt_proxy/client/registry.h"
#include "xla/python/ifrt_proxy/server/grpc_server.h"
#include "xla/python/pjrt_ifrt/pjrt_client.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace ifrt {
namespace proxy {
namespace {

constexpr absl::StatusCode kInternal = absl::StatusCode::kInternal;

constexpr absl::Duration kSomeTime = absl::Seconds(1);

using ::testing::NiceMock;

// Proxy client and proxy server, sitting atop a mock IFRT backend.
class ProxyWithMockBackend {
  using ArrayId = uint64_t;

 public:
  explicit ProxyWithMockBackend() { CHECK_OK(Initialize()); }

  using Hook = absl::AnyInvocable<absl::Status()>;

  // Creates a new array on the proxy-client whose associated mock array on the
  // mock-IFRT backend is configured with the given hooks.
  //
  // If get_ready_hook is provided, it will be invoked whenever the mock array's
  // GetReadyFuture method is called, and if the hook returns a non-OK status,
  // the GetReadyFuture method will return that status. Similarly, the
  // copy_host_hook gets invoked when the mock arrays' CopyToHostBuffer method
  // is called.
  absl::StatusOr<xla::ifrt::ArrayRef> NewArray(
      std::optional<Hook> get_ready_hook, std::optional<Hook> copy_host_hook) {
    ArrayId array_id;
    {
      absl::MutexLock l(mu_);
      array_id = next_array_id_++;
      if (get_ready_hook.has_value()) {
        get_ready_hook_[array_id] = *std::move(get_ready_hook);
      }
      if (copy_host_hook.has_value()) {
        copy_host_hook_[array_id] = *std::move(copy_host_hook);
      }
    }

    DType dtype(DType::kU64);
    Shape shape({2, 3});

    auto data = std::make_unique<std::vector<uint64_t>>(6);
    (*data)[0] = array_id;

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

 private:
  absl::Status InvokeGetReadyHook(ArrayId array_id) {
    absl::MutexLock l(mu_);
    auto it = get_ready_hook_.find(array_id);
    if (it != get_ready_hook_.end()) {
      return it->second();
    }
    return absl::OkStatus();
  }

  absl::Status InvokeCopyHostHook(ArrayId array_id) {
    absl::MutexLock l(mu_);
    auto it = copy_host_hook_.find(array_id);
    if (it != copy_host_hook_.end()) {
      return it->second();
    }
    return absl::OkStatus();
  }

  absl::StatusOr<xla::ifrt::ArrayRef> MockBackendMakeArrayFromHostBuffer(
      const void* data, DType dtype, Shape shape,
      std::optional<absl::Span<const int64_t>> byte_strides,
      ShardingRef sharding, Client::HostBufferSemantics semantics,
      std::function<void()> on_done_with_host_buffer) {
    ArrayId array_id = *reinterpret_cast<const uint64_t*>(data);
    TF_ASSIGN_OR_RETURN(auto delegated,
                        mock_backend_->delegated()->MakeArrayFromHostBuffer(
                            data, dtype, shape, byte_strides, sharding,
                            semantics, on_done_with_host_buffer));
    auto result = tsl::MakeRef<NiceMock<MockArray>>(delegated);
    testing::Mock::AllowLeak(result.get());

    ON_CALL(*result, GetReadyFuture)
        .WillByDefault([this, delegated, array_id]() {
          if (auto s = InvokeGetReadyHook(array_id); !s.ok()) {
            return tsl::Future<>(s);
          }
          return delegated->GetReadyFuture();
        });
    ON_CALL(*result, CopyToHostBuffer)
        .WillByDefault([this, delegated, array_id](auto data, auto byte_strides,
                                                   auto semantics) {
          if (auto s = InvokeCopyHostHook(array_id); !s.ok()) {
            return tsl::Future<>(s);
          }
          return delegated->CopyToHostBuffer(data, byte_strides, semantics);
        });
    return result;
  }

  absl::Status Initialize() {
    // TODO(b/292339723): Use reference backend as the delegate while mocking.
    xla::CpuClientOptions options;
    options.asynchronous = true;
    options.cpu_device_count = 2;
    TF_ASSIGN_OR_RETURN(auto pjrt_cpu_client,
                        xla::GetXlaPjrtCpuClient(std::move(options)));

    mock_backend_ = std::make_unique<NiceMock<MockClient>>(
        /*delegate=*/xla::ifrt::PjRtClient::Create(std::move(pjrt_cpu_client)));
    testing::Mock::AllowLeak(mock_backend_.get());

    ON_CALL(*mock_backend_, MakeArrayFromHostBuffer)
        .WillByDefault(absl::bind_front(
            &ProxyWithMockBackend::MockBackendMakeArrayFromHostBuffer, this));

    ON_CALL(*mock_backend_, GetReadyFuture)
        .WillByDefault([](absl::Span<const ValueRef> values) {
          std::vector<tsl::Future<>> futures;
          futures.reserve(values.size());
          for (const auto& value : values) {
            futures.push_back(value->GetReadyFuture());
          }
          return JoinFutures(futures);
        });

    std::string address =
        absl::StrCat("localhost:", tsl::testing::PickUnusedPortOrDie());
    TF_ASSIGN_OR_RETURN(server_,
                        GrpcServer::CreateFromIfrtClientFactory(
                            address, [this](AttributeMap initialization_data) {
                              return this->mock_backend_;
                            }));
    TF_ASSIGN_OR_RETURN(client_,
                        CreateClient(absl::StrCat("grpc://", address)));
    return absl::OkStatus();
  }

  std::shared_ptr<MockClient> mock_backend_;
  std::unique_ptr<GrpcServer> server_;
  std::unique_ptr<xla::ifrt::Client> client_;

  absl::Mutex mu_;
  absl::flat_hash_map<ArrayId, Hook> get_ready_hook_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<ArrayId, Hook> copy_host_hook_ ABSL_GUARDED_BY(mu_);
  ArrayId next_array_id_ ABSL_GUARDED_BY(mu_) = 0;
};

ProxyWithMockBackend* Singleton() {
  static absl::NoDestructor<ProxyWithMockBackend> result;
  return result.get();
}

TEST(MockArrayTest, ReadyFutureWaitsUntilReady) {
  absl::Notification wait_ready;

  auto get_ready_hook = [&]() {
    wait_ready.WaitForNotification();
    return absl::OkStatus();
  };

  TF_ASSERT_OK_AND_ASSIGN(
      auto arr, Singleton()->NewArray(std::move(get_ready_hook),
                                      /*copy_host_hook=*/std::nullopt));

  auto ready = arr->GetReadyFuture();

  absl::SleepFor(kSomeTime);
  EXPECT_FALSE(ready.IsReady());

  wait_ready.Notify();
  EXPECT_THAT(ready.Await(), absl_testing::IsOk());
}

TEST(MockArrayTest, ReadyFuturePropagatesError) {
  auto get_ready_hook = [&]() { return absl::InternalError("testing"); };

  TF_ASSERT_OK_AND_ASSIGN(
      auto arr, Singleton()->NewArray(std::move(get_ready_hook),
                                      /*copy_host_hook=*/std::nullopt));

  EXPECT_THAT(arr->GetReadyFuture().Await(), absl_testing::StatusIs(kInternal));
}

TEST(MockArrayTest, CopyToHostFutureWaitsUntilCopied) {
  absl::Notification wait_ready;
  auto copy_host_hook = [&]() {
    wait_ready.WaitForNotification();
    return absl::OkStatus();
  };

  TF_ASSERT_OK_AND_ASSIGN(auto arr,
                          Singleton()->NewArray(/*get_ready_hook=*/std::nullopt,
                                                std::move(copy_host_hook)));

  char data[1000];
  auto copied = arr->CopyToHostBuffer(data, /*byte_strides=*/std::nullopt,
                                      ArrayCopySemantics::kAlwaysCopy);

  absl::SleepFor(kSomeTime);
  EXPECT_FALSE(copied.IsReady());

  wait_ready.Notify();
  EXPECT_THAT(copied.Await(), absl_testing::IsOk());
}

TEST(MockArrayTest, CopyToHostFuturePropagatesError) {
  auto copy_host_hook = [&]() { return absl::InternalError("testing"); };

  TF_ASSERT_OK_AND_ASSIGN(auto arr,
                          Singleton()->NewArray(/*get_ready_hook=*/std::nullopt,
                                                std::move(copy_host_hook)));

  char data[1000];
  auto copied = arr->CopyToHostBuffer(data, /*byte_strides=*/std::nullopt,
                                      ArrayCopySemantics::kAlwaysCopy);

  EXPECT_THAT(copied.Await(), absl_testing::StatusIs(kInternal));
}

}  // namespace
}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
