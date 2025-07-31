/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/python/ifrt/test_util.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/user_context.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace xla {
namespace ifrt {
namespace test_util {

namespace {

class ClientFactory {
 public:
  void Register(
      std::function<absl::StatusOr<std::shared_ptr<Client>>()> factory) {
    absl::MutexLock lock(&mu_);
    CHECK(!factory_) << "Client factory has been already registered.";
    factory_ = std::move(factory);
  }

  std::function<absl::StatusOr<std::shared_ptr<Client>>()> Get() const {
    absl::MutexLock lock(&mu_);
    return factory_;
  }

 private:
  mutable absl::Mutex mu_;
  std::function<absl::StatusOr<std::shared_ptr<Client>>()> factory_
      ABSL_GUARDED_BY(mu_);
};

ClientFactory& GetGlobalClientFactory() {
  static auto* const factory = new ClientFactory;
  return *factory;
}

}  // namespace

void RegisterClientFactory(
    std::function<absl::StatusOr<std::shared_ptr<Client>>()> factory) {
  GetGlobalClientFactory().Register(std::move(factory));
}

absl::StatusOr<std::shared_ptr<Client>> GetClient() {
  auto factory = GetGlobalClientFactory().Get();
  CHECK(factory) << "Client factory has not been registered.";
  return factory();
}

void SetTestFilterIfNotUserSpecified(absl::string_view custom_filter) {
  static constexpr absl::string_view kDefaultTestFilter = "*";
#ifdef GTEST_FLAG_SET
  if (GTEST_FLAG_GET(filter) == kDefaultTestFilter) {
    GTEST_FLAG_SET(filter, custom_filter);
  }
#else
  if (testing::GTEST_FLAG(filter) == kDefaultTestFilter) {
    testing::GTEST_FLAG(filter) = custom_filter;
  }
#endif
}

absl::StatusOr<DeviceListRef> GetDevices(Client* client,
                                         absl::Span<const int> device_indices) {
  absl::InlinedVector<xla::ifrt::Device*, 1> devices;
  devices.reserve(device_indices.size());
  const absl::Span<Device* const> client_devices = client->devices();
  for (int device_index : device_indices) {
    if (device_index < 0 || device_index >= client_devices.size()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Out of range device index: ", device_index));
    }
    devices.push_back(client_devices[device_index]);
  }
  return client->MakeDeviceList(devices);
}

absl::StatusOr<DeviceListRef> GetAddressableDevices(
    Client* client, absl::Span<const int> device_indices) {
  absl::InlinedVector<xla::ifrt::Device*, 1> devices;
  devices.reserve(device_indices.size());
  const absl::Span<Device* const> client_devices =
      client->addressable_devices();
  for (int device_index : device_indices) {
    if (device_index < 0 || device_index >= client_devices.size()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Out of range device index: ", device_index));
    }
    devices.push_back(client_devices[device_index]);
  }
  return client->MakeDeviceList(std::move(devices));
}

namespace {

class TestUserContext : public llvm::RTTIExtends<TestUserContext, UserContext> {
 public:
  explicit TestUserContext(uint64_t id) : id_(id) {}

  uint64_t Fingerprint() const override { return id_; }

  std::string DebugString() const override {
    return absl::StrCat("TestUserContext(", id_, ")");
  }

  // No new `ID` is not defined because tests below do not exercise RTTI.

 private:
  uint64_t id_;
};

}  // namespace

UserContextRef MakeUserContext(uint64_t id) {
  return tsl::MakeRef<TestUserContext>(id);
}

}  // namespace test_util
}  // namespace ifrt
}  // namespace xla
