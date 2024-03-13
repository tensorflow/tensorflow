/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/python/ifrt/sharding_test_util.h"

#include <memory>
#include <utility>
#include <vector>

#include "xla/pjrt/pjrt_common.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/mock.h"
#include "xla/python/ifrt/test_util.h"
#include "tsl/platform/test.h"

namespace xla {
namespace ifrt {
namespace test_util {

namespace {

using ::testing::Return;

// Internal state of a client for sharding tests.
struct ShardingTestClientState {
  // Mapping from a device ID to the mock device object.
  absl::flat_hash_map<int, std::unique_ptr<Device>> device_map;
  // Raw pointers to mock devices.
  std::vector<Device*> devices;
};

// Creates a mock client for sharding tests. The client will have a specified
// number of fake addressable and non-addressable devices. Client implements
// `devices()` and `LookupDevice()`. Device implements `id()`, with an arbitrary
// deterministic device ids assigned.
std::shared_ptr<MockClient> MakeShardingTestClient(
    int num_devices, int num_addressable_devices) {
  auto state = std::make_shared<ShardingTestClientState>();
  state->device_map.reserve(num_devices);
  state->devices.reserve(num_devices);

  for (int i = 0; i < num_addressable_devices; ++i) {
    auto device = std::make_unique<MockDevice>();
    ON_CALL(*device, global_device_id)
        .WillByDefault(Return(PjRtGlobalDeviceId(i + 10)));
    ON_CALL(*device, IsAddressable).WillByDefault(Return(true));
    state->devices.push_back(device.get());
    state->device_map.insert({i + 10, std::move(device)});
  }
  for (int i = num_addressable_devices; i < num_devices; ++i) {
    auto device = std::make_unique<MockDevice>();
    ON_CALL(*device, global_device_id)
        .WillByDefault(Return(PjRtGlobalDeviceId(i + 10)));
    ON_CALL(*device, IsAddressable).WillByDefault(Return(false));
    state->devices.push_back(device.get());
    state->device_map.insert({i + 10, std::move(device)});
  }

  auto client = std::make_shared<MockClient>();
  ON_CALL(*client, devices)
      .WillByDefault(
          [state]() -> absl::Span<Device* const> { return state->devices; });
  ON_CALL(*client, LookupDevice)
      .WillByDefault([state](int device_id) -> absl::StatusOr<Device*> {
        auto it = state->device_map.find(device_id);
        if (it == state->device_map.end()) {
          return InvalidArgument("Unexpected device id: %d", device_id);
        }
        return it->second.get();
      });
  return client;
}

}  // namespace

void ShardingTest::SetUp() {
  const auto [num_devices, num_addressable_devices] = GetParam();
  client_ = MakeShardingTestClient(num_devices, num_addressable_devices);
}

DeviceList ShardingTest::GetDevices(absl::Span<const int> device_indices) {
  return test_util::GetDevices(client_.get(), device_indices).value();
}

}  // namespace test_util
}  // namespace ifrt
}  // namespace xla
