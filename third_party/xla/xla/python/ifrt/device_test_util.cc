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

#include "xla/python/ifrt/device_test_util.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/python/ifrt/basic_device_list.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/mock.h"
#include "xla/python/ifrt/test_util.h"
#include "xla/tsl/platform/test.h"
#include "xla/util.h"

namespace xla {
namespace ifrt {
namespace test_util {

namespace {

using ::testing::Return;
using ::testing::ReturnPointee;
using ::testing::ReturnRef;

// Internal state of a client for device tests.
struct DeviceTestClientState {
  Client* client;

  // Shared MemoryKind objects.
  MemoryKind host_memory_kind;

  // Mapping from a memory ID to the mock memory object.
  absl::flat_hash_map<MemoryId, std::unique_ptr<Memory>> memory_map;
  std::vector<Memory*> memories;
  std::vector<absl::Span<Device* const>> memory_devices;

  // Mapping from a device ID to the mock device object.
  absl::flat_hash_map<DeviceId, std::unique_ptr<Device>> device_map;
  // Raw pointers to mock devices.
  std::vector<Device*> devices;
  std::vector<Device*> addressable_devices;
  std::vector<absl::Span<Memory* const>> device_memories;
};

// Creates a mock client for device tests. The client will have a specified
// number of fake addressable and non-addressable devices. Client implements
// `devices()` and `LookupDevice()`. Device implements `id()`, with an
// arbitrary deterministic device ids assigned. Each device has "host" memory
// (which is also its default memory), and each memory has a single device.
std::shared_ptr<MockClient> MakeDeviceTestClient(int num_devices,
                                                 int num_addressable_devices) {
  CHECK_GE(num_devices, num_addressable_devices);
  auto state = std::make_shared<DeviceTestClientState>();

  state->host_memory_kind = MemoryKind("host");

  state->memory_map.reserve(num_devices);
  state->memories.reserve(num_devices);
  state->memory_devices.resize(num_devices);

  state->device_map.reserve(num_devices);
  state->devices.reserve(num_devices);
  state->addressable_devices.reserve(num_addressable_devices);
  state->device_memories.resize(num_devices);

  for (int i = 0; i < num_devices; ++i) {
    const bool addressable = i < num_addressable_devices;
    auto memory = std::make_unique<MockMemory>();
    ON_CALL(*memory, Id).WillByDefault(Return(MemoryId(i + 10)));
    ON_CALL(*memory, Kind).WillByDefault(ReturnRef(state->host_memory_kind));
    // memory_devices will be filled in at the end of the loop.
    ON_CALL(*memory, Devices)
        .WillByDefault(ReturnPointee(&state->memory_devices[i]));
    state->memories.push_back(memory.get());
    state->memory_map.insert({MemoryId(i + 10), std::move(memory)});

    auto device = std::make_unique<MockDevice>();
    // client will be filled in at the end of the loop.
    ON_CALL(*device, client).WillByDefault(ReturnPointee(&state->client));
    ON_CALL(*device, Id).WillByDefault(Return(DeviceId(i + 10)));
    ON_CALL(*device, IsAddressable).WillByDefault(Return(addressable));
    ON_CALL(*device, DefaultMemory).WillByDefault(Return(state->memories[i]));
    // device_memories will be filled in at the end of the loop.
    ON_CALL(*device, Memories)
        .WillByDefault(ReturnPointee(&state->device_memories[i]));
    state->devices.push_back(device.get());
    if (addressable) {
      state->addressable_devices.push_back(device.get());
    }
    state->device_map.insert({DeviceId(i + 10), std::move(device)});

    state->device_memories[i] = absl::MakeConstSpan(&state->memories[i], 1);
    state->memory_devices[i] = absl::MakeConstSpan(&state->devices[i], 1);
  }

  auto client = std::make_shared<MockClient>();
  state->client = client.get();
  ON_CALL(*client, devices)
      .WillByDefault(
          [state]() -> absl::Span<Device* const> { return state->devices; });
  ON_CALL(*client, addressable_devices)
      .WillByDefault([state]() -> absl::Span<Device* const> {
        return state->addressable_devices;
      });
  ON_CALL(*client, LookupDevice)
      .WillByDefault([state](DeviceId device_id) -> absl::StatusOr<Device*> {
        auto it = state->device_map.find(device_id);
        if (it == state->device_map.end()) {
          return InvalidArgument("Unexpected device id: %d", device_id.value());
        }
        return it->second.get();
      });
  ON_CALL(*client, MakeDeviceList)
      .WillByDefault([](absl::Span<Device* const> devices) -> DeviceListRef {
        return BasicDeviceList::Create(devices);
      });
  ON_CALL(*client, GetTopologyForDevices)
      .WillByDefault([](const DeviceListRef&) { return nullptr; });
  return client;
}

}  // namespace

DeviceTestFixture::DeviceTestFixture(const DeviceTestParam& param) {
  const auto& [num_devices, num_addressable_devices] = param;
  client_ = MakeDeviceTestClient(num_devices, num_addressable_devices);
}

DeviceListRef DeviceTestFixture::GetDevices(
    absl::Span<const int> device_indices) {
  return test_util::GetDevices(client_.get(), device_indices).value();
}

DeviceListRef DeviceTestFixture::GetAddressableDevices(
    absl::Span<const int> device_indices) {
  return test_util::GetAddressableDevices(client_.get(), device_indices)
      .value();
}

}  // namespace test_util
}  // namespace ifrt
}  // namespace xla
