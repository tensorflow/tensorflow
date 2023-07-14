/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/compiler/xla/hlo/ir/hlo_sharding.h"
#include "tensorflow/compiler/xla/python/ifrt/mock.h"
#include "tensorflow/compiler/xla/python/ifrt/sharding_serdes.h"
#include "tensorflow/compiler/xla/python/pjrt_ifrt/xla_sharding.h"

namespace xla {
namespace ifrt {
namespace {

using ::testing::ElementsAreArray;

// Test fixture for sharding serialization and deserialization.  It makes a mock
// client with a number of fake devices. Client implements `devices()` and
// `LookupDevice()`, and Device implements `id()`, with an arbitrary device ids
// assigned.
class XlaShardingSerDesTest : public ::testing::TestWithParam<int> {
 public:
  void SetUp() override {
    const int num_devices = GetParam();
    device_map_.reserve(num_devices);
    devices_.reserve(num_devices);
    for (int i = 0; i < num_devices; ++i) {
      auto device = std::make_unique<MockDevice>();
      ON_CALL(*device, id).WillByDefault([i]() { return i + 10; });
      devices_.push_back(device.get());
      device_map_.insert({i + 10, std::move(device)});
    }
    client_ = std::make_unique<MockClient>();
    ON_CALL(*client_, devices)
        .WillByDefault(
            [this]() -> absl::Span<Device* const> { return devices_; });
    ON_CALL(*client_, LookupDevice)
        .WillByDefault([this](int device_id) -> StatusOr<Device*> {
          auto it = device_map_.find(device_id);
          if (it == device_map_.end()) {
            return InvalidArgument("Unexpected device id: %d", device_id);
          }
          return it->second.get();
        });
  }
  Client* client() { return client_.get(); }

 private:
  std::unique_ptr<MockClient> client_;
  absl::flat_hash_map<int, std::unique_ptr<Device>> device_map_;
  std::vector<Device*> devices_;
};

TEST_P(XlaShardingSerDesTest, HloShardingRoundTrip) {
  auto xla_hlo_sharding = xla::HloSharding::Tile(xla::TileAssignment({2, 1}));
  auto sharding = HloSharding::Create(
      DeviceList(DeviceList::Devices(client()->devices().begin(),
                                     client()->devices().end())),
      /*xla_hlo_sharding=*/xla_hlo_sharding);

  TF_ASSERT_OK_AND_ASSIGN(auto serialized, Serialize(*sharding));

  auto deserialized_options =
      std::make_unique<DeserializeShardingOptions>(client());
  TF_ASSERT_OK_AND_ASSIGN(
      auto deserialized,
      Deserialize(serialized, std::move(deserialized_options)));

  const auto* out_sharding = llvm::dyn_cast<HloSharding>(deserialized.get());
  ASSERT_NE(out_sharding, nullptr);
  EXPECT_THAT(out_sharding->devices(), ElementsAreArray(sharding->devices()));
  EXPECT_EQ(out_sharding->xla_hlo_sharding(), sharding->xla_hlo_sharding());
}

INSTANTIATE_TEST_SUITE_P(NumDevices, XlaShardingSerDesTest, testing::Values(2));

}  // namespace
}  // namespace ifrt
}  // namespace xla
