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

#ifndef XLA_PYTHON_IFRT_SHARDING_TEST_UTIL_H_
#define XLA_PYTHON_IFRT_SHARDING_TEST_UTIL_H_

#include <memory>

#include "xla/python/ifrt/client.h"
#include "tsl/platform/test.h"

namespace xla {
namespace ifrt {
namespace test_util {

// Parameters for ShardingTest.
// Requests `num_devices` total devices, where `num_addressable_devices` of them
// are addressable, and the rest of devices are non-addressable.
struct ShardingTestParam {
  int num_devices;
  int num_addressable_devices;
};

// Test fixture for sharding tests.
class ShardingTest : public testing::TestWithParam<ShardingTestParam> {
 public:
  void SetUp() override;
  Client* client() { return client_.get(); }

  // Returns `DeviceList` containing devices at given indexes (not ids) within
  // `client.devices()`.
  // REQUIRES: 0 <= device_indices[i] < num_devices
  DeviceList GetDevices(absl::Span<const int> device_indices);

 private:
  std::shared_ptr<Client> client_;
};

}  // namespace test_util
}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_SHARDING_TEST_UTIL_H_
