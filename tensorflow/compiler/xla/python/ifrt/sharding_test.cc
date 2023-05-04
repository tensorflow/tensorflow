/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/python/ifrt/sharding.h"

#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "llvm/Support/Casting.h"
#include "tensorflow/compiler/xla/python/ifrt/device.h"
#include "tensorflow/compiler/xla/python/ifrt/ir/sharding_param.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/status_matchers.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace xla {
namespace ifrt {
namespace {

using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::SizeIs;
using ::tsl::testing::StatusIs;

DeviceList CreateDummyDevices(int count) {
  DeviceList::Devices devices;
  devices.reserve(count);
  for (int i = 0; i < count; ++i) {
    devices.push_back(reinterpret_cast<Device*>(i + 1));
  }
  return DeviceList(std::move(devices));
}

TEST(SingleDeviceShardingTest, IndexDomains) {
  std::shared_ptr<const Sharding> sharding =
      SingleDeviceSharding::Create(reinterpret_cast<Device*>(1));

  Shape shape({10, 20});
  TF_ASSERT_OK_AND_ASSIGN(auto index_domains, sharding->IndexDomains(shape));
  EXPECT_THAT(index_domains, ElementsAre(IndexDomain(shape)));
}

TEST(OpaqueShardingTest, Disassemble) {
  DeviceList device_list = CreateDummyDevices(2);

  std::vector<Shape> shapes;
  shapes.reserve(2);
  shapes.push_back(Shape({10}));
  shapes.push_back(Shape({20}));
  OpaqueSharding::DisassembleFunc disassemble_func =
      OpaqueSharding::MakeDisassembleFuncFromShapes(shapes);

  std::shared_ptr<const Sharding> opaque_sharding =
      OpaqueSharding::Create(device_list, std::move(disassemble_func));

  TF_ASSERT_OK_AND_ASSIGN(auto exploded,
                          opaque_sharding->Disassemble(Shape({30})));

  ASSERT_THAT(exploded, SizeIs(2));
  for (int i = 0; i < 2; ++i) {
    const auto& [shape, sharding] = exploded[i];
    EXPECT_EQ(shape, shapes[i]);
    EXPECT_TRUE(llvm::isa<SingleDeviceSharding>(*sharding));
    EXPECT_THAT(sharding->devices().devices(),
                ElementsAre(device_list.devices()[i]));
  }
}

TEST(ShardingParamShardingTest, FailToCreateWhenDeviceCountNotMatch) {
  DeviceList device_list = CreateDummyDevices(2);
  ShardingParam param{{2, 3}, {{1, 0}, {3, 2}}};

  EXPECT_THAT(ShardingParamSharding::Create(param, device_list),
              StatusIs(tsl::error::FAILED_PRECONDITION,
                       HasSubstr("Device counts don't match. From "
                                 "ShardingParam 6 vs from DeviceList 2")));
}

TEST(ShardingParamShardingTest, Disassemble) {
  DeviceList device_list = CreateDummyDevices(6);
  ShardingParam param{{2, 3}, {{1, 0}, {3, 2}}};
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<const Sharding> param_sharding,
                          ShardingParamSharding::Create(param, device_list));

  TF_ASSERT_OK_AND_ASSIGN(auto exploded,
                          param_sharding->Disassemble(Shape({6, 6})));
  ASSERT_THAT(exploded, SizeIs(6));
  for (int i = 0; i < 6; ++i) {
    const auto& [shape, sharding] = exploded[i];
    EXPECT_EQ(shape, Shape({3, 2}));
    EXPECT_TRUE(llvm::isa<SingleDeviceSharding>(*sharding));
    EXPECT_THAT(sharding->devices().devices(),
                ElementsAre(device_list.devices()[i]));
  }
}

TEST(ShardingParamShardingTest, DisassembleFailsWhenRankNotMatch) {
  DeviceList device_list = CreateDummyDevices(6);
  ShardingParam param{{2, 3}, {{1, 0}, {3, 2}}};
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<const Sharding> param_sharding,
                          ShardingParamSharding::Create(param, device_list));

  EXPECT_THAT(
      param_sharding->Disassemble(Shape({6, 6, 6})),
      StatusIs(tsl::error::FAILED_PRECONDITION,
               HasSubstr(
                   "Ranks don't match. From Shape 3 vs from ShardingParam 2")));
}

TEST(ShardingParamShardingTest, DisassembleFailsForUnevenSharding) {
  DeviceList device_list = CreateDummyDevices(6);
  ShardingParam param{{2, 3}, {{1, 0}, {3, 2}}};
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<const Sharding> param_sharding,
                          ShardingParamSharding::Create(param, device_list));

  EXPECT_THAT(
      param_sharding->Disassemble(Shape({7, 6})),
      StatusIs(
          tsl::error::FAILED_PRECONDITION,
          HasSubstr("Uneven shard is not supported. dim: 7, dim_shards: 2")));
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
