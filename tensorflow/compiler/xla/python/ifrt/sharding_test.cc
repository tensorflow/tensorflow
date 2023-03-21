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
#include "tensorflow/tsl/lib/core/status_test_util.h"

namespace xla {
namespace ifrt {
namespace {

using ::testing::ElementsAre;

TEST(SingleDeviceShardingTest, IndexDomains) {
  std::shared_ptr<const Sharding> sharding =
      SingleDeviceSharding::Create(reinterpret_cast<Device*>(1));

  Shape shape({10, 20});
  auto index_domains = sharding->IndexDomains(shape);
  TF_ASSERT_OK(index_domains.status());
  EXPECT_THAT(*index_domains, ElementsAre(IndexDomain(shape)));
}

TEST(OpaqueShardingTest, Disassemble) {
  DeviceList::Devices devices;
  devices.reserve(2);
  devices.push_back(reinterpret_cast<Device*>(1));
  devices.push_back(reinterpret_cast<Device*>(2));
  DeviceList device_list(std::move(devices));

  std::vector<Shape> shapes;
  shapes.reserve(2);
  shapes.push_back(Shape({10}));
  shapes.push_back(Shape({20}));
  OpaqueSharding::DisassembleFunc disassemble_func =
      OpaqueSharding::MakeDisassembleFuncFromShapes(shapes);

  std::shared_ptr<const Sharding> sharding =
      OpaqueSharding::Create(device_list, std::move(disassemble_func));

  auto exploded = sharding->Disassemble(Shape({30}));
  TF_ASSERT_OK(exploded.status());

  ASSERT_THAT(*exploded, testing::SizeIs(2));
  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ((*exploded)[i].first, shapes[i]);
    EXPECT_TRUE(llvm::isa<SingleDeviceSharding>((*exploded)[i].second.get()));
    EXPECT_THAT((*exploded)[i].second->devices().devices(),
                testing::ElementsAre(device_list.devices()[i]));
  }
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
