/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/python/ifrt/cast.h"

#include <memory>

#include <gtest/gtest.h>
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/sharding.h"
#include "tsl/platform/test.h"

namespace xla {
namespace ifrt {
namespace {

TEST(CastTest, ShardingCast) {
  std::shared_ptr<const Sharding> sharding =
      SingleDeviceSharding::Create(/*device=*/nullptr, MemoryKind());
  EXPECT_TRUE(::xla::ifrt::isa<SingleDeviceSharding>(*sharding));
  EXPECT_FALSE(::xla::ifrt::isa<OpaqueSharding>(*sharding));

  {
    const auto* single_device_sharding =
        ::xla::ifrt::dyn_cast<SingleDeviceSharding>(sharding.get());
    EXPECT_TRUE(single_device_sharding->IsFullyReplicated());
  }

  EXPECT_EQ(::xla::ifrt::dyn_cast<OpaqueSharding>(sharding.get()), nullptr);

  {
    const auto& single_device_sharding =
        ::xla::ifrt::cast<SingleDeviceSharding>(*sharding);
    EXPECT_TRUE(single_device_sharding.IsFullyReplicated());
  }
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
