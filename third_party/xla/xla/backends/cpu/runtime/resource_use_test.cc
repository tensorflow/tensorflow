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

#include "xla/backends/cpu/runtime/resource_use.h"

#include "tsl/platform/test.h"

namespace xla::cpu {
namespace {

TEST(ResourceUseTest, Equality) {
  auto token = Resource::Create(Resource::kToken);
  auto use0 = ResourceUse::Read(token);
  auto use1 = ResourceUse::Write(token);
  auto use2 = ResourceUse::Read(token);

  EXPECT_NE(use0, use1);
  EXPECT_EQ(use0, use2);
}

TEST(ResourceUseTest, ReadWriteSet) {
  ResourceUse::ReadWriteSet rwset;

  auto token0 = Resource::Create(Resource::kToken);
  auto token1 = Resource::Create(Resource::kToken);

  rwset.Add(ResourceUse::Read(token0));
  EXPECT_FALSE(rwset.HasConflicts({ResourceUse::Read(token0)}));
  EXPECT_TRUE(rwset.HasConflicts({ResourceUse::Write(token0)}));
  EXPECT_FALSE(rwset.HasConflicts({ResourceUse::Read(token1)}));
  EXPECT_FALSE(rwset.HasConflicts({ResourceUse::Write(token1)}));

  rwset.Add(ResourceUse::Write(token0));
  EXPECT_TRUE(rwset.HasConflicts({ResourceUse::Read(token0)}));
  EXPECT_TRUE(rwset.HasConflicts({ResourceUse::Write(token0)}));
  EXPECT_FALSE(rwset.HasConflicts({ResourceUse::Read(token1)}));
  EXPECT_FALSE(rwset.HasConflicts({ResourceUse::Write(token1)}));
}

}  // namespace
}  // namespace xla::cpu
