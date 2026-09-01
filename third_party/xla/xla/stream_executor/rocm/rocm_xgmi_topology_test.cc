/* Copyright 2025 The OpenXLA Authors.
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

#include "xla/stream_executor/rocm/rocm_xgmi_topology.h"

#include <gtest/gtest.h>

namespace stream_executor::gpu {
namespace {

TEST(XgmiTopologyInfoTest, DefaultValues) {
  XgmiTopologyInfo info;
  EXPECT_EQ(info.active_links, 0);
  EXPECT_EQ(info.hive_id, 0);
}

TEST(GetRocmXgmiTopologyTest, InvalidBdfReturnsDefault) {
  // An invalid PCI bus ID should return default topology (0 links).
  XgmiTopologyInfo info = GetRocmXgmiTopology("invalid");
  EXPECT_EQ(info.active_links, 0);
  EXPECT_EQ(info.hive_id, 0);
}

TEST(GetRocmXgmiTopologyTest, EmptyBdfReturnsDefault) {
  XgmiTopologyInfo info = GetRocmXgmiTopology("");
  EXPECT_EQ(info.active_links, 0);
  EXPECT_EQ(info.hive_id, 0);
}

}  // namespace
}  // namespace stream_executor::gpu
