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
#include "xla/stream_executor/device_description.h"

#include <gtest/gtest.h>
#include "xla/stream_executor/semantic_version.h"

namespace stream_executor {
namespace {

TEST(DeviceDescription, DefaultConstruction) {
  DeviceDescription desc;
  EXPECT_EQ(desc.device_address_bits(), -1);
  EXPECT_EQ(desc.device_memory_size(), -1);
  EXPECT_EQ(desc.clock_rate_ghz(), -1);
  EXPECT_EQ(desc.name(), "<undefined>");
  EXPECT_EQ(desc.platform_version(), "<undefined>");
  constexpr SemanticVersion kZeroVersion = {0, 0, 0};
  EXPECT_EQ(desc.driver_version(), kZeroVersion);
  EXPECT_EQ(desc.runtime_version(), kZeroVersion);
  EXPECT_EQ(desc.pci_bus_id(), "<undefined>");
}

}  // namespace
}  // namespace stream_executor
