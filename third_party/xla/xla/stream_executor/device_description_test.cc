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

#include "xla/stream_executor/semantic_version.h"
#include "tsl/platform/test.h"

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

TEST(CudaComputeCapability, GenerationNumericTest) {
  EXPECT_TRUE(CudaComputeCapability(7, 5).IsAtLeastVolta());
  EXPECT_TRUE(CudaComputeCapability(8, 0).IsAtLeastAmpere());
  EXPECT_TRUE(CudaComputeCapability(9, 0).IsAtLeastHopper());
  EXPECT_TRUE(CudaComputeCapability(10, 0).IsAtLeastBlackwell());
}

TEST(CudaComputeCapability, GenerationLiteralTest) {
  EXPECT_TRUE(CudaComputeCapability::Volta().IsAtLeast(7));
  EXPECT_TRUE(CudaComputeCapability::Ampere().IsAtLeast(8));
  EXPECT_TRUE(CudaComputeCapability::Hopper().IsAtLeast(9));
  EXPECT_TRUE(CudaComputeCapability::Blackwell().IsAtLeast(10));
}

TEST(CudaComputeCapability, ComparisonTest) {
  CudaComputeCapability lower{1, 0};
  CudaComputeCapability slightly_higher{1, 1};
  CudaComputeCapability higher{2, 0};

  EXPECT_TRUE(lower == lower);
  EXPECT_FALSE(lower == slightly_higher);
  EXPECT_FALSE(lower == higher);

  EXPECT_TRUE(lower <= lower);
  EXPECT_TRUE(lower < slightly_higher);
  EXPECT_TRUE(lower <= slightly_higher);

  EXPECT_FALSE(lower < lower);
  EXPECT_FALSE(slightly_higher <= lower);
  EXPECT_FALSE(slightly_higher < lower);

  EXPECT_TRUE(slightly_higher >= slightly_higher);
  EXPECT_TRUE(slightly_higher > lower);
  EXPECT_TRUE(slightly_higher >= lower);

  EXPECT_FALSE(slightly_higher > slightly_higher);
  EXPECT_FALSE(lower > slightly_higher);
  EXPECT_FALSE(lower >= slightly_higher);

  EXPECT_TRUE(higher > slightly_higher);
  EXPECT_TRUE(higher >= slightly_higher);
  EXPECT_TRUE(higher >= higher);
}

}  // namespace
}  // namespace stream_executor
