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

#include "xla/stream_executor/cuda/cuda_compute_capability.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/hash/hash_testing.h"
#include "absl/status/status.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.pb.h"
#include "xla/tsl/platform/status_matchers.h"

namespace stream_executor {
namespace {
using tsl::testing::IsOkAndHolds;
using tsl::testing::StatusIs;

TEST(CudaComputeCapabilityTest, ToString) {
  EXPECT_EQ(CudaComputeCapability(100, 52).ToString(), "100.52");
}

TEST(CudaComputeCapabilityTest, FromString) {
  EXPECT_THAT(CudaComputeCapability::FromString("100.52"),
              IsOkAndHolds(CudaComputeCapability(100, 52)));
  EXPECT_THAT(CudaComputeCapability::FromString("1"),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(CudaComputeCapability::FromString("12"),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(CudaComputeCapability::FromString("x"),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(CudaComputeCapability::FromString("1."),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(CudaComputeCapability::FromString("1.x"),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(CudaComputeCapabilityTest, ToProto) {
  CudaComputeCapabilityProto proto = CudaComputeCapability(100, 5).ToProto();
  EXPECT_EQ(proto.major(), 100);
  EXPECT_EQ(proto.minor(), 5);
}

TEST(CudaComputeCapabilityTest, FromProto) {
  CudaComputeCapabilityProto proto;
  proto.set_major(100);
  proto.set_minor(5);
  CudaComputeCapability cc(proto);
  EXPECT_EQ(cc.major, 100);
  EXPECT_EQ(cc.minor, 5);
}

TEST(CudaComputeCapabilityTest, Hash) {
  EXPECT_TRUE(absl::VerifyTypeImplementsAbslHashCorrectly({
      CudaComputeCapability(0, 0),
      CudaComputeCapability(0, 1),
      CudaComputeCapability(1, 0),
      CudaComputeCapability(1, 1),
  }));
}

TEST(CudaComputeCapabilityTest, GenerationNumericTest) {
  EXPECT_TRUE(CudaComputeCapability(7, 5).IsAtLeastVolta());
  EXPECT_TRUE(CudaComputeCapability(8, 0).IsAtLeastAmpere());
  EXPECT_TRUE(CudaComputeCapability(9, 0).IsAtLeastHopper());
  EXPECT_TRUE(CudaComputeCapability(10, 0).IsAtLeastBlackwell());
}

TEST(CudaComputeCapabilityTest, GenerationLiteralTest) {
  EXPECT_TRUE(CudaComputeCapability::Volta().IsAtLeast(7));
  EXPECT_TRUE(CudaComputeCapability::Ampere().IsAtLeast(8));
  EXPECT_TRUE(CudaComputeCapability::Hopper().IsAtLeast(9));
  EXPECT_TRUE(CudaComputeCapability::Blackwell().IsAtLeast(10));
}

TEST(CudaComputeCapabilityTest, ComparisonTest) {
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
