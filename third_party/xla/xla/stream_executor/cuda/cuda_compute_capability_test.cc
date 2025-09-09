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
#include "absl/status/status_matchers.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.pb.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor {
namespace {
using absl_testing::IsOkAndHolds;
using absl_testing::StatusIs;

TEST(CudaComputeCapabilityTest, ToString) {
  EXPECT_EQ(CudaComputeCapability(
                100, 52, CudaComputeCapability::FeatureExtension::kNone)
                .ToString(),
            "100.52");
  // For all compute capabilities of at least 9.x we expect an accelerated
  // feature set to exist
  EXPECT_EQ(CudaComputeCapability(100, 52).ToString(), "100.52");
  EXPECT_EQ(CudaComputeCapability(
                100, 52,
                CudaComputeCapability::FeatureExtension::kAcceleratedFeatures)
                .ToString(),
            "100.52a");
  EXPECT_EQ(
      CudaComputeCapability(
          100, 52,
          CudaComputeCapability::FeatureExtension::kForwardCompatibleFeatures)
          .ToString(),
      "100.52f");
}

TEST(CudaComputeCapabilityTest, FromString) {
  using FeatureExtension = CudaComputeCapability::FeatureExtension;

  EXPECT_THAT(CudaComputeCapability::FromString("100.52"),
              IsOkAndHolds(CudaComputeCapability(100, 52)));
  EXPECT_THAT(CudaComputeCapability::FromString("100.52a"),
              IsOkAndHolds(CudaComputeCapability(
                  100, 52, FeatureExtension::kAcceleratedFeatures)));
  EXPECT_THAT(CudaComputeCapability::FromString("100.52A"),
              IsOkAndHolds(CudaComputeCapability(
                  100, 52, FeatureExtension::kAcceleratedFeatures)));
  EXPECT_THAT(CudaComputeCapability::FromString("100.52 a"),
              IsOkAndHolds(CudaComputeCapability(
                  100, 52, FeatureExtension::kAcceleratedFeatures)));
  EXPECT_THAT(CudaComputeCapability::FromString("100.52f"),
              IsOkAndHolds(CudaComputeCapability(
                  100, 52, FeatureExtension::kForwardCompatibleFeatures)));
  EXPECT_THAT(CudaComputeCapability::FromString("100.52F"),
              IsOkAndHolds(CudaComputeCapability(
                  100, 52, FeatureExtension::kForwardCompatibleFeatures)));
  EXPECT_THAT(CudaComputeCapability::FromString("100.52 f"),
              IsOkAndHolds(CudaComputeCapability(
                  100, 52, FeatureExtension::kForwardCompatibleFeatures)));
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
  CudaComputeCapabilityProto proto0 =
      CudaComputeCapability(100, 5,
                            CudaComputeCapability::FeatureExtension::kNone)
          .ToProto();
  EXPECT_EQ(proto0.major(), 100);
  EXPECT_EQ(proto0.minor(), 5);
  EXPECT_EQ(proto0.feature_extension(), CudaComputeCapabilityProto::NONE);
  CudaComputeCapabilityProto proto1 =
      CudaComputeCapability(
          100, 5, CudaComputeCapability::FeatureExtension::kAcceleratedFeatures)
          .ToProto();
  EXPECT_EQ(proto1.major(), 100);
  EXPECT_EQ(proto1.minor(), 5);
  EXPECT_EQ(proto1.feature_extension(),
            CudaComputeCapabilityProto::ACCELERATED_FEATURES);
  CudaComputeCapabilityProto proto2 =
      CudaComputeCapability(
          100, 5,
          CudaComputeCapability::FeatureExtension::kForwardCompatibleFeatures)
          .ToProto();
  EXPECT_EQ(proto2.major(), 100);
  EXPECT_EQ(proto2.minor(), 5);
  EXPECT_EQ(proto2.feature_extension(),
            CudaComputeCapabilityProto::FORWARD_COMPATIBLE_FEATURES);
}

TEST(CudaComputeCapabilityTest, FromProtoWithFeatureExtensionUnspecified) {
  using FeatureExtension = CudaComputeCapability::FeatureExtension;

  // An unspecified feature extension field should be interpreted as NONE - no
  // feature extension enabled.
  CudaComputeCapabilityProto proto;
  proto.set_major(100);
  proto.set_minor(5);
  TF_ASSERT_OK_AND_ASSIGN(auto cc, CudaComputeCapability::FromProto(proto));
  EXPECT_EQ(cc.major, 100);
  EXPECT_EQ(cc.minor, 5);
  EXPECT_EQ(cc.feature_extension, FeatureExtension::kNone);

  // On Hopper we expect accelerated features to be the default as this is how
  // XLA treated Hopper GPUs before we could handle feature extensions
  // explicitly.
  proto.set_major(9);
  proto.set_minor(5);
  TF_ASSERT_OK_AND_ASSIGN(cc, CudaComputeCapability::FromProto(proto));
  EXPECT_EQ(cc.major, 9);
  EXPECT_EQ(cc.minor, 5);
  EXPECT_EQ(cc.feature_extension, FeatureExtension::kAcceleratedFeatures);

  // On Blackwell we expect accelerated features to be the default as this is
  // how XLA treated Blackwell GPUs before we could handle feature extensions
  // explicitly.
  proto.set_major(10);
  proto.set_minor(2);
  TF_ASSERT_OK_AND_ASSIGN(cc, CudaComputeCapability::FromProto(proto));
  EXPECT_EQ(cc.major, 10);
  EXPECT_EQ(cc.minor, 2);
  EXPECT_EQ(cc.feature_extension, FeatureExtension::kAcceleratedFeatures);
}

TEST(CudaComputeCapabilityTest, FromProtoWithFeatureExtensionSpecified) {
  using FeatureExtension = CudaComputeCapability::FeatureExtension;

  CudaComputeCapabilityProto proto;
  proto.set_major(100);
  proto.set_minor(5);
  proto.set_feature_extension(CudaComputeCapabilityProto::ACCELERATED_FEATURES);
  TF_ASSERT_OK_AND_ASSIGN(auto cc, CudaComputeCapability::FromProto(proto));
  EXPECT_EQ(cc.major, 100);
  EXPECT_EQ(cc.minor, 5);
  EXPECT_EQ(cc.feature_extension, FeatureExtension::kAcceleratedFeatures);
}

TEST(CudaComputeCapabilityTest, Hash) {
  using FeatureExtension = CudaComputeCapability::FeatureExtension;

  EXPECT_TRUE(absl::VerifyTypeImplementsAbslHashCorrectly({
      CudaComputeCapability(0, 0),
      CudaComputeCapability(0, 0, FeatureExtension::kAcceleratedFeatures),
      CudaComputeCapability(0, 0, FeatureExtension::kForwardCompatibleFeatures),
      CudaComputeCapability(0, 1),
      CudaComputeCapability(0, 1, FeatureExtension::kAcceleratedFeatures),
      CudaComputeCapability(0, 1, FeatureExtension::kForwardCompatibleFeatures),
      CudaComputeCapability(1, 0),
      CudaComputeCapability(1, 0, FeatureExtension::kAcceleratedFeatures),
      CudaComputeCapability(1, 0, FeatureExtension::kForwardCompatibleFeatures),
      CudaComputeCapability(1, 1),
      CudaComputeCapability(1, 1, FeatureExtension::kAcceleratedFeatures),
      CudaComputeCapability(1, 1, FeatureExtension::kForwardCompatibleFeatures),
  }));
}

TEST(CudaComputeCapabilityTest, GenerationNumericTest) {
  EXPECT_TRUE(CudaComputeCapability(7, 5).IsAtLeastVolta());
  EXPECT_TRUE(CudaComputeCapability(8, 0).IsAtLeastAmpere());
}

TEST(CudaComputeCapabilityTest, ComparisonTest) {
  using FeatureExtension = CudaComputeCapability::FeatureExtension;

  CudaComputeCapability base{1, 0};
  CudaComputeCapability base_but_accelerated{
      1, 0, FeatureExtension::kAcceleratedFeatures};
  CudaComputeCapability base_but_forward_compatible{
      1, 0, FeatureExtension::kForwardCompatibleFeatures};
  CudaComputeCapability newer_but_same_generation{1, 1};
  CudaComputeCapability newer_but_same_generation_accelerated{
      1, 1, FeatureExtension::kAcceleratedFeatures};
  CudaComputeCapability newer_but_same_generation_compatible{
      1, 1, FeatureExtension::kForwardCompatibleFeatures};
  CudaComputeCapability next_generation{2, 0};

  EXPECT_TRUE(base == base);
  EXPECT_TRUE(base_but_accelerated == base_but_accelerated);
  EXPECT_FALSE(base == base_but_accelerated);
  EXPECT_FALSE(base == newer_but_same_generation);
  EXPECT_FALSE(base == next_generation);

  // sm_10 kernels can run sm_10, sm_11, and sm_20 GPUs.
  // But sm_10a kernels can only run on sm_10 GPUs.
  // sm_10f kernels can run on any sm_10 and sm_11 GPUs.
  EXPECT_TRUE(base.CanRunOn(base));
  EXPECT_TRUE(base.SupportsAllFeaturesOf(base));
  EXPECT_TRUE(base.CanRunOn(newer_but_same_generation));
  EXPECT_TRUE(base.CanRunOn(newer_but_same_generation_accelerated));
  EXPECT_TRUE(base.CanRunOn(newer_but_same_generation_compatible));
  EXPECT_FALSE(base.SupportsAllFeaturesOf(newer_but_same_generation));
  EXPECT_FALSE(
      base.SupportsAllFeaturesOf(newer_but_same_generation_accelerated));
  EXPECT_FALSE(
      base.SupportsAllFeaturesOf(newer_but_same_generation_compatible));
  EXPECT_TRUE(base.CanRunOn(next_generation));
  EXPECT_FALSE(base.SupportsAllFeaturesOf(next_generation));

  EXPECT_TRUE(base_but_accelerated.CanRunOn(base));
  EXPECT_TRUE(base_but_accelerated.SupportsAllFeaturesOf(base));
  EXPECT_FALSE(base_but_accelerated.CanRunOn(newer_but_same_generation));
  EXPECT_FALSE(
      base_but_accelerated.SupportsAllFeaturesOf(newer_but_same_generation));
  EXPECT_FALSE(
      base_but_accelerated.CanRunOn(newer_but_same_generation_accelerated));
  EXPECT_FALSE(base_but_accelerated.SupportsAllFeaturesOf(
      newer_but_same_generation_accelerated));
  EXPECT_FALSE(
      base_but_accelerated.CanRunOn(newer_but_same_generation_compatible));
  EXPECT_FALSE(base_but_accelerated.SupportsAllFeaturesOf(
      newer_but_same_generation_compatible));
  EXPECT_FALSE(base_but_accelerated.CanRunOn(next_generation));
  EXPECT_FALSE(base_but_accelerated.SupportsAllFeaturesOf(next_generation));

  EXPECT_TRUE(base_but_forward_compatible.CanRunOn(base));
  EXPECT_TRUE(base_but_forward_compatible.SupportsAllFeaturesOf(base));
  EXPECT_TRUE(base_but_forward_compatible.CanRunOn(newer_but_same_generation));
  EXPECT_FALSE(base_but_forward_compatible.SupportsAllFeaturesOf(
      newer_but_same_generation));
  EXPECT_TRUE(base_but_forward_compatible.CanRunOn(
      newer_but_same_generation_accelerated));
  EXPECT_FALSE(base_but_forward_compatible.SupportsAllFeaturesOf(
      newer_but_same_generation_accelerated));
  EXPECT_TRUE(base_but_forward_compatible.CanRunOn(
      newer_but_same_generation_compatible));
  EXPECT_FALSE(base_but_forward_compatible.SupportsAllFeaturesOf(
      newer_but_same_generation_compatible));
  EXPECT_FALSE(base_but_forward_compatible.CanRunOn(next_generation));
  EXPECT_FALSE(
      base_but_forward_compatible.SupportsAllFeaturesOf(next_generation));
}

TEST(CudaComputeCapabilityTest, GetPtxAsTargetName) {
  EXPECT_EQ(CudaComputeCapability::Ampere().GetPtxAsTargetName(
                CudaComputeCapability::CompileMode::kPtx),
            "compute_80");
  EXPECT_EQ(CudaComputeCapability::Ampere().GetPtxAsTargetName(
                CudaComputeCapability::CompileMode::kLto),
            "lto_80");
  EXPECT_EQ(CudaComputeCapability::Ampere().GetPtxAsTargetName(
                CudaComputeCapability::CompileMode::kSass),
            "sm_80");

  EXPECT_EQ(CudaComputeCapability::Hopper().GetPtxAsTargetName(), "sm_90");
  EXPECT_EQ(
      CudaComputeCapability(
          9, 0, CudaComputeCapability::FeatureExtension::kAcceleratedFeatures)
          .GetPtxAsTargetName(),
      "sm_90a");
  EXPECT_EQ(
      CudaComputeCapability(
          10, 0,
          CudaComputeCapability::FeatureExtension::kForwardCompatibleFeatures)
          .GetPtxAsTargetName(),
      "sm_100f");
}

TEST(CudaComputeCapabilityTest, WithoutAnyFeatureExtension) {
  EXPECT_EQ(CudaComputeCapability(
                100, 52, CudaComputeCapability::FeatureExtension::kNone)
                .WithoutAnyFeatureExtension(),
            CudaComputeCapability(100, 52));
  EXPECT_EQ(CudaComputeCapability(
                100, 52,
                CudaComputeCapability::FeatureExtension::kAcceleratedFeatures)
                .WithoutAnyFeatureExtension(),
            CudaComputeCapability(100, 52));
  EXPECT_EQ(
      CudaComputeCapability(
          100, 52,
          CudaComputeCapability::FeatureExtension::kForwardCompatibleFeatures)
          .WithoutAnyFeatureExtension(),
      CudaComputeCapability(100, 52));
}

}  // namespace
}  // namespace stream_executor
