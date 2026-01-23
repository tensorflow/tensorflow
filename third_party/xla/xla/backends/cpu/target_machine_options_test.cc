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

#include "xla/backends/cpu/target_machine_options.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "xla/service/cpu/executable.pb.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla.pb.h"

namespace xla {
namespace cpu {
namespace {

TEST(TargetMachineOptionsTest, ToProto) {
  DebugOptions debug_options;
  TargetMachineOptions options(debug_options);
  TargetMachineOptionsProto proto = options.ToProto();

  EXPECT_EQ(proto.triple(), options.triple());
  EXPECT_EQ(proto.cpu(), options.cpu());
  EXPECT_EQ(proto.features(), options.GetTargetMachineFeatures());
}

TEST(TargetMachineOptionsTest, FromProto) {
  TargetMachineOptionsProto proto;
  proto.set_triple("test_triple");
  proto.set_cpu("test_cpu");
  proto.set_features("+enabled_feature,-disabled_feature");

  TF_ASSERT_OK_AND_ASSIGN(TargetMachineOptions options,
                          TargetMachineOptions::FromProto(proto));

  EXPECT_EQ(options.triple(), "test_triple");
  EXPECT_EQ(options.cpu(), "test_cpu");
  EXPECT_THAT(options.enabled_features(),
              testing::ElementsAre("enabled_feature"));
  EXPECT_THAT(options.disabled_features(),
              testing::ElementsAre("disabled_feature"));
  EXPECT_EQ(options.GetTargetMachineFeatures(),
            "+enabled_feature,-disabled_feature");
}

TEST(TargetMachineOptionsTest, ProtoRoundTrip) {
  DebugOptions debug_options;
  TargetMachineOptions options(debug_options);
  TargetMachineOptionsProto proto = options.ToProto();
  TF_ASSERT_OK_AND_ASSIGN(TargetMachineOptions new_options,
                          TargetMachineOptions::FromProto(proto));

  EXPECT_EQ(new_options.triple(), options.triple());
  EXPECT_EQ(new_options.cpu(), options.cpu());
  EXPECT_EQ(new_options.GetTargetMachineFeatures(),

            options.GetTargetMachineFeatures());
}

TEST(TargetMachineOptionsTest, ConstructorWithFeatures) {
  TargetMachineOptions options("test_triple", "test_cpu", "+avx2,-avx512");

  EXPECT_EQ(options.triple(), "test_triple");
  EXPECT_EQ(options.cpu(), "test_cpu");
  EXPECT_THAT(options.enabled_features(), testing::ElementsAre("avx2"));
  EXPECT_THAT(options.disabled_features(), testing::ElementsAre("avx512"));
  EXPECT_EQ(options.GetTargetMachineFeatures(), "+avx2,-avx512");
}

TEST(TargetMachineOptionsTest, GetTargetMachineFeaturesFormat) {
  TargetMachineOptions options1("t", "c", "+f1,-f2");
  EXPECT_EQ(options1.GetTargetMachineFeatures(), "+f1,-f2");

  TargetMachineOptions options2("t", "c", "-f2,+f1");
  EXPECT_EQ(options2.GetTargetMachineFeatures(), "+f1,-f2");

  TargetMachineOptions options3("t", "c", "+f1");
  EXPECT_EQ(options3.GetTargetMachineFeatures(), "+f1");

  TargetMachineOptions options4("t", "c", "-f2");
  EXPECT_EQ(options4.GetTargetMachineFeatures(), "-f2");

  TargetMachineOptions options5("t", "c", "");
  EXPECT_EQ(options5.GetTargetMachineFeatures(), "");

  TargetMachineOptions options6("t", "c", "+f1,-f2,+f3,-f4");
  EXPECT_EQ(options6.GetTargetMachineFeatures(), "+f1,+f3,-f2,-f4");
}

TEST(TargetMachineOptionsTest, FromProtoWithMalformedFeatures) {
  TargetMachineOptionsProto proto;
  proto.set_triple("test_triple");
  proto.set_cpu("test_cpu");
  proto.set_features("malformed");

  auto options = TargetMachineOptions::FromProto(proto);

  EXPECT_EQ(options.status().code(), absl::StatusCode::kInternal);
}

TEST(TargetMachineOptionsTest, FromProtoWithEmptyFeatureAfterPlus) {
  TargetMachineOptionsProto proto;
  proto.set_triple("test_triple");
  proto.set_cpu("test_cpu");
  proto.set_features("+");

  auto options = TargetMachineOptions::FromProto(proto);

  EXPECT_EQ(options.status().code(), absl::StatusCode::kInternal);
}

TEST(TargetMachineOptionsTest, SetFeatures) {
  TargetMachineOptions options("test_triple", "test_cpu", "");
  TF_ASSERT_OK(options.SetFeatures("+avx2,-avx512"));

  EXPECT_EQ(options.GetTargetMachineFeatures(), "+avx2,-avx512");
}

TEST(TargetMachineOptionsTest, AVX512ImpliesNoScatterAndNoGather) {
  TargetMachineOptions options("test_triple", "test_cpu", "+avx512");
  EXPECT_EQ(options.GetTargetMachineFeatures(),
            "+avx512,+prefer-no-scatter,+prefer-no-gather");
}

TEST(TargetMachineOptionsTest, GetTargetMachineFeaturesVector) {
  TargetMachineOptions options("test_triple", "test_cpu", "+avx2,-avx512");
  EXPECT_THAT(options.GetTargetMachineFeaturesVector(),
              testing::ElementsAre("+avx2", "-avx512"));
}

TEST(TargetMachineOptionsTest, TestTargetMachineOptionsEquality) {
  TargetMachineOptions options1("test_triple", "test_cpu", "+avx2,-avx512");
  TargetMachineOptions options2("test_triple", "test_cpu", "+avx2,-avx512");
  EXPECT_EQ(options1, options2);
  TargetMachineOptions options3("test_triple", "test_cpu", "+avx2");
  EXPECT_NE(options1, options3);
}

}  // namespace
}  // namespace cpu
}  // namespace xla
