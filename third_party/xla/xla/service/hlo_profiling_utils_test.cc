/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/service/hlo_profiling_utils.h"

#include <memory>
#include <optional>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"

namespace xla {
namespace {

using ::testing::Eq;
using ::testing::Field;
using ::testing::Optional;

using HloProfilingUtilsTest = HloHardwareIndependentTestBase;

constexpr absl::string_view kHlo = R"(
HloModule m

ENTRY e {
  p0 = f32[8] parameter(0)
  ROOT add = f32[8] add(p0, p0)
})";

TEST_F(HloProfilingUtilsTest, FingerprintRoundTripsThroughModuleAndProto) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  EXPECT_EQ(GetHloModuleFingerprint(
                *module, HloModuleFingerprintStage::kBeforeScheduling),
            std::nullopt);

  std::string fingerprint = TagWithFingerprint(
      module.get(), HloModuleFingerprintStage::kBeforeScheduling);
  EXPECT_FALSE(fingerprint.empty());
  EXPECT_THAT(GetHloModuleFingerprint(
                  *module, HloModuleFingerprintStage::kBeforeScheduling),
              Optional(fingerprint));
  EXPECT_THAT(
      GetHloModuleFingerprint(module->ToProto(),
                              HloModuleFingerprintStage::kBeforeScheduling),
      Optional(fingerprint));
}

TEST_F(HloProfilingUtilsTest, DifferentModulesHaveDifferentFingerprints) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> a,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> b,
                       ParseAndReturnVerifiedModule(R"(
HloModule m

ENTRY e {
  p0 = f32[16] parameter(0)
  ROOT add = f32[16] add(p0, p0)
})"));
  EXPECT_NE(
      TagWithFingerprint(a.get(), HloModuleFingerprintStage::kBeforeScheduling),
      TagWithFingerprint(b.get(),
                         HloModuleFingerprintStage::kBeforeScheduling));
}

TEST(FingerprintedCostNameTest, SplitsWhatItJoins) {
  EXPECT_THAT(
      SplitFingerprintedCostName(FingerprintedCostName("08a5", "fusion.1")),
      Optional(
          AllOf(Field(&FingerprintedCostNameParts::fingerprint, Eq("08a5")),
                Field(&FingerprintedCostNameParts::op_name, Eq("fusion.1")))));
}

TEST(FingerprintedCostNameTest, NameWithoutFingerprintDoesNotSplit) {
  EXPECT_EQ(SplitFingerprintedCostName("fusion.1"), std::nullopt);
}

}  // namespace
}  // namespace xla
