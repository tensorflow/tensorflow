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

#include "xla/service/debug/unstable_reduction_finder.h"

#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/primitive_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

class ReductionStabilityTest : public HloHardwareIndependentTestBase,
                               public ::testing::WithParamInterface<HloOpcode> {
};

constexpr absl::string_view kHloReductionTemplate = R"(
red {
    p0 = $2[] parameter(0)
    p1 = $2[] parameter(1)
    ROOT red = $2[] $0(p0, p1)
}

ENTRY main {
    p0 = $2[164] parameter(0)
    init = $2[] constant($1)
    ROOT red = $2[] reduce(p0, init), to_apply=red, dimensions={0}
})";

TEST_F(ReductionStabilityTest,
       FindsUnstableReductionsIgnoresMaxAndMinReductions) {
  const std::string bf16 = primitive_util::LowercasePrimitiveTypeName(BF16);
  TF_ASSERT_OK_AND_ASSIGN(
      auto min_module,
      ParseAndReturnVerifiedModule(absl::Substitute(
          kHloReductionTemplate, HloOpcodeString(HloOpcode::kMinimum), "-inf",
          bf16)));
  EXPECT_TRUE(FindUnstableReductionInstructions(min_module.get()).empty());

  TF_ASSERT_OK_AND_ASSIGN(
      auto max_module, ParseAndReturnVerifiedModule(absl::Substitute(
                           kHloReductionTemplate,
                           HloOpcodeString(HloOpcode::kMaximum), "inf", bf16)));
  EXPECT_TRUE(FindUnstableReductionInstructions(max_module.get()).empty());
}

TEST_F(ReductionStabilityTest,
       FindsUnstableReductionsFlagsLowPrecisionSumAndProductReductions) {
  const std::string bf16 = primitive_util::LowercasePrimitiveTypeName(BF16);
  TF_ASSERT_OK_AND_ASSIGN(
      auto min_module,
      ParseAndReturnVerifiedModule(absl::Substitute(
          kHloReductionTemplate, HloOpcodeString(HloOpcode::kAdd), "0", bf16)));
  EXPECT_FALSE(FindUnstableReductionInstructions(min_module.get()).empty());

  TF_ASSERT_OK_AND_ASSIGN(
      auto max_module, ParseAndReturnVerifiedModule(absl::Substitute(
                           kHloReductionTemplate,
                           HloOpcodeString(HloOpcode::kMultiply), "1", bf16)));
  EXPECT_FALSE(FindUnstableReductionInstructions(max_module.get()).empty());
}

TEST_F(ReductionStabilityTest,
       FindsUnstableReductionsIgnoresF32SumAndProductReductions) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto min_module,
      ParseAndReturnVerifiedModule(absl::Substitute(
          kHloReductionTemplate, HloOpcodeString(HloOpcode::kAdd), "0",
          primitive_util::LowercasePrimitiveTypeName(F32))));
  EXPECT_TRUE(FindUnstableReductionInstructions(min_module.get()).empty());
}

}  // anonymous namespace
}  // namespace xla
