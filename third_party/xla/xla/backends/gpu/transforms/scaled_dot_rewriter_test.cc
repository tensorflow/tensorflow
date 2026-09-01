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
#include "xla/backends/gpu/transforms/scaled_dot_rewriter.h"

#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/tests/restricted/hlo_test_base_legacy.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

struct ScaledDotRewriterTestCase {
  PrimitiveType operand_type;
  PrimitiveType scale_type;
};

using ScaledDotRewriterTest =
    ::testing::WithParamInterface<ScaledDotRewriterTestCase>;

class ScaledDotRewriterTestFixture : public HloTestBaseLegacy,
                                     public ScaledDotRewriterTest {
 public:
  void SetUp() override {}
};

TEST_P(ScaledDotRewriterTestFixture, ScaledDot) {
  const ScaledDotRewriterTestCase& test_case = GetParam();

  for (auto output_type :
       {PrimitiveType::F32, PrimitiveType::BF16, PrimitiveType::F16}) {
    // lhs_scale should have two dim
    const std::string hlo_string = absl::Substitute(
        R"(
        HloModule module

        ENTRY main {
          lhs = $0[1024,512] parameter(0)
          rhs = $0[64,512] parameter(1)
          lhs_scale = $1[32,2] parameter(2)
          rhs_scale = $1[64,2] parameter(3)
          ROOT dot = $2[1024,64] scaled-dot(lhs, rhs, lhs_scale, rhs_scale),
            lhs_contracting_dims={1},
            rhs_contracting_dims={1}
        }
      )",
        absl::AsciiStrToLower(PrimitiveType_Name(test_case.operand_type)),
        absl::AsciiStrToLower(PrimitiveType_Name(test_case.scale_type)),
        absl::AsciiStrToLower(PrimitiveType_Name(output_type)));
    ASSERT_OK_AND_ASSIGN(auto module,
                         ParseAndReturnUnverifiedModule(hlo_string));

    ScaledDotRewriter rewriter;
    ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
    EXPECT_TRUE(changed);

    // Verify that the module is still valid after the rewrite.
    auto status_or_module = ParseAndReturnVerifiedModule(module->ToString());
    EXPECT_TRUE(status_or_module.status().ok()) << status_or_module.status();

    const HloInstruction* root =
        module->entry_computation()->root_instruction();

    if (output_type == PrimitiveType::F16) {
      EXPECT_EQ(root->opcode(), HloOpcode::kConvert);
      root = root->operand(0);
    }

    EXPECT_EQ(root->opcode(), HloOpcode::kDot);
    for (const HloInstruction* operand : root->operands()) {
      std::vector<HloOpcode> actual_op_codes{};
      while (operand->opcode() != HloOpcode::kParameter) {
        actual_op_codes.push_back(operand->opcode());
        if (operand->opcode() == HloOpcode::kMultiply) {
          operand = operand->operand(1);
        } else {
          operand = operand->operand(0);
        }
      }
      actual_op_codes = std::vector<HloOpcode>(actual_op_codes.rbegin(),
                                               actual_op_codes.rend());

      if (test_case.scale_type == PrimitiveType::BF16) {
        const std::vector<HloOpcode> expected_op_codes{
            HloOpcode::kBroadcast, HloOpcode::kReshape, HloOpcode::kMultiply};
        EXPECT_THAT(actual_op_codes, expected_op_codes);
      } else {
        const std::vector<HloOpcode> expected_op_codes_with_convert{
            HloOpcode::kConvert, HloOpcode::kBroadcast, HloOpcode::kReshape,
            HloOpcode::kMultiply};
        EXPECT_THAT(actual_op_codes, expected_op_codes_with_convert);
      }
    }

    EXPECT_TRUE(RunAndCompare(std::move(module),
                              ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
  }
}

using ScaledDotRewriterElementSizeTest = HloHardwareIndependentTestBase;

// Upcasting an fp4 operand (whose layout carries element_size_in_bits=4) to
// BF16 must drop the custom sub-byte element size; otherwise the rewriter emits
// an illegal bf16[...]{:E(4)} shape that the CpuGpuShapeVerifier rejects.
TEST_F(ScaledDotRewriterElementSizeTest, Fp4OperandDropsSubByteElementSize) {
  const std::string hlo_string = R"(
    HloModule module

    ENTRY main {
      lhs = f4e2m1fn[1024,512]{1,0:E(4)} parameter(0)
      rhs = f8e4m3fn[64,512] parameter(1)
      lhs_scale = f8e8m0fnu[1024,16] parameter(2)
      rhs_scale = f8e8m0fnu[64,16] parameter(3)
      ROOT dot = f32[1024,64] scaled-dot(lhs, rhs, lhs_scale, rhs_scale),
        lhs_contracting_dims={1},
        rhs_contracting_dims={1}
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo_string));

  ScaledDotRewriter rewriter;
  ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
  EXPECT_TRUE(changed);

  for (const HloComputation* computation : module->computations()) {
    for (const HloInstruction* instruction : computation->instructions()) {
      const Shape& shape = instruction->shape();
      if (!shape.IsArray() || !shape.has_layout()) {
        continue;
      }
      if (primitive_util::IsSubByteNonPredType(shape.element_type())) {
        continue;
      }
      EXPECT_EQ(shape.layout().element_size_in_bits(), 0)
          << "Non-sub-byte instruction retains custom element size: "
          << instruction->ToString();
    }
  }
}

TEST_F(ScaledDotRewriterElementSizeTest, FilterCallbackBehavior) {
  const std::string hlo_string = R"(
    HloModule module
    ENTRY main {
      lhs = f8e4m3fn[1024,512] parameter(0)
      rhs = f8e4m3fn[64,512] parameter(1)
      lhs_scale = bf16[1024,16] parameter(2)
      rhs_scale = bf16[64,16] parameter(3)
      ROOT dot = f32[1024,64] scaled-dot(lhs, rhs, lhs_scale, rhs_scale),
        lhs_contracting_dims={1},
        rhs_contracting_dims={1}
    }
  )";

  // Filter callback returning false: skip rewriting.
  ScaledDotRewriter rewriter_skip(
      [](const HloInstruction* instr) { return false; });
  ASSERT_OK_AND_ASSIGN(auto module_skip,
                       ParseAndReturnUnverifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(bool changed_skip, rewriter_skip.Run(module_skip.get()));
  EXPECT_FALSE(changed_skip);

  // Filter callback returning true: perform rewriting.
  ScaledDotRewriter rewriter_apply(
      [](const HloInstruction* instr) { return true; });
  ASSERT_OK_AND_ASSIGN(auto module_apply,
                       ParseAndReturnUnverifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(bool changed_apply,
                       rewriter_apply.Run(module_apply.get()));
  EXPECT_TRUE(changed_apply);
}

TEST_F(ScaledDotRewriterElementSizeTest, SelectiveFilterCallbackBehavior) {
  const std::string hlo_string = R"(
    HloModule module
    ENTRY main {
      lhs1 = f8e4m3fn[1024,512] parameter(0)
      rhs1 = f8e4m3fn[64,512] parameter(1)
      lhs_scale1 = f8e8m0fnu[1024,16] parameter(2)
      rhs_scale1 = f8e8m0fnu[64,16] parameter(3)
      dot1 = f32[1024,64] scaled-dot(lhs1, rhs1, lhs_scale1, rhs_scale1),
        lhs_contracting_dims={1},
        rhs_contracting_dims={1}

      lhs2 = f8e4m3fn[1024,512] parameter(4)
      rhs2 = f8e4m3fn[64,512] parameter(5)
      lhs_scale2 = bf16[1024,16] parameter(6)
      rhs_scale2 = bf16[64,16] parameter(7)
      dot2 = f32[1024,64] scaled-dot(lhs2, rhs2, lhs_scale2, rhs_scale2),
        lhs_contracting_dims={1},
        rhs_contracting_dims={1}

      ROOT add = f32[1024,64] add(dot1, dot2)
    }
  )";

  // Filter that returns false for f8e8m0fnu scale (do not rewrite)
  // and true for bf16 scale (rewrite).
  ScaledDotRewriter rewriter([](const HloInstruction* instr) {
    return instr->operand(2)->shape().element_type() !=
           PrimitiveType::F8E8M0FNU;
  });

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
  EXPECT_TRUE(changed);

  int scaled_dot_count = 0;
  int dot_count = 0;
  for (const HloInstruction* instruction :
       module->entry_computation()->instructions()) {
    if (instruction->opcode() == HloOpcode::kScaledDot) {
      ++scaled_dot_count;
    } else if (instruction->opcode() == HloOpcode::kDot) {
      ++dot_count;
    }
  }

  EXPECT_EQ(scaled_dot_count, 1);
  EXPECT_EQ(dot_count, 1);
}

INSTANTIATE_TEST_SUITE_P(
    ScaledDotRewriterTests, ScaledDotRewriterTestFixture,
    ::testing::ValuesIn<ScaledDotRewriterTestCase>({
        {PrimitiveType::F8E4M3FN, PrimitiveType::F8E8M0FNU},
        {PrimitiveType::F8E5M2, PrimitiveType::F8E8M0FNU},
        {PrimitiveType::BF16, PrimitiveType::BF16},
        {PrimitiveType::S4, PrimitiveType::BF16},
    }),
    [](const ::testing::TestParamInfo<ScaledDotRewriterTest::ParamType>& info) {
      return absl::StrCat(PrimitiveType_Name(info.param.operand_type), "_",
                          PrimitiveType_Name(info.param.scale_type));
    });

}  // namespace
}  // namespace gpu
}  // namespace xla
