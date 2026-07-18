/* Copyright 2021 The OpenXLA Authors.

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

#include "xla/hlo/transforms/simplifiers/dot_merger.h"

#include <cstdint>
#include <functional>
#include <limits>
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/error_spec.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/hlo/transforms/simplifiers/algebraic_simplifier.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tests/literal_test_util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

namespace m = ::xla::match;

class DotMergerTest : public HloHardwareIndependentTestBase {
 public:
  DotMergerTest()
      : HloHardwareIndependentTestBase(
            /*verifier_layout_sensitive=*/false,
            /*allow_mixed_precision_in_hlo_verifier=*/false) {}
};

TEST_F(DotMergerTest, MergeRHS) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    lhs  = f32[200,100] parameter(0)
    rhs0 = f32[100, 10] parameter(1)
    rhs1 = f32[100, 50] parameter(2)
    dot0 = f32[200, 10] dot(lhs, rhs0), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    dot1 = f32[200, 50] dot(lhs, rhs1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    ROOT tuple = (f32[200,10], f32[200,50], f32[200,100]) tuple(dot0, dot1, lhs)
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  const HloInstruction* dot0 = nullptr;
  const HloInstruction* dot1 = nullptr;
  const HloInstruction* lhs = nullptr;
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(m::Slice(m::Op(&dot0)),
                                  m::Slice(m::Op(&dot1)), m::Op(&lhs))));
  EXPECT_EQ(dot0, dot1);
  EXPECT_THAT(dot0,
              GmockMatch(m::Dot(m::Parameter(0),
                                m::Concatenate().WithBinaryOperandsAnyOrder(
                                    m::Transpose(m::Parameter(1)),
                                    m::Transpose(m::Parameter(2))))));
  ASSERT_NE(lhs, nullptr);
  // We want a deterministic first user.
  EXPECT_EQ(lhs->users()[0], dot0);
}

TEST_F(DotMergerTest, MergeRHSSortedByPowerOfTwo) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    lhs  = f32[200,100] parameter(0)
    rhs0 = f32[100, 10] parameter(1)
    rhs1 = f32[100, 50] parameter(2)
    rhs2 = f32[100, 64] parameter(3)
    rhs3 = f32[100, 24] parameter(4)
    dot0 = f32[200, 10] dot(lhs, rhs0), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    dot1 = f32[200, 50] dot(lhs, rhs1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    dot2 = f32[200, 64] dot(lhs, rhs2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    dot3 = f32[200, 24] dot(lhs, rhs3), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    ROOT tuple = (f32[200,10], f32[200,50], f32[200,64], f32[200,24]) tuple(dot0, dot1, dot2, dot3)
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  const HloInstruction* dot0 = nullptr;
  const HloInstruction* dot1 = nullptr;
  const HloInstruction* dot2 = nullptr;
  const HloInstruction* dot3 = nullptr;
  SCOPED_TRACE(module->ToString());
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(m::Slice(m::Op(&dot0)), m::Slice(m::Op(&dot1)),
                          m::Slice(m::Op(&dot2)), m::Slice(m::Op(&dot3)))));
  EXPECT_EQ(dot0, dot1);
  EXPECT_EQ(dot0, dot2);
  EXPECT_EQ(dot0, dot3);
  // Expected order: RHS2 (64, TZ 6), RHS3 (24, TZ 3), RHS1 (50, TZ 1), RHS0
  // (10, TZ 1)
  EXPECT_THAT(
      dot0, GmockMatch(m::Dot(m::Parameter(0),
                              m::Concatenate(m::Transpose(m::Parameter(3)),
                                             m::Transpose(m::Parameter(4)),
                                             m::Transpose(m::Parameter(2)),
                                             m::Transpose(m::Parameter(1))))));
}

TEST_F(DotMergerTest, MergeRHSWithLHS) {
  absl::string_view module_string = R"(
ENTRY main {
  common = bf16[32,4] parameter(0)
  lhs0 = bf16[2,32] parameter(1)
  dot0 = bf16[2,4] dot(lhs0, common), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  rhs1 = bf16[32,8] parameter(2)
  common_t = bf16[4,32] transpose(common), dimensions={1,0}
  dot1 = bf16[4,8] dot(common_t, rhs1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT tuple = (bf16[2,4], bf16[4,8]) tuple(dot0, dot1)
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  const HloInstruction* dot0 = nullptr;
  const HloInstruction* dot1 = nullptr;
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(m::Slice(m::Dot(&dot0, m::Concatenate(), m::Op())),
                          m::Transpose(m::Slice(m::Op(&dot1))))))
      << module->ToString();
  EXPECT_EQ(dot0, dot1);
}

TEST_F(DotMergerTest, MergeRHSWithLayouts) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    lhs  = f32[200,100] parameter(0)
    rhs0 = f32[100, 10]{0,1} parameter(1)
    rhs1 = f32[100, 50]{0,1} parameter(2)
    dot0 = f32[200, 10] dot(lhs, rhs0), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    dot1 = f32[200, 50] dot(lhs, rhs1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    ROOT tuple = (f32[200,10], f32[200,50]) tuple(dot0, dot1)
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  const HloInstruction* dot0 = nullptr;
  const HloInstruction* dot1 = nullptr;
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(m::Slice(m::Op(&dot0)), m::Slice(m::Op(&dot1)))));
  EXPECT_EQ(dot0, dot1);
  Shape expected_concat_shape =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {60, 100}, {1, 0});
  EXPECT_THAT(dot0,
              GmockMatch(m::Dot(
                  m::Parameter(0),
                  m::Concatenate()
                      .WithBinaryOperandsAnyOrder(m::Transpose(m::Parameter(1)),
                                                  m::Transpose(m::Parameter(2)))
                      .WithShapeEqualTo(&expected_concat_shape))));
}

TEST_F(DotMergerTest, MergeDifferentLayoutRHS) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    lhs  = f32[200,100] parameter(0)
    rhs0 = f32[100, 10]{0,1} parameter(1)
    rhs1 = f32[100, 50]{1,0} parameter(2)
    dot0 = f32[200, 10] dot(lhs, rhs0), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    dot1 = f32[200, 50] dot(lhs, rhs1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    ROOT tuple = (f32[200,10], f32[200,50]) tuple(dot0, dot1)
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  const HloInstruction* dot0 = nullptr;
  const HloInstruction* dot1 = nullptr;
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(m::Slice(m::Op(&dot0)), m::Slice(m::Op(&dot1)))));
  EXPECT_EQ(dot0, dot1);
}

TEST_F(DotMergerTest, MergeLHS) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    lhs0 = f32[100,200] parameter(0)
    lhs1 = f32[300,200] parameter(1)
    rhs  = f32[200, 50] parameter(2)
    dot0 = f32[100, 50] dot(lhs0, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    dot1 = f32[300, 50] dot(lhs1, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    ROOT tuple = (f32[100,50], f32[300,50]) tuple(dot0, dot1)
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(m::Slice(), m::Slice())));
}

TEST_F(DotMergerTest, MergeLHSRemovesDeadTransposes) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    lhs0 = f32[20,100] parameter(0)
    lhs0_t = f32[100,20] transpose(lhs0), dimensions={1,0}
    lhs1 = f32[20,300] parameter(1)
    lhs1_t = f32[300,20] transpose(lhs1), dimensions={1,0}
    rhs  = f32[20, 50] parameter(2)
    dot0 = f32[100, 50] dot(lhs0_t, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    dot1 = f32[300, 50] dot(lhs1_t, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    ROOT tuple = (f32[100,50], f32[300,50]) tuple(dot0, dot1)
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));

  // Find the original transposes and record their unique IDs.
  int64_t lhs0_t_id = -1;
  int64_t lhs1_t_id = -1;
  for (const HloInstruction* inst :
       module->entry_computation()->instructions()) {
    if (inst->name() == "lhs0_t") {
      lhs0_t_id = inst->unique_id();
    } else if (inst->name() == "lhs1_t") {
      lhs1_t_id = inst->unique_id();
    }
  }
  ASSERT_NE(lhs0_t_id, -1);
  ASSERT_NE(lhs1_t_id, -1);

  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  // Verify that the merged dot exists.
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(m::Slice(), m::Slice())));

  // Verify that the original transposes (lhs0_t, lhs1_t) are removed.
  for (const HloInstruction* inst :
       module->entry_computation()->instructions()) {
    EXPECT_NE(inst->unique_id(), lhs0_t_id);
    EXPECT_NE(inst->unique_id(), lhs1_t_id);
  }
}

TEST_F(DotMergerTest, MergeLHSWithRHS) {
  absl::string_view module_string = R"(
ENTRY main {
  common = bf16[4,32] parameter(0)
  rhs0 = bf16[32,8] parameter(2)
  dot0 = bf16[4,8] dot(common, rhs0), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  lhs1 = bf16[2,32] parameter(1)
  common_t = bf16[32,4] transpose(common), dimensions={1,0}
  dot1 = bf16[2,4] dot(lhs1, common_t), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT tuple = (bf16[4,8], bf16[2,4]) tuple(dot0, dot1)
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  const HloInstruction* dot0 = nullptr;
  const HloInstruction* dot1 = nullptr;
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(
          m::Transpose(m::Slice(m::Dot(&dot0, m::Concatenate(), m::Op()))),
          m::Slice(m::Op(&dot1)))))
      << module->ToString();
  EXPECT_EQ(dot0, dot1);
}

TEST_F(DotMergerTest, MergeLHSDotsWithNonDefaultLayout) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    lhs0 = f32[100,200] parameter(0)
    lhs1 = f32[300,200] parameter(1)
    rhs  = f32[200, 50] parameter(2)
    dot0 = f32[100, 50]{0,1} dot(lhs0, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    dot1 = f32[300, 50]{0,1} dot(lhs1, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    ROOT tuple = (f32[100,50]{0,1}, f32[300,50]{0,1}) tuple(dot0, dot1)
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  Shape expected_dot_shape =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {400, 50}, {0, 1});
  const HloInstruction* dot0 = nullptr;
  const HloInstruction* dot1 = nullptr;
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(m::Slice(m::Dot(&dot0, m::Op(), m::Op())),
                                  m::Slice(m::Op(&dot1)))));
  EXPECT_EQ(dot0, dot1);
}

TEST_F(DotMergerTest, MergeDifferentLayoutLHS) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    lhs0 = f32[100,200]{1,0} parameter(0)
    lhs1 = f32[300,200]{0,1} parameter(1)
    rhs  = f32[200, 50] parameter(2)
    dot0 = f32[100, 50] dot(lhs0, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    dot1 = f32[300, 50] dot(lhs1, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    ROOT tuple = (f32[100,50], f32[300,50]) tuple(dot0, dot1)
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  const HloInstruction* dot0 = nullptr;
  const HloInstruction* dot1 = nullptr;
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(m::Slice(m::Dot(&dot0, m::Op(), m::Op())),
                                  m::Slice(m::Op(&dot1)))));
  EXPECT_EQ(dot0, dot1);
}

TEST_F(DotMergerTest, MergeDifferentDotLayout) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    lhs0 = f32[100,200] parameter(0)
    lhs1 = f32[300,200] parameter(1)
    rhs  = f32[200, 50] parameter(2)
    dot0 = f32[100, 50]{0,1} dot(lhs0, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    dot1 = f32[300, 50]{1,0} dot(lhs1, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    ROOT tuple = (f32[100,50]{0,1}, f32[300,50]{1,0}) tuple(dot0, dot1)
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  const HloInstruction* dot0 = nullptr;
  const HloInstruction* dot1 = nullptr;
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(m::Slice(m::Dot(&dot0, m::Op(), m::Op())),
                                  m::Slice(m::Op(&dot1)))));
  EXPECT_EQ(dot0, dot1);
}

TEST_F(DotMergerTest, MergeThree) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    lhs0 = f32[100,200] parameter(0)
    lhs1 = f32[300,200] parameter(1)
    lhs2 = f32[500,200] parameter(2)
    rhs  = f32[200, 50] parameter(3)
    dot0 = f32[100, 50] dot(lhs0, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    dot1 = f32[300, 50] dot(lhs1, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    dot2 = f32[500, 50] dot(lhs2, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    ROOT tuple = (f32[100,50], f32[300,50], f32[500,50]) tuple(dot0, dot1, dot2)
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  // Clean up some redundant slice-of-slices so it's easier to pattern-match.
  AlgebraicSimplifier algsimp{AlgebraicSimplifierOptions{}};
  ASSERT_OK(this->RunHloPass(&algsimp, module.get()).status());

  const HloInstruction* s0 = nullptr;
  const HloInstruction* s1 = nullptr;
  const HloInstruction* s2 = nullptr;
  SCOPED_TRACE(module->ToString());
  ASSERT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(
          m::Slice(m::Dot(
              &s0,
              m::Concatenate(m::Parameter(2), m::Parameter(1), m::Parameter(0)),
              m::Parameter(3))),
          m::Slice(m::Op(&s1)), m::Slice(m::Op(&s2)))));

  // There should be just one dot op.
  EXPECT_EQ(s0, s1);
  EXPECT_EQ(s1, s2);
}

TEST_F(DotMergerTest, NoMergeThreeDueToCycle) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    lhs0 = f32[100,200] parameter(0)
    lhs1 = f32[300,200] parameter(1)
    rhs  = f32[200, 50] parameter(2)
    dot0 = f32[100, 50] dot(lhs0, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    dot1 = f32[300, 50] dot(lhs1, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    zero = f32[] constant(0)
    lhs2 = f32[500,200] pad(dot0, zero), padding=400_0x150_0
    dot2 = f32[500, 50] dot(lhs2, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    ROOT tuple = (f32[100,50], f32[300,50], f32[500,50]) tuple(dot0, dot1, dot2)
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  AlgebraicSimplifier algsimp{AlgebraicSimplifierOptions{}};
  ASSERT_OK(this->RunHloPass(&algsimp, module.get()).status());

  const HloInstruction* s0 = nullptr;
  const HloInstruction* s1 = nullptr;
  const HloInstruction* s2 = nullptr;
  SCOPED_TRACE(module->ToString());
  ASSERT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(
                  m::Dot(&s0, m::Parameter(0), m::Parameter(2)),
                  m::Slice(m::Dot(&s1, m::Concatenate(m::Op(), m::Parameter(1)),
                                  m::Parameter(2))),
                  m::Slice(m::Op(&s2)))));

  // There should be two dot ops.
  EXPECT_EQ(s1, s2);
  EXPECT_NE(s0, s1);
}

TEST_F(DotMergerTest, NoMergeDataDependency) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    lhs0 = f32[100,200] parameter(0)
    rhs  = f32[200, 50] parameter(1)
    dot0 = f32[100, 50] dot(lhs0, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    zero = f32[] constant(0)
    lhs1 = f32[300,200] pad(dot0, zero), padding=200_0x150_0
    dot1 = f32[300, 50] dot(lhs1, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    ROOT tuple = (f32[100,50], f32[300,50]) tuple(dot0, dot1)
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(DotMergerTest, MergeSameContractingDimsOnBothSides) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    lhs0 = f32[100,200] parameter(0)
    lhs1 = f32[300,200] parameter(1)
    rhs  = f32[50, 200] parameter(2)
    dot0 = f32[100, 50] dot(lhs0, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={1}
    dot1 = f32[300, 50] dot(lhs1, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={1}
    ROOT tuple = (f32[100,50], f32[300,50]) tuple(dot0, dot1)
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(m::Slice(), m::Slice())));
}

TEST_F(DotMergerTest, MergeWithBatchDims) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    lhs0 = f32[2,4,100,200] parameter(0)
    lhs1 = f32[2,4,300,200] parameter(1)
    rhs  = f32[2,4,200, 50] parameter(2)
    dot0 = f32[2,4,100, 50] dot(lhs0, rhs), lhs_batch_dims={0,1}, rhs_batch_dims={0,1},
                                            lhs_contracting_dims={3}, rhs_contracting_dims={2}
    dot1 = f32[2,4,300, 50] dot(lhs1, rhs), lhs_batch_dims={0,1}, rhs_batch_dims={0,1},
                                            lhs_contracting_dims={3}, rhs_contracting_dims={2}
    ROOT tuple = (f32[2,4,100,50], f32[2,4,300,50]) tuple(dot0, dot1)
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(m::Slice(), m::Slice())));
}

TEST_F(DotMergerTest, MergeLHSWithRHSWithBatchDims) {
  absl::string_view module_string = R"(
ENTRY main {
  common = bf16[16,4,32] parameter(0)
  rhs0 = bf16[16,32,8] parameter(2)
  dot0 = bf16[16,4,8] dot(common, rhs0), lhs_contracting_dims={2},
      rhs_contracting_dims={1}, lhs_batch_dims={0}, rhs_batch_dims={0}
  lhs1 = bf16[16,2,32] parameter(1)
  common_t = bf16[16,32,4] transpose(common), dimensions={0,2,1}
  dot1 = bf16[16,2,4] dot(lhs1, common_t), lhs_contracting_dims={2},
      rhs_contracting_dims={1}, lhs_batch_dims={0}, rhs_batch_dims={0}
  ROOT tuple = (bf16[16,4,8], bf16[16,2,4]) tuple(dot0, dot1)
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  const HloInstruction* dot0 = nullptr;
  const HloInstruction* dot1 = nullptr;
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(m::Transpose(m::Slice(m::Dot(&dot0, m::Concatenate(),
                                                       m::Parameter(0)))),
                          m::Slice(m::Op(&dot1)))))
      << module->ToString();
  EXPECT_EQ(dot0, dot1);
}

TEST_F(DotMergerTest, MergeWithBatchDimsAndMultipleContractingDims) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    lhs  = f32[2,3,4,5] parameter(0)
    rhs0 = f32[2,6,3,4,5] parameter(1)
    rhs1 = f32[2,7,3,4,5] parameter(2)
    dot0 = f32[2,4,6] dot(lhs, rhs0), lhs_batch_dims={0,2}, rhs_batch_dims={0,3},
                                      lhs_contracting_dims={1,3}, rhs_contracting_dims={2,4}
    dot1 = f32[2,4,7] dot(lhs, rhs1), lhs_batch_dims={0,2}, rhs_batch_dims={0,3},
                                      lhs_contracting_dims={1,3}, rhs_contracting_dims={2,4}
    ROOT tuple = (f32[2,4,6], f32[2,4,7]) tuple(dot0, dot1)
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  ASSERT_OK(verifier().Run(module.get()).status());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(m::Slice(), m::Slice())));
}

TEST_F(DotMergerTest, MergeWithUnsortedBatchDims) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    lhs0 = f32[2,4,100,200] parameter(0)
    lhs1 = f32[2,4,300,200] parameter(1)
    rhs  = f32[2,4,200, 50] parameter(2)
    dot0 = f32[4,2,100, 50] dot(lhs0, rhs), lhs_batch_dims={1,0}, rhs_batch_dims={1,0},
                                            lhs_contracting_dims={3}, rhs_contracting_dims={2}
    dot1 = f32[4,2,300, 50] dot(lhs1, rhs), lhs_batch_dims={1,0}, rhs_batch_dims={1,0},
                                            lhs_contracting_dims={3}, rhs_contracting_dims={2}
    ROOT tuple = (f32[4,2,100,50], f32[4,2,300,50]) tuple(dot0, dot1)
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(m::Transpose(m::Slice()), m::Transpose(m::Slice()))));
}

TEST_F(DotMergerTest, MergeWithMultipleNonCandidates) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    lhs0 = f32[100,200] parameter(0)
    lhs1 = f32[300,200] parameter(1)
    lhs2 = f32[500,200] parameter(2)
    rhs  = f32[200, 50] parameter(3)
    dot0 = f32[100, 50] dot(lhs0, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    dot1 = f32[300, 50] dot(lhs1, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    dot2 = f32[500, 50] dot(lhs2, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    ROOT tuple = (f32[100,50], f32[300,50], f32[500,50]) tuple(dot0, dot1, dot2)
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  // dot0 size: (100 * 50 + 100 * 200 + 200 * 50) * sizeof(float) = 140000.
  // Only dot0 is a candidate. dot1 and dot2 are not.
  // We should merge dot0 and dot1 (since we allow at most one non-candidate),
  // leaving dot2 unmerged.
  DotMerger pass(/*max_size_to_merge=*/(100 * 50 + 100 * 200 + 200 * 50) *
                 sizeof(float));
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  AlgebraicSimplifier algsimp{AlgebraicSimplifierOptions{}};
  ASSERT_OK(this->RunHloPass(&algsimp, module.get()).status());

  const HloInstruction* s0 = nullptr;
  const HloInstruction* s1 = nullptr;
  const HloInstruction* s2 = nullptr;
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(
          m::Slice(m::Dot(&s0, m::Concatenate(m::Parameter(0), m::Parameter(2)),
                          m::Parameter(3))),
          m::Dot(&s2, m::Parameter(1), m::Parameter(3)),
          m::Slice(m::Op(&s1)))));
  EXPECT_EQ(s0, s1);
  EXPECT_NE(s0, s2);
}

TEST_F(DotMergerTest, MergeWithOneNonCandidate) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    lhs0 = f32[100,200] parameter(0)
    lhs1 = f32[300,200] parameter(1)
    rhs  = f32[200, 50] parameter(2)
    dot0 = f32[100, 50] dot(lhs0, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    dot1 = f32[300, 50] dot(lhs1, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    ROOT tuple = (f32[100,50], f32[300,50]) tuple(dot0, dot1)
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  // Only dot0 is a candidate. dot1 is not.
  // We should merge both since we allow at most one non-candidate.
  DotMerger pass(/*max_size_to_merge=*/(100 * 50 + 100 * 200 + 200 * 50) *
                 sizeof(float));
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  AlgebraicSimplifier algsimp{AlgebraicSimplifierOptions{}};
  ASSERT_OK(this->RunHloPass(&algsimp, module.get()).status());

  const HloInstruction* s0 = nullptr;
  const HloInstruction* s1 = nullptr;
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(
          m::Slice(m::Dot(&s0, m::Concatenate(m::Parameter(0), m::Parameter(1)),
                          m::Parameter(2))),
          m::Slice(m::Op(&s1)))));
  EXPECT_EQ(s0, s1);
}

TEST_F(DotMergerTest, NoMergeWithOnlyNonCandidates) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    lhs0 = f32[300,200] parameter(0)
    lhs1 = f32[300,200] parameter(1)
    rhs  = f32[200, 50] parameter(2)
    dot0 = f32[300, 50] dot(lhs0, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    dot1 = f32[300, 50] dot(lhs1, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    ROOT tuple = (f32[300,50], f32[300,50]) tuple(dot0, dot1)
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  // Both dot0 and dot1 are non-candidates.
  // We should NOT merge them.
  DotMerger pass(/*max_size_to_merge=*/(100 * 50 + 100 * 200 + 200 * 50) *
                 sizeof(float));
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(DotMergerTest, MergeMultipleCandidatesWithOneNonCandidate) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    lhs0 = f32[100,200] parameter(0)
    lhs1 = f32[100,200] parameter(1)
    lhs2 = f32[300,200] parameter(2)
    rhs  = f32[200, 50] parameter(3)
    dot0 = f32[100, 50] dot(lhs0, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    dot1 = f32[100, 50] dot(lhs1, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    dot2 = f32[300, 50] dot(lhs2, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    ROOT tuple = (f32[100,50], f32[100,50], f32[300,50]) tuple(dot0, dot1, dot2)
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  // dot0 and dot1 are candidates. dot2 is not.
  // We should merge all three.
  DotMerger pass(/*max_size_to_merge=*/(100 * 50 + 100 * 200 + 200 * 50) *
                 sizeof(float));
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  AlgebraicSimplifier algsimp{AlgebraicSimplifierOptions{}};
  ASSERT_OK(this->RunHloPass(&algsimp, module.get()).status());

  const HloInstruction* s0 = nullptr;
  const HloInstruction* s1 = nullptr;
  const HloInstruction* s2 = nullptr;
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(
          m::Slice(m::Dot(
              &s0,
              m::Concatenate(m::Parameter(0), m::Parameter(1), m::Parameter(2)),
              m::Parameter(3))),
          m::Slice(m::Op(&s1)), m::Slice(m::Op(&s2)))));
  EXPECT_EQ(s0, s1);
  EXPECT_EQ(s1, s2);
}

TEST_F(DotMergerTest, MergeDifferentLhsBatchDims) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    lhs0 = f32[10,10,10,10] parameter(0)
    lhs1 = f32[10,10,10,10] parameter(1)
    rhs  = f32[10,10,10,10] parameter(2)
    dot0 = f32[10,10,10,10] dot(lhs0, rhs), lhs_batch_dims={0,1}, rhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_contracting_dims={2}
    dot1 = f32[10,10,10,10] dot(lhs1, rhs), lhs_batch_dims={0,2}, rhs_batch_dims={0,1}, lhs_contracting_dims={1}, rhs_contracting_dims={2}
    ROOT tuple = (f32[10,10,10,10], f32[10,10,10,10]) tuple(dot0, dot1)
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  const HloInstruction* dot0 = nullptr;
  const HloInstruction* dot1 = nullptr;
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(m::Slice(m::Dot(&dot0, m::Concatenate(), m::Op())),
                          m::Slice(m::Op(&dot1)))))
      << module->ToString();
  EXPECT_EQ(dot0, dot1);
}

TEST_F(DotMergerTest, NoMergeDifferentRhsBatchDims) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    lhs0 = f32[10,10,10,10] parameter(0)
    lhs1 = f32[10,10,10,10] parameter(1)
    rhs  = f32[10,10,10,10] parameter(2)
    dot0 = f32[10,10,10,10] dot(lhs0, rhs), lhs_batch_dims={0,1}, rhs_batch_dims={0,1}, lhs_contracting_dims={2}, rhs_contracting_dims={2}
    dot1 = f32[10,10,10,10] dot(lhs1, rhs), lhs_batch_dims={0,1}, rhs_batch_dims={0,2}, lhs_contracting_dims={2}, rhs_contracting_dims={1}
    ROOT tuple = (f32[10,10,10,10], f32[10,10,10,10]) tuple(dot0, dot1)
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(DotMergerTest, MergeMultipleContractingDims) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    lhs0 = f32[10,10,10] parameter(0)
    lhs1 = f32[10,10,10] parameter(1)
    rhs  = f32[10,10,10] parameter(2)
    dot0 = f32[10,10] dot(lhs0, rhs), lhs_contracting_dims={0,1}, rhs_contracting_dims={0,1}
    dot1 = f32[10,10] dot(lhs1, rhs), lhs_contracting_dims={0,1}, rhs_contracting_dims={0,1}
    ROOT tuple = (f32[10,10], f32[10,10]) tuple(dot0, dot1)
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  const HloInstruction* s0 = nullptr;
  const HloInstruction* s1 = nullptr;
  SCOPED_TRACE(module->ToString());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(m::Slice(m::Dot(&s0, m::Concatenate(),
                                                  m::Reshape(m::Parameter(2)))),
                                  m::Slice(m::Op(&s1)))));
  EXPECT_EQ(s0, s1);
}

TEST_F(DotMergerTest, MergeMultipleNonContractingDimsInRhsSharedOperand) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    lhs0 = f32[8,9,10] parameter(0)
    lhs1 = f32[8,9,11] parameter(1)
    rhs  = f32[8,9,12,13] parameter(2)
    dot0 = f32[10,12,13] dot(lhs0, rhs), lhs_contracting_dims={0,1}, rhs_contracting_dims={0,1}
    dot1 = f32[11,12,13] dot(lhs1, rhs), lhs_contracting_dims={0,1}, rhs_contracting_dims={0,1}
    ROOT tuple = (f32[10,12,13], f32[11,12,13]) tuple(dot0, dot1)
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  ASSERT_OK(verifier().Run(module.get()).status());

  const HloInstruction* s0 = nullptr;
  const HloInstruction* s1 = nullptr;
  SCOPED_TRACE(module->ToString());
  ASSERT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(m::Slice(m::Dot(&s0, m::Concatenate(),
                                                  m::Reshape(m::Parameter(2)))),
                                  m::Slice(m::Op(&s1)))));
  EXPECT_EQ(s0, s1);
}

TEST_F(DotMergerTest, MergeMultipleOuterDims) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    lhs0 = f32[10,10,10] parameter(0)
    lhs1 = f32[10,10,10] parameter(1)
    rhs  = f32[10,10,10] parameter(2)
    dot0 = f32[10,10,10,10] dot(lhs0, rhs), lhs_contracting_dims={0}, rhs_contracting_dims={0}
    dot1 = f32[10,10,10,10] dot(lhs1, rhs), lhs_contracting_dims={0}, rhs_contracting_dims={0}
    ROOT tuple = (f32[10,10,10,10], f32[10,10,10,10]) tuple(dot0, dot1)
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  const HloInstruction* dot0 = nullptr;
  const HloInstruction* dot1 = nullptr;
  ASSERT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(m::Reshape(m::Slice(m::Dot(&dot0, m::Concatenate(),
                                                     m::Parameter(2)))),
                          m::Reshape(m::Slice(m::Op(&dot1))))))
      << module->ToString();
  EXPECT_EQ(dot0, dot1);
}

TEST_F(DotMergerTest, MergeDifferentLhsContractingDims) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    lhs0 = f32[10,10] parameter(0)
    lhs1 = f32[10,10] parameter(1)
    rhs  = f32[10,10] parameter(2)
    dot0 = f32[10,10] dot(lhs0, rhs), lhs_contracting_dims={0}, rhs_contracting_dims={0}
    dot1 = f32[10,10] dot(lhs1, rhs), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    ROOT tuple = (f32[10,10], f32[10,10]) tuple(dot0, dot1)
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  const HloInstruction* dot0 = nullptr;
  const HloInstruction* dot1 = nullptr;
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(m::Slice(m::Dot(&dot0, m::Concatenate(), m::Op())),
                          m::Slice(m::Op(&dot1)))))
      << module->ToString();
  EXPECT_EQ(dot0, dot1);
}

TEST_F(DotMergerTest, NoMergeDifferentRhsContractingDims) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    lhs0 = f32[10,10] parameter(0)
    lhs1 = f32[10,10] parameter(1)
    rhs  = f32[10,10] parameter(2)
    dot0 = f32[10,10] dot(lhs0, rhs), lhs_contracting_dims={0}, rhs_contracting_dims={0}
    dot1 = f32[10,10] dot(lhs1, rhs), lhs_contracting_dims={0}, rhs_contracting_dims={1}
    ROOT tuple = (f32[10,10], f32[10,10]) tuple(dot0, dot1)
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(DotMergerTest, NoMergeControlPredecessor) {
  // Don't evem merge dot0 and dot1, because dot1 has a control *successor*.
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    lhs0 = f32[10,10] parameter(0)
    lhs1 = f32[10,10] parameter(1)
    rhs  = f32[10,10] parameter(2)
    dot0 = f32[10,10] dot(lhs0, rhs), lhs_contracting_dims={0}, rhs_contracting_dims={0}
    dot1 = f32[10,10] dot(lhs1, rhs), lhs_contracting_dims={0}, rhs_contracting_dims={0}
    dot2 = f32[10,10] dot(lhs1, rhs), lhs_contracting_dims={0}, rhs_contracting_dims={0}, control-predecessors={dot1}
    ROOT tuple = (f32[10,10], f32[10,10], f32[10,10]) tuple(dot0, dot1, dot2)
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(DotMergerTest, NoMergeDifferentLhsTypes) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    lhs0 = f32[10,10] parameter(0)
    lhs1 = f16[10,10] parameter(1)
    rhs  = f32[10,10] parameter(2)
    dot0 = f32[10,10] dot(lhs0, rhs), lhs_contracting_dims={0}, rhs_contracting_dims={0}
    dot1 = f32[10,10] dot(lhs1, rhs), lhs_contracting_dims={0}, rhs_contracting_dims={0}
    ROOT tuple = (f32[10,10], f32[10,10]) tuple(dot0, dot1)
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(DotMergerTest, NoMergeDifferentRhsTypes) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    lhs  = f32[10,10] parameter(0)
    rhs0 = f32[10,10] parameter(1)
    rhs1 = f16[10,10] parameter(2)
    dot0 = f32[10,10] dot(lhs, rhs0), lhs_contracting_dims={0}, rhs_contracting_dims={0}
    dot1 = f32[10,10] dot(lhs, rhs1), lhs_contracting_dims={0}, rhs_contracting_dims={0}
    ROOT tuple = (f32[10,10], f32[10,10]) tuple(dot0, dot1)
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(DotMergerTest, NoMergeRHSWithLHSDifferentTypes) {
  absl::string_view module_string = R"(
ENTRY main {
  common = f32[32,4] parameter(0)
  lhs0 = bf16[2,32] parameter(1)
  dot0 = f32[2,4] dot(lhs0, common), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  rhs1 = f32[32,8] parameter(2)
  common_t = f32[4,32] transpose(common), dimensions={1,0}
  dot1 = f32[4,8] dot(common_t, rhs1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT tuple = (f32[2,4], f32[4,8]) tuple(dot0, dot1)
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(DotMergerTest, NoMergeDifferentReturnTypes) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    lhs0 = f16[10,10] parameter(0)
    lhs1 = f16[10,10] parameter(1)
    rhs  = f16[10,10] parameter(2)
    dot0 = f16[10,10] dot(lhs0, rhs), lhs_contracting_dims={0}, rhs_contracting_dims={0}
    dot1 = f32[10,10] dot(lhs1, rhs), lhs_contracting_dims={0}, rhs_contracting_dims={0}
    ROOT tuple = (f16[10,10], f32[10,10]) tuple(dot0, dot1)
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(DotMergerTest, MergeWithTypeUpgrade) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    lhs0 = f16[10,10] parameter(0)
    lhs1 = f16[10,10] parameter(1)
    rhs  = f16[10,10] parameter(2)
    dot0 = f32[10,10] dot(lhs0, rhs), lhs_contracting_dims={0}, rhs_contracting_dims={0}
    dot1 = f32[10,10] dot(lhs1, rhs), lhs_contracting_dims={0}, rhs_contracting_dims={0}
    ROOT tuple = (f32[10,10], f32[10,10]) tuple(dot0, dot1)
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  SCOPED_TRACE(module->ToString());

  EXPECT_TRUE(changed);
  const HloInstruction* d0 = nullptr;
  const HloInstruction* d1 = nullptr;
  ASSERT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(
                  m::Slice(m::Dot(&d0,
                                  m::Concatenate(m::Transpose(m::Parameter(0)),
                                                 m::Transpose(m::Parameter(1))),
                                  m::Parameter(2))
                               .WithShape(F32, {20, 10})),
                  m::Slice(m::Op(&d1)))));
  EXPECT_EQ(d0, d1);
}

TEST_F(DotMergerTest, MergeMultipleContractingDimsWithMismatchedConcatLayout) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    lhs  = f32[10,100,10] parameter(0)
    rhs0 = f32[10,5,10] parameter(1)
    rhs1 = f32[10,5,10,6] parameter(2)

    dot0 = f32[100,5] dot(lhs, rhs0), lhs_contracting_dims={0,2}, rhs_contracting_dims={0,2}
    dot1 = f32[100,5,6] dot(lhs, rhs1), lhs_contracting_dims={0,2}, rhs_contracting_dims={0,2}

    ROOT tuple = (f32[100,5], f32[100,5,6]) tuple(dot0, dot1)
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  ASSERT_OK(verifier().Run(module.get()).status());
}

TEST_F(DotMergerTest, MergeMultipleBatchDimsWithMismatchedConcatLayout) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    lhs  = f32[2,100,3,10] parameter(0)
    rhs0 = f32[2,3,10,5] parameter(1)
    rhs1 = f32[2,5,3,10,6] parameter(2)

    dot0 = f32[2,3,100,5] dot(lhs, rhs0), lhs_batch_dims={0,2}, rhs_batch_dims={0,1},
                                          lhs_contracting_dims={3}, rhs_contracting_dims={2}
    dot1 = f32[2,3,100,5,6] dot(lhs, rhs1), lhs_batch_dims={0,2}, rhs_batch_dims={0,2},
                                            lhs_contracting_dims={3}, rhs_contracting_dims={3}

    ROOT tuple = (f32[2,3,100,5], f32[2,3,100,5,6]) tuple(dot0, dot1)
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  ASSERT_OK(verifier().Run(module.get()).status());
}

TEST_F(DotMergerTest, NoMergeWithFalseCompatibility) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    lhs0 = f32[2,4,100,200] parameter(0)
    lhs1 = f32[2,4,300,200] parameter(1)
    rhs  = f32[2,4,200, 50] parameter(2)
    dot0 = f32[2,4,100, 50] dot(lhs0, rhs), lhs_batch_dims={0,1}, rhs_batch_dims={0,1},
        lhs_contracting_dims={3}, rhs_contracting_dims={2}, backend_config={"operation_queue_id":"0"}
    dot1 = f32[2,4,300, 50] dot(lhs1, rhs), lhs_batch_dims={0,1}, rhs_batch_dims={0,1},
        lhs_contracting_dims={3}, rhs_contracting_dims={2}, backend_config={"operation_queue_id":"1"}
    ROOT tuple = (f32[2,4,100,50], f32[2,4,300,50]) tuple(dot0, dot1)
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  std::function<int64_t(const HloInstruction* dot)> queue_id =
      [&](const HloInstruction* dot) -> int64_t {
    // The queue_id will typically be taken from the backend_config, but deps on
    // backend-specific protos is avoided for testing.
    return dot->name() == "dot1" ? 1 : 0;
  };
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max(),
                 queue_id);
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(DotMergerTest, NoMergeLHSWithRHSDifferentDimM) {
  absl::string_view module_string = R"(
ENTRY main {
  common = f32[128,10] parameter(0)
  rhs0 = f32[10,20] parameter(1)
  dot0 = f32[/*m dim*/128,20] dot(common, rhs0),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
  lhs1 = f32[20,128] parameter(2)
  dot1 = f32[/*m dim*/20,10] dot(lhs1, common),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT tuple = (f32[128,20], f32[20,10]) tuple(dot0, dot1)
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(DotMergerTest, NoMergeLHSWithRHSSubtlyDifferentDimM) {
  absl::string_view module_string = R"(
ENTRY main {
  c0 = f16[/*Used as m AND k_b*/2,2] constant({ { 1, 2 }, { 3, 4 } })
  c1 = f16[2,2] constant({ { 5, 6 }, { 7, 8 } })
  dot0 = f16[/*m_a*/2,2]{1,0} dot(c0, c1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  dot1 = f16[2,2] dot(c1, c0), lhs_contracting_dims={/*k_b*/1}, rhs_contracting_dims={0}
  ROOT add = f16[2,2] add(dot0, dot1)
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(DotMergerTest, MergeRHSWithIdenticalCustomLayout) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    lhs  = f32[200,100] parameter(0)
    rhs0 = f32[10, 100] parameter(1)
    rhs1 = f32[50, 100] parameter(2)
    // Contracting dim is 1, Non-contracting dim is 0 for both.
    dot0 = f32[200, 10] dot(lhs, rhs0), lhs_contracting_dims={1}, rhs_contracting_dims={1}
    dot1 = f32[200, 50] dot(lhs, rhs1), lhs_contracting_dims={1}, rhs_contracting_dims={1}
    ROOT tuple = (f32[200,10], f32[200,50]) tuple(dot0, dot1)
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  const HloInstruction* dot0 = nullptr;
  const HloInstruction* dot1 = nullptr;
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(m::Slice(m::Op(&dot0)), m::Slice(m::Op(&dot1)))));
  EXPECT_EQ(dot0, dot1);
  // The concatenated operand should be concatenated along dim 0 (the
  // non-contracting dim) and the dot should have rhs_contracting_dims={1}.
  EXPECT_THAT(dot0,
              GmockMatch(m::Dot(m::Parameter(0),
                                m::Concatenate().WithBinaryOperandsAnyOrder(
                                    m::Parameter(1), m::Parameter(2)))));
}

TEST_F(DotMergerTest, NoCrashOnCategoryMixingReshape) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    lhs = f32[4, 2] parameter(0)
    rhs = f32[4] parameter(1)
    rhs_reshape = f32[2, 2] reshape(rhs)
    dot0 = f32[4, 2] dot(lhs, rhs_reshape), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    ROOT tuple = (f32[4, 2]) tuple(dot0)
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(DotMergerTest, MatchedSharedMismatchedConcat) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    lhs = f32[100,200,300] parameter(0)
    rhs0 = f32[100,5,6,300] parameter(1)
    rhs1 = f32[300,50,100] parameter(2)
    dot0 = f32[100,200,5,6] dot(lhs, rhs0),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={3}
    dot1 = f32[100,50,200] dot(rhs1, lhs),
      lhs_batch_dims={2}, rhs_batch_dims={0},
      lhs_contracting_dims={0}, rhs_contracting_dims={2}
    ROOT tuple = (f32[100,200,5,6], f32[100,50,200]) tuple(dot0, dot1)
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  ASSERT_OK(verifier().Run(module.get()).status());
}

TEST_F(DotMergerTest, MatchedSharedMismatchedConcatLayout) {
  absl::string_view module_string = R"(
   HloModule module

   ENTRY main {
     lhs = f32[100,200,300] parameter(0)
     rhs0 = f32[100,30,300]{0,1,2} parameter(1)
     rhs1 = f32[100,50,300]{2,1,0} parameter(2)
     dot0 = f32[100,200,30] dot(lhs, rhs0),
       lhs_batch_dims={0}, rhs_batch_dims={0},
       lhs_contracting_dims={2}, rhs_contracting_dims={2}
     dot1 = f32[100,200,50] dot(lhs, rhs1),
       lhs_batch_dims={0}, rhs_batch_dims={0},
       lhs_contracting_dims={2}, rhs_contracting_dims={2}
     ROOT tuple = (f32[100,200,30], f32[100,200,50]) tuple(dot0, dot1)
   })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  ASSERT_OK(verifier().Run(module.get()).status());
}
TEST_F(DotMergerTest, MergeMultiRound) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    lhs  = f32[200,100] parameter(0)
    rhs0 = f32[100, 10] parameter(1)
    rhs1 = f32[100, 20] parameter(2)
    dot0 = f32[200, 10] dot(lhs, rhs0), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    dot1 = f32[200, 20] dot(lhs, rhs1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    rhs2 = f32[100, 10] slice(dot0), slice={[0:100], [0:10]}
    rhs3 = f32[100, 20] slice(dot1), slice={[0:100], [0:20]}
    dot2 = f32[200, 10] dot(lhs, rhs2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    dot3 = f32[200, 20] dot(lhs, rhs3), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    ROOT tuple = (f32[200,10], f32[200,20], f32[200,10], f32[200,20]) tuple(dot0, dot1, dot2, dot3)
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  ASSERT_OK(verifier().Run(module.get()).status());

  int dot_count = 0;
  for (const HloInstruction* instr :
       module->entry_computation()->instructions()) {
    if (instr->opcode() == HloOpcode::kDot) {
      dot_count++;
    }
  }
  EXPECT_EQ(dot_count, 2);
}

TEST_F(DotMergerTest, MergeWithDependency) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    lhs = f32[10, 10] parameter(0)
    rhs0 = f32[10, 5] parameter(1)
    rhs1 = f32[10, 5] parameter(2)

    dot0 = f32[10, 5] dot(lhs, rhs0), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    dot1 = f32[10, 5] dot(lhs, rhs1), lhs_contracting_dims={1}, rhs_contracting_dims={0}

    rhs2 = f32[5, 2] parameter(3)
    rhs3 = f32[5, 2] parameter(4)
    dot2 = f32[10, 2] dot(dot0, rhs2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    dot3 = f32[10, 2] dot(dot0, rhs3), lhs_contracting_dims={1}, rhs_contracting_dims={0}

    ROOT tuple = (f32[10, 5], f32[10, 5], f32[10, 2], f32[10, 2]) tuple(dot0, dot1, dot2, dot3)
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  ASSERT_OK(verifier().Run(module.get()).status());
}

TEST_F(DotMergerTest, MergeLHSBatchDimsMissingOnAllDots) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    lhs  = f32[2,10,20] parameter(0)
    rhs0 = f32[20,30] parameter(1)
    rhs1 = f32[20,40] parameter(2)
    // No batch dims on either dot, but lhs is 3D (NC={0,1}, C={2})
    dot0 = f32[2,10,30] dot(lhs, rhs0), lhs_contracting_dims={2}, rhs_contracting_dims={0}
    dot1 = f32[2,10,40] dot(lhs, rhs1), lhs_contracting_dims={2}, rhs_contracting_dims={0}
    ROOT tuple = (f32[2,10,30], f32[2,10,40]) tuple(dot0, dot1)
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  const HloInstruction* dot0 = nullptr;
  const HloInstruction* dot1 = nullptr;
  ASSERT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(m::Slice(m::Op(&dot0)), m::Slice(m::Op(&dot1)))));
  EXPECT_EQ(dot0, dot1);
  EXPECT_TRUE(dot0->dot_dimension_numbers().lhs_batch_dimensions().empty());
  EXPECT_TRUE(dot0->dot_dimension_numbers().rhs_batch_dimensions().empty());
}

TEST_F(DotMergerTest, MergeNoContractingDimsPreserveLayout) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    lhs  = f32[100] parameter(0)
    rhs0 = f32[10] parameter(1)
    rhs1 = f32[50] parameter(2)
    dot0 = f32[100, 10] dot(lhs, rhs0), lhs_contracting_dims={}, rhs_contracting_dims={}
    dot1 = f32[100, 50] dot(lhs, rhs1), lhs_contracting_dims={}, rhs_contracting_dims={}
    ROOT tuple = (f32[100,10], f32[100,50]) tuple(dot0, dot1)
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  const HloInstruction* dot0 = nullptr;
  const HloInstruction* dot1 = nullptr;
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(m::Slice(m::Op(&dot0)), m::Slice(m::Op(&dot1)))));
  EXPECT_EQ(dot0, dot1);
}

TEST_F(DotMergerTest, MergeNoContractingDimsNormalizeToBNC) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    lhs  = f32[10,20] parameter(0)
    rhs0 = f32[30] parameter(1)
    rhs1 = f32[50] parameter(2)
    dot0 = f32[10, 20, 30] dot(lhs, rhs0), lhs_contracting_dims={}, rhs_contracting_dims={}
    dot1 = f32[10, 20, 50] dot(lhs, rhs1), lhs_contracting_dims={}, rhs_contracting_dims={}
    ROOT tuple = (f32[10,20,30], f32[10,20,50]) tuple(dot0, dot1)
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  SCOPED_TRACE(module->ToString());
  EXPECT_TRUE(changed);
  const HloInstruction* dot0 = nullptr;
  const HloInstruction* dot1 = nullptr;
  ASSERT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(m::Slice(m::Op(&dot0)), m::Slice(m::Op(&dot1)))));
  EXPECT_EQ(dot0, dot1);
  EXPECT_THAT(dot0, GmockMatch(m::Dot(m::Parameter(0), m::Concatenate())
                                   .WithContractingDims({}, {})));
}

TEST_F(DotMergerTest, MergeNoNonContractingDimsAtSharedSide) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    lhs  = f32[2, 100] parameter(0)
    rhs0 = f32[2, 100, 10] parameter(1)
    rhs1 = f32[2, 100, 50] parameter(2)
    dot0 = f32[2, 10] dot(lhs, rhs0),
                      lhs_batch_dims={0}, rhs_batch_dims={0},
                      lhs_contracting_dims={1}, rhs_contracting_dims={1}
    dot1 = f32[2, 50] dot(lhs, rhs1),
                      lhs_batch_dims={0}, rhs_batch_dims={0},
                      lhs_contracting_dims={1}, rhs_contracting_dims={1}
    ROOT tuple = (f32[2,10], f32[2,50]) tuple(dot0, dot1)
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  const HloInstruction* dot0 = nullptr;
  const HloInstruction* dot1 = nullptr;
  ASSERT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(m::Slice(m::Op(&dot0)), m::Slice(m::Op(&dot1)))));
  EXPECT_EQ(dot0, dot1);
}

TEST_F(DotMergerTest, MergeMismatchedContractingDimsOrder) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    lhs = f32[100,200] parameter(0)
    rhs0 = f32[100,200,5] parameter(1)
    rhs1 = f32[200,100,6] parameter(2)
    dot0 = f32[5] dot(lhs, rhs0), lhs_contracting_dims={0,1}, rhs_contracting_dims={0,1}
    dot1 = f32[6] dot(lhs, rhs1), lhs_contracting_dims={0,1}, rhs_contracting_dims={1,0}
    ROOT tuple = (f32[5], f32[6]) tuple(dot0, dot1)
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  SCOPED_TRACE(module->ToString());
  EXPECT_TRUE(changed);

  const HloInstruction* dot0 = nullptr;
  const HloInstruction* dot1 = nullptr;
  ASSERT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(m::Slice(m::Op(&dot0)), m::Slice(m::Op(&dot1)))));
  EXPECT_EQ(dot0, dot1);

  EXPECT_THAT(dot0, GmockMatch(m::Dot(m::Op(), m::Concatenate())));
}

TEST_F(DotMergerTest, MergeMismatchedBatchDimsOrder) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    lhs = f32[3,4,100] parameter(0)
    rhs0 = f32[3,4,100,5] parameter(1)
    rhs1 = f32[4,3,100,6] parameter(2)
    dot0 = f32[3,4,5] dot(lhs, rhs0), lhs_batch_dims={0,1}, rhs_batch_dims={0,1},
                                      lhs_contracting_dims={2}, rhs_contracting_dims={2}
    dot1 = f32[3,4,6] dot(lhs, rhs1), lhs_batch_dims={0,1}, rhs_batch_dims={1,0},
                                      lhs_contracting_dims={2}, rhs_contracting_dims={2}
    ROOT tuple = (f32[3,4,5], f32[3,4,6]) tuple(dot0, dot1)
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  SCOPED_TRACE(module->ToString());
  EXPECT_TRUE(changed);

  const HloInstruction* dot0 = nullptr;
  const HloInstruction* dot1 = nullptr;
  ASSERT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(m::Slice(m::Op(&dot0)), m::Slice(m::Op(&dot1)))));
  EXPECT_EQ(dot0, dot1);

  EXPECT_THAT(dot0, GmockMatch(m::Dot(m::Op(), m::Concatenate())));
}

TEST_F(DotMergerTest, MergeOneConcatOperandMissingNonContractingDim) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    lhs  = f32[2, 100, 10] parameter(0)
    rhs0 = f32[2, 100, 20] parameter(1)
    rhs1 = f32[2, 100] parameter(2)
    dot0 = f32[2, 10, 20] dot(lhs, rhs0),
                          lhs_batch_dims={0}, rhs_batch_dims={0},
                          lhs_contracting_dims={1}, rhs_contracting_dims={1}
    dot1 = f32[2, 10] dot(lhs, rhs1),
                      lhs_batch_dims={0}, rhs_batch_dims={0},
                      lhs_contracting_dims={1}, rhs_contracting_dims={1}
    ROOT tuple = (f32[2, 10, 20], f32[2, 10]) tuple(dot0, dot1)
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  const HloInstruction* dot0 = nullptr;
  const HloInstruction* dot1 = nullptr;
  ASSERT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(m::Slice(m::Op(&dot0)),
                                  m::Reshape(m::Slice(m::Op(&dot1))))));
  EXPECT_EQ(dot0, dot1);

  EXPECT_THAT(dot0,
              GmockMatch(m::Dot(m::Parameter(0),
                                m::Concatenate(m::Transpose(m::Parameter(1)),
                                               m::Reshape(m::Parameter(2))))));
}

TEST_F(DotMergerTest, MergeWithConsumerNormalization) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    lhs  = f32[200,100] parameter(0)
    rhs0 = f32[100, 10] parameter(1)
    rhs1 = f32[100, 50] parameter(2)
    dot0 = f32[200, 10] dot(lhs, rhs0), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    dot1 = f32[200, 50] dot(lhs, rhs1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    // Redundant transposes after dot0
    dot0_t = f32[10, 200] transpose(dot0), dimensions={1,0}
    dot0_t_inv = f32[200, 10] transpose(dot0_t), dimensions={1,0}
    ROOT tuple = (f32[200,10], f32[200,50]) tuple(dot0_t_inv, dot1)
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  SCOPED_TRACE(module->ToString());

  const HloInstruction* dot0 = nullptr;
  const HloInstruction* dot1 = nullptr;
  // Expect that the redundant double transposes are completely folded,
  // and the ROOT tuple uses the slice directly!
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(m::Slice(m::Op(&dot0)), m::Slice(m::Op(&dot1)))));
  EXPECT_EQ(dot0, dot1);
}

TEST_F(DotMergerTest, MergeWithRedundantReshapes) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    lhs  = f32[200,100] parameter(0)
    rhs0 = f32[100, 10] parameter(1)
    rhs1 = f32[100, 50] parameter(2)
    dot0 = f32[200, 10] dot(lhs, rhs0), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    dot1 = f32[200, 50] dot(lhs, rhs1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    // Redundant reshapes after dot0
    dot0_r1 = f32[20, 10, 10] reshape(dot0)
    dot0_r2 = f32[200, 10] reshape(dot0_r1)
    ROOT tuple = (f32[200,10], f32[200,50]) tuple(dot0_r2, dot1)
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  SCOPED_TRACE(module->ToString());

  const HloInstruction* dot0 = nullptr;
  const HloInstruction* dot1 = nullptr;
  // Expect that the redundant reshapes are folded away, and ROOT tuple uses the
  // slice directly!
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(m::Slice(m::Op(&dot0)), m::Slice(m::Op(&dot1)))));
  EXPECT_EQ(dot0, dot1);
}

TEST_F(DotMergerTest, MergeWithMultipleConsumerChains) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    lhs  = f32[200,100] parameter(0)
    rhs0 = f32[100, 10] parameter(1)
    rhs1 = f32[100, 50] parameter(2)
    dot0 = f32[200, 10] dot(lhs, rhs0), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    dot1 = f32[200, 50] dot(lhs, rhs1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    // Redundant transposes after dot0 (Chain 1)
    dot0_t = f32[10, 200] transpose(dot0), dimensions={1,0}
    dot0_t_inv = f32[200, 10] transpose(dot0_t), dimensions={1,0}
    // Redundant reshapes after dot0 (Chain 2)
    dot0_r1 = f32[20, 10, 10] reshape(dot0)
    dot0_r2 = f32[200, 10] reshape(dot0_r1)
    ROOT tuple = (f32[200,10], f32[200,10], f32[200,50]) tuple(dot0_t_inv, dot0_r2, dot1)
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  SCOPED_TRACE(module->ToString());

  const HloInstruction* dot0 = nullptr;
  const HloInstruction* dot1 = nullptr;
  // Expect both redundant consumer chains are completely simplified to the
  // slice directly.
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(m::Slice(m::Op(&dot0)), m::Slice(m::Op()),
                                  m::Slice(m::Op(&dot1)))));
  EXPECT_EQ(dot0, dot1);
}

TEST_F(DotMergerTest, MergeComplexShapesAndEvaluate) {
  absl::string_view module_string = R"(
  HloModule t

  ENTRY main {
    // Shared parameter: [B0, B1, C0, C1, N0, N1]
    shared_param = f32[4,5,10,20,6,7] parameter(0)

    // LHS of dot_0: parameter directly, different category locations [N, B0, B1, C0, C1]
    param_lhs_0 = f32[512,4,5,20,10] parameter(1)

    // Dot 0 path (RHS shared): transpose contracting dims to [B, C1, C0, N]
    shared_transpose_0 = f32[4,5,20,10,6,7] transpose(shared_param), dimensions={0,1,3,2,4,5}

    shared_reshape_0 = f32[4,5,20,10,3,14] reshape(shared_transpose_0)

    dot_0 = f32[4,5,512,3,14] dot(param_lhs_0, shared_reshape_0),
      lhs_batch_dims={1,2}, rhs_batch_dims={0,1},
      lhs_contracting_dims={3,4}, rhs_contracting_dims={2,3}

    // RHS of dot_1: parameter directly, multiple non-consecutive NC dims [C0, N0, B1, B0, N1, C1]
    param_rhs_1 = f32[10,2,5,4,3,20] parameter(2)

    // Shared parameter: [B0, B1, C0, C1, N0, N1]
    shared_transpose_1 = f32[4,5,10,20,7,6] transpose(shared_param), dimensions={0,1,2,3,5,4}

    // Dot 1 path (LHS shared): direct use of shared_param (no transpose, no reshape)
    dot_1 = f32[5,4,7,6,2,3] dot(shared_transpose_1, param_rhs_1),
      lhs_batch_dims={1,0}, rhs_batch_dims={2,3},
      lhs_contracting_dims={2,3}, rhs_contracting_dims={0,5}

    // Consumer chain of dot_1: splits B=4 to [2,2], and transposes to [B2, B1, N_non_shared, B0, N_shared]
    dot_1_reshape = f32[5,2,2,14,18] reshape(dot_1)
    dot_1_transpose = f32[5,2,18,2,14] transpose(dot_1_reshape), dimensions={0,2,4,1,3}

    ROOT res = (f32[4,5,512,3,14], f32[5,2,18,2,14]) tuple(dot_0, dot_1_transpose)
  }
  )";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));

  // Generate input literals for evaluation.
  ASSERT_OK_AND_ASSIGN(
      Literal shared_param_literal,
      MakeFakeLiteral(ShapeUtil::MakeShape(F32, {4, 5, 10, 20, 6, 7})));
  ASSERT_OK_AND_ASSIGN(
      Literal param_lhs_0_literal,
      MakeFakeLiteral(ShapeUtil::MakeShape(F32, {512, 4, 5, 20, 10})));
  ASSERT_OK_AND_ASSIGN(
      Literal param_rhs_1_literal,
      MakeFakeLiteral(ShapeUtil::MakeShape(F32, {10, 2, 5, 4, 3, 20})));

  // Evaluate the module before running the pass to establish expected results.
  HloEvaluator evaluator;
  ASSERT_OK_AND_ASSIGN(
      Literal expected_result,
      evaluator.Evaluate(*module, {&shared_param_literal, &param_lhs_0_literal,
                                   &param_rhs_1_literal}));

  // Run the DotMerger pass.
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  // Verify HLO verification passes on the merged structure.
  ASSERT_OK(verifier().Run(module.get()).status());

  // Evaluate the optimized module after running the pass.
  HloEvaluator evaluator_after;
  ASSERT_OK_AND_ASSIGN(
      Literal actual_result,
      evaluator_after.Evaluate(
          *module,
          {&shared_param_literal, &param_lhs_0_literal, &param_rhs_1_literal}));

  // Assert that both pre-pass and post-pass execution results match exactly.
  EXPECT_TRUE(
      LiteralTestUtil::Near(expected_result, actual_result, ErrorSpec{1e-4}));
}

TEST_F(DotMergerTest, MergeComplexShapesAndEvaluateSwapConsumer) {
  absl::string_view module_string = R"(
  HloModule t

  ENTRY main {
    // Shared parameter: [B0, B1, C0, C1, N]
    shared_param = f32[4,3,2,2,2] parameter(0)

    // LHS of dot_0: parameter directly, different category locations [N, B0, B1, C0, C1]
    param_lhs_0 = f32[5,4,3,2,2] parameter(1)

    // Dot 0 path (RHS shared): transpose contracting dims to [B, C1, C0, N]
    shared_transpose = f32[4,3,2,2,2] transpose(shared_param), dimensions={0,1,3,2,4}

    dot_0 = f32[4,3,5,2] dot(param_lhs_0, shared_transpose),
      lhs_batch_dims={1,2}, rhs_batch_dims={0,1},
      lhs_contracting_dims={3,4}, rhs_contracting_dims={2,3}

    // Consumer chain of dot_0: splits B=4 to [2,2], and transposes to [B1, B0_0, N_lhs, N_rhs, B0_1]
    dot_0_reshape = f32[2,2,3,5,2] reshape(dot_0)
    dot_0_transpose = f32[3,2,5,2,2] transpose(dot_0_reshape), dimensions={2,0,3,4,1}

    // RHS of dot_1: parameter directly, multiple non-consecutive NC dims [C0, N0, B1, B0, N1, C1]
    param_rhs_1 = f32[2,2,3,4,2,2] parameter(2)

    // Dot 1 path (LHS shared): direct use of shared_param (no transpose, no reshape)
    dot_1 = f32[3,4,2,2,2] dot(shared_param, param_rhs_1),
      lhs_batch_dims={1,0}, rhs_batch_dims={2,3},
      lhs_contracting_dims={2,3}, rhs_contracting_dims={0,5}

    ROOT res = (f32[3,2,5,2,2], f32[3,4,2,2,2]) tuple(dot_0_transpose, dot_1)
  }
  )";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));

  // Generate input literals for evaluation.
  ASSERT_OK_AND_ASSIGN(
      Literal shared_param_literal,
      MakeFakeLiteral(ShapeUtil::MakeShape(F32, {4, 3, 2, 2, 2})));
  ASSERT_OK_AND_ASSIGN(
      Literal param_lhs_0_literal,
      MakeFakeLiteral(ShapeUtil::MakeShape(F32, {5, 4, 3, 2, 2})));
  ASSERT_OK_AND_ASSIGN(
      Literal param_rhs_1_literal,
      MakeFakeLiteral(ShapeUtil::MakeShape(F32, {2, 2, 3, 4, 2, 2})));

  // Evaluate the module before running the pass to establish expected results.
  HloEvaluator evaluator;
  ASSERT_OK_AND_ASSIGN(
      Literal expected_result,
      evaluator.Evaluate(*module, {&shared_param_literal, &param_lhs_0_literal,
                                   &param_rhs_1_literal}));

  // Run the DotMerger pass.
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  // Verify HLO verification passes on the merged structure.
  ASSERT_OK(verifier().Run(module.get()).status());

  // Evaluate the optimized module after running the pass.
  HloEvaluator evaluator_after;
  ASSERT_OK_AND_ASSIGN(
      Literal actual_result,
      evaluator_after.Evaluate(
          *module,
          {&shared_param_literal, &param_lhs_0_literal, &param_rhs_1_literal}));

  // Assert that both pre-pass and post-pass execution results match exactly.
  EXPECT_TRUE(
      LiteralTestUtil::Near(expected_result, actual_result, ErrorSpec{1e-4}));
}

TEST_F(DotMergerTest, MergeFailureReproducer) {
  absl::string_view module_string = R"(
HloModule t, entry_computation_layout={(bf16[4,5,10,20,3,7]{5,4,3,2,1,0}, bf16[512,4,5,20,10]{4,3,2,1,0}, bf16[10,2,5,4,3,20]{5,4,3,2,1,0})->(bf16[4,5,512,3,7]{4,3,2,1,0}, bf16[5,2,42,2,3]{4,3,2,1,0})}, num_partitions=8

ENTRY main {
  param_lhs_0 = bf16[512,4,5,20,10]{4,3,2,1,0} parameter(1)
  reshape = bf16[512,4,5,200]{3,2,1,0} reshape(param_lhs_0)
  shared_param = bf16[4,5,10,20,3,7]{5,4,3,2,1,0} parameter(0)
  shared_transpose_0 = bf16[4,5,20,10,3,7]{5,4,3,2,1,0} transpose(shared_param), dimensions={0,1,3,2,4,5}
  reshape.1 = bf16[4,5,200,3,7]{4,3,2,1,0} reshape(shared_transpose_0)
  dot_0 = bf16[4,5,512,3,7]{4,3,2,1,0} dot(reshape, reshape.1), lhs_batch_dims={1,2}, lhs_contracting_dims={3}, rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  shared_transpose_1 = bf16[4,5,10,20,7,3]{5,4,3,2,1,0} transpose(shared_param), dimensions={0,1,2,3,5,4}
  reshape.2 = bf16[4,5,200,7,3]{4,3,2,1,0} reshape(shared_transpose_1)
  param_rhs_1 = bf16[10,2,5,4,3,20]{5,4,3,2,1,0} parameter(2)
  transpose = bf16[2,5,4,3,10,20]{5,3,2,1,0,4} transpose(param_rhs_1), dimensions={1,2,3,4,0,5}
  reshape.3 = bf16[2,5,4,3,200]{3,2,1,0,4} reshape(transpose)
  dot_1 = bf16[5,4,7,3,2,3]{5,4,3,2,1,0} dot(reshape.2, reshape.3), lhs_batch_dims={1,0}, lhs_contracting_dims={2}, rhs_batch_dims={1,2}, rhs_contracting_dims={4}
  dot_1_reshape = bf16[5,2,2,3,42]{4,3,2,1,0} reshape(dot_1)
  dot_1_transpose = bf16[5,2,42,2,3]{4,3,2,1,0} transpose(dot_1_reshape), dimensions={0,2,4,1,3}
  ROOT res = (bf16[4,5,512,3,7]{4,3,2,1,0}, bf16[5,2,42,2,3]{4,3,2,1,0}) tuple(dot_0, dot_1_transpose)
}
  )";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));

  // Generate input literals for evaluation.
  ASSERT_OK_AND_ASSIGN(
      Literal shared_param_literal,
      MakeFakeLiteral(ShapeUtil::MakeShape(BF16, {4, 5, 10, 20, 3, 7})));
  ASSERT_OK_AND_ASSIGN(
      Literal param_lhs_0_literal,
      MakeFakeLiteral(ShapeUtil::MakeShape(BF16, {512, 4, 5, 20, 10})));
  ASSERT_OK_AND_ASSIGN(
      Literal param_rhs_1_literal,
      MakeFakeLiteral(ShapeUtil::MakeShape(BF16, {10, 2, 5, 4, 3, 20})));

  // Evaluate the module before running the pass to establish expected results.
  HloEvaluator evaluator;
  ASSERT_OK_AND_ASSIGN(
      Literal expected_result,
      evaluator.Evaluate(*module, {&shared_param_literal, &param_lhs_0_literal,
                                   &param_rhs_1_literal}));

  // Run the DotMerger pass.
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  // Verify HLO verification passes on the merged structure.
  ASSERT_OK(verifier().Run(module.get()).status());

  // Evaluate the optimized module after running the pass.
  HloEvaluator evaluator_after;
  ASSERT_OK_AND_ASSIGN(
      Literal actual_result,
      evaluator_after.Evaluate(
          *module,
          {&shared_param_literal, &param_lhs_0_literal, &param_rhs_1_literal}));

  // Assert that both pre-pass and post-pass execution results match exactly.
  EXPECT_TRUE(
      LiteralTestUtil::Near(expected_result, actual_result, ErrorSpec{5e-3}));
}

TEST_F(DotMergerTest, MergeWithSharedReshapedLhs) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    common = f32[14,15] parameter(0)
    reshaped_common = f32[5,7,3,2] reshape(common)
    rhs0 = f32[5,7,3,2,4] parameter(1)
    rhs1 = f32[5,7,3,2,8] parameter(2)
    dot0 = f32[4] dot(reshaped_common, rhs0), lhs_contracting_dims={0,1,2,3}, rhs_contracting_dims={0,1,2,3}
    dot1 = f32[8] dot(reshaped_common, rhs1), lhs_contracting_dims={0,1,2,3}, rhs_contracting_dims={0,1,2,3}
    ROOT tuple = (f32[4], f32[8]) tuple(dot0, dot1)
  })";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  const HloInstruction* dot0 = nullptr;
  const HloInstruction* dot1 = nullptr;
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(m::Slice(m::Dot(&dot0, m::Op(), m::Concatenate())),
                          m::Slice(m::Op(&dot1)))));
  EXPECT_EQ(dot0, dot1);
}

TEST_F(DotMergerTest, MismatchBatchRank) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    lhs_source = f32[1, 10, 20] parameter(0)
    rhs0 = f32[1, 20, 30] parameter(1)
    rhs1 = f32[20, 40] parameter(2)

    lhs0 = f32[1, 10, 20] reshape(lhs_source)
    lhs1 = f32[10, 20] reshape(lhs_source)

    dot0 = f32[1, 10, 30] dot(lhs0, rhs0), lhs_batch_dims={0}, lhs_contracting_dims={2}, rhs_batch_dims={0}, rhs_contracting_dims={1}
    dot1 = f32[10, 40] dot(lhs1, rhs1), lhs_contracting_dims={1}, rhs_contracting_dims={0}

    ROOT tuple = (f32[1, 10, 30], f32[10, 40]) tuple(dot0, dot1)
  })";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
}

TEST_F(DotMergerTest, UserExampleMerge) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    shared = f32[1, 10, 20] parameter(0)
    op1 = f32[1, 20, 30] parameter(1)
    op2 = f32[20, 40] parameter(2)
    dot1 = f32[1, 10, 30] dot(shared, op1), lhs_batch_dims={0}, lhs_contracting_dims={2}, rhs_batch_dims={0}, rhs_contracting_dims={1}
    shared_reshaped = f32[10, 20] reshape(shared)
    dot2 = f32[10, 40] dot(shared_reshaped, op2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    ROOT tuple = (f32[1, 10, 30], f32[10, 40]) tuple(dot1, dot2)
  })";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  const HloInstruction* dot0 = nullptr;
  const HloInstruction* dot1 = nullptr;
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(
          m::Reshape(m::Slice(m::Dot(&dot0, m::Op(), m::Concatenate()))),
          m::Slice(m::Op(&dot1)))));
  EXPECT_EQ(dot0, dot1);
}

TEST_F(DotMergerTest, MergeWithDegenerateNonContractingDimsInSharedOperand) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    lhs  = f32[1,100] parameter(0)
    rhs0 = f32[100, 10] parameter(1)
    rhs1 = f32[100, 50] parameter(2)
    dot0 = f32[1, 10] dot(lhs, rhs0), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    dot1 = f32[1, 50] dot(lhs, rhs1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    ROOT tuple = (f32[1,10], f32[1,50]) tuple(dot0, dot1)
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  const HloInstruction* dot0 = nullptr;
  const HloInstruction* dot1 = nullptr;
  // Verify that we slice and then reshape (re-inflate) the output
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(m::Reshape(m::Slice(m::Op(&dot0))),
                                  m::Reshape(m::Slice(m::Op(&dot1))))));
  EXPECT_EQ(dot0, dot1);
  // Verify the merged dot has the degenerate dimension reshaped out from LHS
  EXPECT_THAT(
      dot0,
      GmockMatch(m::Dot(
          m::Reshape(m::Parameter(0)),  // LHS is reshaped to [100]
          m::Concatenate().WithBinaryOperandsAnyOrder(
              m::Transpose(m::Parameter(1)), m::Transpose(m::Parameter(2))))));
}

TEST_F(DotMergerTest, MergeWithConsecutiveContractingDims) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    lhs  = f32[10,20,100] parameter(0)
    rhs0 = f32[10,20,10] parameter(1)
    rhs1 = f32[10,20,50] parameter(2)
    dot0 = f32[100,10] dot(lhs, rhs0), lhs_contracting_dims={0,1}, rhs_contracting_dims={0,1}
    dot1 = f32[100,50] dot(lhs, rhs1), lhs_contracting_dims={0,1}, rhs_contracting_dims={0,1}
    ROOT tuple = (f32[100,10], f32[100,50]) tuple(dot0, dot1)
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  ASSERT_OK(verifier().Run(module.get()).status());

  const HloInstruction* dot0 = nullptr;
  const HloInstruction* dot1 = nullptr;
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(m::Slice(m::Op(&dot0)), m::Slice(m::Op(&dot1)))));
  EXPECT_EQ(dot0, dot1);
  EXPECT_THAT(dot0, GmockMatch(m::Dot(
                        m::Reshape(m::Parameter(0)).WithShape(F32, {200, 100}),
                        m::Concatenate().WithBinaryOperandsAnyOrder(
                            m::Reshape(m::Transpose(m::Parameter(1)))
                                .WithShape(F32, {10, 200}),
                            m::Reshape(m::Transpose(m::Parameter(2)))
                                .WithShape(F32, {50, 200})))));
}

TEST_F(DotMergerTest, MergeWithConsecutiveContractingDimsAndDegenerate) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    lhs  = f32[10,1,20,100] parameter(0)
    rhs0 = f32[10,1,20,10] parameter(1)
    rhs1 = f32[10,1,20,50] parameter(2)
    dot0 = f32[1,100,10] dot(lhs, rhs0), lhs_contracting_dims={0,2}, rhs_contracting_dims={0,2}, lhs_batch_dims={1}, rhs_batch_dims={1}
    dot1 = f32[1,100,50] dot(lhs, rhs1), lhs_contracting_dims={0,2}, rhs_contracting_dims={0,2}, lhs_batch_dims={1}, rhs_batch_dims={1}
    ROOT tuple = (f32[1,100,10], f32[1,100,50]) tuple(dot0, dot1)
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  ASSERT_OK(verifier().Run(module.get()).status());

  const HloInstruction* dot0 = nullptr;
  const HloInstruction* dot1 = nullptr;
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(m::Reshape(m::Slice(m::Op(&dot0))),
                                  m::Reshape(m::Slice(m::Op(&dot1))))));
  EXPECT_EQ(dot0, dot1);
  EXPECT_THAT(dot0, GmockMatch(m::Dot(
                        m::Reshape(m::Parameter(0)).WithShape(F32, {200, 100}),
                        m::Concatenate().WithBinaryOperandsAnyOrder(
                            m::Reshape(m::Transpose(m::Parameter(1)))
                                .WithShape(F32, {10, 200}),
                            m::Reshape(m::Transpose(m::Parameter(2)))
                                .WithShape(F32, {50, 200})))));
}

TEST_F(DotMergerTest, MergeRHSPreservesMetadata) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    lhs  = f32[200,100] parameter(0)
    rhs0 = f32[100, 10] parameter(1)
    rhs1 = f32[100, 50] parameter(2)
    dot0 = f32[200, 10] dot(lhs, rhs0), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_name="op1"}
    dot1 = f32[200, 50] dot(lhs, rhs1), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_name="op2"}
    ROOT tuple = (f32[200,10], f32[200,50]) tuple(dot0, dot1)
  })";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  HloInstruction* merged_dot = nullptr;
  for (HloInstruction* inst : module->entry_computation()->instructions()) {
    if (inst->opcode() == HloOpcode::kDot) {
      merged_dot = inst;
      break;
    }
  }
  ASSERT_NE(merged_dot, nullptr);
  EXPECT_FALSE(merged_dot->metadata().op_name().empty());
  // dot1 (size 50) should be sorted before dot0 (size 10), so we expect "op2".
  EXPECT_EQ(merged_dot->metadata().op_name(), "op2");
}

}  // namespace
}  // namespace xla
