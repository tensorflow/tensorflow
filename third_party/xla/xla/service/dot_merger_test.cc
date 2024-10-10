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

#include "xla/service/dot_merger.h"

#include <cstdint>
#include <limits>
#include <memory>

#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/algebraic_simplifier.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/pattern_matcher_gmock.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

namespace m = ::xla::match;

class DotMergerTest : public HloTestBase {
 public:
  DotMergerTest()
      : HloTestBase(/*verifier_layout_sensitive=*/false,
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
    ROOT tuple = (f32[200,10], f32[200,50]) tuple(dot0, dot1)
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  const HloInstruction* dot0 = nullptr;
  const HloInstruction* dot1 = nullptr;
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(m::Slice(m::Op(&dot0)), m::Slice(m::Op(&dot1)))));
  EXPECT_EQ(dot0, dot1);
  EXPECT_THAT(dot0,
              GmockMatch(m::Dot(m::Parameter(0),
                                m::Concatenate().WithBinaryOperandsAnyOrder(
                                    m::Parameter(1), m::Parameter(2)))));
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
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  const HloInstruction* dot0 = nullptr;
  const HloInstruction* dot1 = nullptr;
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(m::Slice(m::Op(&dot0)), m::Slice(m::Op(&dot1)))));
  EXPECT_EQ(dot0, dot1);
  Shape expected_concat_shape =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {100, 60}, {0, 1});
  EXPECT_THAT(
      dot0, GmockMatch(m::Dot(m::Parameter(0),
                              m::Concatenate()
                                  .WithBinaryOperandsAnyOrder(m::Parameter(1),
                                                              m::Parameter(2))
                                  .WithShapeEqualTo(&expected_concat_shape))));
}

TEST_F(DotMergerTest, NoMergeDifferentLayoutRHS) {
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
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_FALSE(changed);
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
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(m::Slice(), m::Slice())));
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
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  Shape expected_dot_shape =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {400, 50}, {0, 1});
  const HloInstruction* dot0 = nullptr;
  const HloInstruction* dot1 = nullptr;
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(m::Slice(m::Dot(&dot0, m::Op(), m::Op())
                                       .WithShapeEqualTo(&expected_dot_shape)),
                          m::Slice(m::Dot(&dot1, m::Op(), m::Op())))));
  EXPECT_EQ(dot0, dot1);
}

TEST_F(DotMergerTest, NoMergeDifferentLayoutLHS) {
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
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(DotMergerTest, NoMergeDifferentDotLayout) {
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
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_FALSE(changed);
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
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  // Clean up some redundant slice-of-slices so it's easier to pattern-match.
  AlgebraicSimplifier algsimp{AlgebraicSimplifierOptions{}};
  TF_ASSERT_OK(this->RunHloPass(&algsimp, module.get()).status());

  const HloInstruction* s0 = nullptr;
  const HloInstruction* s1 = nullptr;
  const HloInstruction* s2 = nullptr;
  SCOPED_TRACE(module->ToString());
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(
          m::Slice(m::Dot(
              &s0,
              m::Concatenate(m::Parameter(0), m::Parameter(1), m::Parameter(2)),
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
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  AlgebraicSimplifier algsimp{AlgebraicSimplifierOptions{}};
  TF_ASSERT_OK(this->RunHloPass(&algsimp, module.get()).status());

  const HloInstruction* s0 = nullptr;
  const HloInstruction* s1 = nullptr;
  const HloInstruction* s2 = nullptr;
  SCOPED_TRACE(module->ToString());
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(
          m::Slice(m::Dot(&s0, m::Concatenate(m::Parameter(0), m::Parameter(1)),
                          m::Parameter(2))),
          m::Slice(m::Op(&s1)),  //
          m::Dot(&s2, m::Op(), m::Parameter(2)))));

  // There should be two dot ops.
  EXPECT_EQ(s0, s1);
  EXPECT_NE(s0, s2);
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
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
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
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
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
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(m::Slice(), m::Slice())));
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
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  TF_ASSERT_OK(verifier().Run(module.get()).status());
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
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(m::Slice(), m::Slice())));
}

TEST_F(DotMergerTest, NoMergeDueToIsMergeCandidate) {
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
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  // We can merge dot0 with either dot1 or dot2 (because only one dot needs to
  // be a merge candidate in order for it to go forward), but we can't merge
  // dot1 and dot2 together, because neither is a candidate.
  //
  // The pass should be deterministic and choose to merge dot0 with dot1.
  DotMerger pass(/*max_size_to_merge=*/(100 * 50 + 100 * 200 + 200 * 50) *
                 sizeof(float));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  const HloInstruction* s0 = nullptr;
  const HloInstruction* s1 = nullptr;
  const HloInstruction* s2 = nullptr;
  SCOPED_TRACE(module->ToString());
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(
          m::Slice(m::Dot(&s0, m::Concatenate(m::Parameter(0), m::Parameter(1)),
                          m::Parameter(3))),
          m::Slice(m::Op(&s1)),
          m::Dot(&s2, m::Parameter(2), m::Parameter(3)))));

  // There should be two unique dot ops.
  EXPECT_EQ(s0, s1);
  EXPECT_NE(s0, s2);
}

TEST_F(DotMergerTest, NoMergeDifferentLhsBatchDims) {
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
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_FALSE(changed);
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
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
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
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);

  const HloInstruction* s0 = nullptr;
  const HloInstruction* s1 = nullptr;
  SCOPED_TRACE(module->ToString());
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(
          m::Slice(m::Dot(&s0, m::Concatenate(m::Parameter(0), m::Parameter(1)),
                          m::Parameter(2))),
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
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  TF_ASSERT_OK(verifier().Run(module.get()).status());

  const HloInstruction* s0 = nullptr;
  const HloInstruction* s1 = nullptr;
  SCOPED_TRACE(module->ToString());
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(
          m::Slice(m::Dot(&s0, m::Concatenate(m::Parameter(0), m::Parameter(1)),
                          m::Parameter(2))),
          m::Slice(m::Op(&s1)))));
  EXPECT_EQ(s0, s1);
}

TEST_F(DotMergerTest, NoMergeMultipleOuterDims) {
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
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(DotMergerTest, NoMergeDifferentLhsContractingDims) {
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
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_FALSE(changed);
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
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
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
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
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
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
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
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
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
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
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
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  SCOPED_TRACE(module->ToString());

  EXPECT_TRUE(changed);
  const HloInstruction* d0 = nullptr;
  const HloInstruction* d1 = nullptr;
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(
          m::Slice(m::Dot(&d0, m::Concatenate(m::Parameter(0), m::Parameter(1)),
                          m::Parameter(2))
                       .WithShape(F32, {20, 10})),
          m::Slice(m::Op(&d1)))));
  EXPECT_EQ(d0, d1);
}

TEST_F(DotMergerTest, MergeSparseDotsSameMetadata) {
  absl::string_view kHlo = R"(
  HloModule test
  ENTRY main {
    lhs0 = f16[5,10,32] parameter(0)
    lhs1 = f16[5,10,32] parameter(1)
    rhs  = f16[5,10,16] parameter(2)
    meta = u16[5,10,2] parameter(3)
    dot0 = f32[5,10,10] dot(lhs0, rhs, meta), sparsity=R.2@2:4,
        lhs_batch_dims={0}, rhs_batch_dims={0},
        lhs_contracting_dims={2}, rhs_contracting_dims={2}
    dot1 = f32[5,10,10] dot(lhs1, rhs, meta), sparsity=R.2@2:4,
        lhs_batch_dims={0}, rhs_batch_dims={0},
        lhs_contracting_dims={2}, rhs_contracting_dims={2}
    ROOT tuple = (f32[5,10,10], f32[5,10,10]) tuple(dot0, dot1)
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  const HloInstruction *d0, *d1;
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(
                  m::Slice(m::Op(&d0)
                               .WithOpcode(HloOpcode::kDot)
                               .WithOperand(0, m::Concatenate(m::Parameter(0),
                                                              m::Parameter(1)))
                               .WithOperand(1, m::Parameter(2))
                               .WithOperand(2, m::Parameter(3))
                               .WithShape(F32, {5, 20, 10})),
                  m::Slice(m::Op(&d1)))));
  EXPECT_EQ(d0, d1);
  EXPECT_EQ(d0->operand(2)->shape(), ShapeUtil::MakeShape(U16, {5, 10, 2}));
}

TEST_F(DotMergerTest, MergeSparseDotsConcatMetadata) {
  absl::string_view kHlo = R"(
  HloModule test
  ENTRY main {
    lhs0 = f16[5,10,16] parameter(0)
    lhs1 = f16[5,10,16] parameter(1)
    rhs  = f16[5,10,32] parameter(2)
    meta0 = u16[5,10,2] parameter(3)
    meta1 = u16[5,10,2] parameter(4)
    dot0 = f32[5,10,10] dot(lhs0, rhs, meta0), sparsity=L.2@2:4,
        lhs_batch_dims={0}, rhs_batch_dims={0},
        lhs_contracting_dims={2}, rhs_contracting_dims={2}
    dot1 = f32[5,10,10] dot(lhs1, rhs, meta1), sparsity=L.2@2:4,
        lhs_batch_dims={0}, rhs_batch_dims={0},
        lhs_contracting_dims={2}, rhs_contracting_dims={2}
    ROOT tuple = (f32[5,10,10], f32[5,10,10]) tuple(dot0, dot1)
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  const HloInstruction *d0, *d1;
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(
                  m::Slice(m::Op(&d0)
                               .WithOpcode(HloOpcode::kDot)
                               .WithOperand(0, m::Concatenate(m::Parameter(0),
                                                              m::Parameter(1)))
                               .WithOperand(1, m::Parameter(2))
                               .WithOperand(2, m::Concatenate(m::Parameter(3),
                                                              m::Parameter(4)))
                               .WithShape(F32, {5, 20, 10})),
                  m::Slice(m::Op(&d1)))));
  EXPECT_EQ(d0, d1);
  EXPECT_EQ(d0->operand(2)->shape(), ShapeUtil::MakeShape(U16, {5, 20, 2}));
}

TEST_F(DotMergerTest, MergeSparseDotsDifferentMetadata) {
  absl::string_view kHlo = R"(
  HloModule test
  ENTRY main {
    lhs0 = f16[5,10,32] parameter(0)
    lhs1 = f16[5,10,32] parameter(1)
    rhs  = f16[5,10,16] parameter(2)
    meta1 = u16[5,10,2] parameter(3)
    meta2 = u16[5,10,2] parameter(4)
    dot0 = f32[5,10,10] dot(lhs0, rhs, meta1), sparsity=R.2@2:4,
        lhs_batch_dims={0}, rhs_batch_dims={0},
        lhs_contracting_dims={2}, rhs_contracting_dims={2}
    dot1 = f32[5,10,10] dot(lhs1, rhs, meta2), sparsity=R.2@2:4,
        lhs_batch_dims={0}, rhs_batch_dims={0},
        lhs_contracting_dims={2}, rhs_contracting_dims={2}
    ROOT tuple = (f32[5,10,10], f32[5,10,10]) tuple(dot0, dot1)
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(DotMergerTest, NoMergeWithFalseCompatibility) {
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
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  std::function<bool(const HloInstruction* dot_a, const HloInstruction* dot_b)>
      can_merge = [&](const HloInstruction* dot_a,
                      const HloInstruction* dot_b) -> bool { return false; };
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max(),
                 can_merge);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_FALSE(changed);
}

}  // namespace
}  // namespace xla
