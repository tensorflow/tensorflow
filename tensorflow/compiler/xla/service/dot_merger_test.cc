/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/dot_merger.h"

#include "tensorflow/compiler/xla/service/algebraic_simplifier.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/service/pattern_matcher_gmock.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace {

using ::testing::ElementsAre;
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
  EXPECT_FALSE(changed);
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

TEST_F(DotMergerTest, NoMergeNonCanonicalLhsBatch) {
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

TEST_F(DotMergerTest, NoMergeNonCanonicalRhsBatch) {
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

TEST_F(DotMergerTest, NoMergeMultipleContractingDims) {
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
  EXPECT_FALSE(changed);
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

TEST_F(DotMergerTest, MergeBatchDotNoBatchDims) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    dot0 = f32[200, 50] dot(f32[200,100] parameter(0), f32[100,50] parameter(1)),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
    dot1 = f32[200, 50] dot(f32[200,100] parameter(2), f32[100,50] parameter(3)),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
    ROOT tuple = tuple(dot0, dot1)
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  SCOPED_TRACE(module->ToString());
  const HloInstruction* dot0 = nullptr;
  const HloInstruction* dot1 = nullptr;
  const HloInstruction* slice0 = nullptr;
  const HloInstruction* slice1 = nullptr;
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(m::Reshape(m::Slice(&slice0, m::Op(&dot0))),
                          m::Reshape(m::Slice(&slice1, m::Op(&dot1))))));
  EXPECT_EQ(dot0, dot1);

  EXPECT_THAT(slice0->slice_starts(), ElementsAre(0, 0, 0));
  EXPECT_THAT(slice1->slice_starts(), ElementsAre(1, 0, 0));
  EXPECT_THAT(slice0->slice_limits(), ElementsAre(1, 200, 50));
  EXPECT_THAT(slice1->slice_limits(), ElementsAre(2, 200, 50));
  EXPECT_THAT(slice0->slice_strides(), ElementsAre(1, 1, 1));
  EXPECT_THAT(slice1->slice_strides(), ElementsAre(1, 1, 1));

  EXPECT_THAT(dot0,
              GmockMatch(m::Dot(m::Concatenate(m::Reshape(m::Parameter(0)),
                                               m::Reshape(m::Parameter(2))),
                                m::Concatenate(m::Reshape(m::Parameter(1)),
                                               m::Reshape(m::Parameter(3))))));
}

TEST_F(DotMergerTest, MergeBatchDotSameBatchDims) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    dot0 = f32[3,3,200,50] dot(f32[3,3,200,100] parameter(0), f32[3,3,100,50] parameter(1)),
      lhs_batch_dims={0,1}, rhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_contracting_dims={2}
    dot1 = f32[3,3,200, 50] dot(f32[3,3,200,100] parameter(2), f32[3,3,100,50] parameter(3)),
      lhs_batch_dims={0,1}, rhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_contracting_dims={2}
    ROOT tuple = tuple(dot0, dot1)
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  SCOPED_TRACE(module->ToString());
  const HloInstruction* dot0 = nullptr;
  const HloInstruction* dot1 = nullptr;
  const HloInstruction* slice0 = nullptr;
  const HloInstruction* slice1 = nullptr;
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(m::Slice(&slice0, m::Op(&dot0)),
                                  m::Slice(&slice1, m::Op(&dot1)))));
  EXPECT_EQ(dot0, dot1);

  EXPECT_THAT(slice0->slice_starts(), ElementsAre(0, 0, 0, 0));
  EXPECT_THAT(slice1->slice_starts(), ElementsAre(3, 0, 0, 0));
  EXPECT_THAT(slice0->slice_limits(), ElementsAre(3, 3, 200, 50));
  EXPECT_THAT(slice1->slice_limits(), ElementsAre(6, 3, 200, 50));
  EXPECT_THAT(slice0->slice_strides(), ElementsAre(1, 1, 1, 1));
  EXPECT_THAT(slice1->slice_strides(), ElementsAre(1, 1, 1, 1));

  EXPECT_THAT(dot0, GmockMatch(m::Dot(
                        m::Concatenate(m::Parameter(0), m::Parameter(2)),
                        m::Concatenate(m::Parameter(1), m::Parameter(3)))));
}

TEST_F(DotMergerTest, MergeBatchDotBatchDimsSameRank) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    dot0 = f32[3,3,200,50] dot(f32[3,3,200,100] parameter(0), f32[3,3,100,50] parameter(1)),
      lhs_batch_dims={0,1}, rhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_contracting_dims={2}
    dot1 = f32[3,6,200,50] dot(f32[3,6,200,100] parameter(2), f32[3,6,100,50] parameter(3)),
      lhs_batch_dims={0,1}, rhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_contracting_dims={2}
    ROOT tuple = tuple(dot0, dot1)
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  SCOPED_TRACE(module->ToString());
  const HloInstruction* dot0 = nullptr;
  const HloInstruction* dot1 = nullptr;
  const HloInstruction* slice0 = nullptr;
  const HloInstruction* slice1 = nullptr;
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(m::Slice(&slice0, m::Op(&dot0)),
                                  m::Slice(&slice1, m::Op(&dot1)))));
  EXPECT_EQ(dot0, dot1);

  EXPECT_THAT(slice0->slice_starts(), ElementsAre(0, 0, 0, 0));
  EXPECT_THAT(slice1->slice_starts(), ElementsAre(0, 3, 0, 0));
  EXPECT_THAT(slice0->slice_limits(), ElementsAre(3, 3, 200, 50));
  EXPECT_THAT(slice1->slice_limits(), ElementsAre(3, 9, 200, 50));
  EXPECT_THAT(slice0->slice_strides(), ElementsAre(1, 1, 1, 1));
  EXPECT_THAT(slice1->slice_strides(), ElementsAre(1, 1, 1, 1));

  EXPECT_THAT(dot0, GmockMatch(m::Dot(
                        m::Concatenate(m::Parameter(0), m::Parameter(2)),
                        m::Concatenate(m::Parameter(1), m::Parameter(3)))));
}

TEST_F(DotMergerTest, MergeBatchDotsOneIsNotBatched) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    dot0 = f32[200,50] dot(f32[200,100] parameter(0), f32[100,50] parameter(1)),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
    dot1 = f32[3,200,50] dot(f32[3,200,100] parameter(2), f32[3,100,50] parameter(3)),
      lhs_batch_dims={0}, rhs_batch_dims={0}, lhs_contracting_dims={2}, rhs_contracting_dims={1}
    ROOT tuple = tuple(dot0, dot1)
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  SCOPED_TRACE(module->ToString());
  const HloInstruction* dot0 = nullptr;
  const HloInstruction* dot1 = nullptr;
  const HloInstruction* slice0 = nullptr;
  const HloInstruction* slice1 = nullptr;
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(m::Reshape(m::Slice(&slice0, m::Op(&dot0))),
                                  m::Slice(&slice1, m::Op(&dot1)))));
  EXPECT_EQ(dot0, dot1);

  EXPECT_THAT(slice0->slice_starts(), ElementsAre(0, 0, 0));
  EXPECT_THAT(slice1->slice_starts(), ElementsAre(1, 0, 0));
  EXPECT_THAT(slice0->slice_limits(), ElementsAre(1, 200, 50));
  EXPECT_THAT(slice1->slice_limits(), ElementsAre(4, 200, 50));
  EXPECT_THAT(slice0->slice_strides(), ElementsAre(1, 1, 1));
  EXPECT_THAT(slice1->slice_strides(), ElementsAre(1, 1, 1));

  EXPECT_THAT(
      dot0, GmockMatch(m::Dot(
                m::Concatenate(m::Reshape(m::Parameter(0)), m::Parameter(2)),
                m::Concatenate(m::Reshape(m::Parameter(1)), m::Parameter(3)))));
}

TEST_F(DotMergerTest, MergeBatchDotsAddDegenerateDimInMiddle) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    dot0 = f32[3,5,200,50] dot(f32[3,5,200,100] parameter(0), f32[3,5,100,50] parameter(1)),
      lhs_batch_dims={0,1}, rhs_batch_dims={0,1}, lhs_contracting_dims={3}, rhs_contracting_dims={2}
    dot1 = f32[3,4,5,200,50] dot(f32[3,4,5,200,100] parameter(2), f32[3,4,5,100,50] parameter(3)),
      lhs_batch_dims={0,1,2}, rhs_batch_dims={0,1,2}, lhs_contracting_dims={4}, rhs_contracting_dims={3}
    ROOT tuple = tuple(dot0, dot1)
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  DotMerger pass(/*max_size_to_merge=*/std::numeric_limits<int64_t>::max());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, this->RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  SCOPED_TRACE(module->ToString());
  const HloInstruction* dot0 = nullptr;
  const HloInstruction* dot1 = nullptr;
  const HloInstruction* slice0 = nullptr;
  const HloInstruction* slice1 = nullptr;
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(m::Reshape(m::Slice(&slice0, m::Op(&dot0))),
                                  m::Slice(&slice1, m::Op(&dot1)))));
  EXPECT_EQ(dot0, dot1);

  EXPECT_THAT(slice0->slice_starts(), ElementsAre(0, 0, 0, 0, 0));
  EXPECT_THAT(slice1->slice_starts(), ElementsAre(0, 1, 0, 0, 0));
  EXPECT_THAT(slice0->slice_limits(), ElementsAre(3, 1, 5, 200, 50));
  EXPECT_THAT(slice1->slice_limits(), ElementsAre(3, 5, 5, 200, 50));
  EXPECT_THAT(slice0->slice_strides(), ElementsAre(1, 1, 1, 1, 1));
  EXPECT_THAT(slice1->slice_strides(), ElementsAre(1, 1, 1, 1, 1));

  EXPECT_THAT(
      dot0, GmockMatch(m::Dot(
                m::Concatenate(m::Reshape(m::Parameter(0)), m::Parameter(2)),
                m::Concatenate(m::Reshape(m::Parameter(1)), m::Parameter(3)))));
}

}  // namespace
}  // namespace xla
