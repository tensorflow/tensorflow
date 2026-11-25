/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/hlo/transforms/expanders/dot_decomposer.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/service/pattern_matcher.h"

namespace xla {
namespace {

namespace m = ::xla::match;
namespace op = ::xla::testing::opcode_matchers;

using DotDecomposerTest = HloHardwareIndependentTestBase;

TEST_F(DotDecomposerTest, CanonicalizeMultipleNonContractingDims) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    p0 = f32[64,63,512]{2,1,0} parameter(0)
    p1 = f32[512,512]{1,0} parameter(1)
    ROOT dot = f32[64,63,512]{2,1,0} dot(p0, p1), lhs_contracting_dims={2},
                                                  rhs_contracting_dims={0}
  })";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  ASSERT_OK_AND_ASSIGN(bool canonicalized, DotDecomposer().Run(module.get()));
  EXPECT_TRUE(canonicalized);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Reshape(AllOf(op::Dot(op::Reshape(), op::Reshape(),
                                        /*lhs_contracting_dim=*/1,
                                        /*rhs_contracting_dim=*/0),
                                op::Shape("f32[4032,512]"))));
}

TEST_F(DotDecomposerTest,
       DontCanonicalizeLhsContractingDim0AndRhsContractingDim1) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    p0 = f32[512,64]{1,0} parameter(0)
    p1 = f32[1024,512]{1,0} parameter(1)
    ROOT dot = f32[64,1024]{1,0} dot(p0, p1), lhs_contracting_dims={0},
                                              rhs_contracting_dims={1}
  })";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  ASSERT_OK_AND_ASSIGN(bool canonicalized, DotDecomposer().Run(module.get()));
  EXPECT_FALSE(canonicalized) << module->ToString();
}

TEST_F(DotDecomposerTest, TransposeContractingDimsUponCanonicalization) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    p0 = f32[512,32,32]{2,1,0} parameter(0)
    p1 = f32[1024,512]{1,0} parameter(1)
    // This dot is considered non-canonical because the LHS has two
    // non-contracting dimensions. Both, LHS and RHS operands are canonicalized,
    // which involves transposing the contracting dimensions to be 1 and 0 on
    // the LHS and RHS, respectively.
    // TODO(tjoerg): Consider leaving the RHS alone, since it is canonical.
    ROOT dot = f32[32,32,1024]{2,1,0} dot(p0, p1), lhs_contracting_dims={0},
                                                   rhs_contracting_dims={1}
  })";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  ASSERT_OK_AND_ASSIGN(bool canonicalized, DotDecomposer().Run(module.get()));
  EXPECT_TRUE(canonicalized) << module->ToString();
  const HloInstruction* dot = nullptr;
  const HloInstruction* lhs_transpose = nullptr;
  const HloInstruction* rhs_transpose = nullptr;
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Reshape(
          m::Op(&dot)
              .WithOperand(0, m::Reshape(m::Transpose(&lhs_transpose)))
              .WithOperand(1, m::Reshape(m::Transpose(&rhs_transpose))))));
  EXPECT_THAT(dot, AllOf(op::Dot(op::Reshape(), op::Reshape(),
                                 /*lhs_contracting_dim=*/1,
                                 /*rhs_contracting_dim=*/0),
                         op::Shape("f32[1024,1024]")));
  EXPECT_THAT(lhs_transpose, op::ShapeWithLayout("f32[32,32,512]"));
  EXPECT_THAT(rhs_transpose, op::ShapeWithLayout("f32[512,1024]"));
}

TEST_F(DotDecomposerTest, DontCanonicalizeIfNoNoncontractingDims) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    p0 = f32[64,4]{1,0} parameter(0)
    p1 = f32[64,4]{1,0} parameter(1)
    ROOT dot = f32[64]{0} dot(p0, p1), lhs_batch_dims={0},
                                       lhs_contracting_dims={1},
                                       rhs_batch_dims={0},
                                       rhs_contracting_dims={1}
  })";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  ASSERT_OK_AND_ASSIGN(bool canonicalized, DotDecomposer().Run(module.get()));
  EXPECT_FALSE(canonicalized);
}

TEST_F(DotDecomposerTest, DontAddLhsNonContractingDimIfOne) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    p0 = f32[64,4]{1,0} parameter(0)
    p1 = f32[64,4,2,1]{3,2,1,0} parameter(1)
    ROOT dot = f32[64,2,1]{2,1,0} dot(p0, p1), lhs_batch_dims={0},
                                               lhs_contracting_dims={1},
                                               rhs_batch_dims={0},
                                               rhs_contracting_dims={1}
  })";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  ASSERT_OK_AND_ASSIGN(bool canonicalized, DotDecomposer().Run(module.get()));
  EXPECT_TRUE(canonicalized);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Reshape(AllOf(op::Dot(op::Reshape(), op::Reshape(),
                                        /*lhs_contracting_dim=*/1,
                                        /*rhs_contracting_dim=*/1),
                                op::Shape("f32[64,2]"))));
}

TEST_F(DotDecomposerTest, DontAddRhsNonContractingDimIfOne) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    p0 = f32[64,4,2,1]{3,2,1,0} parameter(0)
    p1 = f32[64,4]{1,0} parameter(1)
    ROOT dot = f32[64,2,1]{2,1,0} dot(p0, p1), lhs_batch_dims={0},
                                               lhs_contracting_dims={1},
                                               rhs_batch_dims={0},
                                               rhs_contracting_dims={1}
  })";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  ASSERT_OK_AND_ASSIGN(bool canonicalized, DotDecomposer().Run(module.get()));
  EXPECT_TRUE(canonicalized);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Reshape(AllOf(op::Dot(op::Reshape(), op::Reshape(),
                                        /*lhs_contracting_dim=*/2,
                                        /*rhs_contracting_dim=*/1),
                                op::Shape("f32[64,2]"))));
}

TEST_F(DotDecomposerTest, AddLhsNonContractingDimIfZero) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    p0 = f32[64,4,2,0]{3,2,1,0} parameter(0)
    p1 = f32[64,4]{1,0} parameter(1)
    ROOT dot = f32[64,2,0]{2,1,0} dot(p0, p1), lhs_batch_dims={0},
                                               lhs_contracting_dims={1},
                                               rhs_batch_dims={0},
                                               rhs_contracting_dims={1}
  })";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  ASSERT_OK_AND_ASSIGN(bool canonicalized, DotDecomposer().Run(module.get()));
  EXPECT_TRUE(canonicalized);

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Reshape(AllOf(op::Dot(op::Reshape(), op::Reshape(),
                                        /*lhs_contracting_dim=*/2,
                                        /*rhs_contracting_dim=*/1),
                                op::Shape("f32[64,0]"))));
}

TEST_F(DotDecomposerTest, AddRhsNonContractingDimIfZero) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    p0 = f32[64,4]{1,0} parameter(0)
    p1 = f32[64,4,2,0]{3,2,1,0} parameter(1)
    ROOT dot = f32[64,2,0]{2,1,0} dot(p0, p1), lhs_batch_dims={0},
                                                 lhs_contracting_dims={1},
                                                 rhs_batch_dims={0},
                                                 rhs_contracting_dims={1}
  })";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  ASSERT_OK_AND_ASSIGN(bool canonicalized, DotDecomposer().Run(module.get()));
  EXPECT_TRUE(canonicalized);

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Reshape(AllOf(op::Dot(op::Reshape(), op::Reshape(),
                                        /*lhs_contracting_dim=*/1,
                                        /*rhs_contracting_dim=*/1),
                                op::Shape("f32[64,0]"))));
}

TEST_F(DotDecomposerTest, CanonicalizeBatchDims) {
  absl::string_view module_string = R"(
  ENTRY main {
    p0 = f32[64,4,32,8] parameter(0)
    p1 = f32[128,4,8,32] parameter(1)
    ROOT dot = f32[32,8,64,128] dot(p0, p1), lhs_batch_dims={2,3},
                                             lhs_contracting_dims={1},
                                             rhs_batch_dims={3,2},
                                             rhs_contracting_dims={1}
  })";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  ASSERT_OK_AND_ASSIGN(bool canonicalized, DotDecomposer().Run(module.get()));
  EXPECT_TRUE(canonicalized);

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Reshape(AllOf(op::Dot(op::Reshape(), op::Reshape(),
                                        /*lhs_contracting_dim=*/3,
                                        /*rhs_contracting_dim=*/2),
                                op::Shape("f32[32,8,64,128]"))));
}

// The lhs batch dimension is dynamic and the rhs one is static. The canonical
// rhs must stay static (https://github.com/openxla/xla/issues/49340).
TEST_F(DotDecomposerTest, CanonicalizeKeepsBatchDimDynamismOfEachOperand) {
  absl::string_view module_string = R"(
  ENTRY main {
    p0 = f32[<=16,<=16,8] parameter(0)
    p1 = f32[4,16,8,1] parameter(1)
    ROOT dot = f32[<=16,<=16,4,1] dot(p0, p1), lhs_batch_dims={0},
                                               lhs_contracting_dims={2},
                                               rhs_batch_dims={1},
                                               rhs_contracting_dims={2}
  })";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  ASSERT_OK_AND_ASSIGN(bool canonicalized, DotDecomposer().Run(module.get()));
  EXPECT_TRUE(canonicalized);

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root,
              op::Reshape(AllOf(
                  op::Dot(AllOf(op::Reshape(), op::Shape("f32[<=16,<=16,8]")),
                          AllOf(op::Reshape(), op::Shape("f32[16,8,4]")),
                          /*lhs_contracting_dim=*/2,
                          /*rhs_contracting_dim=*/1),
                  op::Shape("f32[<=16,<=16,4]"))));
  // op::Shape ignores dynamism, so check it explicitly.
  const HloInstruction* dot = root->operand(0);
  EXPECT_TRUE(dot->operand(0)->shape().is_dynamic_dimension(0));
  EXPECT_TRUE(dot->operand(1)->shape().is_static());
  EXPECT_TRUE(dot->shape().is_dynamic_dimension(0));
}

TEST_F(DotDecomposerTest, CanonicalizeKeepsOrderOfDynamicBatchDims) {
  absl::string_view module_string = R"(
  ENTRY main {
    p0 = f32[<=2,3,5,7] parameter(0)
    p1 = f32[3,<=2,7,11] parameter(1)
    ROOT dot = f32[3,<=2,5,11] dot(p0, p1), lhs_batch_dims={1,0},
                                            lhs_contracting_dims={3},
                                            rhs_batch_dims={0,1},
                                            rhs_contracting_dims={2}
  })";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  ASSERT_OK_AND_ASSIGN(bool canonicalized, DotDecomposer().Run(module.get()));
  EXPECT_TRUE(canonicalized);

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Reshape(AllOf(op::Dot(op::Reshape(), op::Reshape(),
                                              /*lhs_contracting_dim=*/3,
                                              /*rhs_contracting_dim=*/2),
                                      op::Shape("f32[3,<=2,5,11]"))));
  // The batch dimensions are not reshaped, so the dynamic one stays in place.
  const HloInstruction* dot = root->operand(0);
  for (const HloInstruction* instr : {dot, dot->operand(0), dot->operand(1)}) {
    EXPECT_FALSE(instr->shape().is_dynamic_dimension(0)) << instr->ToString();
    EXPECT_TRUE(instr->shape().is_dynamic_dimension(1)) << instr->ToString();
  }
}

TEST_F(DotDecomposerTest, RunOnComputation) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    p0 = f32[64,63,512]{2,1,0} parameter(0)
    p1 = f32[512,512]{1,0} parameter(1)
    ROOT dot = f32[64,63,512]{2,1,0} dot(p0, p1), lhs_contracting_dims={2},
                                                  rhs_contracting_dims={0}
  })";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_string));
  HloComputation* comp = module->entry_computation();
  ASSERT_OK_AND_ASSIGN(bool canonicalized,
                       DotDecomposer::RunOnComputation(comp));
  EXPECT_TRUE(canonicalized);
  EXPECT_THAT(comp->root_instruction(),
              op::Reshape(AllOf(op::Dot(op::Reshape(), op::Reshape(),
                                        /*lhs_contracting_dim=*/1,
                                        /*rhs_contracting_dim=*/0),
                                op::Shape("f32[4032,512]"))));

  // Verify idempotency: running a second time on the canonicalized computation
  // should make no changes and return false.
  ASSERT_OK_AND_ASSIGN(bool canonicalized_again,
                       DotDecomposer::RunOnComputation(comp));
  EXPECT_FALSE(canonicalized_again);
}

template <typename Arg0, typename Arg1, typename Arg2>
auto SparseDotMatcher(Arg0&& arg0, Arg1&& arg1, Arg2&& arg2) {
  return match::Op()
      .WithOpcode(HloOpcode::kDot)
      .WithOperand(0, std::forward<Arg0>(arg0))
      .WithOperand(1, std::forward<Arg1>(arg1))
      .WithOperand(2, std::forward<Arg2>(arg2));
}

}  // namespace
}  // namespace xla
