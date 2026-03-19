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
#include "xla/tsl/platform/statusor.h"

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

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  TF_ASSERT_OK_AND_ASSIGN(bool canonicalized,
                          DotDecomposer().Run(module.get()));
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

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  TF_ASSERT_OK_AND_ASSIGN(bool canonicalized,
                          DotDecomposer().Run(module.get()));
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

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  TF_ASSERT_OK_AND_ASSIGN(bool canonicalized,
                          DotDecomposer().Run(module.get()));
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

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  TF_ASSERT_OK_AND_ASSIGN(bool canonicalized,
                          DotDecomposer().Run(module.get()));
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

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  TF_ASSERT_OK_AND_ASSIGN(bool canonicalized,
                          DotDecomposer().Run(module.get()));
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

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  TF_ASSERT_OK_AND_ASSIGN(bool canonicalized,
                          DotDecomposer().Run(module.get()));
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

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  TF_ASSERT_OK_AND_ASSIGN(bool canonicalized,
                          DotDecomposer().Run(module.get()));
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

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  TF_ASSERT_OK_AND_ASSIGN(bool canonicalized,
                          DotDecomposer().Run(module.get()));
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

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  TF_ASSERT_OK_AND_ASSIGN(bool canonicalized,
                          DotDecomposer().Run(module.get()));
  EXPECT_TRUE(canonicalized);

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Reshape(AllOf(op::Dot(op::Reshape(), op::Reshape(),
                                        /*lhs_contracting_dim=*/3,
                                        /*rhs_contracting_dim=*/2),
                                op::Shape("f32[32,8,64,128]"))));
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
