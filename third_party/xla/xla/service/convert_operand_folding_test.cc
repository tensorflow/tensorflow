/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/service/convert_operand_folding.h"

#include "absl/strings/substitute.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/primitive_util.h"
#include "xla/tests/hlo_test_base.h"
namespace xla {
namespace {

namespace op = ::xla::testing::opcode_matchers;

using ConvertOperandFoldingTest = HloTestBase;

TEST_F(ConvertOperandFoldingTest, IntegralUpcastConvertFolded) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    p0 = s8[2,3]{1,0} parameter(0)
    p1 = s16[3,2]{0,1} parameter(1)
    c0 = s16[2,3]{1,0} convert(p0)
    c1 = s16[3,2]{0,1} convert(p1)
    ROOT dot = s16[2,2]{1,0} dot(c0, c1), lhs_contracting_dims={1},
                                          rhs_contracting_dims={0}
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  TF_ASSERT_OK_AND_ASSIGN(bool folded,
                          ConvertOperandFolding().Run(module.get()));
  EXPECT_TRUE(folded);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              AllOf(op::Dot(op::Parameter(0), op::Parameter(1)),
                    op::Shape("s16[2,2]{1,0}")));
}

TEST_F(ConvertOperandFoldingTest, FloatingUpcastConvertFolded) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    p0 = f16[2,3]{1,0} parameter(0)
    p1 = bf16[3,2]{0,1} parameter(1)
    c0 = f32[2,3]{1,0} convert(p0)
    c1 = f32[3,2]{0,1} convert(p1)
    ROOT dot = f32[2,2]{1,0} dot(c0, c1), lhs_contracting_dims={1},
                                          rhs_contracting_dims={0}
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  TF_ASSERT_OK_AND_ASSIGN(bool folded,
                          ConvertOperandFolding().Run(module.get()));
  EXPECT_TRUE(folded);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              AllOf(op::Dot(op::Parameter(0), op::Parameter(1)),
                    op::Shape("f32[2,2]{1,0}")));
}

TEST_F(ConvertOperandFoldingTest, IntegralToFloatingConvertFolded) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    p0 = s8[2,3]{1,0} parameter(0)
    p1 = s16[3,2]{0,1} parameter(1)
    c0 = f16[2,3]{1,0} convert(p0)
    c1 = f32[3,2]{0,1} convert(p1)
    ROOT dot = f32[2,2]{1,0} dot(c0, c1), lhs_contracting_dims={1},
                                          rhs_contracting_dims={0}
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  TF_ASSERT_OK_AND_ASSIGN(bool folded,
                          ConvertOperandFolding().Run(module.get()));
  EXPECT_TRUE(folded);
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              AllOf(op::Dot(op::Parameter(0), op::Parameter(1)),
                    op::Shape("f32[2,2]{1,0}")));
}

TEST_F(ConvertOperandFoldingTest, DowncastConvertNotFolded) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    p0 = s32[2,3]{1,0} parameter(0)
    p1 = s16[3,2]{0,1} parameter(1)
    c0 = s16[2,3]{1,0} convert(p0)
    c1 = s8[3,2]{0,1} convert(p1)
    ROOT dot = s16[2,2]{1,0} dot(c0, c1), lhs_contracting_dims={1},
                                          rhs_contracting_dims={0}
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  TF_ASSERT_OK_AND_ASSIGN(bool folded,
                          ConvertOperandFolding().Run(module.get()));
  EXPECT_FALSE(folded);
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      AllOf(
          op::Dot(
              AllOf(op::Convert(op::Parameter(0)), op::Shape("s16[2,3]{1,0}")),
              AllOf(op::Convert(op::Parameter(1)), op::Shape("s8[3,2]{0,1}"))),
          op::Shape("s16[2,2]{1,0}")));
}

TEST_F(ConvertOperandFoldingTest, OneOperandFolded) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    p0 = s8[2,3]{1,0} parameter(0)
    p1 = s16[3,2]{0,1} parameter(1)
    c0 = s16[2,3]{1,0} convert(p0)
    c1 = s8[3,2]{0,1} convert(p1)
    ROOT dot = s16[2,2]{1,0} dot(c0, c1), lhs_contracting_dims={1},
                                          rhs_contracting_dims={0}
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  TF_ASSERT_OK_AND_ASSIGN(bool folded,
                          ConvertOperandFolding().Run(module.get()));
  EXPECT_TRUE(folded);
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      AllOf(op::Dot(op::Parameter(0), AllOf(op::Convert(op::Parameter(1)),
                                            op::Shape("s8[3,2]{0,1}"))),
            op::Shape("s16[2,2]{1,0}")));
}

TEST_F(ConvertOperandFoldingTest, FoldedWithFormatting) {
  absl::string_view module_string = R"(
  HloModule module
  sum {
    a = s16[] parameter(0)
    b = s16[] parameter(1)
    ROOT r  = add(a,b)
  }

  ENTRY main {
    p0 = s8[3,10] parameter(0)
    c0 = s16[3,10] convert(p0)
    r0 = s16[3,2,5] reshape(c0)
    t0 = s16[2,5,3] transpose(r0), dimensions={1,2,0}
    s0 = s16[2,1,3] slice(t0), slice={[0:2], [2:3], [0:3]}
    rs0 = s16[2,3] reshape(s0)
    p1 = s8[3,1,2] parameter(1)
    c1 = s16[3,1,2] convert(p1)
    r1 = s16[1,3,2] transpose(c1), dimensions={1,0,2}
    z = s16[] constant(0)
    rr1 = s16[3,2] reduce(r1,z), dimensions={0}, to_apply=sum
    ROOT dot = s16[2,2] dot(rs0, rr1), lhs_contracting_dims={1},
                                          rhs_contracting_dims={0}
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  TF_ASSERT_OK_AND_ASSIGN(bool folded,
                          ConvertOperandFolding().Run(module.get()));
  EXPECT_TRUE(folded);
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Dot(
          op::Reshape(op::Slice(op::Transpose(op::Reshape(op::Parameter(0))))),
          op::Reshape(op::Transpose(op::Parameter(1)))));
}

TEST_F(ConvertOperandFoldingTest, FoldedWithDSAndGather) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    p0 = s8[100,3] parameter(0)
    c0 = s16[100,3] convert(p0)
    ids = s32[20] parameter(2)
    g = s16[20,3] gather(c0, ids), offset_dims={1}, collapsed_slice_dims={0}, start_index_map={0}, index_vector_dim=1, slice_sizes={1,3}
    t = s16[3,20] transpose(g), dimensions={1,0}

    p1 = s8[25,3] parameter(1)
    c1 = s16[25,3] convert(p1)
    z = s32[] constant(0)
    s = s32[] parameter(3)
    ds = s16[20,3] dynamic-slice(c1, s, z), dynamic_slice_sizes={20,3}

    ROOT dot = s16[3,3] dot(t, ds), lhs_contracting_dims={1},
                                          rhs_contracting_dims={0}
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  TF_ASSERT_OK_AND_ASSIGN(bool folded,
                          ConvertOperandFolding().Run(module.get()));
  EXPECT_TRUE(folded);
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      op::Dot(op::Transpose(op::Gather(op::Parameter(0), op::Parameter(2))),
              op::DynamicSlice(op::Parameter(1), op::Parameter(3),
                               op::Constant())));
}

}  // namespace
}  // namespace xla
