/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/convert_operand_folding.h"

#include "absl/strings/substitute.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
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

}  // namespace
}  // namespace xla
