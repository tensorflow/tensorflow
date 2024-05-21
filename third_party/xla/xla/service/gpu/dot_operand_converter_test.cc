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

#include "xla/service/gpu/dot_operand_converter.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/primitive_util.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

namespace op = ::xla::testing::opcode_matchers;

class DotOperandConverterTest : public HloTestBase {
 public:
  void TestConvert(bool left_less_precise, PrimitiveType lhs_type,
                   PrimitiveType rhs_type, PrimitiveType result_type) {
    absl::string_view module_tmpl = R"(
  HloModule module

  ENTRY main {
    p0 = $0[2,3]{1,0} parameter(0)
    p1 = $1[3,2]{1,0} parameter(1)
    ROOT dot = $2[2,2]{1,0} dot(p0, p1), lhs_contracting_dims={1},
                                         rhs_contracting_dims={0}
  })";
    auto module_string = absl::Substitute(
        module_tmpl, primitive_util::LowercasePrimitiveTypeName(lhs_type),
        primitive_util::LowercasePrimitiveTypeName(rhs_type),
        primitive_util::LowercasePrimitiveTypeName(result_type));
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                            ParseAndReturnVerifiedModule(module_string));
    TF_ASSERT_OK_AND_ASSIGN(bool upcasted,
                            DotOperandConverter().Run(module.get()));
    EXPECT_TRUE(upcasted);
    if (left_less_precise) {
      auto original_lhs = op::Parameter(0);
      auto upcasted_lhs =
          AllOf(op::Convert(original_lhs),
                op::Shape(absl::Substitute(
                    "$0[2,3]{1,0}",
                    primitive_util::LowercasePrimitiveTypeName(rhs_type))));
      EXPECT_THAT(
          module->entry_computation()->root_instruction(),
          AllOf(op::Dot(upcasted_lhs, op::Parameter(1)),
                op::Shape(absl::Substitute(
                    "$0[2,2]{1,0}",
                    primitive_util::LowercasePrimitiveTypeName(result_type)))));
    } else {
      auto original_rhs = op::Parameter(1);
      auto upcasted_rhs =
          AllOf(op::Convert(original_rhs),
                op::Shape(absl::Substitute(
                    "$0[3,2]{1,0}",
                    primitive_util::LowercasePrimitiveTypeName(lhs_type))));
      EXPECT_THAT(
          module->entry_computation()->root_instruction(),
          AllOf(op::Dot(op::Parameter(0), upcasted_rhs),
                op::Shape(absl::Substitute(
                    "$0[2,2]{1,0}",
                    primitive_util::LowercasePrimitiveTypeName(result_type)))));
    }
  }
};

TEST_F(DotOperandConverterTest, ConvertsLeftAndRight) {
  TestConvert(/*left_less_precise=*/true, S8, BF16, F32);
  TestConvert(/*left_less_precise=*/false, BF16, S8, F32);
}

TEST_F(DotOperandConverterTest, NoConvertHappensWithSameTypes) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    p0 = s8[2,3]{1,0} parameter(0)
    p1 = s8[3,2]{1,0} parameter(1)
    ROOT dot = bf16[2,2]{1,0} dot(p0, p1), lhs_contracting_dims={1},
                                         rhs_contracting_dims={0}
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  TF_ASSERT_OK_AND_ASSIGN(bool upcasted,
                          DotOperandConverter().Run(module.get()));
  EXPECT_FALSE(upcasted);
}

TEST_F(DotOperandConverterTest, NoConvertFromF8toF8) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    p0 = f8e4m3fn[2,3]{1,0} parameter(0)
    p1 = f8e5m2[3,2]{1,0} parameter(1)
    ROOT dot = bf16[2,2]{1,0} dot(p0, p1), lhs_contracting_dims={1},
                                         rhs_contracting_dims={0}
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  TF_ASSERT_OK_AND_ASSIGN(bool upcasted,
                          DotOperandConverter().Run(module.get()));
  EXPECT_FALSE(upcasted);
}

TEST_F(DotOperandConverterTest, CompilerOptimizesUsingDotOperandConverter) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    p0 = s8[2,3]{1,0} parameter(0)
    p1 = bf16[3,2]{1,0} parameter(1)
    ROOT dot = bf16[2,2]{1,0} dot(p0, p1), lhs_contracting_dims={1},
                                         rhs_contracting_dims={0}
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          GetOptimizedModule(module_string));
}

}  // namespace
}  // namespace xla::gpu
