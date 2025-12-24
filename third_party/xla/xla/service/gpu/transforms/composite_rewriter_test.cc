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

#include "xla/service/gpu/transforms/composite_rewriter.h"

#include <optional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_join.h"
#include "absl/strings/substitute.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

struct TestCase {
  std::string test_name;
  std::string lhs_type;
  std::string rhs_type;
  std::string lhs_scale_type;
  std::string rhs_scale_type;
  std::string lhs_scale_shape;
  std::string rhs_scale_shape;
  std::optional<float> lhs_scale_const_val;
  std::optional<float> rhs_scale_const_val;
  bool expected_rewrite;
};

std::string GenerateHlo(const TestCase& test_case) {
  // Helper to generate scale definition (either parameter or constant)
  // and maintain the list of main parameters.
  std::string main_params_decl;
  std::vector<std::string> call_operands;
  int param_idx = 0;

  // LHS operand (always param 0)
  main_params_decl +=
      absl::Substitute("  %lhs = $0[3,128,256]{2,1,0} parameter($1)\n",
                       test_case.lhs_type, param_idx++);
  call_operands.push_back("%lhs");

  // RHS operand (always param 1)
  main_params_decl +=
      absl::Substitute("  %rhs = $0[3,256,128]{2,1,0} parameter($1)\n",
                       test_case.rhs_type, param_idx++);
  call_operands.push_back("%rhs");

  // LHS Scale
  if (test_case.lhs_scale_const_val.has_value()) {
    std::string val_str = std::to_string(*test_case.lhs_scale_const_val);
    // Remove trailing zeros for cleanliness
    val_str.erase(val_str.find_last_not_of('0') + 1, std::string::npos);
    if (val_str.back() == '.') {
      val_str.pop_back();
    }

    std::string literal;
    if (test_case.lhs_scale_shape == "3,1,1") {
      literal = absl::Substitute("{{{$0}}, {{$0}}, {{$0}}}", val_str);
    } else {
      // Assume rank 3 scalar for 1,1,1 or others
      literal = absl::Substitute("{{{$0}}}", val_str);
    }

    main_params_decl += absl::Substitute(
        "  %lhs_scales = $0[$1]{2,1,0} constant($2)\n",
        test_case.lhs_scale_type, test_case.lhs_scale_shape, literal);
  } else {
    main_params_decl += absl::Substitute(
        "  %lhs_scales = $0[$1]{2,1,0} parameter($2)\n",
        test_case.lhs_scale_type, test_case.lhs_scale_shape, param_idx++);
  }
  call_operands.push_back("%lhs_scales");

  // RHS Scale
  if (test_case.rhs_scale_const_val.has_value()) {
    std::string val_str = std::to_string(*test_case.rhs_scale_const_val);
    val_str.erase(val_str.find_last_not_of('0') + 1, std::string::npos);
    if (val_str.back() == '.') {
      val_str.pop_back();
    }

    std::string literal;
    if (test_case.rhs_scale_shape == "3,1,1") {
      literal = absl::Substitute("{{{$0}}, {{$0}}, {{$0}}}", val_str);
    } else {
      literal = absl::Substitute("{{{$0}}}", val_str);
    }

    main_params_decl += absl::Substitute(
        "  %rhs_scales = $0[$1]{2,1,0} constant($2)\n",
        test_case.rhs_scale_type, test_case.rhs_scale_shape, literal);
  } else {
    main_params_decl += absl::Substitute(
        "  %rhs_scales = $0[$1]{2,1,0} parameter($2)\n",
        test_case.rhs_scale_type, test_case.rhs_scale_shape, param_idx++);
  }
  call_operands.push_back("%rhs_scales");

  // Construct the HLO string
  // Note: We use a dummy body for xla.scaled_dot.1 because the rewriter
  // currently doesn't inspect it, only the call site.
  // We match the parameter types to avoid parser errors.
  std::string hlo_template = R"(
    HloModule test_module

    %xla.scaled_dot.1 {
      %p0 = $0[3,128,256]{2,1,0} parameter(0)
      %p1 = $1[3,256,128]{2,1,0} parameter(1)
      %p2 = $2[$4]{2,1,0} parameter(2)
      %p3 = $3[$5]{2,1,0} parameter(3)
      // Dummy root with correct shape
      ROOT %dummy = bf16[3,128,128]{2,1,0} constant({...})
    }

    ENTRY %main {
      $6
      ROOT %call = bf16[3,128,128]{2,1,0} call($7),
          to_apply=%xla.scaled_dot.1,
          is_composite=true,
          frontend_attributes={
            composite.attributes="{dimension_numbers=[[[2],[1]],[[0],[0]]]}",
            composite.name="xla.scaled_dot",
            composite.version="1"
          }
    }
  )";

  std::string call_operands_str = absl::StrJoin(call_operands, ", ");

  return absl::Substitute(hlo_template, test_case.lhs_type, test_case.rhs_type,
                          test_case.lhs_scale_type, test_case.rhs_scale_type,
                          test_case.lhs_scale_shape, test_case.rhs_scale_shape,
                          main_params_decl, call_operands_str);
}

class CompositeRewriterParameterizedTest
    : public ::testing::TestWithParam<TestCase> {};

TEST_P(CompositeRewriterParameterizedTest, Run) {
  const TestCase& test_case = GetParam();
  std::string hlo_string = GenerateHlo(test_case);
  LOG(INFO) << "HLO string: \n" << hlo_string;

  CompositeRewriter rewriter;
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  auto result = rewriter.Run(module.get());

  if (test_case.expected_rewrite) {
    EXPECT_THAT(result, absl_testing::IsOkAndHolds(true));
    EXPECT_THAT(module->entry_computation()->root_instruction()->opcode(),
                HloOpcode::kScaledDot);
  } else {
    // If it didn't rewrite, it should either be OkAndHolds(false)
    // or arguably just check that the opcode is still Call.
    // The current implementation returns OkAndHolds(false) if no change.
    EXPECT_THAT(result, absl_testing::IsOkAndHolds(false));
    EXPECT_THAT(module->entry_computation()->root_instruction()->opcode(),
                HloOpcode::kCall);
  }
}

INSTANTIATE_TEST_SUITE_P(

    ScaledDotTests, CompositeRewriterParameterizedTest,

    ::testing::Values(
        TestCase{
            /*test_name=*/"FP8_Standard_Case",
            /*lhs_type=*/"f8e4m3fn",
            /*rhs_type=*/"f8e4m3fn",
            /*lhs_scale_type=*/"f8e8m0fnu",
            /*rhs_scale_type=*/"f8e8m0fnu",
            /*lhs_scale_shape=*/"3,128,8",
            /*rhs_scale_shape=*/"3,8,128",
            /*lhs_scale_const_val=*/std::nullopt,
            /*rhs_scale_const_val=*/std::nullopt,
            /*expected_rewrite=*/true,
        },
        TestCase{
            /*test_name=*/"BF16_Identity_Case",
            /*lhs_type=*/"bf16",
            /*rhs_type=*/"bf16",
            /*lhs_scale_type=*/"bf16",
            /*rhs_scale_type=*/"bf16",
            /*lhs_scale_shape=*/"1,1,1",
            /*rhs_scale_shape=*/"1,1,1",
            /*lhs_scale_const_val=*/1.0f,
            /*rhs_scale_const_val=*/1.0f,
            /*expected_rewrite=*/true,
        },
        TestCase{
            /*test_name=*/"BF16_Invalid_Scale_Value",
            /*lhs_type=*/"bf16",
            /*rhs_type=*/"bf16",
            /*lhs_scale_type=*/"bf16",
            /*rhs_scale_type=*/"bf16",
            /*lhs_scale_shape=*/"1,1,1",
            /*rhs_scale_shape=*/"1,1,1",
            /*lhs_scale_const_val=*/1.0f,
            /*rhs_scale_const_val=*/2.0f,
            /*expected_rewrite=*/false,
        },
        TestCase{
            /*test_name=*/"BF16_Invalid_Scale_Shape",
            /*lhs_type=*/"bf16",
            /*rhs_type=*/"bf16",
            /*lhs_scale_type=*/"bf16",
            /*rhs_scale_type=*/"bf16",
            /*lhs_scale_shape=*/"3,128,1",
            /*rhs_scale_shape=*/"1,1,1",
            /*lhs_scale_const_val=*/std::nullopt,
            /*rhs_scale_const_val=*/1.0f,
            /*expected_rewrite=*/false,
        },
        TestCase{
            /*test_name=*/"Mixed_Type_Fail_BF16_Scale_With_FP8_Op",
            /*lhs_type=*/"f8e4m3fn",
            /*rhs_type=*/"f8e4m3fn",
            /*lhs_scale_type=*/"bf16",
            /*rhs_scale_type=*/"f8e8m0fnu",
            /*lhs_scale_shape=*/"3,128,8",
            /*rhs_scale_shape=*/"3,8,128",
            /*lhs_scale_const_val=*/std::nullopt,
            /*rhs_scale_const_val=*/std::nullopt,
            /*expected_rewrite=*/false,
        },
        TestCase{
            /*test_name=*/"FP8_ScaleFactor_16",
            /*lhs_type=*/"f8e4m3fn",
            /*rhs_type=*/"f8e4m3fn",
            /*lhs_scale_type=*/"f8e8m0fnu",
            /*rhs_scale_type=*/"f8e8m0fnu",
            /*lhs_scale_shape=*/"3,128,16",  // 256 / 16 = 16 (not divisible by
                                             // 32)
            /*rhs_scale_shape=*/"3,8,128",
            /*lhs_scale_const_val=*/std::nullopt,
            /*rhs_scale_const_val=*/std::nullopt,
            /*expected_rewrite=*/false,
        },
        TestCase{
            /*test_name=*/"FP8_ScaleFactor_64",
            /*lhs_type=*/"f8e4m3fn",
            /*rhs_type=*/"f8e4m3fn",
            /*lhs_scale_type=*/"f8e8m0fnu",
            /*rhs_scale_type=*/"f8e8m0fnu",
            /*lhs_scale_shape=*/"3,128,4",  // 256 / 4 = 64 (divisible by 32)
            /*rhs_scale_shape=*/"3,8,128",
            /*lhs_scale_const_val=*/std::nullopt,
            /*rhs_scale_const_val=*/std::nullopt,
            /*expected_rewrite=*/true,
        }),
    [](const ::testing::TestParamInfo<TestCase>& info) {
      return info.param.test_name;
    });

}  // namespace
}  // namespace xla::gpu
