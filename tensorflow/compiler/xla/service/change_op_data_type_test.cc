/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/change_op_data_type.h"

#include <string>
#include <tuple>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/service/pattern_matcher_gmock.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace {

namespace m = ::xla::match;

class ChangeOpDataTypeTest : public HloTestBase {
 public:
  ChangeOpDataTypeTest()
      : HloTestBase(/*verifier_layout_sensitive=*/false,
                    /*allow_mixed_precision_in_hlo_verifier=*/false) {}
};

TEST_F(ChangeOpDataTypeTest, Simple) {
  const char* const kModuleStr = R"(
  HloModule module
  ENTRY entry {
    ROOT op = add(f16[10] parameter(0), f16[10] parameter(1))
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  ChangeOpDataType pass(F16, F32, [](const HloInstruction*) { return true; });
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  SCOPED_TRACE(module->ToString());
  EXPECT_TRUE(changed);
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(
          m::Convert(m::Add(m::Convert(m::Parameter(0)).WithShape(F32, {10}),
                            m::Convert(m::Parameter(1)).WithShape(F32, {10})))
              .WithShape(F16, {10})));
}

TEST_F(ChangeOpDataTypeTest, AllTypesMustBeSame) {
  const char* const kModuleStr = R"(
  HloModule module
  ENTRY entry {
    ROOT op = f16[1] dynamic-slice(f16[10] parameter(0), s32[1] parameter(1)), dynamic_slice_sizes={1}
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  ChangeOpDataType pass(F16, F32, [](const HloInstruction*) { return true; });
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  SCOPED_TRACE(module->ToString());
  EXPECT_FALSE(changed);
}

TEST_F(ChangeOpDataTypeTest, DotAndConv) {
  const char* const kModuleStr = R"(
  HloModule module
  ENTRY entry {
    dot = f16[10,10] dot(f16[10,10] parameter(0), f16[10,10] parameter(1)),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
    conv = f16[1,2,1] convolution(f16[1,2,1] parameter(2), f16[1,1,1] parameter(3)),
      window={size=1}, dim_labels=b0f_0io->b0f
    root = tuple(dot, conv)
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  ChangeOpDataType pass(F16, F32, [](const HloInstruction* instr) {
    return instr->opcode() == HloOpcode::kDot ||
           instr->opcode() == HloOpcode::kConvolution;
  });
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  SCOPED_TRACE(module->ToString());
  EXPECT_TRUE(changed);
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(
          m::Convert(
              m::Dot(m::Convert(m::Parameter(0)).WithShape(F32, {10, 10}),
                     m::Convert(m::Parameter(1)).WithShape(F32, {10, 10})))
              .WithShape(F16, {10, 10}),
          m::Convert(m::Convolution(
                         m::Convert(m::Parameter(2)).WithShape(F32, {1, 2, 1}),
                         m::Convert(m::Parameter(3)).WithShape(F32, {1, 1, 1})))
              .WithShape(F16, {1, 2, 1}))));
}

TEST_F(ChangeOpDataTypeTest, SimpleWithCloner) {
  const char* const kModuleStr = R"(
  HloModule module
  ENTRY entry {
    ROOT op = add(f16[10] parameter(0), f16[10] parameter(1))
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  HloPredicate matcher = HloPredicateTrue;

  int count = 0;
  ChangeOpDataType::HloCloner cloner =
      [&count](const HloInstruction* instr, const Shape& shape,
               absl::Span<HloInstruction* const> operands) {
        count++;
        return instr->CloneWithNewOperands(shape, operands);
      };
  ChangeOpDataType pass(F16, F32, matcher, cloner);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  SCOPED_TRACE(module->ToString());
  EXPECT_TRUE(changed);
  // Verify that the cloner provided was used.
  EXPECT_EQ(count, 1);
}

TEST_F(ChangeOpDataTypeTest, SimpleWithMultipleTypes) {
  const char* const kModuleStr = R"(
  HloModule module
  ENTRY entry {
    op1 = add(f16[10] parameter(0), f16[10] parameter(1))
    op2 = add(u16[10] parameter(2), u16[10] parameter(3))
    ROOT tup = (f16[10], u16[10]) tuple(op1, op2)
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  HloPredicate matcher = [](const HloInstruction*) { return true; };
  ChangeOpDataType pass({{F16, F32}, {U16, U32}}, matcher);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  SCOPED_TRACE(module->ToString());
  EXPECT_TRUE(changed);
  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kTuple);
  EXPECT_EQ(root->operand_count(), 2);
  EXPECT_THAT(
      root->operand(0),
      GmockMatch(
          m::Convert(m::Add(m::Convert(m::Parameter(0)).WithShape(F32, {10}),
                            m::Convert(m::Parameter(1)).WithShape(F32, {10})))
              .WithShape(F16, {10})));
  EXPECT_THAT(
      root->operand(1),
      GmockMatch(
          m::Convert(m::Add(m::Convert(m::Parameter(2)).WithShape(U32, {10}),
                            m::Convert(m::Parameter(3)).WithShape(U32, {10})))
              .WithShape(U16, {10})));
}

}  // anonymous namespace
}  // namespace xla
