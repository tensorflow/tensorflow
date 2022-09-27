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
#include "tensorflow/compiler/xla/tests/test_utils.h"

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

}  // anonymous namespace
}  // namespace xla
