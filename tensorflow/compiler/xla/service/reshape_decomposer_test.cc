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
#include "tensorflow/compiler/xla/service/reshape_decomposer.h"

#include <memory>
#include <optional>

#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/filecheck.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace {

class ReshapeDecomposerTest : public HloTestBase {
 public:
  // Runs reshape decomposer, if `expected` is present, checks it with FileCheck
  // on the output, otherwise checks that the module has not changed.
  //
  // Check that all generated reshapes are bitcasts.
  void CheckReshapeDecomposer(const char* hlo,
                              std::optional<absl::string_view> expected) {
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                            ParseAndReturnVerifiedModule(hlo));

    ReshapeDecomposer pass;
    TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
    EXPECT_EQ(changed, expected.has_value());
    if (changed) {
      TF_ASSERT_OK_AND_ASSIGN(
          bool filecheck_matches,
          RunFileCheck(module->ToString(
                           HloPrintOptions{}.set_print_operand_shape(false)),
                       *expected));
      EXPECT_TRUE(filecheck_matches);
      EXPECT_TRUE(absl::c_all_of(
          module->entry_computation()->instructions(),
          [&](const HloInstruction* instr) {
            return instr->opcode() != HloOpcode::kReshape ||
                   ShapeUtil::ReshapeIsBitcast(instr->operand(0)->shape(),
                                               instr->shape());
          }));
    }
  }
};

TEST_F(ReshapeDecomposerTest, IsBitcast) {
  // Reshape is already a bitcast, nothing should be changed.
  const char* hlo = R"(
HloModule Module

ENTRY main {
  p = f32[8]{0} parameter(0)
  ROOT r = f32[4,2]{1,0} reshape(p)
}
)";
  CheckReshapeDecomposer(hlo, std::nullopt);
}

TEST_F(ReshapeDecomposerTest, AlignableOutput) {
  // Reshape is not a bitcast, but the output is alignable: a transposition on
  // output is inserted.
  const char* hlo = R"(
HloModule Module

ENTRY main {
  p = f32[8,3]{1,0} parameter(0)
  ROOT r = f32[4,2,3]{0,1,2} reshape(p)
}
)";

  CheckReshapeDecomposer(hlo, R"(
// CHECK: [[INSTR_0:%[^ ]+]] = f32[4,2,3]{2,1,0} reshape([[INSTR_1:%[^ ]+]])
// CHECK-NEXT: ROOT [[INSTR_2:%[^ ]+]] = f32[4,2,3]{0,1,2} copy([[INSTR_0]])
)");
}

TEST_F(ReshapeDecomposerTest, AlignableInput) {
  // Reshape is not a bitcast, but the input is alignable: a transposition is
  // inserted on input before reshape.
  const char* hlo = R"(
HloModule Module

ENTRY main {
  p = f32[4,2,3]{0,1,2} parameter(0)
  ROOT r = f32[8,3]{1,0} reshape(p)
}
)";
  CheckReshapeDecomposer(hlo, R"(
// CHECK: [[INSTR_0:%[^ ]+]] = f32[4,2,3]{2,1,0} copy([[INSTR_1:%[^ ]+]])
// CHECK-NEXT: ROOT [[INSTR_2:%[^ ]+]] = f32[8,3]{1,0} reshape([[INSTR_0]])
)");
}

TEST_F(ReshapeDecomposerTest, NotAlignable) {
  // Neither input nor the output are alignable, we expect two transposes to be
  // inserted.
  const char* hlo = R"(
HloModule Module

ENTRY main {
  p = f32[4,2,3,8]{0,2,1,3} parameter(0)
  ROOT r = f32[8,3,2,4]{0,2,1,3} reshape(p)
}
)";
  CheckReshapeDecomposer(hlo, R"(
// CHECK: [[INSTR_0:%[^ ]+]] = f32[4,2,3,8]{3,2,1,0} copy([[INSTR_1:%[^ ]+]])
// CHECK-NEXT: [[INSTR_2:%[^ ]+]] = f32[8,3,2,4]{3,2,1,0} reshape([[INSTR_0]])
// CHECK-NEXT: ROOT [[INSTR_3:%[^ ]+]] = f32[8,3,2,4]{0,2,1,3} copy([[INSTR_2]])
)");
}

}  // namespace
}  // namespace xla
