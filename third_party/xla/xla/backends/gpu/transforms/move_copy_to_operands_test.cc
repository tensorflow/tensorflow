/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/transforms/move_copy_to_operands.h"

#include <memory>
#include <optional>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/layout.h"
#include "xla/service/layout_assignment.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

class MoveCopyToOperandsTest : public HloHardwareIndependentTestBase {
 public:
  MoveCopyToOperandsTest()
      : HloHardwareIndependentTestBase(
            /*verifier_layout_sensitive=*/true,
            /*allow_mixed_precision_in_hlo_verifier=*/true,
            LayoutAssignment::InstructionCanChangeLayout) {}
  void CheckMoveCopyToOperands(absl::string_view hlo,
                               std::optional<absl::string_view> expected) {
    RunAndFilecheckHloRewrite(hlo, MoveCopyToOperands{}, expected);
  }
};

TEST_F(MoveCopyToOperandsTest, Pad) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  input = s8[1,17,9,9]{3,1,2,0} parameter(0)
  constant = s8[] constant(0)
  pad = s8[1,32,9,9]{3,1,2,0} pad(input, constant), padding=0_0x0_15x0_0x0_0
  ROOT copy = s8[1,32,9,9]{1,3,2,0} copy(pad)
}
)";

  CheckMoveCopyToOperands(hlo, R"(
// CHECK: [[input:%[^ ]+]] = s8[1,17,9,9]{3,1,2,0} parameter(0)
// CHECK: [[copy:%[^ ]+]] = s8[1,17,9,9]{1,3,2,0} copy([[input]])
// CHECK: [[constant:%[^ ]+]] = s8[] constant(0)
// CHECK: ROOT [[pad:%[^ ]+]] = s8[1,32,9,9]{1,3,2,0} pad([[copy]], [[constant]]), padding=0_0x0_15x0_0x0_0
)");
}

TEST_F(MoveCopyToOperandsTest, Unary) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  input = f32[1,17,9,9]{3,2,1,0} parameter(0)
  sqrt = f32[1,17,9,9]{3,2,1,0} sqrt(input)
  ROOT copy = f32[1,17,9,9]{1,3,2,0} copy(sqrt)
}
)";

  CheckMoveCopyToOperands(hlo, R"(
// CHECK: [[input:%[^ ]+]] = f32[1,17,9,9]{3,2,1,0} parameter(0)
// CHECK: [[copy:%[^ ]+]] = f32[1,17,9,9]{1,3,2,0} copy([[input]])
// CHECK: ROOT [[sqrt:%[^ ]+]] = f32[1,17,9,9]{1,3,2,0} sqrt([[copy]])
)");
}

TEST_F(MoveCopyToOperandsTest, Reverse) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  input = f32[1,17,9,9]{3,2,1,0} parameter(0)
  reverse = f32[1,17,9,9]{3,2,1,0} reverse(input), dimensions={1,2}
  ROOT copy = f32[1,17,9,9]{1,3,2,0} copy(reverse)
}
)";

  CheckMoveCopyToOperands(hlo, R"(
// CHECK: [[input:%[^ ]+]] = f32[1,17,9,9]{3,2,1,0} parameter(0)
// CHECK: [[copy:%[^ ]+]] = f32[1,17,9,9]{1,3,2,0} copy([[input]])
// CHECK: ROOT [[reverse:%[^ ]+]] = f32[1,17,9,9]{1,3,2,0} reverse([[copy]]), dimensions={1,2}
)");
}

TEST_F(MoveCopyToOperandsTest, Convert) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  input = f32[1,17,9,9]{3,2,1,0} parameter(0)
  converted = f16[1,17,9,9]{3,2,1,0} convert(input)
  ROOT copy = f16[1,17,9,9]{1,3,2,0} copy(converted)
}
)";

  CheckMoveCopyToOperands(hlo, R"(
// CHECK: [[input:%[^ ]+]] = f32[1,17,9,9]{3,2,1,0} parameter(0)
// CHECK: [[copy:%[^ ]+]] = f32[1,17,9,9]{1,3,2,0} copy([[input]])
// CHECK: ROOT [[converted:%[^ ]+]] = f16[1,17,9,9]{1,3,2,0} convert([[copy]])
)");
}

TEST_F(MoveCopyToOperandsTest, BitcastConvertSameElements) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  input = f32[2,2]{1,0} parameter(0)
  bitcast = s32[2,2]{1,0} bitcast-convert(input)
  ROOT copy = s32[2,2]{0,1} copy(bitcast)
}
)";

  CheckMoveCopyToOperands(hlo, R"(
// CHECK: [[input:%[^ ]+]] = f32[2,2]{1,0} parameter(0)
// CHECK: [[copy:%[^ ]+]] = f32[2,2]{0,1} copy([[input]])
// CHECK: ROOT [[bitcast:%[^ ]+]] = s32[2,2]{0,1} bitcast-convert([[copy]])
)");
}

TEST_F(MoveCopyToOperandsTest, BitcastConvertDifferentElements) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  input = f32[4]{0} parameter(0)
  bitcast = f16[4,2]{1,0} bitcast-convert(input)
  ROOT copy = f16[4,2]{0,1} copy(bitcast)
}
)";

  CheckMoveCopyToOperands(hlo, std::nullopt);
}

TEST_F(MoveCopyToOperandsTest, Slice) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  input = f32[1,17,9,9]{3,2,1,0} parameter(0)
  slice = f32[1,4,6,6]{3,2,1,0} slice(input), slice={[0:1],[0:4],[0:6],[0:6]}
  ROOT copy = f32[1,4,6,6]{1,3,2,0} copy(slice)
}
)";

  CheckMoveCopyToOperands(hlo, R"(
// CHECK: [[input:%[^ ]+]] = f32[1,17,9,9]{3,2,1,0} parameter(0)
// CHECK: [[copy:%[^ ]+]] = f32[1,17,9,9]{1,3,2,0} copy([[input]])
// CHECK: ROOT [[slice:%[^ ]+]] = f32[1,4,6,6]{1,3,2,0} slice([[copy]]), slice={[0:1], [0:4], [0:6], [0:6]}
)");
}

TEST_F(MoveCopyToOperandsTest, Binary) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  input = f32[1,17,9,9]{3,2,1,0} parameter(0)
  input2 = f32[1,17,9,9]{3,2,1,0} parameter(1)
  add = f32[1,17,9,9]{3,2,1,0} add(input, input2)
  ROOT copy = f32[1,17,9,9]{1,3,2,0} copy(add)
}
)";

  CheckMoveCopyToOperands(hlo, R"(
// CHECK: [[input:%[^ ]+]] = f32[1,17,9,9]{3,2,1,0} parameter(0)
// CHECK: [[copy1:%[^ ]+]] = f32[1,17,9,9]{1,3,2,0} copy([[input]])
// CHECK: [[input2:%[^ ]+]] = f32[1,17,9,9]{3,2,1,0} parameter(1)
// CHECK: [[copy2:%[^ ]+]] = f32[1,17,9,9]{1,3,2,0} copy([[input2]])
// CHECK: ROOT [[add:%[^ ]+]] = f32[1,17,9,9]{1,3,2,0} add([[copy1]], [[copy2]])
)");
}

TEST_F(MoveCopyToOperandsTest, Concat) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  input = f32[1,17,9,9]{3,2,1,0} parameter(0)
  input2 = f32[5,17,9,9]{3,2,1,0} parameter(1)
  concat = f32[6,17,9,9]{3,2,1,0} concatenate(input, input2), dimensions={0}
  ROOT copy = f32[6,17,9,9]{1,3,2,0} copy(concat)
}
)";

  CheckMoveCopyToOperands(hlo, R"(
// CHECK: [[input:%[^ ]+]] = f32[1,17,9,9]{3,2,1,0} parameter(0)
// CHECK: [[copy1:%[^ ]+]] = f32[1,17,9,9]{1,3,2,0} copy([[input]])
// CHECK: [[input2:%[^ ]+]] = f32[5,17,9,9]{3,2,1,0} parameter(1)
// CHECK: [[copy2:%[^ ]+]] = f32[5,17,9,9]{1,3,2,0} copy([[input2]])
// CHECK: ROOT [[concat:%[^ ]+]] = f32[6,17,9,9]{1,3,2,0} concatenate([[copy1]], [[copy2]]), dimensions={0}
)");
}

TEST_F(MoveCopyToOperandsTest, FoldParameter) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  input = f32[1,17,9,9]{3,2,1,0} parameter(0)
  ROOT copy = f32[1,17,9,9]{1,3,2,0} copy(input)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  module->mutable_config()
      .mutable_debug_options()
      .set_xla_pjrt_allow_auto_layout_in_hlo(true);

  MoveCopyToOperands pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(root->shape().layout(), Layout({1, 3, 2, 0}));
}

TEST_F(MoveCopyToOperandsTest, DoNotFoldParameterWithNonCopyUser) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  input = f32[1,17,9,9]{3,2,1,0} parameter(0)
  copy = f32[1,17,9,9]{1,3,2,0} copy(input)
  sqrt = f32[1,17,9,9]{3,2,1,0} sqrt(input)
  ROOT tuple = (f32[1,17,9,9]{1,3,2,0}, f32[1,17,9,9]{3,2,1,0}) tuple(copy, sqrt)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  module->mutable_config()
      .mutable_debug_options()
      .set_xla_pjrt_allow_auto_layout_in_hlo(true);

  MoveCopyToOperands pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(MoveCopyToOperandsTest, FoldParameterWithMultipleCopyUsersSameLayout) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  input = f32[1,17,9,9]{3,2,1,0} parameter(0)
  copy1 = f32[1,17,9,9]{1,3,2,0} copy(input)
  copy2 = f32[1,17,9,9]{1,3,2,0} copy(input)
  ROOT tuple = (f32[1,17,9,9]{1,3,2,0}, f32[1,17,9,9]{1,3,2,0}) tuple(copy1, copy2)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  module->mutable_config()
      .mutable_debug_options()
      .set_xla_pjrt_allow_auto_layout_in_hlo(true);

  MoveCopyToOperands pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->operand(0), root->operand(1));
  EXPECT_EQ(root->operand(0)->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(root->operand(0)->shape().layout(), Layout({1, 3, 2, 0}));
}

TEST_F(MoveCopyToOperandsTest,
       DoNotFoldParameterWithMultipleCopyUsersDifferentLayout) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  input = f32[1,17,9,9]{3,2,1,0} parameter(0)
  copy1 = f32[1,17,9,9]{1,3,2,0} copy(input)
  copy2 = f32[1,17,9,9]{0,1,2,3} copy(input)
  ROOT tuple = (f32[1,17,9,9]{1,3,2,0}, f32[1,17,9,9]{0,1,2,3}) tuple(copy1, copy2)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  module->mutable_config()
      .mutable_debug_options()
      .set_xla_pjrt_allow_auto_layout_in_hlo(true);

  MoveCopyToOperands pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(MoveCopyToOperandsTest, MoveOverMultipleOps) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  input = f32[1,17,9,9]{3,2,1,0} parameter(0)
  sqrt1 = f32[1,17,9,9]{3,2,1,0} sqrt(input)
  sqrt2 = f32[1,17,9,9]{3,2,1,0} sqrt(sqrt1)
  ROOT copy = f32[1,17,9,9]{1,3,2,0} copy(sqrt2)
}
)";

  CheckMoveCopyToOperands(hlo, R"(
// CHECK: [[input:%[^ ]+]] = f32[1,17,9,9]{3,2,1,0} parameter(0)
// CHECK: [[copy:%[^ ]+]] = f32[1,17,9,9]{1,3,2,0} copy([[input]])
// CHECK: [[sqrt1:%[^ ]+]] = f32[1,17,9,9]{1,3,2,0} sqrt([[copy]])
// CHECK: ROOT [[sqrt2:%[^ ]+]] = f32[1,17,9,9]{1,3,2,0} sqrt([[sqrt1]])
)");
}

TEST_F(MoveCopyToOperandsTest, Deduplication) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  input = f32[1,17,9,9]{3,2,1,0} parameter(0)
  sqrt = f32[1,17,9,9]{3,2,1,0} sqrt(input)
  copy1 = f32[1,17,9,9]{1,3,2,0} copy(sqrt)
  copy2 = f32[1,17,9,9]{1,3,2,0} copy(sqrt)
  ROOT tuple = (f32[1,17,9,9]{1,3,2,0}, f32[1,17,9,9]{1,3,2,0}) tuple(copy1, copy2)
}
)";

  CheckMoveCopyToOperands(hlo, R"(
// CHECK: [[input:%[^ ]+]] = f32[1,17,9,9]{3,2,1,0} parameter(0)
// CHECK: [[copy:%[^ ]+]] = f32[1,17,9,9]{1,3,2,0} copy([[input]])
// CHECK: [[sqrt:%[^ ]+]] = f32[1,17,9,9]{1,3,2,0} sqrt([[copy]])
// CHECK: ROOT [[tuple:%[^ ]+]] = (f32[1,17,9,9]{1,3,2,0}, f32[1,17,9,9]{1,3,2,0}) tuple([[sqrt]], [[sqrt]])
)");
}

TEST_F(MoveCopyToOperandsTest, DoNotFoldParameterWithTransposeUser) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  p0 = f32[2,2]{1,0} parameter(0)
  copy = f32[2,2]{0,1} copy(p0)
  transpose = f32[2,2]{0,1} transpose(p0), dimensions={1,0}
  ROOT tuple = (f32[2,2]{0,1}, f32[2,2]{0,1}) tuple(copy, transpose)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  module->mutable_config()
      .mutable_debug_options()
      .set_xla_pjrt_allow_auto_layout_in_hlo(true);

  MoveCopyToOperands pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(MoveCopyToOperandsTest, DoNotFoldNonParameterEvenWithAutoLayout) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  p0 = f32[2,2]{1,0} parameter(0)
  negate = f32[2,2]{1,0} negate(p0)
  other_user = f32[2,2]{1,0} negate(p0)
  ROOT copy = f32[2,2]{0,1} copy(negate)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  module->mutable_config()
      .mutable_debug_options()
      .set_xla_pjrt_allow_auto_layout_in_hlo(true);

  MoveCopyToOperands pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kNegate);
  EXPECT_EQ(root->shape().layout(), Layout({0, 1}));
  EXPECT_EQ(root->operand(0)->opcode(), HloOpcode::kCopy);
  EXPECT_EQ(root->operand(0)->operand(0)->opcode(), HloOpcode::kParameter);
}

TEST_F(MoveCopyToOperandsTest, MoveCopyAcrossScatter) {
  const char* hlo = R"(
HloModule module

update_op (a: f32[], b: f32[]) -> f32[] {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add = f32[] add(a, b)
}

ENTRY main {
  p0 = f32[2,2,2]{2,1,0} parameter(0)
  indices = s32[2,1]{1,0} constant({{0}, {1}})
  updates = f32[2,2]{1,0} constant({{1.0, 2.0}, {3.0, 4.0}})
  scatter = f32[2,2,2]{2,1,0} scatter(p0, indices, updates),
      update_window_dims={1},
      inserted_window_dims={0,1},
      scatter_dims_to_operand_dims={2},
      index_vector_dim=1,
      to_apply=update_op
  ROOT copy = f32[2,2,2]{0,1,2} copy(scatter)
}
)";

  CheckMoveCopyToOperands(hlo, R"(
// CHECK: [[p0:%[^ ]+]] = f32[2,2,2]{2,1,0} parameter(0)
// CHECK: [[copy:%[^ ]+]] = f32[2,2,2]{0,1,2} copy([[p0]])
// CHECK: [[indices:%[^ ]+]] = s32[2,1]{1,0} constant({{.*}})
// CHECK: [[updates:%[^ ]+]] = f32[2,2]{1,0} constant({{.*}})
// CHECK: ROOT [[scatter:%[^ ]+]] = f32[2,2,2]{0,1,2} scatter([[copy]], [[indices]], [[updates]])
// CHECK-SAME: scatter_dims_to_operand_dims={2}
)");
}

TEST_F(MoveCopyToOperandsTest, DoNotMoveCopyAcrossVariadicScatter) {
  const char* hlo = R"(
HloModule module

update_op (a: f32[], b: f32[], c: f32[], d: f32[]) -> (f32[], f32[]) {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  c = f32[] parameter(2)
  d = f32[] parameter(3)
  add1 = f32[] add(a, c)
  add2 = f32[] add(b, d)
  ROOT tuple = (f32[], f32[]) tuple(add1, add2)
}

ENTRY main {
  p0 = f32[4,4]{1,0} parameter(0)
  p1 = f32[4,4]{1,0} parameter(1)
  indices = s32[2,1]{1,0} constant({{0}, {2}})
  updates0 = f32[2,4]{1,0} constant({{1.0, 2.0, 3.0, 4.0}, {5.0, 6.0, 7.0, 8.0}})
  updates1 = f32[2,4]{1,0} constant({{1.0, 2.0, 3.0, 4.0}, {5.0, 6.0, 7.0, 8.0}})
  scatter = (f32[4,4]{1,0}, f32[4,4]{1,0}) scatter(p0, p1, indices, updates0, updates1),
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1,
      to_apply=update_op
  gte0 = f32[4,4]{1,0} get-tuple-element(scatter), index=0
  ROOT copy = f32[4,4]{0,1} copy(gte0)
}
)";

  CheckMoveCopyToOperands(hlo, std::nullopt);
}

TEST_F(MoveCopyToOperandsTest, MoveCopyAcrossScatterSameRankOperands) {
  const char* hlo = R"(
HloModule module

update_op (a: f32[], b: f32[]) -> f32[] {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add = f32[] add(a, b)
}

ENTRY main {
  p0 = f32[4,4]{1,0} parameter(0)
  indices = s32[2,1]{1,0} constant({{0}, {2}})
  updates = f32[2,4]{1,0} iota(), iota_dimension=1
  scatter = f32[4,4]{1,0} scatter(p0, indices, updates),
      update_window_dims={1},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=1,
      to_apply=update_op
  ROOT copy = f32[4,4]{0,1} copy(scatter)
}
)";

  CheckMoveCopyToOperands(hlo, R"(
// CHECK: [[p0:%[^ ]+]] = f32[4,4]{1,0} parameter(0)
// CHECK: [[copy_p0:%[^ ]+]] = f32[4,4]{0,1} copy([[p0]])
// CHECK: [[indices:%[^ ]+]] = s32[2,1]{1,0} constant({{.*}})
// CHECK: [[copy_indices:%[^ ]+]] = s32[2,1]{0,1} copy([[indices]])
// CHECK: [[updates:%[^ ]+]] = f32[2,4]{1,0} iota()
// CHECK: [[copy_updates:%[^ ]+]] = f32[2,4]{0,1} copy([[updates]])
// CHECK: ROOT [[scatter:%[^ ]+]] = f32[4,4]{0,1} scatter([[copy_p0]], [[copy_indices]], [[copy_updates]])
)");
}

TEST_F(MoveCopyToOperandsTest, MoveCopyAcrossBroadcast) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  p0 = f32[2,3]{0,1} parameter(0)
  broadcast = f32[4,2,5,3]{0,1,2,3} broadcast(p0), dimensions={1,3}
  ROOT copy = f32[4,2,5,3]{3,1,2,0} copy(broadcast)
}
)";

  CheckMoveCopyToOperands(hlo, R"(
// CHECK: [[p0:%[^ ]+]] = f32[2,3]{0,1} parameter(0)
// CHECK: [[copy_p0:%[^ ]+]] = f32[2,3]{1,0} copy([[p0]])
// CHECK: ROOT [[broadcast:%[^ ]+]] = f32[4,2,5,3]{3,1,2,0} broadcast([[copy_p0]]), dimensions={1,3}
)");
}

TEST_F(MoveCopyToOperandsTest, MoveCopyAcrossBroadcastNoCopy) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  p0 = f32[2,3]{1,0} parameter(0)
  broadcast = f32[4,2,5,3]{0,1,2,3} broadcast(p0), dimensions={1,3}
  ROOT copy = f32[4,2,5,3]{3,1,2,0} copy(broadcast)
}
)";

  CheckMoveCopyToOperands(hlo, R"(
// CHECK: [[p0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
// CHECK-NOT: copy([[p0]])
// CHECK: ROOT [[broadcast:%[^ ]+]] = f32[4,2,5,3]{3,1,2,0} broadcast([[p0]]), dimensions={1,3}
)");
}

}  // namespace
}  // namespace xla::gpu
