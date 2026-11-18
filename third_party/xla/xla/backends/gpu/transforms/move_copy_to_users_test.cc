/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/backends/gpu/transforms/move_copy_to_users.h"

#include <memory>
#include <optional>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/layout_assignment.h"

namespace xla {
namespace {

class MoveCopyToUsersTest : public HloHardwareIndependentTestBase {
 public:
  MoveCopyToUsersTest()
      : HloHardwareIndependentTestBase(
            /*verifier_layout_sensitive=*/true,
            /*allow_mixed_precision_in_hlo_verifier=*/true,
            LayoutAssignment::InstructionCanChangeLayout) {}
  void CheckMoveCopyToUsers(absl::string_view hlo,
                            std::optional<absl::string_view> expected) {
    RunAndFilecheckHloRewrite(hlo, MoveCopyToUsers{}, expected);
  }
};

TEST_F(MoveCopyToUsersTest, Pad) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  input = s8[1,17,9,9]{3,1,2,0} parameter(0)
  copy = s8[1,17,9,9]{1,3,2,0} copy(input)
  constant = s8[] constant(0)
  ROOT pad = s8[1,32,9,9]{1,3,2,0} pad(copy, constant), padding=0_0x0_15x0_0x0_0
}
)";

  CheckMoveCopyToUsers(hlo, R"(
// CHECK: [[constant_0:%[^ ]+]] = s8[] constant(0)
// CHECK: [[pad_1_1:%[^ ]+]] = s8[1,32,9,9]{3,1,2,0} pad([[input_2:%[^ ]+]], [[constant_0]]), padding=0_0x0_15x0_0x0_0
// CHECK: ROOT [[copy_1_3:%[^ ]+]] = s8[1,32,9,9]{1,3,2,0} copy([[pad_1_1]])
)");
}

TEST_F(MoveCopyToUsersTest, Unary) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  input = f32[1,17,9,9]{3,2,1,0} parameter(0)
  copy = f32[1,17,9,9]{1,3,2,0} copy(input)
  ROOT pad = f32[1,17,9,9]{1,3,2,0} sqrt(copy)
}
)";

  CheckMoveCopyToUsers(hlo, R"(
// CHECK: [[input_0:%[^ ]+]] = f32[1,17,9,9]{3,2,1,0} parameter(0)
// CHECK: [[sqrt_1:%[^ ]+]] = f32[1,17,9,9]{3,2,1,0} sqrt([[input_0]])
// CHECK: ROOT [[copy_1_2:%[^ ]+]] = f32[1,17,9,9]{1,3,2,0} copy([[sqrt_1]])
)");
}

TEST_F(MoveCopyToUsersTest, Reverse) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  input = f32[1,17,9,9]{3,2,1,0} parameter(0)
  copy = f32[1,17,9,9]{1,3,2,0} copy(input)
  ROOT pad = f32[1,17,9,9]{1,3,2,0} reverse(copy), dimensions={1,2}
}
)";

  CheckMoveCopyToUsers(hlo, R"(
// CHECK: [[input_0:%[^ ]+]] = f32[1,17,9,9]{3,2,1,0} parameter(0)
// CHECK: [[sqrt_1:%[^ ]+]] = f32[1,17,9,9]{3,2,1,0} reverse([[input_0]]), dimensions={1,2}
// CHECK: ROOT [[copy_1_2:%[^ ]+]] = f32[1,17,9,9]{1,3,2,0} copy([[sqrt_1]])
)");
}

TEST_F(MoveCopyToUsersTest, Convert) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  input = f32[1,17,9,9]{3,2,1,0} parameter(0)
  copy = f32[1,17,9,9]{1,3,2,0} copy(input)
  ROOT converted = f16[1,17,9,9]{1,3,2,0} convert(copy)
}
)";

  CheckMoveCopyToUsers(hlo, R"(
// CHECK: [[input_0:%[^ ]+]] = f32[1,17,9,9]{3,2,1,0} parameter(0)
// CHECK: [[sqrt_1:%[^ ]+]] = f16[1,17,9,9]{3,2,1,0} convert([[input_0]])
// CHECK: ROOT [[copy_1_2:%[^ ]+]] = f16[1,17,9,9]{1,3,2,0} copy([[sqrt_1]])
)");
}

TEST_F(MoveCopyToUsersTest, Slice) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  input = f32[1,17,9,9]{3,2,1,0} parameter(0)
  copy = f32[1,17,9,9]{1,3,2,0} copy(input)
  ROOT slice = f32[1,4,6,6]{1,3,2,0} slice(copy), slice={[0:1],[0:4],[0:6],[0:6]}
}
)";

  CheckMoveCopyToUsers(hlo, R"(
// CHECK: [[slice_0:%[^ ]+]] = f32[1,4,6,6]{3,2,1,0} slice([[input_1:%[^ ]+]]), slice={[0:1], [0:4], [0:6], [0:6]}
// CHECK-NEXT: ROOT [[copy_1_2:%[^ ]+]] = f32[1,4,6,6]{1,3,2,0} copy([[slice_0]])
)");
}

TEST_F(MoveCopyToUsersTest, DynamicSlice) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  input = f32[1,17,9,9]{3,2,1,0} parameter(0)
  copy = f32[1,17,9,9]{1,3,2,0} copy(input)
  p0 = s32[] parameter(1)
  p1 = s32[] parameter(2)
  p2 = s32[] parameter(3)
  p3 = s32[] parameter(4)
  ROOT ds = f32[1,4,6,6]{1,3,2,0} dynamic-slice(copy, p0, p1, p2, p3), dynamic_slice_sizes={1,4,6,6}
}
)";

  CheckMoveCopyToUsers(hlo, R"(
// CHECK: [[ds:%[^ ]+]] = f32[1,4,6,6]{3,2,1,0} dynamic-slice({{.*}}), dynamic_slice_sizes={1,4,6,6}
// CHECK-NEXT: ROOT {{.*}} = f32[1,4,6,6]{1,3,2,0} copy([[ds]])
)");
}

TEST_F(MoveCopyToUsersTest, ReduceWindow) {
  const char* hlo = R"(
HloModule R2Window

mul {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT mul = f32[] multiply(lhs, rhs)
}

ENTRY R2Window {
  operand = f32[256,384]{1,0} parameter(0)
  c = f32[256,384]{0,1} copy(operand)
  constant = f32[] constant(1)
  ROOT reduce-window = f32[256,384]{0,1} reduce-window(c, constant), window={size=2x3 pad=0_1x1_1}, to_apply=mul
}
)";

  CheckMoveCopyToUsers(hlo, R"(
// CHECK: [[reduce_window_1_0:%[^ ]+]] = f32[256,384]{1,0} reduce-window([[operand_1:%[^ ]+]], [[constant_2:%[^ ]+]]), window={size=2x3 pad=0_1x1_1}, to_apply=[[mul_3:%[^ ]+]]
// CHECK-NEXT: ROOT [[copy_4:%[^ ]+]] = f32[256,384]{0,1} copy([[reduce_window_1_0]])
)");
}

TEST_F(MoveCopyToUsersTest, Reduce) {
  const char* hlo = R"(
HloModule R2

mul {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT mul = f32[] multiply(lhs, rhs)
}

ENTRY R2 {
  operand = f32[256,384,10]{2,1,0} parameter(0)
  c = f32[256,384,10]{0,1,2} copy(operand)
  constant = f32[] constant(1)
  ROOT reduce = f32[384,10]{0,1} reduce(c, constant), dimensions={0}, to_apply=mul
}
)";

  CheckMoveCopyToUsers(hlo, R"(
// CHECK: [[operand:%[^ ]+]] = f32[256,384,10]{2,1,0} parameter(0)
// CHECK: ROOT [[reduce:%[^ ]+]] = f32[384,10]{0,1} reduce([[operand]], [[constant_2:%[^ ]+]]), dimensions={0}, to_apply=[[mul_3:%[^ ]+]]
)");
}

TEST_F(MoveCopyToUsersTest, Binary) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  input = f32[1,17,9,9]{3,2,1,0} parameter(0)
  input2 = f32[1,17,9,9]{3,2,1,0} parameter(1)
  copy = f32[1,17,9,9]{1,3,2,0} copy(input)
  copy2 = f32[1,17,9,9]{1,3,2,0} copy(input2)
  ROOT add = f32[1,17,9,9]{1,3,2,0} add(copy, copy2)
}
)";

  CheckMoveCopyToUsers(hlo, R"(
// CHECK: [[input_0:%[^ ]+]] = f32[1,17,9,9]{3,2,1,0} parameter(0)
// CHECK: [[input2_1:%[^ ]+]] = f32[1,17,9,9]{3,2,1,0} parameter(1)
// CHECK: [[add_1_2:%[^ ]+]] = f32[1,17,9,9]{3,2,1,0} add([[input_0]], [[input2_1]])
// CHECK: ROOT [[copy_1_3:%[^ ]+]] = f32[1,17,9,9]{1,3,2,0} copy([[add_1_2]])
)");
}

TEST_F(MoveCopyToUsersTest, BinaryDifferentLayoutNoChange) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  input = f32[1,17,9,9]{3,2,0,1} parameter(0)
  input2 = f32[1,17,9,9]{3,2,1,0} parameter(1)
  copy = f32[1,17,9,9]{1,3,2,0} copy(input)
  copy2 = f32[1,17,9,9]{1,3,2,0} copy(input2)
  ROOT add = f32[1,17,9,9]{1,3,2,0} add(copy, copy2)
}
)";

  CheckMoveCopyToUsers(hlo, std::nullopt);
}

TEST_F(MoveCopyToUsersTest, Concat) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  input = f32[1,17,9,9]{3,2,1,0} parameter(0)
  input2 = f32[5,17,9,9]{3,2,1,0} parameter(1)
  copy = f32[1,17,9,9]{1,3,2,0} copy(input)
  copy2 = f32[5,17,9,9]{1,3,2,0} copy(input2)
  ROOT add = f32[6,17,9,9]{1,3,2,0} concatenate(copy, copy2), dimensions={0}
}
)";

  CheckMoveCopyToUsers(hlo, R"(
// CHECK: [[input_0:%[^ ]+]] = f32[1,17,9,9]{3,2,1,0} parameter(0)
// CHECK: [[input2_1:%[^ ]+]] = f32[5,17,9,9]{3,2,1,0} parameter(1)
// CHECK: [[concat:%[^ ]+]] = f32[6,17,9,9]{3,2,1,0} concatenate([[input_0]], [[input2_1]])
// CHECK: ROOT [[copy_1_3:%[^ ]+]] = f32[6,17,9,9]{1,3,2,0} copy([[concat]])
)");
}

TEST_F(MoveCopyToUsersTest, ConcatDifferentLayoutNoChange) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  input = f32[1,17,9,9]{3,2,0,1} parameter(0)
  input2 = f32[1,17,9,9]{3,2,1,0} parameter(1)
  copy = f32[1,17,9,9]{1,3,2,0} copy(input)
  copy2 = f32[1,17,9,9]{1,3,2,0} copy(input2)
  ROOT add = f32[2,17,9,9]{1,3,2,0} concatenate(copy, copy2), dimensions={0}
}
)";

  CheckMoveCopyToUsers(hlo, std::nullopt);
}

TEST_F(MoveCopyToUsersTest, ConvergesInSecondStep) {
  const char* const kHloString = R"(
HloModule ConvergeTest

ENTRY main {
  p0 = c64[3,3]{1,0} parameter(0)
  copy = c64[3,3]{0,1} copy(p0)
  real = f32[3,3]{0,1} real(copy)
  ROOT res = f32[3,3]{1,0} copy(real)
}
)";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloString));

  MoveCopyToUsers pass;
  ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get()));
  EXPECT_TRUE(changed);
  ASSERT_OK_AND_ASSIGN(changed, pass.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(MoveCopyToUsersTest, ConvertConsumedByConvolutionNoChange) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  input = f32[1,9,9,17]{3,2,1,0} parameter(0)
  copy = f32[1,9,9,17]{3,2,1,0} copy(input)
  converted = f16[1,9,9,17]{3,2,1,0} convert(copy)
  filter = f16[3,3,17,32]{3,2,1,0} parameter(1)
  ROOT conv = f16[1,9,9,32]{3,2,1,0} convolution(converted, filter), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f
}
)";

  CheckMoveCopyToUsers(hlo, std::nullopt);
}

// Returns a module with a while loop that runs `trip_count` iterations. The
// loop state is (i, stack, acc), where stack is a copy of a parameter.
// `body_instructions` computes `layer` from `stack` and `i`, and `passed_on` is
// the instruction the body passes on as stack. `after_loop` is added to the
// entry computation after the loop.
std::string WhileLoopOverCopiedStack(absl::string_view body_instructions,
                                     absl::string_view passed_on = "stack",
                                     int trip_count = 8,
                                     absl::string_view after_loop = "") {
  return absl::Substitute(R"(
HloModule module

body {
  p = (s32[], f32[8,16,32]{1,0,2}, f32[1,16,32]{1,0,2}) parameter(0)
  i = s32[] get-tuple-element(p), index=0
  stack = f32[8,16,32]{1,0,2} get-tuple-element(p), index=1
  zero = s32[] constant(0)
  $0
  one = s32[] constant(1)
  next = s32[] add(i, one)
  ROOT result = (s32[], f32[8,16,32]{1,0,2}, f32[1,16,32]{1,0,2}) tuple(next, $1, layer)
}

condition {
  p = (s32[], f32[8,16,32]{1,0,2}, f32[1,16,32]{1,0,2}) parameter(0)
  i = s32[] get-tuple-element(p), index=0
  limit = s32[] constant($2)
  ROOT lt = pred[] compare(i, limit), direction=LT
}

ENTRY main {
  weights = f32[8,16,32]{2,1,0} parameter(0)
  acc = f32[1,16,32]{1,0,2} parameter(1)
  copy = f32[8,16,32]{1,0,2} copy(weights)
  zero = s32[] constant(0)
  init = (s32[], f32[8,16,32]{1,0,2}, f32[1,16,32]{1,0,2}) tuple(zero, copy, acc)
  loop = (s32[], f32[8,16,32]{1,0,2}, f32[1,16,32]{1,0,2}) while(init), condition=condition, body=body
  $3
  ROOT out = f32[1,16,32]{1,0,2} get-tuple-element(loop), index=2
}
)",
                          body_instructions, passed_on, trip_count, after_loop);
}

// Returns `hlo`, a module from WhileLoopOverCopiedStack, with the initial value
// of the induction variable of the loop replaced by a parameter, so that the
// trip count of the loop can not be computed.
std::string WithUnknownInitialInductionVariable(absl::string_view hlo) {
  return absl::StrReplaceAll(hlo, {{"zero = s32[] constant(0)\n  init",
                                    "zero = s32[] parameter(2)\n  init"}});
}

// Returns `hlo`, a module from WhileLoopOverCopiedStack, with the loop
// annotated with `trip_count` as its known trip count.
std::string WithKnownTripCount(absl::string_view hlo,
                               absl::string_view trip_count) {
  return absl::StrReplaceAll(
      hlo, {{"condition=condition, body=body",
             absl::StrCat("condition=condition, body=body, "
                          "backend_config={\"known_trip_count\":{\"n\":\"",
                          trip_count, "\"}}")}});
}

constexpr absl::string_view kDynamicSliceOfStack =
    "layer = f32[1,16,32]{1,0,2} dynamic-slice(stack, i, zero, zero), "
    "dynamic_slice_sizes={1,16,32}";

TEST_F(MoveCopyToUsersTest, DynamicSliceInWhileBody) {
  // The loop carries the stack in the layout of the parameter, and only the
  // layer that is sliced in the body is copied.
  CheckMoveCopyToUsers(WhileLoopOverCopiedStack(kDynamicSliceOfStack), R"(
// CHECK:       body
// CHECK:         [[p:%[^ ]+]] = (s32[], f32[8,16,32]{2,1,0}, f32[1,16,32]{1,0,2}) parameter(0)
// CHECK:         [[stack:%[^ ]+]] = f32[8,16,32]{2,1,0} get-tuple-element([[p]]), index=1
// CHECK:         [[slice:%[^ ]+]] = f32[1,16,32]{2,1,0} dynamic-slice([[stack]], {{.*}}), dynamic_slice_sizes={1,16,32}
// CHECK:         [[layer:%[^ ]+]] = f32[1,16,32]{1,0,2} copy([[slice]])
// CHECK:         ROOT {{.*}} = (s32[], f32[8,16,32]{2,1,0}, f32[1,16,32]{1,0,2}) tuple({{.*}}, [[stack]], [[layer]])
// CHECK:       condition
// CHECK:         (s32[], f32[8,16,32]{2,1,0}, f32[1,16,32]{1,0,2}) parameter(0)
// CHECK:       ENTRY
// CHECK:         [[weights:%[^ ]+]] = f32[8,16,32]{2,1,0} parameter(0)
// CHECK-NOT:     copy(
// CHECK:         [[init:%[^ ]+]] = (s32[], f32[8,16,32]{2,1,0}, f32[1,16,32]{1,0,2}) tuple({{.*}}, [[weights]], {{.*}})
// CHECK:         (s32[], f32[8,16,32]{2,1,0}, f32[1,16,32]{1,0,2}) while([[init]])
)");
}

TEST_F(MoveCopyToUsersTest, SliceInWhileBody) {
  // A loop with one iteration copies the slice once.
  CheckMoveCopyToUsers(
      WhileLoopOverCopiedStack("layer = f32[1,16,32]{1,0,2} slice(stack), "
                               "slice={[3:4], [0:16], [0:32]}",
                               "stack", /*trip_count=*/1),
      R"(
// CHECK:       body
// CHECK:         [[stack:%[^ ]+]] = f32[8,16,32]{2,1,0} get-tuple-element({{.*}}), index=1
// CHECK:         [[slice:%[^ ]+]] = f32[1,16,32]{2,1,0} slice([[stack]]), slice={[3:4], [0:16], [0:32]}
// CHECK:         f32[1,16,32]{1,0,2} copy([[slice]])
// CHECK:       ENTRY
// CHECK-NOT:     copy(
)");
}

TEST_F(MoveCopyToUsersTest, SlicesInWhileBodyAndAfterWhileLoop) {
  // The body reads two layers in each of 7 iterations, and one more layer is
  // read after the loop. Each of the slices is copied: 15 layers in total,
  // which is not more than twice the 8 layers of the stack.
  CheckMoveCopyToUsers(
      WhileLoopOverCopiedStack(
          absl::StrCat(kDynamicSliceOfStack, "\n",
                       "other = f32[1,16,32]{1,0,2} slice(stack), "
                       "slice={[0:1], [0:16], [0:32]}"),
          "stack", /*trip_count=*/7,
          "stack_after = f32[8,16,32]{1,0,2} get-tuple-element(loop), index=1\n"
          "last = f32[1,16,32]{1,0,2} slice(stack_after), "
          "slice={[7:8], [0:16], [0:32]}"),
      R"(
// CHECK:       body
// CHECK:         [[stack:%[^ ]+]] = f32[8,16,32]{2,1,0} get-tuple-element({{.*}}), index=1
// CHECK-DAG:     [[slice:%[^ ]+]] = f32[1,16,32]{2,1,0} dynamic-slice([[stack]], {{.*}}), dynamic_slice_sizes={1,16,32}
// CHECK-DAG:     f32[1,16,32]{1,0,2} copy([[slice]])
// CHECK-DAG:     [[other:%[^ ]+]] = f32[1,16,32]{2,1,0} slice([[stack]]), slice={[0:1], [0:16], [0:32]}
// CHECK-DAG:     f32[1,16,32]{1,0,2} copy([[other]])
// CHECK:       ENTRY
// CHECK-NOT:     f32[8,16,32]{1,0,2}
// CHECK:         [[loop:%[^ ]+]] = (s32[], f32[8,16,32]{2,1,0}, f32[1,16,32]{1,0,2}) while(
// CHECK:         [[after:%[^ ]+]] = f32[8,16,32]{2,1,0} get-tuple-element([[loop]]), index=1
// CHECK:         [[last:%[^ ]+]] = f32[1,16,32]{2,1,0} slice([[after]]), slice={[7:8], [0:16], [0:32]}
// CHECK:         f32[1,16,32]{1,0,2} copy([[last]])
)");
}

TEST_F(MoveCopyToUsersTest, CopyWithOtherUserIsMovedIntoWhileBody) {
  // The copy is also read outside of the loop. The loop carries the stack in
  // the layout of the parameter, and the copy is kept for its other user.
  CheckMoveCopyToUsers(
      WhileLoopOverCopiedStack(kDynamicSliceOfStack, "stack", /*trip_count=*/8,
                               "flat = f32[4096]{0} reshape(copy)"),
      R"(
// CHECK:       body
// CHECK:         [[stack:%[^ ]+]] = f32[8,16,32]{2,1,0} get-tuple-element({{.*}}), index=1
// CHECK:         [[slice:%[^ ]+]] = f32[1,16,32]{2,1,0} dynamic-slice([[stack]], {{.*}}), dynamic_slice_sizes={1,16,32}
// CHECK:         f32[1,16,32]{1,0,2} copy([[slice]])
// CHECK:       ENTRY
// CHECK:         [[weights:%[^ ]+]] = f32[8,16,32]{2,1,0} parameter(0)
// CHECK-DAG:     [[copy:%[^ ]+]] = f32[8,16,32]{1,0,2} copy([[weights]])
// CHECK-DAG:     f32[4096]{0} reshape([[copy]])
// CHECK-DAG:     (s32[], f32[8,16,32]{2,1,0}, f32[1,16,32]{1,0,2}) tuple({{.*}}, [[weights]], {{.*}})
)");
}

TEST_F(MoveCopyToUsersTest, SliceCopiesTwiceTheArrayInWhileBody) {
  // The slice copies one layer in each of 16 iterations, which is twice the 8
  // layers of the stack.
  CheckMoveCopyToUsers(
      WhileLoopOverCopiedStack("layer = f32[1,16,32]{1,0,2} slice(stack), "
                               "slice={[3:4], [0:16], [0:32]}",
                               "stack", /*trip_count=*/16),
      R"(
// CHECK:       body
// CHECK:         [[stack:%[^ ]+]] = f32[8,16,32]{2,1,0} get-tuple-element({{.*}}), index=1
// CHECK:         [[slice:%[^ ]+]] = f32[1,16,32]{2,1,0} slice([[stack]]), slice={[3:4], [0:16], [0:32]}
// CHECK:         f32[1,16,32]{1,0,2} copy([[slice]])
// CHECK:       ENTRY
// CHECK-NOT:     copy(
)");
}

TEST_F(MoveCopyToUsersTest, SliceCopiesMoreThanTwiceTheArrayNoChange) {
  // The slice copies one layer in each of 17 iterations, which is more than
  // twice the 8 layers of the stack.
  CheckMoveCopyToUsers(
      WhileLoopOverCopiedStack("layer = f32[1,16,32]{1,0,2} slice(stack), "
                               "slice={[3:4], [0:16], [0:32]}",
                               "stack", /*trip_count=*/17),
      std::nullopt);
}

TEST_F(MoveCopyToUsersTest, SlicesTogetherCopyMoreThanTwiceTheArrayNoChange) {
  // The body reads two layers in each of 8 iterations, and one more layer is
  // read after the loop: 17 layers in total, which is more than twice the 8
  // layers of the stack.
  CheckMoveCopyToUsers(
      WhileLoopOverCopiedStack(
          absl::StrCat(kDynamicSliceOfStack, "\n",
                       "other = f32[1,16,32]{1,0,2} slice(stack), "
                       "slice={[0:1], [0:16], [0:32]}"),
          "stack", /*trip_count=*/8,
          "stack_after = f32[8,16,32]{1,0,2} get-tuple-element(loop), index=1\n"
          "last = f32[1,16,32]{1,0,2} slice(stack_after), "
          "slice={[7:8], [0:16], [0:32]}"),
      std::nullopt);
}

TEST_F(MoveCopyToUsersTest, WhileBodyReadsWholeArrayNoChange) {
  CheckMoveCopyToUsers(WhileLoopOverCopiedStack(absl::StrCat(
                           kDynamicSliceOfStack, "\n",
                           "negated = f32[8,16,32]{1,0,2} negate(stack)")),
                       std::nullopt);
}

TEST_F(MoveCopyToUsersTest, WhileBodyUpdatesArrayNoChange) {
  CheckMoveCopyToUsers(
      WhileLoopOverCopiedStack(
          absl::StrCat(kDynamicSliceOfStack, "\n",
                       "updated = f32[8,16,32]{1,0,2} dynamic-update-slice("
                       "stack, layer, i, zero, zero)"),
          "updated"),
      std::nullopt);
}

TEST_F(MoveCopyToUsersTest, UnknownTripCountNoChange) {
  // The number of layers the body copies is not known.
  CheckMoveCopyToUsers(WithUnknownInitialInductionVariable(
                           WhileLoopOverCopiedStack(kDynamicSliceOfStack)),
                       std::nullopt);
}

TEST_F(MoveCopyToUsersTest, TripCountFromBackendConfig) {
  // The trip count can not be computed, but the loop is annotated with it.
  CheckMoveCopyToUsers(
      WithKnownTripCount(WithUnknownInitialInductionVariable(
                             WhileLoopOverCopiedStack(kDynamicSliceOfStack)),
                         "8"),
      R"(
// CHECK:       body
// CHECK:         [[stack:%[^ ]+]] = f32[8,16,32]{2,1,0} get-tuple-element({{.*}}), index=1
// CHECK:         [[slice:%[^ ]+]] = f32[1,16,32]{2,1,0} dynamic-slice([[stack]], {{.*}}), dynamic_slice_sizes={1,16,32}
// CHECK:         f32[1,16,32]{1,0,2} copy([[slice]])
// CHECK:       ENTRY
// CHECK-NOT:     copy(
)");
}

TEST_F(MoveCopyToUsersTest, LargeTripCountNoChange) {
  // The size of the slices of all 2^62 iterations does not fit into 64 bits.
  CheckMoveCopyToUsers(
      WithKnownTripCount(WithUnknownInitialInductionVariable(
                             WhileLoopOverCopiedStack(kDynamicSliceOfStack)),
                         "4611686018427387904"),
      std::nullopt);
}

TEST_F(MoveCopyToUsersTest, WhileConditionReadsArrayNoChange) {
  CheckMoveCopyToUsers(
      absl::StrReplaceAll(
          WhileLoopOverCopiedStack(kDynamicSliceOfStack),
          {{"  limit = s32[] constant(",
            "  stack = f32[8,16,32]{1,0,2} get-tuple-element(p), index=1\n"
            "  first = f32[1,16,32]{1,0,2} slice(stack), "
            "slice={[0:1], [0:16], [0:32]}\n"
            "  limit = s32[] constant("}}),
      std::nullopt);
}

TEST_F(MoveCopyToUsersTest, WhileBodyCalledByTwoLoopsNoChange) {
  // The layout of the stack can not be changed in a body and a condition that
  // are shared with another loop.
  CheckMoveCopyToUsers(
      WhileLoopOverCopiedStack(
          kDynamicSliceOfStack, "stack", /*trip_count=*/8,
          "init2 = (s32[], f32[8,16,32]{1,0,2}, f32[1,16,32]{1,0,2}) "
          "tuple(zero, copy, acc)\n"
          "loop2 = (s32[], f32[8,16,32]{1,0,2}, f32[1,16,32]{1,0,2}) "
          "while(init2), condition=condition, body=body"),
      std::nullopt);
}

TEST_F(MoveCopyToUsersTest, WhileBodyReadsReturnedStateNoChange) {
  // The state the body returns is read in the body.
  CheckMoveCopyToUsers(
      absl::StrReplaceAll(
          WhileLoopOverCopiedStack(kDynamicSliceOfStack),
          {{"tuple(next, stack, layer)\n",
            "tuple(next, stack, layer)\n"
            "  returned = f32[8,16,32]{1,0,2} get-tuple-element(result), "
            "index=1\n"}}),
      std::nullopt);
}

TEST_F(MoveCopyToUsersTest, ArrayReadAfterWhileLoopNoChange) {
  CheckMoveCopyToUsers(
      WhileLoopOverCopiedStack(
          kDynamicSliceOfStack, "stack", /*trip_count=*/8,
          "stack_after = f32[8,16,32]{1,0,2} get-tuple-element(loop), index=1\n"
          "negated = f32[8,16,32]{1,0,2} negate(stack_after)"),
      std::nullopt);
}

TEST_F(MoveCopyToUsersTest, NegativeTripCountNoChange) {
  CheckMoveCopyToUsers(
      WithKnownTripCount(WithUnknownInitialInductionVariable(
                             WhileLoopOverCopiedStack(kDynamicSliceOfStack)),
                         "-8"),
      std::nullopt);
}

TEST_F(MoveCopyToUsersTest, CopyToOtherMemorySpaceNoChange) {
  // The copy changes the memory space, so it is not a layout choice.
  CheckMoveCopyToUsers(
      absl::StrReplaceAll(
          WhileLoopOverCopiedStack(kDynamicSliceOfStack),
          {{"f32[8,16,32]{1,0,2}", "f32[8,16,32]{1,0,2:S(5)}"}}),
      std::nullopt);
}

TEST_F(MoveCopyToUsersTest, CopyToPackedLayoutNoChange) {
  // The copy packs the elements, so it is not a layout choice.
  CheckMoveCopyToUsers(
      absl::StrReplaceAll(WhileLoopOverCopiedStack(kDynamicSliceOfStack),
                          {{"f32[", "s4["}, {"{1,0,2}", "{1,0,2:E(4)}"}}),
      std::nullopt);
}

TEST_F(MoveCopyToUsersTest, CopyWithControlDependencyIsMovedIntoWhileBody) {
  // The control dependency of the removed copy must not make the removal fail.
  CheckMoveCopyToUsers(
      absl::StrReplaceAll(
          WhileLoopOverCopiedStack(kDynamicSliceOfStack),
          {{"copy(weights)", "copy(weights), control-predecessors={acc}"}}),
      R"(
// CHECK:       ENTRY
// CHECK-NOT:     copy(
// CHECK:         while(
)");
}

}  // namespace
}  // namespace xla
