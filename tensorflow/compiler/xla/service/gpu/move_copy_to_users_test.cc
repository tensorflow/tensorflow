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

#include "tensorflow/compiler/xla/service/gpu/move_copy_to_users.h"

#include <optional>
#include <utility>

#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/tsl/platform/test.h"

namespace xla {
namespace {

class MoveCopyToUsersTest : public HloTestBase {
 public:
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
  ROOT converted = f32[1,4,6,6] slice(copy), slice={[0:1],[0:4],[0:6],[0:6]}
}
)";

  CheckMoveCopyToUsers(hlo, R"(
// CHECK: [[slice_0:%[^ ]+]] = f32[1,4,6,6]{3,2,1,0} slice([[input_1:%[^ ]+]]), slice={[0:1], [0:4], [0:6], [0:6]}
// CHECK-NEXT: ROOT [[copy_1_2:%[^ ]+]] = f32[1,4,6,6]{3,2,1,0} copy([[slice_0]])
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

}  // namespace
}  // namespace xla
