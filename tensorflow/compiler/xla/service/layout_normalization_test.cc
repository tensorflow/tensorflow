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

#include "tensorflow/compiler/xla/service/layout_normalization.h"

#include <optional>
#include <utility>

#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/tsl/platform/test.h"

namespace xla {
namespace {

class LayoutNormalizationTest : public HloTestBase {
 public:
  void CheckLayoutNormalization(absl::string_view hlo,
                                std::optional<absl::string_view> expected) {
    RunAndFilecheckHloRewrite(hlo, LayoutNormalization{}, expected);
  }
};

TEST_F(LayoutNormalizationTest, TestDefault) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  p = f32[5,4]{0,1} parameter(0)
  ROOT o = f32[5,4]{0,1} abs(p)
}
)";
  CheckLayoutNormalization(hlo, R"(
// CHECK:  [[p_0:%[^ ]+]] = f32[5,4]{0,1} parameter(0)
// CHECK:  [[bitcast_1:%[^ ]+]] = f32[4,5]{1,0} bitcast([[p_0]])
// CHECK:  [[abs_2:%[^ ]+]] = f32[4,5]{1,0} abs([[bitcast_1]])
// CHECK:  ROOT [[bitcast_2_3:%[^ ]+]] = f32[5,4]{0,1} bitcast([[abs_2]])
)");
}

TEST_F(LayoutNormalizationTest, TestUnary) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  p = f32[5,4]{0,1} parameter(0)
  a = f32[5,4]{0,1} abs(p)
  ROOT b = f32[5,4]{0,1} sqrt(a)
}
)";
  CheckLayoutNormalization(hlo, R"(
// CHECK:  [[p_0:%[^ ]+]] = f32[5,4]{0,1} parameter(0)
// CHECK:  [[bitcast_1:%[^ ]+]] = f32[4,5]{1,0} bitcast([[p_0]])
// CHECK:  [[abs_2:%[^ ]+]] = f32[4,5]{1,0} abs([[bitcast_1]])
// CHECK:  [[sqrt_3:%[^ ]+]] = f32[4,5]{1,0} sqrt([[abs_2]])
// CHECK:  ROOT [[bitcast_3_4:%[^ ]+]] = f32[5,4]{0,1} bitcast([[sqrt_3]])
)");
}

TEST_F(LayoutNormalizationTest, TestUnaryDegenerateDimensions) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  p = f32[5,1,4,1]{0,1,2,3} parameter(0)
  ROOT o = f32[5,1,4,1]{0,1,2,3} abs(p)
}
)";
  CheckLayoutNormalization(hlo, R"(
// CHECK:  [[p_0:%[^ ]+]] = f32[5,1,4,1]{0,1,2,3} parameter(0)
// CHECK:  [[bitcast_1:%[^ ]+]] = f32[4,5]{1,0} bitcast([[p_0]])
// CHECK:  [[abs_2:%[^ ]+]] = f32[4,5]{1,0} abs([[bitcast_1]])
// CHECK:  ROOT [[bitcast_2_3:%[^ ]+]] = f32[5,1,4,1]{0,1,2,3} bitcast([[abs_2]])
)");
}

TEST_F(LayoutNormalizationTest, TestBinary) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  a = f32[5,4]{0,1} parameter(0)
  b = f32[5,4]{0,1} parameter(1)
  c = add(a, b)
  ROOT out = sqrt(c)
}
)";

  CheckLayoutNormalization(hlo, R"(
// CHECK:  [[a_0:%[^ ]+]] = f32[5,4]{0,1} parameter(0)
// CHECK:  [[bitcast_1:%[^ ]+]] = f32[4,5]{1,0} bitcast([[a_0]])
// CHECK:  [[b_2:%[^ ]+]] = f32[5,4]{0,1} parameter(1)
// CHECK:  [[bitcast_2_3:%[^ ]+]] = f32[4,5]{1,0} bitcast([[b_2]])
// CHECK:  [[add_4:%[^ ]+]] = f32[4,5]{1,0} add([[bitcast_1]], [[bitcast_2_3]])
// CHECK:  [[sqrt_5:%[^ ]+]] = f32[4,5]{1,0} sqrt([[add_4]])
// CHECK:  ROOT [[bitcast_5_6:%[^ ]+]] = f32[5,4]{0,1} bitcast([[sqrt_5]])
)");
}

TEST_F(LayoutNormalizationTest, Reshape) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  a = f32[5,4]{0,1} parameter(0)
  ROOT b = f32[5,2,2]{0,2,1} reshape(a)
})";

  CheckLayoutNormalization(hlo, R"(
// CHECK:  [[a_0:%[^ ]+]] = f32[5,4]{0,1} parameter(0)
// CHECK:  [[bitcast_1:%[^ ]+]] = f32[4,5]{1,0} bitcast([[a_0]])
// CHECK:  [[reshape_2:%[^ ]+]] = f32[2,2,5]{2,1,0} reshape([[bitcast_1]])
// CHECK:  ROOT [[bitcast_2_3:%[^ ]+]] = f32[5,2,2]{0,2,1} bitcast([[reshape_2]])
)");
}

TEST_F(LayoutNormalizationTest, Transpose) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  a = f32[5,4]{1,0} parameter(0)
  t = f32[4,5]{0,1} transpose(a), dimensions={1,0}
  ROOT out = abs(t)
}
)";

  CheckLayoutNormalization(hlo, R"(
// CHECK: [[a_0:%[^ ]+]] = f32[5,4]{1,0} parameter(0)
// CHECK: [[bitcast_1:%[^ ]+]] = f32[5,4]{1,0} bitcast([[a_0]])
// CHECK: [[abs_2:%[^ ]+]] = f32[5,4]{1,0} abs([[bitcast_1]])
// CHECK: ROOT [[bitcast_3_3:%[^ ]+]] = f32[4,5]{0,1} bitcast([[abs_2]])
)");
}

TEST_F(LayoutNormalizationTest, PhysicalTranspose) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  p = f64[3,4,5]{0,1,2} parameter(0)
  t = f32[5,4,3]{2,0,1} transpose(p), dimensions={2,1,0}
  ROOT out = abs(t)
}
)";

  CheckLayoutNormalization(hlo, R"(
// CHECK: [[bitcast_0:%[^ ]+]] = f64[5,4,3]{2,1,0} bitcast([[p_1:%[^ ]+]])
// CHECK: [[transpose_2:%[^ ]+]] = f32[4,5,3]{2,1,0} transpose([[bitcast_0]]), dimensions={1,0,2}
// CHECK: [[abs_3:%[^ ]+]] = f32[4,5,3]{2,1,0} abs([[transpose_2]])
)");
}

TEST_F(LayoutNormalizationTest, PhysicalTransposeDegenerate) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  p = f32[3,4,5,1]{0,1,2,3} parameter(0)
  t = f32[5,1,4,3]{3,2,0,1} transpose(p), dimensions={2,3,1,0}
  ROOT out = abs(t)
}
)";

  CheckLayoutNormalization(hlo, R"(
// CHECK: [[bitcast_0:%[^ ]+]] = f32[5,4,3]{2,1,0} bitcast([[p_1:%[^ ]+]])
// CHECK: [[transpose_2:%[^ ]+]] = f32[5,4,3]{2,1,0} transpose([[bitcast_0]]), dimensions={0,1,2}
// CHECK: [[abs_3:%[^ ]+]] = f32[5,4,3]{2,1,0} abs([[transpose_2]])
)");
}

TEST_F(LayoutNormalizationTest, Copy) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  p = f32[3,4,5]{0,1,2} parameter(0)
  t = f32[3,4,5]{2,1,0} copy(p)
  ROOT out = abs(t)
}
)";

  CheckLayoutNormalization(hlo, R"(
// CHECK: [[p_0:%[^ ]+]] = f32[3,4,5]{0,1,2} parameter(0)
// CHECK: [[bitcast_1:%[^ ]+]] = f32[5,4,3]{2,1,0} bitcast([[p_0]])
// CHECK: [[transpose_2:%[^ ]+]] = f32[3,4,5]{2,1,0} transpose([[bitcast_1]]), dimensions={2,1,0}
// CHECK: [[abs_3:%[^ ]+]] = f32[3,4,5]{2,1,0} abs([[transpose_2]])
)");
}

TEST_F(LayoutNormalizationTest, CopyDegenerate) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  p = f32[3,1,4,1,5]{0,1,2,3,4} parameter(0)
  t = f32[3,1,4,1,5]{4,3,2,1,0} copy(p)
  ROOT out = abs(t)
}
)";

  CheckLayoutNormalization(hlo, R"(
// CHECK: [[p_0:%[^ ]+]] = f32[3,1,4,1,5]{0,1,2,3,4} parameter(0)
// CHECK: [[bitcast_1:%[^ ]+]] = f32[5,4,3]{2,1,0} bitcast([[p_0]])
// CHECK: [[transpose_2:%[^ ]+]] = f32[3,4,5]{2,1,0} transpose([[bitcast_1]]), dimensions={2,1,0}
// CHECK: [[abs_3:%[^ ]+]] = f32[3,4,5]{2,1,0} abs([[transpose_2]])
)");
}

TEST_F(LayoutNormalizationTest, Broadcast) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  a = f32[4,5]{0,1} parameter(0)
  b = f32[4,3,2,5]{0,1,2,3} broadcast(a), dimensions={0,3}
  ROOT out = abs(b)
}
)";

  CheckLayoutNormalization(hlo, R"(
// CHECK: [[broadcast_0:%[^ ]+]] = f32[5,2,3,4]{3,2,1,0} broadcast([[bitcast_1:%[^ ]+]]), dimensions={0,3}
// CHECK: [[abs_2:%[^ ]+]] = f32[5,2,3,4]{3,2,1,0} abs([[broadcast_0]])
)");
}

TEST_F(LayoutNormalizationTest, BroadcastCustomOutputLayout) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  a = f32[2,3]{1,0} parameter(0)
  b = f32[2,4,3]{1,2,0} broadcast(a), dimensions={0,2}
  ROOT out = abs(b)
}
)";

  CheckLayoutNormalization(hlo, R"(
// CHECK: [[a_0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
// CHECK: [[bitcast_1:%[^ ]+]] = f32[2,3]{1,0} bitcast([[a_0]])
// CHECK: [[broadcast_2:%[^ ]+]] = f32[2,3,4]{2,1,0} broadcast([[bitcast_1]]), dimensions={0,1}
// CHECK: [[abs_3:%[^ ]+]] = f32[2,3,4]{2,1,0} abs([[broadcast_2]])
// CHECK: ROOT [[bitcast_3_4:%[^ ]+]] = f32[2,4,3]{1,2,0} bitcast([[abs_3]])
)");
}

TEST_F(LayoutNormalizationTest, BroadcastCustomOutputLayoutWithDegenerate) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  a = f32[9]{0} parameter(0)
  b = f32[2,1,4,9]{2,0,1,3} broadcast(a), dimensions={3}
  ROOT out = abs(b)
}
)";

  CheckLayoutNormalization(hlo, R"(
// CHECK: [[bitcast_0:%[^ ]+]] = f32[9]{0} bitcast([[a_1:%[^ ]+]])
// CHECK: [[broadcast_2:%[^ ]+]] = f32[9,2,4]{2,1,0} broadcast([[bitcast_0]]), dimensions={0}
// CHECK: [[abs_3:%[^ ]+]] = f32[9,2,4]{2,1,0} abs([[broadcast_2]])
// CHECK: ROOT [[bitcast_3_4:%[^ ]+]] = f32[2,1,4,9]{2,0,1,3} bitcast([[abs_3]])
)");
}

TEST_F(LayoutNormalizationTest, BroadcastWithDegenerate) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  a = f32[1,4,5]{0,1,2} parameter(0)
  b = f32[1,4,3,1,2,5,1]{0,1,2,3,4,5,6} broadcast(a), dimensions={0,1,5}
  ROOT out = abs(b)
}
)";

  CheckLayoutNormalization(hlo, R"(
// CHECK:  [[broadcast_0:%[^ ]+]] = f32[5,2,3,4]{3,2,1,0} broadcast([[bitcast_1:%[^ ]+]]), dimensions={0,3}
// CHECK:  [[abs_2:%[^ ]+]] = f32[5,2,3,4]{3,2,1,0} abs([[broadcast_0]])
)");
}

TEST_F(LayoutNormalizationTest, Concatenate) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  a = f32[4,5]{0,1} parameter(0)
  b = f32[4,5]{0,1} parameter(1)
  c = f32[8,5]{0,1} concatenate(a, b), dimensions={0}
  ROOT out = abs(c)
}
)";

  CheckLayoutNormalization(hlo, R"(
// CHECK: [[a_0:%[^ ]+]] = f32[4,5]{0,1} parameter(0)
// CHECK: [[bitcast_1:%[^ ]+]] = f32[5,4]{1,0} bitcast([[a_0]])
// CHECK: [[b_2:%[^ ]+]] = f32[4,5]{0,1} parameter(1)
// CHECK: [[bitcast_2_3:%[^ ]+]] = f32[5,4]{1,0} bitcast([[b_2]])
// CHECK: [[concatenate_4:%[^ ]+]] = f32[5,8]{1,0} concatenate([[bitcast_1]], [[bitcast_2_3]]), dimensions={1}
// CHECK: [[abs_5:%[^ ]+]] = f32[5,8]{1,0} abs([[concatenate_4]])
)");
}

TEST_F(LayoutNormalizationTest, ConcatenateDegenerateDim) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  a = f32[1,4,5]{0,1,2} parameter(0)
  b = f32[1,4,5]{0,1,2} parameter(1)
  c = f32[2,4,5]{0,1,2} concatenate(a, b), dimensions={0}
  ROOT out = abs(c)
}
)";

  CheckLayoutNormalization(hlo, R"(
// CHECK: [[a_0:%[^ ]+]] = f32[1,4,5]{0,1,2} parameter(0)
// CHECK: [[bitcast_1:%[^ ]+]] = f32[5,4]{1,0} bitcast([[a_0]])
// CHECK: [[bitcast_4_2:%[^ ]+]] = f32[5,4,1]{2,1,0} bitcast([[bitcast_1]])
// CHECK: [[b_3:%[^ ]+]] = f32[1,4,5]{0,1,2} parameter(1)
// CHECK: [[bitcast_2_4:%[^ ]+]] = f32[5,4]{1,0} bitcast([[b_3]])
// CHECK: [[bitcast_5_5:%[^ ]+]] = f32[5,4,1]{2,1,0} bitcast([[bitcast_2_4]])
// CHECK: [[concatenate_6:%[^ ]+]] = f32[5,4,2]{2,1,0} concatenate([[bitcast_4_2]], [[bitcast_5_5]]), dimensions={2}
// CHECK: [[abs_7:%[^ ]+]] = f32[5,4,2]{2,1,0} abs([[concatenate_6]])
)");
}

TEST_F(LayoutNormalizationTest, ConcatenateOneDegenerateDim) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  a = f32[1,5]{0,1} parameter(0)
  b = f32[2,5]{0,1} parameter(1)
  c = f32[3,5]{0,1} concatenate(a, b), dimensions={0}
  ROOT out = abs(c)
}
)";

  CheckLayoutNormalization(hlo, R"(
// CHECK: [[a_0:%[^ ]+]] = f32[1,5]{0,1} parameter(0)
// CHECK: [[bitcast_1:%[^ ]+]] = f32[5]{0} bitcast([[a_0]])
// CHECK: [[bitcast_4_2:%[^ ]+]] = f32[5,1]{1,0} bitcast([[bitcast_1]])
// CHECK: [[b_3:%[^ ]+]] = f32[2,5]{0,1} parameter(1)
// CHECK: [[bitcast_2_4:%[^ ]+]] = f32[5,2]{1,0} bitcast([[b_3]])
// CHECK: [[concatenate_5:%[^ ]+]] = f32[5,3]{1,0} concatenate([[bitcast_4_2]], [[bitcast_2_4]]), dimensions={1}
// CHECK: [[abs_6:%[^ ]+]] = f32[5,3]{1,0} abs([[concatenate_5]])
// CHECK: ROOT [[bitcast_6_7:%[^ ]+]] = f32[3,5]{0,1} bitcast([[abs_6]])
)");
}

TEST_F(LayoutNormalizationTest, ConcatenateOneDegenerateDimOfMany) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  a = f32[1,5,1,4]{0,1,3,2} parameter(0)
  b = f32[1,5,1,4]{0,1,3,2} parameter(1)
  c = f32[2,5,1,4]{0,1,3,2} concatenate(a, b), dimensions={0}
  ROOT out = abs(c)
}
)";

  CheckLayoutNormalization(hlo, R"(
// CHECK: [[a_0:%[^ ]+]] = f32[1,5,1,4]{0,1,3,2} parameter(0)
// CHECK: [[bitcast_1:%[^ ]+]] = f32[4,5]{1,0} bitcast([[a_0]])
// CHECK: [[bitcast_4_2:%[^ ]+]] = f32[4,5,1]{2,1,0} bitcast([[bitcast_1]])
// CHECK: [[b_3:%[^ ]+]] = f32[1,5,1,4]{0,1,3,2} parameter(1)
// CHECK: [[bitcast_2_4:%[^ ]+]] = f32[4,5]{1,0} bitcast([[b_3]])
// CHECK: [[bitcast_5_5:%[^ ]+]] = f32[4,5,1]{2,1,0} bitcast([[bitcast_2_4]])
// CHECK: [[concatenate_6:%[^ ]+]] = f32[4,5,2]{2,1,0} concatenate([[bitcast_4_2]], [[bitcast_5_5]]), dimensions={2}
// CHECK: [[abs_7:%[^ ]+]] = f32[4,5,2]{2,1,0} abs([[concatenate_6]])
// CHECK: ROOT [[bitcast_7_8:%[^ ]+]] = f32[2,5,1,4]{0,1,3,2} bitcast([[abs_7]])
)");
}

TEST_F(LayoutNormalizationTest, ConcatenateOtherDegenerateDim) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  a = f32[1,5]{0,1} parameter(0)
  b = f32[1,5]{0,1} parameter(1)
  c = f32[1,10]{0,1} concatenate(a, b), dimensions={1}
  ROOT out = abs(c)
}
)";

  CheckLayoutNormalization(hlo, R"(
// CHECK: [[a_0:%[^ ]+]] = f32[1,5]{0,1} parameter(0)
// CHECK: [[bitcast_1:%[^ ]+]] = f32[5]{0} bitcast([[a_0]])
// CHECK: [[b_2:%[^ ]+]] = f32[1,5]{0,1} parameter(1)
// CHECK: [[bitcast_2_3:%[^ ]+]] = f32[5]{0} bitcast([[b_2]])
// CHECK: [[concatenate_4:%[^ ]+]] = f32[10]{0} concatenate([[bitcast_1]], [[bitcast_2_3]]), dimensions={0}
// CHECK: [[abs_5:%[^ ]+]] = f32[10]{0} abs([[concatenate_4]])
// CHECK: ROOT [[bitcast_5_6:%[^ ]+]] = f32[1,10]{0,1} bitcast([[abs_5]])
)");
}

TEST_F(LayoutNormalizationTest, Reverse) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  a = f32[2,3,5]{0,2,1} parameter(0)
  b = f32[2,3,5]{0,2,1} reverse(a), dimensions={0,1}
  ROOT out = abs(b)
}
)";

  CheckLayoutNormalization(hlo, R"(
// CHECK: [[a_0:%[^ ]+]] = f32[2,3,5]{0,2,1} parameter(0)
// CHECK: [[bitcast_1:%[^ ]+]] = f32[3,5,2]{2,1,0} bitcast([[a_0]])
// CHECK: [[reverse_2:%[^ ]+]] = f32[3,5,2]{2,1,0} reverse([[bitcast_1]]), dimensions={0,2}
// CHECK: [[abs_3:%[^ ]+]] = f32[3,5,2]{2,1,0} abs([[reverse_2]])
)");
}

TEST_F(LayoutNormalizationTest, ReverseDegenerateDimensions) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  a = f32[1,3,5]{0,2,1} parameter(0)
  b = f32[1,3,5]{1,2,0} reverse(a), dimensions={0,1}
  ROOT out = abs(b)
}
)";

  CheckLayoutNormalization(hlo, R"(
// CHECK: [[reverse_0:%[^ ]+]] = f32[3,5]{1,0} reverse([[bitcast_1:%[^ ]+]]), dimensions={1}
// CHECK: [[abs_2:%[^ ]+]] = f32[3,5]{1,0} abs([[reverse_0]])
)");
}

TEST_F(LayoutNormalizationTest, Pad) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  a = f32[1,3,5,7]{0,2,1,3} parameter(0)
  z = f32[] constant(0)
  b = f32[1,13,15,7]{0,2,1,3} pad(a, z), padding=0_0x5_5x5_5x0_0
  ROOT out = abs(b)
}
)";

  CheckLayoutNormalization(hlo, R"(
// CHECK: [[pad_0:%[^ ]+]] = f32[7,13,15]{2,1,0} pad([[bitcast_1:%[^ ]+]], [[bitcast_3_2:%[^ ]+]]), padding=0_0x5_5x5_5
// CHECK: [[abs_3:%[^ ]+]] = f32[7,13,15]{2,1,0} abs([[pad_0]])
// CHECK: ROOT [[bitcast_5_4:%[^ ]+]] = f32[1,13,15,7]{0,2,1,3} bitcast([[abs_3]])
)");
}

TEST_F(LayoutNormalizationTest, PadDegenerate) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  a = f32[1,3,5]{0,2,1} parameter(0)
  z = f32[] constant(0)
  b = f32[11,13,15]{0,2,1} pad(a, z), padding=5_5x5_5x5_5
  ROOT out = abs(b)
}

)";
  CheckLayoutNormalization(hlo, R"(
// CHECK: [[bitcast_2_0:%[^ ]+]] = f32[] bitcast([[z_1:%[^ ]+]])
// CHECK: [[bitcast_3_2:%[^ ]+]] = f32[] bitcast([[bitcast_2_0]])
// CHECK: [[pad_3:%[^ ]+]] = f32[13,15,11]{2,1,0} pad([[bitcast_4_4:%[^ ]+]], [[bitcast_3_2]]), padding=5_5x5_5x5_5
// CHECK: [[abs_5:%[^ ]+]] = f32[13,15,11]{2,1,0} abs([[pad_3]])
// CHECK: ROOT [[bitcast_6_6:%[^ ]+]] = f32[11,13,15]{0,2,1} bitcast([[abs_5]])
)");
}

TEST_F(LayoutNormalizationTest, PadOtherDimDegenerate) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  a = f32[1,3,5,1]{0,2,1,3} parameter(0)
  z = f32[] constant(0)
  b = f32[11,13,7,1]{0,2,1,3} pad(a, z), padding=5_5x5_5x1_1x0_0
  ROOT out = abs(b)
}

)";
  CheckLayoutNormalization(hlo, R"(
// CHECK: [[bitcast_0:%[^ ]+]] = f32[3,5]{1,0} bitcast([[a_1:%[^ ]+]])
// CHECK: [[bitcast_4_2:%[^ ]+]] = f32[3,5,1]{2,1,0} bitcast([[bitcast_0]])
// CHECK: [[z_3:%[^ ]+]] = f32[] constant(0)
// CHECK: [[bitcast_2_4:%[^ ]+]] = f32[] bitcast([[z_3]])
// CHECK: [[bitcast_3_5:%[^ ]+]] = f32[] bitcast([[bitcast_2_4]])
// CHECK: [[pad_6:%[^ ]+]] = f32[13,7,11]{2,1,0} pad([[bitcast_4_2]], [[bitcast_3_5]]), padding=5_5x1_1x5_5
// CHECK: [[abs_7:%[^ ]+]] = f32[13,7,11]{2,1,0} abs([[pad_6]])
// CHECK: ROOT [[bitcast_6_8:%[^ ]+]] = f32[11,13,7,1]{0,2,1,3} bitcast([[abs_7]])
)");
}

}  // namespace
}  // namespace xla
