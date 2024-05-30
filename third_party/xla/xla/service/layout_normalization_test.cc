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

#include "xla/service/layout_normalization.h"

#include <functional>
#include <optional>
#include <utility>

#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/scatter_simplifier.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/status.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {

class LayoutNormalizationTest : public HloTestBase {
 public:
  void CheckLayoutNormalization(
      absl::string_view hlo, std::optional<absl::string_view> expected,
      std::function<void(HloModule*)> after_pass_checks = nullptr) {
    RunAndFilecheckHloRewrite(hlo, LayoutNormalization{}, expected,
                              after_pass_checks);
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
// CHECK:  [[bitcast_1:%[^ ]+]] = f32[1,4,1,5]{3,2,1,0} bitcast([[p_0]])
// CHECK:  [[abs_2:%[^ ]+]] = f32[1,4,1,5]{3,2,1,0} abs([[bitcast_1]])
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
// CHECK: [[bitcast_0:%[^ ]+]] = f32[1,5,4,3]{3,2,1,0} bitcast([[p_1:%[^ ]+]])
// CHECK: [[transpose_2:%[^ ]+]] = f32[1,5,4,3]{3,2,1,0} transpose([[bitcast_0]]), dimensions={0,1,2,3}
// CHECK: [[abs_3:%[^ ]+]] = f32[1,5,4,3]{3,2,1,0} abs([[transpose_2]])
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
// CHECK: [[bitcast_1:%[^ ]+]] = f32[5,1,4,1,3]{4,3,2,1,0} bitcast([[p_0]])
// CHECK: [[transpose_2:%[^ ]+]] = f32[3,1,4,1,5]{4,3,2,1,0} transpose([[bitcast_1]]), dimensions={4,3,2,1,0}
// CHECK: [[abs_3:%[^ ]+]] = f32[3,1,4,1,5]{4,3,2,1,0} abs([[transpose_2]])
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

TEST_F(LayoutNormalizationTest, BroadcastOperandLayoutNotInverseOfItself) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  a = f32[4,3,5]{0,2,1} parameter(0)
  b = f32[4,3,2,5]{0,1,2,3} broadcast(a), dimensions={0,1,3}
  ROOT out = abs(b)
}
)";

  CheckLayoutNormalization(hlo, R"(
// CHECK: [[bitcast_1:%[^ ]+]] = f32[3,5,4]{2,1,0} bitcast
// CHECK: [[broadcast_0:%[^ ]+]] = f32[5,2,3,4]{3,2,1,0} broadcast([[bitcast_1]]), dimensions={2,0,3}
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

TEST_F(LayoutNormalizationTest, BroadcastUnsortedDimensions) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  a = f32[2,3]{1,0} parameter(0)
  b = f32[3,4,2]{2,1,0} broadcast(a), dimensions={2,0}
  ROOT out = abs(b)
}
)";

  CheckLayoutNormalization(hlo, R"(
// CHECK: [[a_0:%[^ ]+]] = f32[2,3]{1,0} parameter(0)
// CHECK: [[bitcast_1:%[^ ]+]] = f32[2,3]{1,0} bitcast([[a_0]])
// CHECK: [[broadcast_2:%[^ ]+]] = f32[3,4,2]{2,1,0} broadcast([[bitcast_1]]), dimensions={2,0}
// CHECK: [[abs_3:%[^ ]+]] = f32[3,4,2]{2,1,0} abs([[broadcast_2]])
// CHECK: ROOT [[bitcast_3_4:%[^ ]+]] = f32[3,4,2]{2,1,0} bitcast([[abs_3]])
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
// CHECK: [[broadcast_2:%[^ ]+]] = f32[9,1,2,4]{3,2,1,0} broadcast([[bitcast_0]]), dimensions={0}
// CHECK: [[abs_3:%[^ ]+]] = f32[9,1,2,4]{3,2,1,0} abs([[broadcast_2]])
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
// CHECK:  [[broadcast_0:%[^ ]+]] = f32[1,5,2,1,3,4,1]{6,5,4,3,2,1,0} broadcast([[bitcast_1:%[^ ]+]]), dimensions={1,5,6}
// CHECK:  [[abs_2:%[^ ]+]] = f32[1,5,2,1,3,4,1]{6,5,4,3,2,1,0} abs([[broadcast_0]])
)");
}

TEST_F(LayoutNormalizationTest, IotaCustomOutputLayout) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  a = f32[2,4,3]{1,2,0} iota(), iota_dimension=2
  ROOT out = abs(a)
}
)";

  CheckLayoutNormalization(hlo, R"(
// CHECK: [[iota_2:%[^ ]+]] = f32[2,3,4]{2,1,0} iota(), iota_dimension=1
// CHECK: [[abs_3:%[^ ]+]] = f32[2,3,4]{2,1,0} abs([[iota_2]])
// CHECK: ROOT [[bitcast_3_4:%[^ ]+]] = f32[2,4,3]{1,2,0} bitcast([[abs_3]])
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
// CHECK: [[bitcast_1:%[^ ]+]] = f32[5,4,1]{2,1,0} bitcast([[a_0]])
// CHECK: [[b_3:%[^ ]+]] = f32[1,4,5]{0,1,2} parameter(1)
// CHECK: [[bitcast_2:%[^ ]+]] = f32[5,4,1]{2,1,0} bitcast([[b_3]])
// CHECK: [[concatenate_6:%[^ ]+]] = f32[5,4,2]{2,1,0} concatenate([[bitcast_1]], [[bitcast_2]]), dimensions={2}
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
// CHECK: [[bitcast_1:%[^ ]+]] = f32[5,1]{1,0} bitcast([[a_0]])
// CHECK: [[b_3:%[^ ]+]] = f32[2,5]{0,1} parameter(1)
// CHECK: [[bitcast_2_4:%[^ ]+]] = f32[5,2]{1,0} bitcast([[b_3]])
// CHECK: [[concatenate_5:%[^ ]+]] = f32[5,3]{1,0} concatenate([[bitcast_1]], [[bitcast_2_4]]), dimensions={1}
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
// CHECK: [[bitcast_1:%[^ ]+]] = f32[1,4,5,1]{3,2,1,0} bitcast([[a_0]])
// CHECK: [[b_3:%[^ ]+]] = f32[1,5,1,4]{0,1,3,2} parameter(1)
// CHECK: [[bitcast_2_4:%[^ ]+]] = f32[1,4,5,1]{3,2,1,0} bitcast([[b_3]])
// CHECK: [[concatenate_6:%[^ ]+]] = f32[1,4,5,2]{3,2,1,0} concatenate([[bitcast_1]], [[bitcast_2_4]]), dimensions={3}
// CHECK: [[abs_7:%[^ ]+]] = f32[1,4,5,2]{3,2,1,0} abs([[concatenate_6]])
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
// CHECK: [[bitcast_1:%[^ ]+]] = f32[5,1]{1,0} bitcast([[a_0]])
// CHECK: [[b_2:%[^ ]+]] = f32[1,5]{0,1} parameter(1)
// CHECK: [[bitcast_2_3:%[^ ]+]] = f32[5,1]{1,0} bitcast([[b_2]])
// CHECK: [[concatenate_4:%[^ ]+]] = f32[10,1]{1,0} concatenate([[bitcast_1]], [[bitcast_2_3]]), dimensions={0}
// CHECK: [[abs_5:%[^ ]+]] = f32[10,1]{1,0} abs([[concatenate_4]])
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
// CHECK: [[reverse_0:%[^ ]+]] = f32[3,5,1]{2,1,0} reverse([[bitcast_1:%[^ ]+]]), dimensions={0,2}
// CHECK: [[abs_2:%[^ ]+]] = f32[3,5,1]{2,1,0} abs([[reverse_0]])
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
// CHECK: [[pad_0:%[^ ]+]] = f32[7,13,15,1]{3,2,1,0} pad([[bitcast_1:%[^ ]+]], [[bitcast_3_2:%[^ ]+]]), padding=0_0x5_5x5_5x0_0
// CHECK: [[abs_3:%[^ ]+]] = f32[7,13,15,1]{3,2,1,0} abs([[pad_0]])
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
// CHECK: [[pad_3:%[^ ]+]] = f32[13,15,11]{2,1,0} pad([[bitcast_4_4:%[^ ]+]], [[bitcast_3_2:%[^ ]+]]), padding=5_5x5_5x5_5
// CHECK: [[abs_5:%[^ ]+]] = f32[13,15,11]{2,1,0} abs([[pad_3]])
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
// CHECK: [[pad_6:%[^ ]+]] = f32[1,13,7,11]{3,2,1,0} pad([[bitcast_4_2:%[^ ]+]], [[bitcast_3_5:%[^ ]+]]), padding=0_0x5_5x1_1x5_5
// CHECK: [[abs_7:%[^ ]+]] = f32[1,13,7,11]{3,2,1,0} abs([[pad_6]])
)");
}

TEST_F(LayoutNormalizationTest, ReduceWindow) {
  const char* hlo = R"(
HloModule R2Window

mul {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT mul = f32[] multiply(lhs, rhs)
}

ENTRY R2Window {
  operand = f32[256,384]{0,1} parameter(0)
  constant = f32[] constant(1)
  ROOT reduce-window = f32[256,384]{0,1} reduce-window(operand, constant), window={size=2x3 pad=0_1x1_1}, to_apply=mul
}
)";
  CheckLayoutNormalization(hlo, R"(
// CHECK: [[reduce_window_1_0:%[^ ]+]] = f32[384,256]{1,0} reduce-window([[bitcast_5_1:%[^ ]+]], [[bitcast_8_2:%[^ ]+]]), window={size=3x2 pad=1_1x0_1}, to_apply=[[mul_3:%[^ ]+]]
  )");
}

TEST_F(LayoutNormalizationTest, Constant) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  p = f32[5,4]{0,1} parameter(0)
  c = f32[5,4]{0,1} constant({...})
  ROOT o = f32[5,4]{0,1} add(p, c)
}
)";
  CheckLayoutNormalization(hlo, R"(
// CHECK: [[p_0:%[^ ]+]] = f32[5,4]{0,1} parameter(0)
// CHECK-NEXT: [[bitcast_1:%[^ ]+]] = f32[4,5]{1,0} bitcast([[p_0]])
// CHECK-NEXT: [[constant_2:%[^ ]+]] = f32[4,5]{1,0} constant({...})
// CHECK-NEXT: [[add_3:%[^ ]+]] = f32[4,5]{1,0} add([[bitcast_1]], [[constant_2]])
// CHECK-NEXT: ROOT [[bitcast_3_4:%[^ ]+]] = f32[5,4]{0,1} bitcast([[add_3]])
  )");
}

TEST_F(LayoutNormalizationTest, ConstantAvoidRevisitOfUser) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  c = f32[5,4]{0,1} constant({...})
  s = f32[5,4]{0,1} sine(c)
  t = f32[5,4]{0,1} tanh(s)
  ROOT o = f32[5,4]{0,1} add(s, t)
}
)";
  // If we allowed visiting the normalized user 's' of the constant, we would
  // run into a CHECK failure, because the constant was normalized in-place and
  // therefore would not be revisited.
  CheckLayoutNormalization(hlo, R"(
// CHECK: [[constant_2:%[^ ]+]] = f32[4,5]{1,0} constant({...})
// CHECK-NEXT: [[sine:%[^ ]+]] = f32[4,5]{1,0} sine([[constant_2]])
// CHECK-NEXT: [[bitcast_1:%[^ ]+]] = f32[5,4]{0,1} bitcast([[sine]])
// CHECK-NEXT: [[bitcast_2:%[^ ]+]] = f32[4,5]{1,0} bitcast([[bitcast_1]])
// CHECK-NEXT: [[tanh:%[^ ]+]] = f32[4,5]{1,0} tanh([[bitcast_2]])
// CHECK-NEXT: [[add_3:%[^ ]+]] = f32[4,5]{1,0} add([[bitcast_2]], [[tanh]])
// CHECK-NEXT: ROOT [[bitcast_3_4:%[^ ]+]] = f32[5,4]{0,1} bitcast([[add_3]])
  )");
}

TEST_F(LayoutNormalizationTest, Slice) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  input = f32[1,17,9,9]{1,3,2,0} parameter(0)
  ROOT converted = f32[1,4,6,6]{1,3,2,0} slice(input), slice={[0:1],[0:4],[0:6],[0:6]}
}
)";
  CheckLayoutNormalization(hlo, R"(
// CHECK: [[input_0:%[^ ]+]] = f32[1,17,9,9]{1,3,2,0} parameter(0)
// CHECK-NEXT: [[bitcast_1:%[^ ]+]] = f32[1,9,9,17]{3,2,1,0} bitcast([[input_0]])
// CHECK-NEXT: [[slice_2:%[^ ]+]] = f32[1,6,6,4]{3,2,1,0} slice([[bitcast_1]]), slice={[0:1], [0:6], [0:6], [0:4]}
// CHECK-NEXT: ROOT [[bitcast_3_4:%[^ ]+]] = f32[1,4,6,6]{1,3,2,0} bitcast([[slice_2]])
  )");
}

TEST_F(LayoutNormalizationTest, Select) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  p0 = f32[1,17,9,9]{1,3,2,0} parameter(0)
  p1 = f32[1,17,9,9]{1,3,2,0} parameter(1)
  b = pred[1,17,9,9]{1,3,2,0} parameter(2)
  ROOT out = f32[1,17,9,9]{1,3,2,0} select(b, p0, p1), metadata={op_name="test"}
}
)";
  CheckLayoutNormalization(hlo, R"(
// CHECK: f32[1,9,9,17]{3,2,1,0} select({{.*}}, {{.*}}, {{.*}}), metadata={op_name="test"}
)");
}

TEST_F(LayoutNormalizationTest, DynamicSlice) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  input = f32[3,4,32]{1,0,2} parameter(0)
  p1 = s32[] parameter(1)
  p2 = s32[] parameter(2)
  p3 = s32[] parameter(3)
  ROOT out = f32[1,4,32]{1,0,2} dynamic-slice(input, p1, p2, p3), dynamic_slice_sizes={1,4,32}, metadata={op_name="test"}
}
  )";
  CheckLayoutNormalization(hlo, R"(
// CHECK: f32[32,1,4]{2,1,0} dynamic-slice({{.*}}, {{.*}}, {{.*}}, {{.*}}), dynamic_slice_sizes={32,1,4}, metadata={op_name="test"}
)");
}

TEST_F(LayoutNormalizationTest, DynamicSliceHasDegenerate) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  input = f32[1,4,32]{1,0,2} parameter(0)
  p1 = s32[] parameter(1)
  p2 = s32[] parameter(2)
  p3 = s32[] parameter(3)
  ROOT out = f32[1,4,32]{1,0,2} dynamic-slice(input, p1, p2, p3), dynamic_slice_sizes={1,4,32}, metadata={op_name="test"}
}
  )";
  CheckLayoutNormalization(hlo, R"(
// CHECK: f32[32,1,4]{2,1,0} dynamic-slice({{.*}}, {{.*}}, {{.*}}), dynamic_slice_sizes={32,1,4}, metadata={op_name="test"}
)");
}

TEST_F(LayoutNormalizationTest, DynamicUpdateSlice) {
  const char* hlo = R"(
HloModule m

ENTRY main {
  to_update = f32[3,1,32]{1,0,2} parameter(0)
  updates = f32[1,1,32]{1,0,2} parameter(1)
  p0 = s32[] parameter(2)
  p1 = s32[] parameter(3)
  p2 = s32[] parameter(4)

  ROOT out = f32[3,1,32]{1,0,2} dynamic-update-slice(to_update, updates, p0, p1, p2), metadata={op_name="test"}
}
)";

  CheckLayoutNormalization(hlo, R"(
// CHECK: f32[32,3,1]{2,1,0} dynamic-update-slice({{.*}}, {{.*}}, {{.*}}, {{.*}}), metadata={op_name="test"}
)");
}

TEST_F(LayoutNormalizationTest, DynamicUpdateSliceNonDeg) {
  const char* hlo = R"(
HloModule m

ENTRY main {
  to_update = f32[5,3,32]{1,0,2} parameter(0)
  updates = f32[1,1,32]{1,0,2} parameter(1)
  p0 = s32[] parameter(2)
  p1 = s32[] parameter(3)
  p2 = s32[] parameter(4)

  ROOT out = f32[5,3,32]{1,0,2} dynamic-update-slice(to_update, updates, p0, p1, p2), metadata={op_name="test"}
}
)";

  CheckLayoutNormalization(hlo, R"(
// CHECK: f32[32,5,3]{2,1,0} dynamic-update-slice
)");
}

TEST_F(LayoutNormalizationTest, Clamp) {
  const char* hlo = R"(
HloModule m

ENTRY main {
  p0 = f32[64,1,32]{1,0,2} parameter(0)
  p1 = f32[64,1,32]{1,0,2} parameter(1)
  p2 = f32[64,1,32]{1,0,2} parameter(2)
  ROOT out = f32[64,1,32]{1,0,2} clamp(f32[64,1,32]{1,0,2} p0, f32[64,1,32]{1,0,2} p1, f32[64,1,32]{1,0,2} p2), metadata={op_name="test"}
}
)";

  CheckLayoutNormalization(hlo, R"(
// CHECK: f32[32,64,1]{2,1,0} clamp({{.*}}, {{.*}}, {{.*}}), metadata={op_name="test"}
)");
}

TEST_F(LayoutNormalizationTest, BitcastConvertToBiggerType) {
  const char* hlo = R"(
HloModule m

ENTRY main {
  p0 = u32[4,2]{0,1} parameter(0)
  ROOT out = u64[4]{0} bitcast-convert(u32[4,2]{0,1} p0), metadata={op_name="test"}
}
)";

  CheckLayoutNormalization(hlo, R"(
// CHECK: bitcast-convert({{.*}}), metadata={op_name="test"}
)");
}

TEST_F(LayoutNormalizationTest, BitcastConvertToSmallerType) {
  const char* hlo = R"(
HloModule m

ENTRY main {
  p0 = u64[4]{0} parameter(0)
  ROOT out = u32[4,2]{0,1} bitcast-convert(u64[4]{0} p0), metadata={op_name="test"}
}
)";

  CheckLayoutNormalization(hlo, R"(
// CHECK: bitcast-convert({{.*}}), metadata={op_name="test"}
)");
}

TEST_F(LayoutNormalizationTest, Scatter) {
  const char* hlo = R"(
HloModule simplified_scatter

region_0.10 {
  Arg_0.11 = s16[] parameter(0)
  Arg_1.12 = s16[] parameter(1)
  ROOT maximum.13 = s16[] maximum(Arg_0.11, Arg_1.12)
}

ENTRY main.17 {
  p0 = s16[3,2,2,14,16]{0,1,4,3,2} parameter(0)
  p1 = s32[2,11]{0,1} parameter(1)
  p2 = s16[11,3,5]{2,0,1} parameter(2)
  ROOT scatter = s16[3,2,2,14,16]{0,1,4,3,2} scatter(p0, p1, p2), update_window_dims={1,2}, inserted_window_dims={1,2,3}, scatter_dims_to_operand_dims={4,0}, index_vector_dim=0, to_apply=region_0.10
}
)";

  CheckLayoutNormalization(
      hlo, R"(
// CHECK: scatter({{.*}}),
// CHECK-SAME: update_window_dims={2,0}, inserted_window_dims={0,1,3}, scatter_dims_to_operand_dims={2,4}, index_vector_dim=1, to_apply=%region_0.10
)",
      // Run the ScatterSimplifier afterwards, otherwise the verifier will
      // complain!
      [](HloModule* module) {
        TF_CHECK_OK(ScatterSimplifier().Run(module).status());
      });
}

TEST_F(LayoutNormalizationTest, SimplifiedScatter) {
  const char* hlo = R"(
HloModule simplified_scatter

region_0.10 {
  Arg_0.11 = s16[] parameter(0)
  Arg_1.12 = s16[] parameter(1)
  ROOT maximum.13 = s16[] maximum(Arg_0.11, Arg_1.12)
}

ENTRY main.17 {
  p0 = s16[16,3,2,2,14]{0,4,3,2,1} parameter(0)
  p1 = s32[528,2]{1,0} parameter(1)
  p2 = s16[528,5,3,1,1,1]{1,2,0,5,4,3} parameter(2)
  ROOT scatter = s16[16,3,2,2,14]{0,4,3,2,1} scatter(p0, p1, p2), update_window_dims={1,2,3,4,5}, inserted_window_dims={}, scatter_dims_to_operand_dims={0,1}, index_vector_dim=1, to_apply=region_0.10
}
)";

  CheckLayoutNormalization(
      hlo, R"(
// CHECK: scatter({{.*}}),
// CHECK-SAME: update_window_dims={4,0,1,2,5}, inserted_window_dims={}, scatter_dims_to_operand_dims={4,0}, index_vector_dim=1, to_apply=%region_0.10
)",
      // Run the ScatterSimplifier afterwards, otherwise the verifier will
      // complain!
      [](HloModule* module) {
        TF_CHECK_OK(ScatterSimplifier().Run(module).status());
      });
}

TEST_F(LayoutNormalizationTest, VariadicScatter) {
  const char* hlo = R"(
HloModule simplified_scatter

region_0.10 {
  Arg_0.11 = s16[] parameter(0)
  Arg_1.12 = s16[] parameter(1)
  Arg_2.13 = s16[] parameter(2)
  Arg_3.14 = s16[] parameter(3)
  maximum.15 = s16[] maximum(Arg_0.11, Arg_1.12)
  maximum.16 = s16[] maximum(Arg_2.13, Arg_3.14)
  ROOT res = (s16[], s16[]) tuple(maximum.15, maximum.16)
}

ENTRY main.17 {
  p0 = s16[16,3,2,2,14]{0,4,3,2,1} parameter(0)
  p1 = s16[16,3,2,2,14]{0,4,3,2,1} parameter(1)
  p2 = s32[528,2]{1,0} parameter(2)
  p3 = s16[528,5,3,1,1,1]{1,2,0,5,4,3} parameter(3)
  p4 = s16[528,5,3,1,1,1]{1,2,0,5,4,3} parameter(4)
  ROOT scatter = (s16[16,3,2,2,14]{0,4,3,2,1}, s16[16,3,2,2,14]{0,4,3,2,1}) scatter(p0, p1, p2, p3, p4), update_window_dims={1,2,3,4,5}, inserted_window_dims={}, scatter_dims_to_operand_dims={0,1}, index_vector_dim=1, to_apply=region_0.10
}
)";

  CheckLayoutNormalization(
      hlo, R"(
// CHECK: scatter({{.*}}),
// CHECK-SAME: update_window_dims={4,0,1,2,5}, inserted_window_dims={}, scatter_dims_to_operand_dims={4,0}, index_vector_dim=1, to_apply=%region_0.10
)",
      // Run the ScatterSimplifier afterwards, otherwise the verifier will
      // complain!
      [](HloModule* module) {
        TF_CHECK_OK(ScatterSimplifier().Run(module).status());
      });
}

}  // namespace
}  // namespace xla
