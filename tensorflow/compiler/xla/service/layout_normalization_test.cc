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

#include <utility>

#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/tests/filecheck.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/llvm_irgen_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

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

}  // namespace
}  // namespace xla
