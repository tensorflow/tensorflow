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

}  // namespace
}  // namespace xla
