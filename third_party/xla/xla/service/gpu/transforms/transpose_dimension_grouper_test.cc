/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/gpu/transforms/transpose_dimension_grouper.h"

#include <optional>

#include "absl/strings/string_view.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/test.h"

namespace xla {

namespace {

class TransposeDimensionGrouperTest : public HloTestBase {
 public:
  void CheckDimensionGrouper(absl::string_view hlo,
                             std::optional<absl::string_view> expected) {
    RunAndFilecheckHloRewrite(hlo, gpu::TransposeDimensionGrouper{}, expected);
  }
};

TEST_F(TransposeDimensionGrouperTest, TransposeWithGrouping) {
  const char* hlo = R"(
HloModule TransposeWithGrouping

ENTRY main {
  input = f32[100,1,10,32,2]{4,3,2,1,0} parameter(0)
  ROOT out = f32[10,1,32,100,2]{4,3,2,1,0} transpose(input), dimensions={2,1,3,0,4}
}
)";

  CheckDimensionGrouper(hlo,
                        R"(
// CHECK:  [[input_0:%[^ ]+]] = f32[100,1,10,32,2]{4,3,2,1,0} parameter(0)
// CHECK:  [[bitcast_1:%[^ ]+]] = f32[100,320,2]{2,1,0} bitcast([[input_0]])
// CHECK:  [[transpose:%[^ ]+]] = f32[320,100,2]{2,1,0} transpose([[bitcast_1]]), dimensions={1,0,2}
// CHECK:  ROOT {{.*}} = f32[10,1,32,100,2]{4,3,2,1,0} bitcast([[transpose]])
      )");
}

}  // namespace
}  // namespace xla
