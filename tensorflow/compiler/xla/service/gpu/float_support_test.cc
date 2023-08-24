/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <string>

#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace gpu {
namespace {

using FloatSupportTest = HloTestBase;

TEST_F(FloatSupportTest, MixedTypeDotIsNotUpcasted) {
  const std::string kHloText = R"(
ENTRY e {
  p0 = bf16[32,32] parameter(0)
  p1 = bf16[32,32] parameter(1)
  ROOT d = f32[32,32] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";

  MatchOptimizedHlo(kHloText, R"(
; CHECK-NOT: convert
)");

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{1e-6, 1e-6}));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
