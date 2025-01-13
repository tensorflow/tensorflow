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

#include "absl/strings/string_view.h"
#include "xla/error_spec.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/test_macros.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace {

using XnnFusionTest = HloTestBase;

XLA_TEST_F(XnnFusionTest, CorrectComputation) {
  constexpr absl::string_view kModuleStr = R"(
    HloModule xnn-fusion

    xnn_fusion {
      %lhs = f32[4] parameter(0)
      %rhs = f32[4] parameter(1)
      %add = f32[4] add(%lhs, %rhs)
      ROOT %mul = f32[4] multiply(%add, %add)
    }

    ENTRY entry {
      %p0 = f32[4] parameter(0)
      %p1 = f32[4] parameter(1)
      ROOT %fusion = f32[4] fusion(%p0, %p1), kind=kCustom, calls=xnn_fusion,
        backend_config={"fusion_config": {kind: "__xnn_fusion"}}
    })";

  EXPECT_TRUE(RunAndCompare(kModuleStr, ErrorSpec{0.0}));
}

}  // namespace
}  // namespace xla
