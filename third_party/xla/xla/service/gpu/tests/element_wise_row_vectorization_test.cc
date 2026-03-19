/* Copyright 2021 The OpenXLA Authors.
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

#include "xla/error_spec.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

using ElementWiseRowVectorizationTest =
    HloPjRtInterpreterReferenceMixin<HloPjRtTestBase>;

TEST_F(ElementWiseRowVectorizationTest, SimpleAddSmallRowBroadcastingTest) {
  const char* hlo_text = R"(
HloModule SimpleAddSmallRowBroadcasting

%fused_computation.0 {
  %param_0 = f32[48]{0} parameter(0)
  %broadcast = f32[256,14,14,48]{3,2,1,0} broadcast(%param_0), dimensions={3}
  %param_1 = f32[256,14,14,48]{3,2,1,0} parameter(1)
  ROOT %add = f32[256,14,14,48]{3,2,1,0} add(%broadcast, %param_1)
}

ENTRY main {
  %param_0 = f32[48]{0} parameter(0)
  %param_1 = f32[256,14,14,48]{3,2,1,0} parameter(1)

  ROOT %fusion.0_small = f32[256,14,14,48]{3,2,1,0} fusion(%param_0, %param_1), kind=kLoop, calls=%fused_computation.0
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_text).value();
  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
