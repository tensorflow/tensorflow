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

#include <optional>

#include "xla/debug_options_flags.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_pjrt_test_base.h"

namespace xla {
namespace gpu {
namespace {

class InPlaceOpTest : public HloPjRtInterpreterReferenceMixin<HloPjRtTestBase> {
  // Don't override any flags.
  DebugOptions GetDebugOptionsForTest() const override {
    return GetDebugOptionsFromFlags();
  }
};

TEST_F(InPlaceOpTest, DynamicUpdateSliceWithMultipleConsumers) {
  const char* hlo_text = R"(
    HloModule main

    ENTRY %main {
      %param_0 = s32[4,4] parameter(0)
      %constant_0 = s32[] constant(0)
      %constant_1 = s32[] constant(1)
      %constant_1x1_1 = s32[1,1] constant({ {1} })
      %slice = s32[1,1] slice(%param_0), slice={[1:2], [0:1]}
      %add = add(%slice, %constant_1x1_1)
      %updated = s32[4,4] dynamic-update-slice(%param_0, %add, %constant_1, %constant_0)
      %transpose = s32[4,4] transpose(%updated), dimensions={1,0}
      ROOT %tuple = tuple(%transpose, %updated)
    }
  )";

  EXPECT_TRUE(RunAndCompare(hlo_text, std::nullopt));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
