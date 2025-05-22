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

#include <cstdint>
#include <utility>

#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

class NopCustomCallTest : public HloPjRtTestBase {};

TEST_F(NopCustomCallTest, RunAllocateBufferAndUpdate) {
  // The test uses a custom call with the AllocateBuffer target (also known as
  // kNopCustomCallTarget) to allocate an output buffer. Then it verifies
  // we can successfully modify the buffer.
  const char* hlo_text = R"(
  HloModule AllocateBuffer, is_scheduled=true

  overwrite_one {
    p0 = s32[1] parameter(0)
    c0 = s32[] constant(0)
    c1 = s32[1] constant({1})
    ROOT dus0 = s32[1] dynamic-update-slice(p0, c1, c0)
  }

  ENTRY main {
    buffer = s32[1] custom-call(), custom_call_target="AllocateBuffer"
    ROOT fusion = s32[1] fusion(buffer), kind=kLoop, calls=overwrite_one
  })";
  auto module = ParseAndReturnVerifiedModule(hlo_text).value();

  Literal result = ExecuteNoHloPasses(std::move(module), {});
  Literal expected = LiteralUtil::CreateR1<int32_t>({1});
  EXPECT_TRUE(LiteralTestUtil::Equal(expected, result));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
