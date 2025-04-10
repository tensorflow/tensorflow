/* Copyright 2019 The OpenXLA Authors.

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

#include <utility>

#include "xla/debug_options_flags.h"
#include "xla/error_spec.h"
#include "xla/hlo/testlib/test.h"
#include "xla/service/hlo_module_config.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace {

using PtxasBugTest = HloPjRtInterpreterReferenceMixin<HloPjRtTestBase>;

// Checks for a bug in ptxas, tracked as Google bug 120501638, and nvidia bug
// 2459377.  We never received an explanation of what exactly was going wrong
// here in ptxas.  Known-bad in ptxas 10.0.145, known-good in ptxas 10.0.249.
TEST_F(PtxasBugTest, DoIt) {
  const char* const kModuleStr = R"(
HloModule test

add_F32.14 {
  lhs.15 = f32[] parameter(0)
  rhs.16 = f32[] parameter(1)
  ROOT add.17 = f32[] add(lhs.15, rhs.16)
}

ENTRY testcase {
  arg0.1 = f32[2,5,2]{2,1,0} parameter(0)
  reshape.2 = f32[2,5,2]{2,1,0} reshape(arg0.1)
  constant.3 = f32[] constant(0)
  pad.4 = f32[2,6,2]{2,1,0} pad(reshape.2, constant.3), padding=0_0x0_1x0_0
  reshape.5 = f32[2,3,2,2]{3,2,1,0} reshape(pad.4)
  transpose.6 = f32[2,2,3,2]{3,0,2,1} transpose(reshape.5), dimensions={2,0,1,3}
  reshape.7 = f32[4,3,2]{2,1,0} reshape(transpose.6)
  reshape.8 = f32[4,1,3,2]{3,2,1,0} reshape(reshape.7)
  transpose.9 = f32[4,2,1,3]{1,3,2,0} transpose(reshape.8), dimensions={0,3,1,2}
  convert.10 = f32[4,2,1,3]{1,3,2,0} convert(transpose.9)
  constant.12 = f32[] constant(0)
  pad.13 = f32[4,2,1,3]{3,2,1,0} pad(convert.10, constant.12), padding=0_0x0_0x0_0x0_0
  constant.11 = f32[] constant(0)
  reduce-window.18 = f32[4,2,1,3]{3,2,1,0} reduce-window(pad.13, constant.11),
    window={size=1x1x1x1}, to_apply=add_F32.14
  constant.19 = f32[] constant(1)
  broadcast.20 = f32[4,2,1,3]{3,2,1,0} broadcast(constant.19), dimensions={}
  divide.21 = f32[4,2,1,3]{3,2,1,0} divide(reduce-window.18, broadcast.20)
  convert.22 = f32[4,2,1,3]{3,2,1,0} convert(divide.21)
  transpose.23 = f32[4,1,3,2]{2,1,3,0} transpose(convert.22), dimensions={0,2,3,1}
  reshape.24 = f32[4,3,2]{2,1,0} reshape(transpose.23)
  reshape.25 = f32[2,2,3,2]{3,2,1,0} reshape(reshape.24)
  transpose.26 = f32[2,3,2,2]{3,1,0,2} transpose(reshape.25), dimensions={1,2,0,3}
  reshape.27 = f32[2,6,2]{2,1,0} reshape(transpose.26)
  slice.28 = f32[2,5,2]{2,1,0} slice(reshape.27), slice={[0:2], [0:5], [0:2]}
  reshape.29 = f32[2,5,2]{2,1,0} reshape(slice.28)
  tuple.30 = (f32[2,5,2]{2,1,0}) tuple(reshape.29)
  ROOT get-tuple-element.31 = f32[2,5,2]{2,1,0} get-tuple-element(tuple.30), index=0
})";

  // Create a module with the true-default flags, not the default-for-testing
  // flags.  In particular, true-default flags enable unrolling, whereas for
  // testing we disable unrolling, and this bug doesn't trigger without
  // unrolling.
  HloModuleConfig config;
  config.set_debug_options(DefaultDebugOptionsIgnoringFlags());
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr, config));
  EXPECT_TRUE(RunAndCompare(std::move(module), ErrorSpec{0.01, 0.01}));
}

}  // anonymous namespace
}  // namespace xla
