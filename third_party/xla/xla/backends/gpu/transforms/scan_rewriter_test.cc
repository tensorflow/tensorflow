/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/transforms/scan_rewriter.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/service/pattern_matcher.h"

namespace xla::gpu {
namespace {

namespace m = ::xla::match;

using ScanRewriterTest = HloHardwareIndependentTestBase;

TEST_F(ScanRewriterTest, BasicScan) {
  const char* hlo_text = R"(
HloModule module

add {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  add = f32[] add(p0, p1)
  ROOT tuple = (f32[], f32[]) tuple(add, add)
}

ENTRY entry {
  p0 = f32[100] parameter(0)
  p1 = f32[] parameter(1)
  ROOT scan = (f32[100], f32[]) scan(p0, p1), 
    dimensions={0}, num_carries=1, is_associative=true, to_apply=add
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));

  ScanRewriter pass;
  ASSERT_OK(pass.Run(module.get()));

  // Check that the scan is rewritten.
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              ::xla::GmockMatch(m::GetTupleElement(
                  m::CustomCall(m::Parameter(0), m::Parameter(1)), 0)));
}

}  // namespace
}  // namespace xla::gpu
