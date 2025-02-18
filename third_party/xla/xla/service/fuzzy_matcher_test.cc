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

#include "xla/service/fuzzy_matcher.h"

#include <gtest/gtest.h>
#include "xla/service/pattern_matcher.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {

using FuzzyMatcherTest = HloTestBase;

TEST_F(FuzzyMatcherTest, IgnoreConvert) {
  constexpr char kModuleStr[] = R"(
    HloModule test_module
    ENTRY test {
      x = f16[8,3] parameter(0)
      y = f16[8,3] parameter(1)
      div = f16[8,3] divide(x, y)
      ROOT convert = f32[8,3] convert(div)
    })";
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  auto* root = hlo_module->entry_computation()->root_instruction();
  EXPECT_TRUE(
      Match(root, fm::Divide(match::Parameter(0), match::Parameter(1))));
}

}  // namespace

}  // namespace xla
