/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/real_imag_expander.h"

#include <memory>
#include <utility>

#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/service/pattern_matcher_gmock.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {
namespace {

namespace m = match;

class RealImagExpanderTest : public HloTestBase {};

TEST_F(RealImagExpanderTest, RealWithNonComplexInput) {
  const char* kModuleStr = R"(
    HloModule real_float
    ENTRY main {
      input = f32[4] parameter(0)
      ROOT real = real(input)
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  RealImagExpander expander;
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&expander, module.get()));
  EXPECT_TRUE(result);

  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Parameter(0)));
}

TEST_F(RealImagExpanderTest, ImagWithNonComplexInput) {
  const char* kModuleStr = R"(
    HloModule imag_float
    ENTRY main {
      input = f32[4,2,8] parameter(0)
      ROOT imag = imag(input)
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  RealImagExpander expander;
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&expander, module.get()));
  EXPECT_TRUE(result);

  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Broadcast()));

  std::cerr << module->ToString();
}

TEST_F(RealImagExpanderTest, RealImagWithComplexInput) {
  const char* kModuleStr = R"(
    HloModule real_float
    ENTRY main {
      input = c64[4] parameter(0)
      real = real(input)
      imag = imag(input)
      ROOT t = tuple(real, imag)
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  RealImagExpander expander;
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&expander, module.get()));

  // If inputs are complex, the pass should not change anything.
  EXPECT_FALSE(result);
}

}  // namespace
}  // namespace xla
