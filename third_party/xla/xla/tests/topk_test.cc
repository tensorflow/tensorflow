/* Copyright 2023 The OpenXLA Authors.

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

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/error_spec.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/test_macros.h"

namespace xla {
namespace {

class TopkTest : public HloTestBase {};

XLA_TEST_F(TopkTest, LargestTopK) {
  absl::string_view hlo_text_module = R"(
    HloModule topk

    ENTRY TopK {
      x = bf16[10,10] parameter(0)
      ROOT topk = (bf16[10,2], s32[10,2]) topk(x), k=2, largest=true
    }
  )";
  EXPECT_TRUE(RunAndCompare(hlo_text_module, ErrorSpec{1e-5, 1e-5}));
}

XLA_TEST_F(TopkTest, SmallestTopK) {
  absl::string_view hlo_text_module = R"(
    HloModule topk

    ENTRY TopK {
      x = bf16[10,10] parameter(0)
      ROOT topk = (bf16[10,2], s32[10,2]) topk(x), k=2, largest=false
    }
  )";
  EXPECT_TRUE(RunAndCompare(hlo_text_module, ErrorSpec{1e-5, 1e-5}));
}

XLA_TEST_F(TopkTest, TopKOfTranspose) {
  // Regression test for b/362565176
  absl::string_view hlo_text_module = R"(
    HloModule topk

    ENTRY main {
      %Arg_0.1 = f32[2048,6]{1,0} parameter(0)
      t = f32[6,2048]{0,1} transpose(%Arg_0.1), dimensions={1,0}
      ROOT %topk.2 = (f32[6,8]{1,0}, s32[6,8]{1,0}) topk(t), k=8, largest=true
    }
  )";
  EXPECT_TRUE(RunAndCompare(hlo_text_module, ErrorSpec{1e-5, 1e-5}));
}

}  // namespace
}  // namespace xla
