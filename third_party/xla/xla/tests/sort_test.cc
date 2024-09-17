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

#include <string_view>

#include <gtest/gtest.h>
#include "xla/error_spec.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/test_macros.h"

namespace xla {
namespace {

class SortTest : public HloTestBase {};

XLA_TEST_F(SortTest, SortDim0) {
  std::string_view hlo_text_module = R"(
    HloModule sort

    compare {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT lt = pred[] compare(p0, p1), direction=LT
    }

    ENTRY e {
      x = f32[32,64] parameter(0)
      ROOT sort = f32[32,64] sort(x), dimensions={0}, to_apply=compare
    }
  )";

  EXPECT_TRUE(RunAndCompare(hlo_text_module, ErrorSpec{0.0, 0.0}));
}

XLA_TEST_F(SortTest, SortDim1) {
  std::string_view hlo_text_module = R"(
    HloModule sort

    compare {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT lt = pred[] compare(p0, p1), direction=LT
    }

    ENTRY e {
      x = f32[32,64] parameter(0)
      ROOT sort = f32[32,64] sort(x), dimensions={1}, to_apply=compare
    }
  )";

  EXPECT_TRUE(RunAndCompare(hlo_text_module, ErrorSpec{0.0, 0.0}));
}

XLA_TEST_F(SortTest, SortTwiceWithSameComparator) {
  std::string_view hlo_text_module = R"(
    HloModule sort

    compare {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT lt = pred[] compare(p0, p1), direction=LT
    }

    ENTRY e {
      x = f32[32,64] parameter(0)
      y = f32[64,32] parameter(1)
      sort_x = f32[32,64] sort(x), dimensions={0}, to_apply=compare
      sort_y = f32[64,32] sort(y), dimensions={1}, to_apply=compare
      ROOT tuple = (f32[32,64], f32[64,32]) tuple(sort_x, sort_y)
    }
  )";

  EXPECT_TRUE(RunAndCompare(hlo_text_module, ErrorSpec{0.0, 0.0}));
}

}  // namespace
}  // namespace xla
