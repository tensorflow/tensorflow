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
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/test.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla::cpu {
namespace {

TEST_F(HloTestBase, ConcatR3Axis1) {
  const std::string hlo_text_module = R"(
    HloModule module

    ENTRY entry {
      %x = s32[3,3,2] parameter(0)
      %y = s32[3,3,2] parameter(1)
      ROOT %cat_axis1 = s32[3,6,2] concatenate(x, y), dimensions={1}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_text_module));

  Literal x_input =
      LiteralUtil::CreateR3<int32_t>({{{0, 1}, {2, 3}, {4, 5}},
                                      {{6, 7}, {8, 9}, {10, 11}},
                                      {{12, 13}, {14, 15}, {16, 17}}});
  Literal y_input =
      LiteralUtil::CreateR3<int32_t>({{{0, -1}, {-2, -3}, {-4, -5}},
                                      {{-6, -7}, {-8, -9}, {-10, -11}},
                                      {{-12, -13}, {-14, -15}, {-16, -17}}});

  TF_ASSERT_OK_AND_ASSIGN(auto result,
                          Execute(std::move(module), {&x_input, &y_input}));

  LiteralTestUtil::ExpectR3Equal(
      {{{0, 1}, {2, 3}, {4, 5}, {0, -1}, {-2, -3}, {-4, -5}},
       {{6, 7}, {8, 9}, {10, 11}, {-6, -7}, {-8, -9}, {-10, -11}},
       {{12, 13}, {14, 15}, {16, 17}, {-12, -13}, {-14, -15}, {-16, -17}}},
      result);
}

}  // namespace
}  // namespace xla::cpu
