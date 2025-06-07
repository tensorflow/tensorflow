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

#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tests/test_macros.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::cpu {
namespace {

class TopkTest : public HloPjRtTestBase {};

TEST_F(TopkTest, CustomCallTarget) {
  absl::string_view hlo_text_module = R"(
  HloModule topk

  ENTRY TopK {
    x = f32[10,10] parameter(0)
    ROOT topk = (f32[10,3], s32[10,3]) custom-call(x), custom_call_target="TopK"
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_text_module));

  auto input =
      LiteralUtil::CreateR2<float>({{98, 21, 67, 27, 54, 67, 98, 84, 9, 62},
                                    {65, 68, 49, 3, 9, 0, 52, 78, 36, 96},
                                    {44, 50, 35, 62, 33, 19, 37, 26, 23, 90},
                                    {34, 55, 10, 98, 19, 35, 11, 77, 25, 1},
                                    {87, 19, 15, 98, 35, 90, 64, 60, 80, 12},
                                    {8, 11, 77, 52, 76, 33, 39, 55, 74, 96},
                                    {75, 69, 2, 85, 85, 65, 48, 29, 91, 25},
                                    {26, 4, 76, 48, 88, 96, 71, 2, 58, 68},
                                    {42, 90, 38, 86, 18, 0, 22, 28, 1, 39},
                                    {90, 34, 63, 92, 30, 54, 3, 98, 85, 4}});
  TF_ASSERT_OK_AND_ASSIGN(auto result, Execute(std::move(module), {&input}));
  std::vector<Literal> results = result.DecomposeTuple();
  ASSERT_EQ(results.size(), 2);
  LiteralTestUtil::ExpectR2Equal<float>({{98, 98, 84},
                                         {96, 78, 68},
                                         {90, 62, 50},
                                         {98, 77, 55},
                                         {98, 90, 87},
                                         {96, 77, 76},
                                         {91, 85, 85},
                                         {96, 88, 76},
                                         {90, 86, 42},
                                         {98, 92, 90}},
                                        results[0]);
  LiteralTestUtil::ExpectR2Equal({{0, 6, 7},
                                  {9, 7, 1},
                                  {9, 3, 1},
                                  {3, 7, 1},
                                  {3, 5, 0},
                                  {9, 2, 4},
                                  {8, 3, 4},
                                  {5, 4, 2},
                                  {1, 3, 0},
                                  {7, 3, 0}},
                                 results[1]);
}

}  // namespace
}  // namespace xla::cpu
