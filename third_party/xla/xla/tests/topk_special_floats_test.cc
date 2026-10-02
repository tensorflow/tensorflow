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

#include <cstdint>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/base/casts.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/array2d.h"
#include "xla/error_spec.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/platform/test.h"
#include "xla/types.h"

namespace xla {
namespace {

using TopkTest = HloPjRtInterpreterReferenceMixin<HloTestBase>;

TEST_F(TopkTest, TopKWithSpecialFloats) {
  // Regression test for TopK TotalOrder with special float values
  // (NaN, +-Inf, +-0).
  float neg_qnan0 = absl::bit_cast<float>(0xFFC00000u);
  float neg_qnan1 = absl::bit_cast<float>(0xFFE00000u);
  float neg_qnan2 = absl::bit_cast<float>(0xFFE00001u);
  float neg_smallest_qnan = absl::bit_cast<float>(0xFFFFFFFFu);
  float pos_qnan0 = absl::bit_cast<float>(0x7FC00000u);
  float pos_qnan1 = absl::bit_cast<float>(0x7FE00000u);
  float pos_qnan2 = absl::bit_cast<float>(0x7FE00001u);

  float pos_inf = absl::bit_cast<float>(0x7F800000u);
  float neg_inf = absl::bit_cast<float>(0xFF800000u);
  float pos_zero = absl::bit_cast<float>(0x00000000u);
  float neg_zero = absl::bit_cast<float>(0x80000000u);

  float pos_dnrm_1e38 = absl::bit_cast<float>(0x00700000u);  // 1.028e-38
  float pos_dnrm_1e40 = absl::bit_cast<float>(0x00020000u);  // 1.836e-40
  float neg_dnrm_1e39 = absl::bit_cast<float>(0x80100000u);  // -1.469e-39
  float neg_dnrm_9e41 = absl::bit_cast<float>(0x80010000u);  // -9.183e-41

  float pos_64_5 = absl::bit_cast<float>(0x42810000u);
  float pos_0_0125 = absl::bit_cast<float>(0x3C4d0000u);
  float neg_64_5 = absl::bit_cast<float>(0xC2810000u);

  std::vector<float> input_row_start = {
      neg_inf,    pos_inf,       neg_dnrm_1e39, pos_zero,      pos_qnan0,
      pos_qnan2,  neg_zero,      neg_64_5,      pos_inf,       neg_qnan0,
      neg_qnan2,  neg_64_5,      neg_zero,      pos_dnrm_1e40, neg_dnrm_9e41,
      neg_qnan1,  pos_zero,      neg_inf,       pos_64_5,      pos_qnan1,
      pos_0_0125, pos_dnrm_1e38, neg_dnrm_1e39, pos_dnrm_1e40, neg_dnrm_9e41};

  std::vector<float> expected_values_64 = {
      pos_qnan2,         pos_qnan1,         pos_qnan0,
      pos_inf,           pos_inf,           pos_64_5,
      pos_0_0125,        pos_dnrm_1e38,     pos_dnrm_1e40,
      pos_dnrm_1e40,     pos_zero,          pos_zero,
      neg_zero,          neg_zero,          neg_dnrm_9e41,
      neg_dnrm_9e41,     neg_dnrm_1e39,     neg_dnrm_1e39,
      neg_64_5,          neg_64_5,          neg_inf,
      neg_inf,           neg_qnan0,         neg_qnan1,
      neg_qnan2,         neg_smallest_qnan, neg_smallest_qnan,
      neg_smallest_qnan, neg_smallest_qnan, neg_smallest_qnan,
      neg_smallest_qnan, neg_smallest_qnan, neg_smallest_qnan,
      neg_smallest_qnan, neg_smallest_qnan, neg_smallest_qnan,
      neg_smallest_qnan, neg_smallest_qnan, neg_smallest_qnan,
      neg_smallest_qnan, neg_smallest_qnan, neg_smallest_qnan,
      neg_smallest_qnan, neg_smallest_qnan, neg_smallest_qnan,
      neg_smallest_qnan, neg_smallest_qnan, neg_smallest_qnan,
      neg_smallest_qnan, neg_smallest_qnan, neg_smallest_qnan,
      neg_smallest_qnan, neg_smallest_qnan, neg_smallest_qnan,
      neg_smallest_qnan, neg_smallest_qnan, neg_smallest_qnan,
      neg_smallest_qnan, neg_smallest_qnan, neg_smallest_qnan,
      neg_smallest_qnan, neg_smallest_qnan, neg_smallest_qnan,
      neg_smallest_qnan};

  std::vector<int32_t> expected_indices_64 = {
      5,  19, 4,  1,  8,  18, 20, 21, 13, 23, 3,  16, 6,  12, 14, 24,
      2,  22, 7,  11, 0,  17, 9,  15, 10, 25, 26, 27, 28, 29, 30, 31,
      32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
      48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};

  for (int n : {512, 1024, 32768, 65536 + 4096}) {
    Array2D<float> input(8, n, neg_smallest_qnan);
    for (int i = 0; i < input_row_start.size(); ++i) {
      for (int j = 0; j < 8; ++j) {
        input(j, i) = input_row_start[i];
      }
    }

    auto literal = LiteralUtil::CreateR2FromArray2D<float>(input);

    for (int k : {2, 4, 8, 16, 32, 64}) {
      const auto hlo_text_module = absl::StrFormat(R"(
      HloModule topk

      ENTRY TopK {
        x = f32[8,%d]{1,0} parameter(0)
        ROOT topk = (f32[8,%d]{1,0}, s32[8,%d]{1,0}) topk(x), k=%d, largest=true
      }
    )",
                                                   n, k, k, k);

      ASSERT_OK_AND_ASSIGN(auto module,
                           ParseAndReturnVerifiedModule(hlo_text_module));

      Array2D<float> expected_vals_arr(8, k);
      Array2D<int32_t> expected_idxs_arr(8, k);
      for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < k; ++j) {
          expected_vals_arr(i, j) = expected_values_64[j];
          expected_idxs_arr(i, j) = expected_indices_64[j];
        }
      }

      auto expected_vals_literal =
          LiteralUtil::CreateR2FromArray2D<float>(expected_vals_arr);
      auto expected_idxs_literal =
          LiteralUtil::CreateR2FromArray2D<int32_t>(expected_idxs_arr);
      auto expected_tuple = LiteralUtil::MakeTuple(
          {&expected_vals_literal, &expected_idxs_literal});

      ASSERT_OK_AND_ASSIGN(Literal actual_tuple,
                           Execute(std::move(module), {&literal}));

      EXPECT_TRUE(LiteralTestUtil::Near(expected_tuple, actual_tuple,
                                        xla::ErrorSpec{0, 0}));
    }
  }
}

bfloat16 ToBfloat16(uint16_t x) { return absl::bit_cast<bfloat16>(x); }

TEST_F(TopkTest, TopKWithSpecialBfloat16) {
  // Regression test for TopK TotalOrder with special float values
  // (NaN, +-Inf, +-0).
  bfloat16 neg_qnan0 = ToBfloat16(0xFFC0u);
  bfloat16 neg_qnan1 = ToBfloat16(0xFFE0u);
  bfloat16 neg_qnan2 = ToBfloat16(0xFFE1u);
  bfloat16 neg_smallest_qnan = ToBfloat16(0xFFFFu);
  bfloat16 pos_qnan0 = ToBfloat16(0x7FC0u);
  bfloat16 pos_qnan1 = ToBfloat16(0x7FE0u);
  bfloat16 pos_qnan2 = ToBfloat16(0x7FE1u);

  bfloat16 pos_inf = ToBfloat16(0x7F80u);
  bfloat16 neg_inf = ToBfloat16(0xFF80u);
  bfloat16 pos_zero = ToBfloat16(0x0000u);
  bfloat16 neg_zero = ToBfloat16(0x8000u);

  bfloat16 pos_dnrm_1e38 = ToBfloat16(0x0070u);  // 1.028e-38
  bfloat16 pos_dnrm_1e40 = ToBfloat16(0x0002u);  // 1.836e-40
  bfloat16 neg_dnrm_1e39 = ToBfloat16(0x8010u);  // -1.469e-39
  bfloat16 neg_dnrm_9e41 = ToBfloat16(0x8001u);  // -9.183e-41

  bfloat16 pos_64_5 = ToBfloat16(0x4281u);
  bfloat16 pos_0_0125 = ToBfloat16(0x3C4du);
  bfloat16 neg_64_5 = ToBfloat16(0xC281u);

  std::vector<bfloat16> input_row_start = {
      neg_inf,    pos_inf,       neg_dnrm_1e39, pos_zero,      pos_qnan0,
      pos_qnan2,  neg_zero,      neg_64_5,      pos_inf,       neg_qnan0,
      neg_qnan2,  neg_64_5,      neg_zero,      pos_dnrm_1e40, neg_dnrm_9e41,
      neg_qnan1,  pos_zero,      neg_inf,       pos_64_5,      pos_qnan1,
      pos_0_0125, pos_dnrm_1e38, neg_dnrm_1e39, pos_dnrm_1e40, neg_dnrm_9e41};

  std::vector<bfloat16> expected_values_64 = {
      pos_qnan0,     pos_qnan0,     pos_qnan0,     pos_inf,       pos_inf,
      pos_64_5,      pos_0_0125,    pos_dnrm_1e38, pos_dnrm_1e40, pos_dnrm_1e40,
      pos_zero,      pos_zero,      neg_zero,      neg_zero,      neg_dnrm_9e41,
      neg_dnrm_9e41, neg_dnrm_1e39, neg_dnrm_1e39, neg_64_5,      neg_64_5,
      neg_inf,       neg_inf,       neg_qnan0,     neg_qnan0,     neg_qnan0,
      neg_qnan0,     neg_qnan0,     neg_qnan0,     neg_qnan0,     neg_qnan0,
      neg_qnan0,     neg_qnan0,     neg_qnan0,     neg_qnan0,     neg_qnan0,
      neg_qnan0,     neg_qnan0,     neg_qnan0,     neg_qnan0,     neg_qnan0,
      neg_qnan0,     neg_qnan0,     neg_qnan0,     neg_qnan0,     neg_qnan0,
      neg_qnan0,     neg_qnan0,     neg_qnan0,     neg_qnan0,     neg_qnan0,
      neg_qnan0,     neg_qnan0,     neg_qnan0,     neg_qnan0,     neg_qnan0,
      neg_qnan0,     neg_qnan0,     neg_qnan0,     neg_qnan0,     neg_qnan0,
      neg_qnan0,     neg_qnan0,     neg_qnan0,     neg_qnan0};

  std::vector<int32_t> expected_indices_64 = {
      5,  19, 4,  1,  8,  18, 20, 21, 13, 23, 3,  16, 6,  12, 14, 24,
      2,  22, 7,  11, 0,  17, 9,  15, 10, 25, 26, 27, 28, 29, 30, 31,
      32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
      48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};

  for (int n : {512, 1024, 32768, 65536 + 4096}) {
    Array2D<bfloat16> input(8, n, neg_smallest_qnan);
    for (int i = 0; i < input_row_start.size(); ++i) {
      for (int j = 0; j < 8; ++j) {
        input(j, i) = input_row_start[i];
      }
    }

    auto literal = LiteralUtil::CreateR2FromArray2D<bfloat16>(input);

    for (int k : {2, 4, 8, 16, 32, 64}) {
      const auto hlo_text_module = absl::StrFormat(R"(
      HloModule topk

      ENTRY TopK {
        x = bf16[8,%d]{1,0} parameter(0)
        ROOT topk = (bf16[8,%d]{1,0}, s32[8,%d]{1,0}) topk(x), k=%d, largest=true
      }
    )",
                                                   n, k, k, k);

      ASSERT_OK_AND_ASSIGN(auto module,
                           ParseAndReturnVerifiedModule(hlo_text_module));

      Array2D<bfloat16> expected_vals_arr(8, k);
      Array2D<int32_t> expected_idxs_arr(8, k);
      for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < k; ++j) {
          expected_vals_arr(i, j) = expected_values_64[j];
          expected_idxs_arr(i, j) = expected_indices_64[j];
        }
      }

      auto expected_vals_literal =
          LiteralUtil::CreateR2FromArray2D<bfloat16>(expected_vals_arr);
      auto expected_idxs_literal =
          LiteralUtil::CreateR2FromArray2D<int32_t>(expected_idxs_arr);
      auto expected_tuple = LiteralUtil::MakeTuple(
          {&expected_vals_literal, &expected_idxs_literal});

      ASSERT_OK_AND_ASSIGN(Literal actual_tuple,
                           Execute(std::move(module), {&literal}));

      EXPECT_TRUE(LiteralTestUtil::Near(expected_tuple, actual_tuple,
                                        xla::ErrorSpec{0, 0}));
    }
  }
}

}  // namespace
}  // namespace xla
