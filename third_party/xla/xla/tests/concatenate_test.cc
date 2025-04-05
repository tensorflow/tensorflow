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
#include <vector>

#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/hlo/testlib/test.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/shape.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {

using ConcatenateTest = HloTestBase;

TEST_F(ConcatenateTest, TwoR3Axis1) {
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

TEST_F(ConcatenateTest, ThreeR2Axis1) {
  const std::string hlo_text_module = R"(
    HloModule module

    ENTRY entry {
      %x = s32[3,2] parameter(0)
      %y = s32[3,2] parameter(1)
      %z = s32[3,2] parameter(2)
      ROOT %cat_axis1 = s32[3,6] concatenate(x, y, z), dimensions={1}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_text_module));

  Literal x_input = LiteralUtil::CreateR2<int32_t>({{0, 1}, {2, 3}, {4, 5}});
  Literal y_input =
      LiteralUtil::CreateR2<int32_t>({{0, -1}, {-2, -3}, {-4, -5}});
  Literal z_input = LiteralUtil::CreateR2<int32_t>({{5, -4}, {-3, -2}, {1, 0}});

  TF_ASSERT_OK_AND_ASSIGN(
      auto result, Execute(std::move(module), {&x_input, &y_input, &z_input}));

  LiteralTestUtil::ExpectR2Equal(
      {{0, 1, 0, -1, 5, -4}, {2, 3, -2, -3, -3, -2}, {4, 5, -4, -5, 1, 0}},
      result);
}

static auto MakeIotaForShape(const Shape& shape) {
  std::vector<int64_t> strides(shape.dimensions().size(), 1);
  for (int i = shape.dimensions().size() - 1; i > 0; --i) {
    strides[i - 1] = strides[i] * shape.dimensions(i);
  }
  return [strides = std::move(strides)](absl::Span<const int64_t> indices) {
    return absl::c_inner_product(indices, strides, 0);
  };
}

static auto MakeNegativeIotaForShape(const Shape& shape) {
  auto iota = MakeIotaForShape(shape);
  return [iota = std::move(iota)](absl::Span<const int64_t> indices) {
    return -iota(indices);
  };
}

TEST_F(ConcatenateTest, TwoR3Axis1Parallel) {
  // This test should run the parallel backend since the shape of the
  // concatenate is big enough to be partitioned. To make sure the parallel
  // backend is used, we provide `outer_dimension_partitions` in the
  // `backend_config` which would not work otherwise.
  const std::string hlo_text_module = R"(
    HloModule module

    ENTRY entry {
      %x = s32[64,64,64] parameter(0)
      %y = s32[64,64,64] parameter(1)
      ROOT %cat_axis1 = s32[64,128,64] concatenate(x, y), dimensions={1}, backend_config={"outer_dimension_partitions": ["1"]}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_text_module));

  Literal x_input = LiteralUtil::CreateFromDimensions(S32, {64, 64, 64});
  TF_CHECK_OK(x_input.Populate<int32_t>(MakeIotaForShape(x_input.shape())));

  Literal y_input = LiteralUtil::CreateFromDimensions(S32, {64, 64, 64});
  TF_CHECK_OK(
      y_input.Populate<int32_t>(MakeNegativeIotaForShape(y_input.shape())));

  TF_ASSERT_OK_AND_ASSIGN(Literal result,
                          Execute(std::move(module), {&x_input, &y_input}));

  // Assert that the result is the concatenation of the two inputs.
  // Iota and negative iota are used to validate that the concatenation produces
  // a correct sequence of elements.
  int64_t positive_value = 0;
  int64_t negative_value = 0;
  for (int x = 0; x < 64; ++x) {
    for (int y = 0; y < 128; ++y) {
      for (int z = 0; z < 64; ++z) {
        SCOPED_TRACE(absl::StrCat("result.Get<int32_t>({",
                                  absl::StrJoin({x, y, z}, ", "), "})"));
        if (y < 64) {
          ASSERT_EQ(result.Get<int32_t>({x, y, z}), positive_value);
          ++positive_value;
        } else {
          ASSERT_EQ(result.Get<int32_t>({x, y, z}), negative_value);
          --negative_value;
        }
      }
    }
  }
}

}  // namespace
}  // namespace xla
