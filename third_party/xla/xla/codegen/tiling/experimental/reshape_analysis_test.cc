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

#include "xla/codegen/tiling/experimental/reshape_analysis.h"

#include <cstdint>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu::experimental {
namespace {

using ::testing::Eq;
using ::testing::Pointwise;

struct ReshapeTestCase {
  std::string name;
  std::vector<int64_t> input_dims;
  std::vector<int64_t> output_dims;
  std::vector<MinimalReshape> expected;
};

class ReshapeAnalysisTest : public ::testing::TestWithParam<ReshapeTestCase> {};

TEST_P(ReshapeAnalysisTest, AnalyzesMinimalReshapes) {
  const auto& param = GetParam();
  Shape input_shape = ShapeUtil::MakeShape(F32, param.input_dims);
  Shape output_shape = ShapeUtil::MakeShape(F32, param.output_dims);
  auto result = GetMinimalReshapes(input_shape, output_shape);
  EXPECT_THAT(result, Pointwise(Eq(), param.expected)) << param.name;
}

INSTANTIATE_TEST_SUITE_P(
    ReshapeAnalysisTests, ReshapeAnalysisTest,
    ::testing::ValuesIn<ReshapeTestCase>({
        {"Identity",
         {10, 20},
         {10, 20},
         {{{0, 1}, {0, 1}, MinimalReshapeCategory::kIdentity},
          {{1, 1}, {1, 1}, MinimalReshapeCategory::kIdentity}}},
        {"IncreaseRank",
         {10},
         {1, 10, 1},
         {{{0, 1}, {0, 3}, MinimalReshapeCategory::kIncreaseRank}}},
        {"DecreaseRank",
         {1, 10, 1},
         {10},
         {{{0, 3}, {0, 1}, MinimalReshapeCategory::kDecreaseRank}}},
        {"IdentityAndDecreaseRank",
         {10, 1, 10, 1},
         {10, 10},
         {{{0, 1}, {0, 1}, MinimalReshapeCategory::kIdentity},
          {{1, 3}, {1, 1}, MinimalReshapeCategory::kDecreaseRank}}},
        {"ExpandShape",
         {12},
         {3, 4},
         {{{0, 1}, {0, 2}, MinimalReshapeCategory::kExpandShape}}},
        {"CollapseShape",
         {3, 4},
         {12},
         {{{0, 2}, {0, 1}, MinimalReshapeCategory::kCollapseShape}}},
        {"Permutation",
         {2, 5, 7},
         {7, 5, 2},
         {{{0, 3}, {0, 3}, MinimalReshapeCategory::kGeneric}}},
        {"Generic",
         {8, 16},
         {4, 32},
         {{{0, 2}, {0, 2}, MinimalReshapeCategory::kGeneric}}},
        {"CollapseShapeAndExpandShape",
         {2, 4, 8},
         {8, 4, 2},
         {{{0, 2}, {0, 1}, MinimalReshapeCategory::kCollapseShape},
          {{2, 1}, {1, 2}, MinimalReshapeCategory::kExpandShape}}},
        {"CollapseShapeAndDecreaseRank",
         {3, 5, 1, 7},
         {15, 7},
         {{{0, 2}, {0, 1}, MinimalReshapeCategory::kCollapseShape},
          {{2, 2}, {1, 1}, MinimalReshapeCategory::kDecreaseRank}}},
        {"CollapseShapeIncludeOne",
         {3, 1, 5},
         {15},
         {{{0, 3}, {0, 1}, MinimalReshapeCategory::kCollapseShape}}},
        {"ExpandShapeIncludeOne",
         {15},
         {3, 1, 5},
         {{{0, 1}, {0, 3}, MinimalReshapeCategory::kExpandShape}}},
        {"SplitsAtSizeOneDimensions",
         {1, 10},
         {1, 10},
         {{{0, 1}, {0, 1}, MinimalReshapeCategory::kIdentity},
          {{1, 1}, {1, 1}, MinimalReshapeCategory::kIdentity}}},
        {"ScalarToTensor",
         {},
         {1, 1},
         {{{0, 0}, {0, 2}, MinimalReshapeCategory::kIncreaseRank}}},
    }),
    [](const ::testing::TestParamInfo<ReshapeAnalysisTest::ParamType>& info) {
      return info.param.name;
    });

}  // namespace
}  // namespace xla::gpu::experimental
