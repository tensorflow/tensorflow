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

#include "xla/codegen/tiling/experimental/tiling_space_utils.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"

namespace xla {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;

TEST(TilingSpaceUtilsTest, EmptyInputSpaceReturnsOneTiling) {
  ASSERT_OK_AND_ASSIGN(auto tilings, GetFlatTilingsForInputSpace({}));
  EXPECT_THAT(tilings, ElementsAre(FlatTiling{}));
}

TEST(TilingSpaceUtilsTest,
     GetFlatTilingsForInputSpace_ReturnsPowersOfTwoAndDimSizeForRankOne) {
  // Rank one, size 1
  {
    ASSERT_OK_AND_ASSIGN(auto tilings, GetFlatTilingsForInputSpace({1}));
    EXPECT_THAT(tilings, ElementsAre(FlatTiling{1}));
  }

  // Rank one, size 4
  {
    ASSERT_OK_AND_ASSIGN(auto tilings, GetFlatTilingsForInputSpace({4}));
    EXPECT_THAT(tilings,
                ElementsAre(FlatTiling{1}, FlatTiling{2}, FlatTiling{4}));
  }

  // Rank one, size 5
  {
    ASSERT_OK_AND_ASSIGN(auto tilings, GetFlatTilingsForInputSpace({5}));
    EXPECT_THAT(tilings, ElementsAre(FlatTiling{1}, FlatTiling{2},
                                     FlatTiling{4}, FlatTiling{5}));
  }

  // Rank one, size 11
  {
    ASSERT_OK_AND_ASSIGN(auto tilings, GetFlatTilingsForInputSpace({11}));
    EXPECT_THAT(tilings,
                ElementsAre(FlatTiling{1}, FlatTiling{2}, FlatTiling{4},
                            FlatTiling{8}, FlatTiling{11}));
  }
}

TEST(TilingSpaceUtilsTest,
     GetFlatTilingsForInputSpace_RankTwoCartesianProduct) {
  ASSERT_OK_AND_ASSIGN(auto tilings, GetFlatTilingsForInputSpace({3, 4}));
  EXPECT_THAT(
      tilings,
      ElementsAre(FlatTiling{1, 1}, FlatTiling{1, 2}, FlatTiling{1, 4},
                  FlatTiling{2, 1}, FlatTiling{2, 2}, FlatTiling{2, 4},
                  FlatTiling{3, 1}, FlatTiling{3, 2}, FlatTiling{3, 4}));
}

TEST(TilingSpaceUtilsTest,
     GetFlatTilingsForInputSpace_RankThreeCartesianProduct) {
  ASSERT_OK_AND_ASSIGN(auto tilings, GetFlatTilingsForInputSpace({3, 4, 2}));
  EXPECT_THAT(
      tilings,
      ElementsAre(FlatTiling{1, 1, 1}, FlatTiling{1, 1, 2}, FlatTiling{1, 2, 1},
                  FlatTiling{1, 2, 2}, FlatTiling{1, 4, 1}, FlatTiling{1, 4, 2},
                  FlatTiling{2, 1, 1}, FlatTiling{2, 1, 2}, FlatTiling{2, 2, 1},
                  FlatTiling{2, 2, 2}, FlatTiling{2, 4, 1}, FlatTiling{2, 4, 2},
                  FlatTiling{3, 1, 1}, FlatTiling{3, 1, 2}, FlatTiling{3, 2, 1},
                  FlatTiling{3, 2, 2}, FlatTiling{3, 4, 1},
                  FlatTiling{3, 4, 2}));
}

TEST(TilingSpaceUtilsTest,
     GetFlatTilingsForInputSpace_FailsForNegativeDimensions) {
  auto tilings_or = GetFlatTilingsForInputSpace({-5});
  EXPECT_THAT(tilings_or.status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Dimension size must be non-negative."));
}

}  // namespace
}  // namespace xla
