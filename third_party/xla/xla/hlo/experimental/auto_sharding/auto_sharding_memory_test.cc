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

#include "xla/hlo/experimental/auto_sharding/auto_sharding_memory.h"

#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/container/btree_set.h"

namespace xla {
namespace spmd {
namespace {

// Converts rows of an int64_t matrix into the repeated fields of a proto.
std::function<tsl::protobuf::RepeatedField<int64_t>(int64_t)>  // NOLINT
Convert(const std::vector<std::vector<int64_t>>& live) {
  return [live](int64_t live_idx) {
    return ::proto2::RepeatedField<int64_t>(live[live_idx].begin(),  // NOLINT
                                            live[live_idx].end());
  };
}

// Returns the interval at the associated primitive index.
std::function<std::pair<int64_t, int64_t>(int64_t)> Convert(
    const std::vector<std::pair<int64_t, int64_t>>& intervals) {
  return [intervals](int64_t prim_idx) { return intervals[prim_idx]; };
}

// clang-format off

//  |    111   ==>  |    111
//  | 000      ==>  | 000
//  +------->  ==>  +------->
//    (time)          (time)
TEST(AutoShardingMemoryTest, WithoutOverlap) {
  const std::vector<std::vector<int64_t>> live =
      {{0   },
       {0   },
       {0   },
       {   1},
       {   1},
       {   1}};

  MemoryTermReducer reducer;
  const auto num_terms =
          reducer.Reduce(live.size(), /*num_primitives=*/2, Convert(live));

  const std::vector<std::vector<int64_t>> expected_reduced_live =
      {{0   },
       {0   },
       {0   },
       {   1},
       {   1},
       {   1}};
  const std::vector<absl::btree_set<int64_t>> expected_reduced_groups = {};
  const std::pair<int64_t, int64_t> expected_num_terms = {6, 6};
  EXPECT_EQ(num_terms, expected_num_terms);
  EXPECT_EQ(reducer.GetReducedLive(), expected_reduced_live);
  EXPECT_EQ(reducer.GetReducedGroups(), expected_reduced_groups);
}

//  |                  2222
//  |  11111   ==>  |  ....1   Groups:
//  | 00000    ==>  | 0....      m[2] = m[0] + m[1]
//  +------->  ==>  +------->
//    (time)          (time)
TEST(AutoShardingMemoryTest, PartialOverlap) {
  const std::vector<std::vector<int64_t>> live =
      {{0   },
       {0, 1},
       {0, 1},
       {0, 1},
       {0, 1},
       {   1}};

  MemoryTermReducer reducer;
  const auto num_terms =
          reducer.Reduce(live.size(), /*num_primitives=*/2, Convert(live));

  const std::vector<std::vector<int64_t>> expected_reduced_live =
      {{0      },
       {      2},
       {      2},
       {      2},
       {      2},
       {   1   }};
  const std::vector<absl::btree_set<int64_t>> expected_reduced_groups =
      {{0, 1}};
  const std::pair<int64_t, int64_t> expected_num_terms = {10, 8};
  EXPECT_EQ(num_terms, expected_num_terms);
  EXPECT_EQ(reducer.GetReducedLive(), expected_reduced_live);
  EXPECT_EQ(reducer.GetReducedGroups(), expected_reduced_groups);
}

//  |                  2222
//  | 11111    ==>  | 1....    Groups:
//  |  00000   ==>  |  ....0     m[2] = m[0] + m[1]
//  +------->  ==>  +------->
//    (time)          (time)
TEST(AutoShardingMemoryTest, PartialOverlapReversed) {
  const std::vector<std::vector<int64_t>> live =
      {{   1},
       {0, 1},
       {0, 1},
       {0, 1},
       {0, 1},
       {0   }};

  MemoryTermReducer reducer;
  const auto num_terms =
          reducer.Reduce(live.size(), /*num_primitives=*/2, Convert(live));

  const std::vector<std::vector<int64_t>> expected_reduced_live =
      {{   1   },
       {      2},
       {      2},
       {      2},
       {      2},
       {0      }};
  const std::vector<absl::btree_set<int64_t>> expected_reduced_groups =
      {{0, 1}};
  const std::pair<int64_t, int64_t> expected_num_terms = {10, 8};
  EXPECT_EQ(num_terms, expected_num_terms);
  EXPECT_EQ(reducer.GetReducedLive(), expected_reduced_live);
  EXPECT_EQ(reducer.GetReducedGroups(), expected_reduced_groups);
}

//  |  1111    ==>  |  1111
//  | 000000   ==>  | 000000
//  +------->  ==>  +------->
//    (time)          (time)
TEST(AutoShardingMemoryTest, DoesNotSplitPrimitive) {
  const std::vector<std::vector<int64_t>> live =
      {{0   },
       {0, 1},
       {0, 1},
       {0, 1},
       {0, 1},
       {0   }};

  MemoryTermReducer reducer;
  const auto num_terms =
          reducer.Reduce(live.size(), /*num_primitives=*/2, Convert(live));

  const std::vector<std::vector<int64_t>> expected_reduced_live =
      {{0   },
       {0, 1},
       {0, 1},
       {0, 1},
       {0, 1},
       {0   }};
  const std::vector<absl::btree_set<int64_t>> expected_reduced_groups = {};
  const std::pair<int64_t, int64_t> expected_num_terms = {10, 10};
  EXPECT_EQ(num_terms, expected_num_terms);
  EXPECT_EQ(reducer.GetReducedLive(), expected_reduced_live);
  EXPECT_EQ(reducer.GetReducedGroups(), expected_reduced_groups);
}

//  |          ==>  |  22222
//  |  11111   ==>  |  .....   Groups:
//  | 000000   ==>  | 0.....     m[2] = m[0] + m[1]
//  +------->  ==>  +------->
//    (time)          (time)
TEST(AutoShardingMemoryTest, OnePrimitiveVanishes) {
  const std::vector<std::vector<int64_t>> live =
      {{0   },
       {0, 1},
       {0, 1},
       {0, 1},
       {0, 1},
       {0, 1}};

  MemoryTermReducer reducer;
  const auto num_terms =
          reducer.Reduce(live.size(), /*num_primitives=*/2, Convert(live));

  const std::vector<std::vector<int64_t>> expected_reduced_live =
      {{0   },
       {   2},
       {   2},
       {   2},
       {   2},
       {   2}};
  const std::vector<absl::btree_set<int64_t>> expected_reduced_groups =
      {{0, 1}};
  const std::pair<int64_t, int64_t> expected_num_terms = {11, 8};
  EXPECT_EQ(num_terms, expected_num_terms);
  EXPECT_EQ(reducer.GetReducedLive(), expected_reduced_live);
  EXPECT_EQ(reducer.GetReducedGroups(), expected_reduced_groups);
}

//  |          ==>  | 222222
//  | 111111   ==>  | ......   Groups:
//  | 000000   ==>  | ......     m[2] = m[0] + m[1]
//  +------->  ==>  +------->
//    (time)          (time)
TEST(AutoShardingMemoryTest, BothPrimitivesVanish) {
  const std::vector<std::vector<int64_t>> live =
      {{0, 1},
       {0, 1},
       {0, 1},
       {0, 1},
       {0, 1},
       {0, 1}};

  MemoryTermReducer reducer;
  const auto num_terms =
          reducer.Reduce(live.size(), /*num_primitives=*/2, Convert(live));

  const std::vector<std::vector<int64_t>> expected_reduced_live =
      {{2},
       {2},
       {2},
       {2},
       {2},
       {2}};
  const std::vector<absl::btree_set<int64_t>> expected_reduced_groups =
      {{0, 1}};
  const std::pair<int64_t, int64_t> expected_num_terms = {12, 8};
  EXPECT_EQ(num_terms, expected_num_terms);
  EXPECT_EQ(reducer.GetReducedLive(), expected_reduced_live);
  EXPECT_EQ(reducer.GetReducedGroups(), expected_reduced_groups);
}

//  |            ==>  | 33333
//  |     22222  ==>  |     22222
//  | 11111111   ==>  | .....111   Groups:
//  | 00000      ==>  | .....        m[3] = m[0] + m[1]
//  +--------->  ==>  +--------->
//    (time)            (time)
TEST(AutoShardingMemoryTest, OneGroupingPreventsAnother) {
  const std::vector<std::vector<int64_t>> live =
      {{0, 1   },
       {0, 1   },
       {0, 1   },
       {0, 1   },
       {0, 1, 2},
       {   1, 2},
       {   1, 2},
       {   1, 2},
       {      2}};

  MemoryTermReducer reducer;
  const auto num_terms =
          reducer.Reduce(live.size(), /*num_primitives=*/3, Convert(live));

  const std::vector<std::vector<int64_t>> expected_reduced_live =
      {{      3},
       {      3},
       {      3},
       {      3},
       {   2, 3},
       {1, 2   },
       {1, 2   },
       {1, 2   },
       {   2   }};
  const std::vector<absl::btree_set<int64_t>> expected_reduced_groups =
      {{0, 1}};
  const std::pair<int64_t, int64_t> expected_num_terms = {18, 15};
  EXPECT_EQ(num_terms, expected_num_terms);
  EXPECT_EQ(reducer.GetReducedLive(), expected_reduced_live);
  EXPECT_EQ(reducer.GetReducedGroups(), expected_reduced_groups);
}

//  |          ==>  | 333444
//  |    222   ==>  |    ...   Groups:
//  | 111      ==>  | ...        m[3] = m[0] + m[1]
//  | 000000   ==>  | ......     m[4] = m[0] + m[2]
//  +------->  ==>  +------->
//    (time)          (time)
TEST(AutoShardingMemoryTest, TwoGroups) {
  const std::vector<std::vector<int64_t>> live =
      {{0, 1   },
       {0, 1   },
       {0, 1   },
       {0,    2},
       {0,    2},
       {0,    2}};

  MemoryTermReducer reducer;
  const auto num_terms =
          reducer.Reduce(live.size(), /*num_primitives=*/3, Convert(live));

  const std::vector<std::vector<int64_t>> expected_reduced_live =
      {{3},
       {3},
       {3},
       {4},
       {4},
       {4}};
  const std::vector<absl::btree_set<int64_t>> expected_reduced_groups =
      {{0, 1}, {0, 2}};
  const std::pair<int64_t, int64_t> expected_num_terms = {12, 10};
  EXPECT_EQ(num_terms, expected_num_terms);
  EXPECT_EQ(reducer.GetReducedLive(), expected_reduced_live);
  EXPECT_EQ(reducer.GetReducedGroups(), expected_reduced_groups);
}

//  |           ==>  |  444555
//  |     3333  ==>  |     ...3
//  |     222   ==>  |     ...   Groups:
//  |  111      ==>  |  ...        m[4] = m[0] + m[1]
//  | 0000      ==>  | 0...        m[5] = m[2] + m[3]
//  +------->   ==>  +------->
//    (time)           (time)
TEST(AutoShardingMemoryTest, TwoGroupsMutuallyExclusive) {
  const std::vector<std::vector<int64_t>> live =
      {{0         },
       {0, 1      },
       {0, 1      },
       {0, 1      },
       {      2, 3},
       {      2, 3},
       {      2, 3},
       {         3}};

  MemoryTermReducer reducer;
  const auto num_terms =
          reducer.Reduce(live.size(), /*num_primitives=*/4, Convert(live));

  const std::vector<std::vector<int64_t>> expected_reduced_live =
      {{0      },
       {      4},
       {      4},
       {      4},
       {      5},
       {      5},
       {      5},
       {   3   }};
  const std::vector<absl::btree_set<int64_t>> expected_reduced_groups =
      {{0, 1}, {2, 3}};
  const std::pair<int64_t, int64_t> expected_num_terms = {14, 12};
  EXPECT_EQ(num_terms, expected_num_terms);
  EXPECT_EQ(reducer.GetReducedLive(), expected_reduced_live);
  EXPECT_EQ(reducer.GetReducedGroups(), expected_reduced_groups);
}

//  | 11      ==>  | 11
//  | 00      ==>  | 00
//  +------>  ==>  +------>
//   (time)         (time)
TEST(AutoShardingMemoryTest, MergingPrimitivesWouldNotReduceTerms) {
  const std::vector<std::vector<int64_t>> live =
      {{0, 1},
       {0, 1}};

  MemoryTermReducer reducer;
  const auto num_terms =
          reducer.Reduce(live.size(), /*num_primitives=*/2, Convert(live));

  const std::vector<std::vector<int64_t>> expected_reduced_live =
      {{0, 1},
       {0, 1}};
  const std::vector<absl::btree_set<int64_t>> expected_reduced_groups = {};
  const std::pair<int64_t, int64_t> expected_num_terms = {4, 4};
  EXPECT_EQ(num_terms, expected_num_terms);
  EXPECT_EQ(reducer.GetReducedLive(), expected_reduced_live);
  EXPECT_EQ(reducer.GetReducedGroups(), expected_reduced_groups);
}

//  |          ==>  | 333333
//  | 222222   ==>  | ......
//  | 111111   ==>  | ......   Groups:
//  | 000000   ==>  | ......     m[3] = m[0] + m[1] + m[2]
//  +------->  ==>  +------->
//    (time)          (time)
TEST(AutoShardingMemoryTest, AllPrimitivesVanish) {
  const std::vector<std::vector<int64_t>> live =
      {{0, 1, 2},
       {0, 1, 2},
       {0, 1, 2},
       {0, 1, 2},
       {0, 1, 2},
       {0, 1, 2}};

  MemoryTermReducer reducer;
  const auto num_terms =
          reducer.Reduce(live.size(), /*num_primitives=*/3, Convert(live));

  const std::vector<std::vector<int64_t>> expected_reduced_live =
      {{3},
       {3},
       {3},
       {3},
       {3},
       {3}};
  const std::vector<absl::btree_set<int64_t>> expected_reduced_groups =
      {{0, 1, 2}};
  const std::pair<int64_t, int64_t> expected_num_terms = {18, 9};
  EXPECT_EQ(num_terms, expected_num_terms);
  EXPECT_EQ(reducer.GetReducedLive(), expected_reduced_live);
  EXPECT_EQ(reducer.GetReducedGroups(), expected_reduced_groups);
}

//  |            ==>  |    555555
//  |            ==>  | 4444444
//  |    333333  ==>  |    ......
//  |    222222  ==>  |    ......  Groups:
//  | 1111111    ==>  | .......      m[4] = m[0] + m[1]
//  | 0000000    ==>  | .......      m[5] = m[2] + m[3]
//  +--------->  ==>  +--------->
//     (time)            (time)
TEST(AutoShardingMemoryTest, MergingGroupsWouldNotReduceTerms) {
  const std::vector<std::vector<int64_t>> live =
      {{0, 1      },
       {0, 1      },
       {0, 1      },
       {0, 1, 2, 3},
       {0, 1, 2, 3},
       {0, 1, 2, 3},
       {0, 1, 2, 3},
       {      2, 3},
       {      2, 3}};

  MemoryTermReducer reducer;
  const auto num_terms =
          reducer.Reduce(live.size(), /*num_primitives=*/4, Convert(live));

  const std::vector<std::vector<int64_t>> expected_reduced_live =
      {{4   },
       {4   },
       {4   },
       {4, 5},
       {4, 5},
       {4, 5},
       {4, 5},
       {   5},
       {   5}};
  const std::vector<absl::btree_set<int64_t>> expected_reduced_groups =
      {{0, 1}, {2, 3}};
  const std::pair<int64_t, int64_t> expected_num_terms = {26, 17};
  EXPECT_EQ(num_terms, expected_num_terms);
  EXPECT_EQ(reducer.GetReducedLive(), expected_reduced_live);
  EXPECT_EQ(reducer.GetReducedGroups(), expected_reduced_groups);
}

//  |                      |  444466666555
//  |      333333333  ==>  |      ........3  Groups:
//  |      22222222   ==>  |      ........     m[4] = m[0] + m[1]
//  |  111111111      ==>  |  .........        m[5] = m[2] + m[3]
//  | 0000000000      ==>  | 0.........        m[6] = m[0] + m[1] + m[2] + m[3]
//  +-------------->  ==>  +-------------->
//       (time)                 (time)
TEST(AutoShardingMemoryTest, ExampleFromDocumentation) {
  const std::vector<std::vector<int64_t>> live =
      {{0         },
       {0, 1      },
       {0, 1      },
       {0, 1      },
       {0, 1      },
       {0, 1, 2, 3},
       {0, 1, 2, 3},
       {0, 1, 2, 3},
       {0, 1, 2, 3},
       {0, 1, 2, 3},
       {      2, 3},
       {      2, 3},
       {      2, 3},
       {         3}};

  MemoryTermReducer reducer;
  const auto num_terms =
          reducer.Reduce(live.size(), /*num_primitives=*/4, Convert(live));

  const std::vector<std::vector<int64_t>> expected_reduced_live =
      {{0      },
       {      4},
       {      4},
       {      4},
       {      4},
       {      6},
       {      6},
       {      6},
       {      6},
       {      6},
       {      5},
       {      5},
       {      5},
       {   3   }};
  const std::vector<absl::btree_set<int64_t>> expected_reduced_groups =
      {{0, 1}, {2, 3}, {0, 1, 2, 3}};
  const std::pair<int64_t, int64_t> expected_num_terms = {36, 22};
  EXPECT_EQ(num_terms, expected_num_terms);
  EXPECT_EQ(reducer.GetReducedLive(), expected_reduced_live);
  EXPECT_EQ(reducer.GetReducedGroups(), expected_reduced_groups);
}

//  |         ==>  | 333
//  | 2222    ==>  | ...2
//  |    1    ==>  |    1    Groups:
//  | 000     ==>  | ...       m[3] = m[0] + m[2]
//  +------>  ==>  +------>
//   (time)         (time)
TEST(AutoShardingMemoryTest, MergesWithRightmost) {
  const std::vector<std::vector<int64_t>> live =
      {{0,    2},
       {0,    2},
       {0,    2},
       {   1, 2}};

  MemoryTermReducer reducer;
  const auto num_terms =
          reducer.Reduce(live.size(), /*num_primitives=*/3, Convert(live));

  const std::vector<std::vector<int64_t>> expected_reduced_live =
      {{      3},
       {      3},
       {      3},
       {1, 2   }};
  const std::vector<absl::btree_set<int64_t>> expected_reduced_groups =
      {{0, 2}};
  const std::pair<int64_t, int64_t> expected_num_terms = {8, 7};
  EXPECT_EQ(num_terms, expected_num_terms);
  EXPECT_EQ(reducer.GetReducedLive(), expected_reduced_live);
  EXPECT_EQ(reducer.GetReducedGroups(), expected_reduced_groups);
}

//  |                      |  444466666555
//  |      333333333  ==>  |      ........3  Groups:
//  |      22222222   ==>  |      ........     m[4] = m[0] + m[1]
//  |  111111111      ==>  |  .........        m[5] = m[2] + m[3]
//  | 0000000000      ==>  | 0.........        m[6] = m[0] + m[1] + m[2] + m[3]
//  +-------------->  ==>  +-------------->
//       (time)                 (time)
TEST(AutoShardingMemoryTest, ExampleFromDocumentationUsingIntervals) {
  const std::vector<std::pair<int64_t, int64_t>> intervals =
      {{0, 9}, {1, 9}, {5, 12}, {5, 13}};

  MemoryTermReducer reducer;
  const auto num_terms = reducer.Reduce(/*num_lives=*/14, /*num_primitives=*/4,
                                        Convert(intervals));

  const std::vector<std::vector<int64_t>> expected_reduced_live = {};
  const std::vector<std::pair<int64_t, int64_t>> expected_reduced_intervals =
      {{0, 0}, {10, 0}, {13, 4}, {13, 13}, {1, 4}, {10, 12}, {5, 9}};
  const std::vector<absl::btree_set<int64_t>> expected_reduced_groups =
      {{0, 1}, {2, 3}, {0, 1, 2, 3}};
  const std::pair<int64_t, int64_t> expected_num_terms = {36, 22};
  EXPECT_EQ(num_terms, expected_num_terms);
  EXPECT_EQ(reducer.GetReducedLive(), expected_reduced_live);
  EXPECT_EQ(reducer.GetReducedIntervals(), expected_reduced_intervals);
  EXPECT_EQ(reducer.GetReducedGroups(), expected_reduced_groups);
}

TEST(AutoShardingMemoryTest, InvalidIntervals) {
  const std::vector<std::pair<int64_t, int64_t>> intervals =
      {{0, 4}, {9223372036854775807, 0}, {9223372036854775807, 0}};

  MemoryTermReducer reducer;
  const auto num_terms = reducer.Reduce(/*num_lives=*/5, /*num_primitives=*/3,
                                        Convert(intervals));

  const std::vector<std::vector<int64_t>> expected_reduced_live = {};
  const std::vector<std::pair<int64_t, int64_t>> expected_reduced_intervals =
      {{0, 4}, {9223372036854775807, 0}, {9223372036854775807, 0}};
  const std::vector<absl::btree_set<int64_t>> expected_reduced_groups = {};
  const std::pair<int64_t, int64_t> expected_num_terms = {5, 5};
  EXPECT_EQ(num_terms, expected_num_terms);
  EXPECT_EQ(reducer.GetReducedLive(), expected_reduced_live);
  EXPECT_EQ(reducer.GetReducedIntervals(), expected_reduced_intervals);
  EXPECT_EQ(reducer.GetReducedGroups(), expected_reduced_groups);
}

//  |                 ==>  |      55555555
//  |                 ==>  |  444444444
//  |      333333333  ==>  |      ........3
//  |      22222222   ==>  |      ........   Groups:
//  |  111111111      ==>  |  .........        m[4] = m[0] + m[1]
//  | 0000000000      ==>  | 0.........        m[5] = m[2] + m[3]
//  +-------------->  ==>  +-------------->
//       (time)                 (time)
TEST(AutoShardingMemoryTest, OneIterationOnly) {
  const std::vector<std::pair<int64_t, int64_t>> intervals =
      {{0, 9}, {1, 9}, {5, 12}, {5, 13}};

  MemoryTermReducer reducer;
  const auto num_terms = reducer.Reduce(/*num_lives=*/14, /*num_primitives=*/4,
                                        Convert(intervals),
                                        /*max_iterations=*/1);

  const std::vector<std::vector<int64_t>> expected_reduced_live = {};
  const std::vector<std::pair<int64_t, int64_t>> expected_reduced_intervals =
      {{0, 0}, {10, 0}, {13, 4}, {13, 13}, {1, 9}, {5, 12}};
  const std::vector<absl::btree_set<int64_t>> expected_reduced_groups =
      {{0, 1}, {2, 3}};
  const std::pair<int64_t, int64_t> expected_num_terms = {36, 23};
  EXPECT_EQ(num_terms, expected_num_terms);
  EXPECT_EQ(reducer.GetReducedLive(), expected_reduced_live);
  EXPECT_EQ(reducer.GetReducedIntervals(), expected_reduced_intervals);
  EXPECT_EQ(reducer.GetReducedGroups(), expected_reduced_groups);
}

//  |                 ==>  | 55555
//  |                 ==>  | 44444444444
//  | 33333           ==>  | .....              Groups:
//  | 22222222        ==>  | .....222            m[4] = m[0] + m[1]
//  | 11111111111     ==>  | ...........         m[5] = m[2] + m[3]
//  | 00000000000000  ==>  | ...........000
//  +-------------->  ==>  +-------------->
//       (time)                 (time)
TEST(AutoShardingMemoryTest, StairsBottomLeft) {
  const std::vector<std::pair<int64_t, int64_t>> intervals =
      {{0, 13}, {0, 10}, {0, 7}, {0, 4}};

  MemoryTermReducer reducer;
  const auto num_terms = reducer.Reduce(/*num_lives=*/14, /*num_primitives=*/4,
                                        Convert(intervals),
                                        /*max_iterations=*/1);

  const std::vector<std::vector<int64_t>> expected_reduced_live = {};
  const std::vector<std::pair<int64_t, int64_t>> expected_reduced_intervals =
      {{11, 13}, {11, -1}, {5, 7}, {5, -1}, {0, 10}, {0, 4}};
  const std::vector<absl::btree_set<int64_t>> expected_reduced_groups =
      {{0, 1}, {2, 3}};
  const std::pair<int64_t, int64_t> expected_num_terms = {38, 26};
  EXPECT_EQ(num_terms, expected_num_terms);
  EXPECT_EQ(reducer.GetReducedLive(), expected_reduced_live);
  EXPECT_EQ(reducer.GetReducedIntervals(), expected_reduced_intervals);
  EXPECT_EQ(reducer.GetReducedGroups(), expected_reduced_groups);
}

//  |                 ==>  | 55555
//  |                 ==>  | 44444444444
//  | 33333333333333  ==>  | ...........333  Groups:
//  | 22222222222     ==>  | ...........      m[4] = m[2] + m[3]
//  | 11111111        ==>  | .....111         m[5] = m[0] + m[1]
//  | 00000           ==>  | .....
//  +-------------->  ==>  +-------------->
//       (time)                 (time)
TEST(AutoShardingMemoryTest, StairsTopLeft) {
  const std::vector<std::pair<int64_t, int64_t>> intervals =
      {{0, 4}, {0, 7}, {0, 10}, {0, 13}};

  MemoryTermReducer reducer;
  const auto num_terms = reducer.Reduce(/*num_lives=*/14, /*num_primitives=*/4,
                                        Convert(intervals),
                                        /*max_iterations=*/1);

  const std::vector<std::vector<int64_t>> expected_reduced_live = {};
  const std::vector<std::pair<int64_t, int64_t>> expected_reduced_intervals =
      {{5, -1}, {5, 7}, {11, -1}, {11, 13}, {0, 10}, {0, 4}};
  const std::vector<absl::btree_set<int64_t>> expected_reduced_groups =
      {{2, 3}, {0, 1}};
  const std::pair<int64_t, int64_t> expected_num_terms = {38, 26};
  EXPECT_EQ(num_terms, expected_num_terms);
  EXPECT_EQ(reducer.GetReducedLive(), expected_reduced_live);
  EXPECT_EQ(reducer.GetReducedIntervals(), expected_reduced_intervals);
  EXPECT_EQ(reducer.GetReducedGroups(), expected_reduced_groups);
}

//  |                 ==>  |          55555
//  |                 ==>  |    44444444444
//  | 33333333333333  ==>  | 333...........  Groups:
//  |    22222222222  ==>  |    ...........    m[4] = m[2] + m[3]
//  |       11111111  ==>  |       111.....    m[5] = m[0] + m[1]
//  |          00000  ==>  |          .....
//  +-------------->  ==>  +-------------->
//       (time)                 (time)
TEST(AutoShardingMemoryTest, StairsTopRight) {
  const std::vector<std::pair<int64_t, int64_t>> intervals =
      {{9, 13}, {6, 13}, {3, 13}, {0, 13}};

  MemoryTermReducer reducer;
  const auto num_terms = reducer.Reduce(/*num_lives=*/14, /*num_primitives=*/4,
                                        Convert(intervals),
                                        /*max_iterations=*/1);

  const std::vector<std::vector<int64_t>> expected_reduced_live = {};
  const std::vector<std::pair<int64_t, int64_t>> expected_reduced_intervals =
      {{14, 8}, {6, 8}, {14, 2}, {0, 2}, {3, 13}, {9, 13}};
  const std::vector<absl::btree_set<int64_t>> expected_reduced_groups =
      {{2, 3}, {0, 1}};
  const std::pair<int64_t, int64_t> expected_num_terms = {38, 26};
  EXPECT_EQ(num_terms, expected_num_terms);
  EXPECT_EQ(reducer.GetReducedLive(), expected_reduced_live);
  EXPECT_EQ(reducer.GetReducedIntervals(), expected_reduced_intervals);
  EXPECT_EQ(reducer.GetReducedGroups(), expected_reduced_groups);
}

//  |                 ==>  |          55555
//  |                 ==>  |    44444444444
//  |          33333  ==>  |          .....  Groups:
//  |       22222222  ==>  |       222.....    m[4] = m[0] + m[1]
//  |    11111111111  ==>  |    ...........    m[5] = m[2] + m[3]
//  | 00000000000000  ==>  | 000...........
//  +-------------->  ==>  +-------------->
//       (time)                 (time)
TEST(AutoShardingMemoryTest, StairsBottomRight) {
  const std::vector<std::pair<int64_t, int64_t>> intervals =
      {{0, 13}, {3, 13}, {6, 13}, {9, 13}};

  MemoryTermReducer reducer;
  const auto num_terms = reducer.Reduce(/*num_lives=*/14, /*num_primitives=*/4,
                                        Convert(intervals),
                                        /*max_iterations=*/1);

  const std::vector<std::vector<int64_t>> expected_reduced_live = {};
  const std::vector<std::pair<int64_t, int64_t>> expected_reduced_intervals =
      {{0, 2}, {14, 2}, {6, 8}, {14, 8}, {3, 13}, {9, 13}};
  const std::vector<absl::btree_set<int64_t>> expected_reduced_groups =
      {{0, 1}, {2, 3}};
  const std::pair<int64_t, int64_t> expected_num_terms = {38, 26};
  EXPECT_EQ(num_terms, expected_num_terms);
  EXPECT_EQ(reducer.GetReducedLive(), expected_reduced_live);
  EXPECT_EQ(reducer.GetReducedIntervals(), expected_reduced_intervals);
  EXPECT_EQ(reducer.GetReducedGroups(), expected_reduced_groups);
}

// clang-format on

}  // namespace
}  // namespace spmd
}  // namespace xla
