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
#include "absl/container/flat_hash_set.h"

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
  const int num_primitives = 2;
  const std::vector<std::vector<int64_t>> live =
      {{0   },
       {0   },
       {0   },
       {   1},
       {   1},
       {   1}};

  MemoryTermReducer reducer;
  const auto num_terms =
          reducer.Reduce(live.size(), num_primitives, Convert(live));

  const std::vector<std::vector<int64_t>> expected_reduced_live =
      {{0   },
       {0   },
       {0   },
       {   1},
       {   1},
       {   1}};
  const std::vector<std::pair<int64_t, int64_t>> expected_reduced_intervals =
      {{0, 2}, {3, 5}};
  const std::vector<absl::btree_set<int64_t>> expected_reduced_groups = {};
  const std::pair<int64_t, int64_t> expected_num_terms = {6, 6};
  const absl::flat_hash_set<int64_t> expected_reduced_times = {0, 3};
  EXPECT_EQ(num_terms, expected_num_terms);
  EXPECT_EQ(reducer.GetReducedLive(), expected_reduced_live);
  EXPECT_EQ(reducer.GetReducedIntervals(), expected_reduced_intervals);
  EXPECT_EQ(reducer.GetReducedGroups(), expected_reduced_groups);
  EXPECT_EQ(reducer.GetReducedTimes(num_primitives), expected_reduced_times);
}

//  |                  2222
//  |  11111   ==>  |  ....1   Groups:
//  | 00000    ==>  | 0....      m[2] = m[0] + m[1]
//  +------->  ==>  +------->
//    (time)          (time)
TEST(AutoShardingMemoryTest, PartialOverlap) {
  const int num_primitives = 2;
  const std::vector<std::vector<int64_t>> live =
      {{0   },
       {0, 1},
       {0, 1},
       {0, 1},
       {0, 1},
       {   1}};

  MemoryTermReducer reducer;
  const auto num_terms =
          reducer.Reduce(live.size(), num_primitives, Convert(live));

  const std::vector<std::vector<int64_t>> expected_reduced_live =
      {{0      },
       {      2},
       {      2},
       {      2},
       {      2},
       {   1   }};
  const std::vector<std::pair<int64_t, int64_t>> expected_reduced_intervals =
      {{0, 0}, {5, 5}, {1, 4}};
  const std::vector<absl::btree_set<int64_t>> expected_reduced_groups =
      {{0, 1}};
  const std::pair<int64_t, int64_t> expected_num_terms = {10, 8};
  const absl::flat_hash_set<int64_t> expected_reduced_times = {1};
  EXPECT_EQ(num_terms, expected_num_terms);
  EXPECT_EQ(reducer.GetReducedLive(), expected_reduced_live);
  EXPECT_EQ(reducer.GetReducedIntervals(), expected_reduced_intervals);
  EXPECT_EQ(reducer.GetReducedGroups(), expected_reduced_groups);
  EXPECT_EQ(reducer.GetReducedTimes(num_primitives), expected_reduced_times);
}

//  |                  2222
//  | 11111    ==>  | 1....    Groups:
//  |  00000   ==>  |  ....0     m[2] = m[0] + m[1]
//  +------->  ==>  +------->
//    (time)          (time)
TEST(AutoShardingMemoryTest, PartialOverlapReversed) {
  const int num_primitives = 2;
  const std::vector<std::vector<int64_t>> live =
      {{   1},
       {0, 1},
       {0, 1},
       {0, 1},
       {0, 1},
       {0   }};

  MemoryTermReducer reducer;
  const auto num_terms =
          reducer.Reduce(live.size(), num_primitives, Convert(live));

  const std::vector<std::vector<int64_t>> expected_reduced_live =
      {{   1   },
       {      2},
       {      2},
       {      2},
       {      2},
       {0      }};
  const std::vector<std::pair<int64_t, int64_t>> expected_reduced_intervals =
      {{5, 5}, {0, 0}, {1, 4}};
  const std::vector<absl::btree_set<int64_t>> expected_reduced_groups =
      {{0, 1}};
  const std::pair<int64_t, int64_t> expected_num_terms = {10, 8};
  const absl::flat_hash_set<int64_t> expected_reduced_times = {1};
  EXPECT_EQ(num_terms, expected_num_terms);
  EXPECT_EQ(reducer.GetReducedLive(), expected_reduced_live);
  EXPECT_EQ(reducer.GetReducedIntervals(), expected_reduced_intervals);
  EXPECT_EQ(reducer.GetReducedGroups(), expected_reduced_groups);
  EXPECT_EQ(reducer.GetReducedTimes(num_primitives), expected_reduced_times);
}

//  |  1111    ==>  |  1111
//  | 000000   ==>  | 000000
//  +------->  ==>  +------->
//    (time)          (time)
TEST(AutoShardingMemoryTest, DoesNotSplitPrimitive) {
  const int num_primitives = 2;
  const std::vector<std::vector<int64_t>> live =
      {{0   },
       {0, 1},
       {0, 1},
       {0, 1},
       {0, 1},
       {0   }};

  MemoryTermReducer reducer;
  const auto num_terms =
          reducer.Reduce(live.size(), num_primitives, Convert(live));

  const std::vector<std::vector<int64_t>> expected_reduced_live =
      {{0   },
       {0, 1},
       {0, 1},
       {0, 1},
       {0, 1},
       {0   }};
  const std::vector<std::pair<int64_t, int64_t>> expected_reduced_intervals =
      {{0, 5}, {1, 4}};
  const std::vector<absl::btree_set<int64_t>> expected_reduced_groups = {};
  const std::pair<int64_t, int64_t> expected_num_terms = {10, 10};
  const absl::flat_hash_set<int64_t> expected_reduced_times = {1};
  EXPECT_EQ(num_terms, expected_num_terms);
  EXPECT_EQ(reducer.GetReducedLive(), expected_reduced_live);
  EXPECT_EQ(reducer.GetReducedIntervals(), expected_reduced_intervals);
  EXPECT_EQ(reducer.GetReducedGroups(), expected_reduced_groups);
  EXPECT_EQ(reducer.GetReducedTimes(num_primitives), expected_reduced_times);
}

//  |          ==>  |  22222
//  |  11111   ==>  |  .....   Groups:
//  | 000000   ==>  | 0.....     m[2] = m[0] + m[1]
//  +------->  ==>  +------->
//    (time)          (time)
TEST(AutoShardingMemoryTest, OnePrimitiveVanishes) {
  const int num_primitives = 2;
  const std::vector<std::vector<int64_t>> live =
      {{0   },
       {0, 1},
       {0, 1},
       {0, 1},
       {0, 1},
       {0, 1}};

  MemoryTermReducer reducer;
  const auto num_terms =
          reducer.Reduce(live.size(), num_primitives, Convert(live));

  const std::vector<std::vector<int64_t>> expected_reduced_live =
      {{0   },
       {   2},
       {   2},
       {   2},
       {   2},
       {   2}};
  const std::vector<std::pair<int64_t, int64_t>> expected_reduced_intervals =
      {{0, 0}, {6, 0}, {1, 5}};
  const std::vector<absl::btree_set<int64_t>> expected_reduced_groups =
      {{0, 1}};
  const std::pair<int64_t, int64_t> expected_num_terms = {11, 8};
  const absl::flat_hash_set<int64_t> expected_reduced_times = {1};
  EXPECT_EQ(num_terms, expected_num_terms);
  EXPECT_EQ(reducer.GetReducedLive(), expected_reduced_live);
  EXPECT_EQ(reducer.GetReducedIntervals(), expected_reduced_intervals);
  EXPECT_EQ(reducer.GetReducedGroups(), expected_reduced_groups);
  EXPECT_EQ(reducer.GetReducedTimes(num_primitives), expected_reduced_times);
}

//  |          ==>  | 222222
//  | 111111   ==>  | ......   Groups:
//  | 000000   ==>  | ......     m[2] = m[0] + m[1]
//  +------->  ==>  +------->
//    (time)          (time)
TEST(AutoShardingMemoryTest, BothPrimitivesVanish) {
  const int num_primitives = 2;
  const std::vector<std::vector<int64_t>> live =
      {{0, 1},
       {0, 1},
       {0, 1},
       {0, 1},
       {0, 1},
       {0, 1}};

  MemoryTermReducer reducer;
  const auto num_terms =
          reducer.Reduce(live.size(), num_primitives, Convert(live));

  const std::vector<std::vector<int64_t>> expected_reduced_live =
      {{2},
       {2},
       {2},
       {2},
       {2},
       {2}};
  const std::vector<std::pair<int64_t, int64_t>> expected_reduced_intervals =
      {{6, -1}, {6, -1}, {0, 5}};
  const std::vector<absl::btree_set<int64_t>> expected_reduced_groups =
      {{0, 1}};
  const std::pair<int64_t, int64_t> expected_num_terms = {12, 8};
  const absl::flat_hash_set<int64_t> expected_reduced_times = {0};
  EXPECT_EQ(num_terms, expected_num_terms);
  EXPECT_EQ(reducer.GetReducedLive(), expected_reduced_live);
  EXPECT_EQ(reducer.GetReducedIntervals(), expected_reduced_intervals);
  EXPECT_EQ(reducer.GetReducedGroups(), expected_reduced_groups);
  EXPECT_EQ(reducer.GetReducedTimes(num_primitives), expected_reduced_times);
}

//  |            ==>  | 33333
//  |     22222  ==>  |     22222
//  | 11111111   ==>  | .....111   Groups:
//  | 00000      ==>  | .....        m[3] = m[0] + m[1]
//  +--------->  ==>  +--------->
//    (time)            (time)
TEST(AutoShardingMemoryTest, OneGroupingPreventsAnother) {
  const int num_primitives = 3;
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
          reducer.Reduce(live.size(), num_primitives, Convert(live));

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
  const std::vector<std::pair<int64_t, int64_t>> expected_reduced_intervals =
      {{5, -1}, {5, 7}, {4, 8}, {0, 4}};
  const std::vector<absl::btree_set<int64_t>> expected_reduced_groups =
      {{0, 1}};
  const std::pair<int64_t, int64_t> expected_num_terms = {18, 15};
  const absl::flat_hash_set<int64_t> expected_reduced_times = {4};
  EXPECT_EQ(num_terms, expected_num_terms);
  EXPECT_EQ(reducer.GetReducedLive(), expected_reduced_live);
  EXPECT_EQ(reducer.GetReducedIntervals(), expected_reduced_intervals);
  EXPECT_EQ(reducer.GetReducedGroups(), expected_reduced_groups);
  EXPECT_EQ(reducer.GetReducedTimes(num_primitives), expected_reduced_times);
}

//  |          ==>  | 333444
//  |    222   ==>  |    ...   Groups:
//  | 111      ==>  | ...        m[3] = m[0] + m[1]
//  | 000000   ==>  | ......     m[4] = m[0] + m[2]
//  +------->  ==>  +------->
//    (time)          (time)
TEST(AutoShardingMemoryTest, TwoGroups) {
  const int num_primitives = 3;
  const std::vector<std::vector<int64_t>> live =
      {{0, 1   },
       {0, 1   },
       {0, 1   },
       {0,    2},
       {0,    2},
       {0,    2}};

  MemoryTermReducer reducer;
  const auto num_terms =
          reducer.Reduce(live.size(), num_primitives, Convert(live));

  const std::vector<std::vector<int64_t>> expected_reduced_live =
      {{3},
       {3},
       {3},
       {4},
       {4},
       {4}};
  const std::vector<std::pair<int64_t, int64_t>> expected_reduced_intervals =
      {{6, 2}, {3, -1}, {6, 2}, {0, 2}, {3, 5}};
  const std::vector<absl::btree_set<int64_t>> expected_reduced_groups =
      {{0, 1}, {0, 2}};
  const std::pair<int64_t, int64_t> expected_num_terms = {12, 10};
  const absl::flat_hash_set<int64_t> expected_reduced_times = {0, 3};
  EXPECT_EQ(num_terms, expected_num_terms);
  EXPECT_EQ(reducer.GetReducedLive(), expected_reduced_live);
  EXPECT_EQ(reducer.GetReducedIntervals(), expected_reduced_intervals);
  EXPECT_EQ(reducer.GetReducedGroups(), expected_reduced_groups);
  EXPECT_EQ(reducer.GetReducedTimes(num_primitives), expected_reduced_times);
}

//  |           ==>  |  444555
//  |     3333  ==>  |     ...3
//  |     222   ==>  |     ...   Groups:
//  |  111      ==>  |  ...        m[4] = m[0] + m[1]
//  | 0000      ==>  | 0...        m[5] = m[2] + m[3]
//  +------->   ==>  +------->
//    (time)           (time)
TEST(AutoShardingMemoryTest, TwoGroupsMutuallyExclusive) {
  const int num_primitives = 4;
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
          reducer.Reduce(live.size(), num_primitives, Convert(live));

  const std::vector<std::vector<int64_t>> expected_reduced_live =
      {{0      },
       {      4},
       {      4},
       {      4},
       {      5},
       {      5},
       {      5},
       {   3   }};
  const std::vector<std::pair<int64_t, int64_t>> expected_reduced_intervals =
      {{0, 0}, {4, 0}, {7, 3}, {7, 7}, {1, 3}, {4, 6}};
  const std::vector<absl::btree_set<int64_t>> expected_reduced_groups =
      {{0, 1}, {2, 3}};
  const std::pair<int64_t, int64_t> expected_num_terms = {14, 12};
  const absl::flat_hash_set<int64_t> expected_reduced_times = {1, 4};
  EXPECT_EQ(num_terms, expected_num_terms);
  EXPECT_EQ(reducer.GetReducedLive(), expected_reduced_live);
  EXPECT_EQ(reducer.GetReducedIntervals(), expected_reduced_intervals);
  EXPECT_EQ(reducer.GetReducedGroups(), expected_reduced_groups);
  EXPECT_EQ(reducer.GetReducedTimes(num_primitives), expected_reduced_times);
}

//  | 11      ==>  | 11
//  | 00      ==>  | 00
//  +------>  ==>  +------>
//   (time)         (time)
TEST(AutoShardingMemoryTest, MergingPrimitivesWouldNotReduceTerms) {
  const int num_primitives = 2;
  const std::vector<std::vector<int64_t>> live =
      {{0, 1},
       {0, 1}};

  MemoryTermReducer reducer;
  const auto num_terms =
          reducer.Reduce(live.size(), num_primitives, Convert(live));

  const std::vector<std::vector<int64_t>> expected_reduced_live =
      {{0, 1},
       {0, 1}};
  const std::vector<std::pair<int64_t, int64_t>> expected_reduced_intervals =
      {{0, 1}, {0, 1}};
  const std::vector<absl::btree_set<int64_t>> expected_reduced_groups = {};
  const std::pair<int64_t, int64_t> expected_num_terms = {4, 4};
  const absl::flat_hash_set<int64_t> expected_reduced_times = {0};
  EXPECT_EQ(num_terms, expected_num_terms);
  EXPECT_EQ(reducer.GetReducedLive(), expected_reduced_live);
  EXPECT_EQ(reducer.GetReducedIntervals(), expected_reduced_intervals);
  EXPECT_EQ(reducer.GetReducedGroups(), expected_reduced_groups);
  EXPECT_EQ(reducer.GetReducedTimes(num_primitives), expected_reduced_times);
}

//  |          ==>  | 333333
//  | 222222   ==>  | ......
//  | 111111   ==>  | ......   Groups:
//  | 000000   ==>  | ......     m[3] = m[0] + m[1] + m[2]
//  +------->  ==>  +------->
//    (time)          (time)
TEST(AutoShardingMemoryTest, AllPrimitivesVanish) {
  const int num_primitives = 3;
  const std::vector<std::vector<int64_t>> live =
      {{0, 1, 2},
       {0, 1, 2},
       {0, 1, 2},
       {0, 1, 2},
       {0, 1, 2},
       {0, 1, 2}};

  MemoryTermReducer reducer;
  const auto num_terms =
          reducer.Reduce(live.size(), num_primitives, Convert(live));

  const std::vector<std::vector<int64_t>> expected_reduced_live =
      {{3},
       {3},
       {3},
       {3},
       {3},
       {3}};
  const std::vector<std::pair<int64_t, int64_t>> expected_reduced_intervals =
      {{6, -1}, {6, -1}, {6, -1}, {0, 5}};
  const std::vector<absl::btree_set<int64_t>> expected_reduced_groups =
      {{0, 1, 2}};
  const std::pair<int64_t, int64_t> expected_num_terms = {18, 9};
  const absl::flat_hash_set<int64_t> expected_reduced_times = {0};
  EXPECT_EQ(num_terms, expected_num_terms);
  EXPECT_EQ(reducer.GetReducedLive(), expected_reduced_live);
  EXPECT_EQ(reducer.GetReducedIntervals(), expected_reduced_intervals);
  EXPECT_EQ(reducer.GetReducedGroups(), expected_reduced_groups);
  EXPECT_EQ(reducer.GetReducedTimes(num_primitives), expected_reduced_times);
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
  const int num_primitives = 4;
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
          reducer.Reduce(live.size(), num_primitives, Convert(live));

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
  const std::vector<std::pair<int64_t, int64_t>> expected_reduced_intervals =
      {{7, -1}, {7, -1}, {9, 2}, {9, 2}, {0, 6}, {3, 8}};
  const std::vector<absl::btree_set<int64_t>> expected_reduced_groups =
      {{0, 1}, {2, 3}};
  const std::pair<int64_t, int64_t> expected_num_terms = {26, 17};
  const absl::flat_hash_set<int64_t> expected_reduced_times = {3};
  EXPECT_EQ(num_terms, expected_num_terms);
  EXPECT_EQ(reducer.GetReducedLive(), expected_reduced_live);
  EXPECT_EQ(reducer.GetReducedIntervals(), expected_reduced_intervals);
  EXPECT_EQ(reducer.GetReducedGroups(), expected_reduced_groups);
  EXPECT_EQ(reducer.GetReducedTimes(num_primitives), expected_reduced_times);
}

//  |                      |  444466666555
//  |      333333333  ==>  |      ........3  Groups:
//  |      22222222   ==>  |      ........     m[4] = m[0] + m[1]
//  |  111111111      ==>  |  .........        m[5] = m[2] + m[3]
//  | 0000000000      ==>  | 0.........        m[6] = m[0] + m[1] + m[2] + m[3]
//  +-------------->  ==>  +-------------->
//       (time)                 (time)
TEST(AutoShardingMemoryTest, ExampleFromDocumentation) {
  const int num_primitives = 4;
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
          reducer.Reduce(live.size(), num_primitives, Convert(live));

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
  const std::vector<std::pair<int64_t, int64_t>> expected_reduced_intervals =
      {{0, 0}, {10, 0}, {13, 4}, {13, 13}, {1, 4}, {10, 12}, {5, 9}};
  const std::vector<absl::btree_set<int64_t>> expected_reduced_groups =
      {{0, 1}, {2, 3}, {0, 1, 2, 3}};
  const std::pair<int64_t, int64_t> expected_num_terms = {36, 22};
  const absl::flat_hash_set<int64_t> expected_reduced_times = {5};
  EXPECT_EQ(num_terms, expected_num_terms);
  EXPECT_EQ(reducer.GetReducedLive(), expected_reduced_live);
  EXPECT_EQ(reducer.GetReducedIntervals(), expected_reduced_intervals);
  EXPECT_EQ(reducer.GetReducedGroups(), expected_reduced_groups);
  EXPECT_EQ(reducer.GetReducedTimes(num_primitives), expected_reduced_times);
}

//  |         ==>  | 333
//  | 2222    ==>  | ...2
//  |    1    ==>  |    1    Groups:
//  | 000     ==>  | ...       m[3] = m[0] + m[2]
//  +------>  ==>  +------>
//   (time)         (time)
TEST(AutoShardingMemoryTest, MergesWithRightmost) {
  const int num_primitives = 3;
  const std::vector<std::vector<int64_t>> live =
      {{0,    2},
       {0,    2},
       {0,    2},
       {   1, 2}};

  MemoryTermReducer reducer;
  const auto num_terms =
          reducer.Reduce(live.size(), num_primitives, Convert(live));

  const std::vector<std::vector<int64_t>> expected_reduced_live =
      {{      3},
       {      3},
       {      3},
       {1, 2   }};
  const std::vector<std::pair<int64_t, int64_t>> expected_reduced_intervals =
      {{3, -1}, {3, 3}, {3, 3}, {0, 2}};
  const std::vector<absl::btree_set<int64_t>> expected_reduced_groups =
      {{0, 2}};
  const std::pair<int64_t, int64_t> expected_num_terms = {8, 7};
  const absl::flat_hash_set<int64_t> expected_reduced_times = {0, 3};
  EXPECT_EQ(num_terms, expected_num_terms);
  EXPECT_EQ(reducer.GetReducedLive(), expected_reduced_live);
  EXPECT_EQ(reducer.GetReducedIntervals(), expected_reduced_intervals);
  EXPECT_EQ(reducer.GetReducedGroups(), expected_reduced_groups);
  EXPECT_EQ(reducer.GetReducedTimes(num_primitives), expected_reduced_times);
}

//  |                      |  444466666555
//  |      333333333  ==>  |      ........3  Groups:
//  |      22222222   ==>  |      ........     m[4] = m[0] + m[1]
//  |  111111111      ==>  |  .........        m[5] = m[2] + m[3]
//  | 0000000000      ==>  | 0.........        m[6] = m[0] + m[1] + m[2] + m[3]
//  +-------------->  ==>  +-------------->
//       (time)                 (time)
TEST(AutoShardingMemoryTest, ExampleFromDocumentationUsingIntervals) {
  const int num_primitives = 4;
  const std::vector<std::pair<int64_t, int64_t>> intervals =
      {{0, 9}, {1, 9}, {5, 12}, {5, 13}};

  MemoryTermReducer reducer;
  const auto num_terms = reducer.Reduce(/*num_lives=*/14, num_primitives,
                                        Convert(intervals));

  const std::vector<std::vector<int64_t>> expected_reduced_live = {};
  const std::vector<std::pair<int64_t, int64_t>> expected_reduced_intervals =
      {{0, 0}, {10, 0}, {13, 4}, {13, 13}, {1, 4}, {10, 12}, {5, 9}};
  const std::vector<absl::btree_set<int64_t>> expected_reduced_groups =
      {{0, 1}, {2, 3}, {0, 1, 2, 3}};
  const std::pair<int64_t, int64_t> expected_num_terms = {36, 22};
  const absl::flat_hash_set<int64_t> expected_reduced_times = {5};
  EXPECT_EQ(num_terms, expected_num_terms);
  EXPECT_EQ(reducer.GetReducedLive(), expected_reduced_live);
  EXPECT_EQ(reducer.GetReducedIntervals(), expected_reduced_intervals);
  EXPECT_EQ(reducer.GetReducedGroups(), expected_reduced_groups);
  EXPECT_EQ(reducer.GetReducedTimes(num_primitives), expected_reduced_times);
}

TEST(AutoShardingMemoryTest, InvalidIntervals) {
  const int num_primitives = 3;
  const std::vector<std::pair<int64_t, int64_t>> intervals =
      {{0, 4}, {9223372036854775807, 0}, {9223372036854775807, 0}};

  MemoryTermReducer reducer;
  const auto num_terms = reducer.Reduce(/*num_lives=*/5, num_primitives,
                                        Convert(intervals));

  const std::vector<std::vector<int64_t>> expected_reduced_live = {};
  const std::vector<std::pair<int64_t, int64_t>> expected_reduced_intervals =
      {{0, 4}, {9223372036854775807, 0}, {9223372036854775807, 0}};
  const std::vector<absl::btree_set<int64_t>> expected_reduced_groups = {};
  const std::pair<int64_t, int64_t> expected_num_terms = {5, 5};
  const absl::flat_hash_set<int64_t> expected_reduced_times = {0};
  EXPECT_EQ(num_terms, expected_num_terms);
  EXPECT_EQ(reducer.GetReducedLive(), expected_reduced_live);
  EXPECT_EQ(reducer.GetReducedIntervals(), expected_reduced_intervals);
  EXPECT_EQ(reducer.GetReducedGroups(), expected_reduced_groups);
  EXPECT_EQ(reducer.GetReducedTimes(num_primitives), expected_reduced_times);
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
  const int num_primitives = 4;
  const std::vector<std::pair<int64_t, int64_t>> intervals =
      {{0, 9}, {1, 9}, {5, 12}, {5, 13}};

  MemoryTermReducer reducer;
  const auto num_terms = reducer.Reduce(/*num_lives=*/14, num_primitives,
                                        Convert(intervals),
                                        /*max_iterations=*/1);

  const std::vector<std::vector<int64_t>> expected_reduced_live = {};
  const std::vector<std::pair<int64_t, int64_t>> expected_reduced_intervals =
      {{0, 0}, {10, 0}, {13, 4}, {13, 13}, {1, 9}, {5, 12}};
  const std::vector<absl::btree_set<int64_t>> expected_reduced_groups =
      {{0, 1}, {2, 3}};
  const std::pair<int64_t, int64_t> expected_num_terms = {36, 23};
  const absl::flat_hash_set<int64_t> expected_reduced_times = {5};
  EXPECT_EQ(num_terms, expected_num_terms);
  EXPECT_EQ(reducer.GetReducedLive(), expected_reduced_live);
  EXPECT_EQ(reducer.GetReducedIntervals(), expected_reduced_intervals);
  EXPECT_EQ(reducer.GetReducedGroups(), expected_reduced_groups);
  EXPECT_EQ(reducer.GetReducedTimes(num_primitives), expected_reduced_times);
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
  const int num_primitives = 4;
  const std::vector<std::pair<int64_t, int64_t>> intervals =
      {{0, 13}, {0, 10}, {0, 7}, {0, 4}};

  MemoryTermReducer reducer;
  const auto num_terms = reducer.Reduce(/*num_lives=*/14, num_primitives,
                                        Convert(intervals),
                                        /*max_iterations=*/1);

  const std::vector<std::vector<int64_t>> expected_reduced_live = {};
  const std::vector<std::pair<int64_t, int64_t>> expected_reduced_intervals =
      {{11, 13}, {11, -1}, {5, 7}, {5, -1}, {0, 10}, {0, 4}};
  const std::vector<absl::btree_set<int64_t>> expected_reduced_groups =
      {{0, 1}, {2, 3}};
  const std::pair<int64_t, int64_t> expected_num_terms = {38, 26};
  const absl::flat_hash_set<int64_t> expected_reduced_times = {0};
  EXPECT_EQ(num_terms, expected_num_terms);
  EXPECT_EQ(reducer.GetReducedLive(), expected_reduced_live);
  EXPECT_EQ(reducer.GetReducedIntervals(), expected_reduced_intervals);
  EXPECT_EQ(reducer.GetReducedGroups(), expected_reduced_groups);
  EXPECT_EQ(reducer.GetReducedTimes(num_primitives), expected_reduced_times);
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
  const int num_primitives = 4;
  const std::vector<std::pair<int64_t, int64_t>> intervals =
      {{0, 4}, {0, 7}, {0, 10}, {0, 13}};

  MemoryTermReducer reducer;
  const auto num_terms = reducer.Reduce(/*num_lives=*/14, num_primitives,
                                        Convert(intervals),
                                        /*max_iterations=*/1);

  const std::vector<std::vector<int64_t>> expected_reduced_live = {};
  const std::vector<std::pair<int64_t, int64_t>> expected_reduced_intervals =
      {{5, -1}, {5, 7}, {11, -1}, {11, 13}, {0, 10}, {0, 4}};
  const std::vector<absl::btree_set<int64_t>> expected_reduced_groups =
      {{2, 3}, {0, 1}};
  const std::pair<int64_t, int64_t> expected_num_terms = {38, 26};
  const absl::flat_hash_set<int64_t> expected_reduced_times = {0};
  EXPECT_EQ(num_terms, expected_num_terms);
  EXPECT_EQ(reducer.GetReducedLive(), expected_reduced_live);
  EXPECT_EQ(reducer.GetReducedIntervals(), expected_reduced_intervals);
  EXPECT_EQ(reducer.GetReducedGroups(), expected_reduced_groups);
  EXPECT_EQ(reducer.GetReducedTimes(num_primitives), expected_reduced_times);
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
  const int num_primitives = 4;
  const std::vector<std::pair<int64_t, int64_t>> intervals =
      {{9, 13}, {6, 13}, {3, 13}, {0, 13}};

  MemoryTermReducer reducer;
  const auto num_terms = reducer.Reduce(/*num_lives=*/14, num_primitives,
                                        Convert(intervals),
                                        /*max_iterations=*/1);

  const std::vector<std::vector<int64_t>> expected_reduced_live = {};
  const std::vector<std::pair<int64_t, int64_t>> expected_reduced_intervals =
      {{14, 8}, {6, 8}, {14, 2}, {0, 2}, {3, 13}, {9, 13}};
  const std::vector<absl::btree_set<int64_t>> expected_reduced_groups =
      {{2, 3}, {0, 1}};
  const std::pair<int64_t, int64_t> expected_num_terms = {38, 26};
  const absl::flat_hash_set<int64_t> expected_reduced_times = {9};
  EXPECT_EQ(num_terms, expected_num_terms);
  EXPECT_EQ(reducer.GetReducedLive(), expected_reduced_live);
  EXPECT_EQ(reducer.GetReducedIntervals(), expected_reduced_intervals);
  EXPECT_EQ(reducer.GetReducedGroups(), expected_reduced_groups);
  EXPECT_EQ(reducer.GetReducedTimes(num_primitives), expected_reduced_times);
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
  const int num_primitives = 4;
  const std::vector<std::pair<int64_t, int64_t>> intervals =
      {{0, 13}, {3, 13}, {6, 13}, {9, 13}};

  MemoryTermReducer reducer;
  const auto num_terms = reducer.Reduce(/*num_lives=*/14, num_primitives,
                                        Convert(intervals),
                                        /*max_iterations=*/1);

  const std::vector<std::vector<int64_t>> expected_reduced_live = {};
  const std::vector<std::pair<int64_t, int64_t>> expected_reduced_intervals =
      {{0, 2}, {14, 2}, {6, 8}, {14, 8}, {3, 13}, {9, 13}};
  const std::vector<absl::btree_set<int64_t>> expected_reduced_groups =
      {{0, 1}, {2, 3}};
  const std::pair<int64_t, int64_t> expected_num_terms = {38, 26};
  const absl::flat_hash_set<int64_t> expected_reduced_times = {9};
  EXPECT_EQ(num_terms, expected_num_terms);
  EXPECT_EQ(reducer.GetReducedLive(), expected_reduced_live);
  EXPECT_EQ(reducer.GetReducedIntervals(), expected_reduced_intervals);
  EXPECT_EQ(reducer.GetReducedGroups(), expected_reduced_groups);
  EXPECT_EQ(reducer.GetReducedTimes(num_primitives), expected_reduced_times);
}

// clang-format on

}  // namespace
}  // namespace spmd
}  // namespace xla
