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
#include <vector>

#include <gtest/gtest.h>
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
  const auto num_reduced_terms =
          reducer.Reduce(live.size(), /*num_primitives=*/2, Convert(live));
  const auto reduced_live = reducer.GetReducedLive();
  const auto reduced_groups = reducer.GetReducedGroups();

  const std::vector<std::vector<int64_t>> expected_reduced_live =
      {{0   },
       {0   },
       {0   },
       {   1},
       {   1},
       {   1}};
  const std::vector<absl::flat_hash_set<int64_t>> expected_reduced_groups = {};
  EXPECT_EQ(num_reduced_terms, 6);
  EXPECT_EQ(reduced_live, expected_reduced_live);
  EXPECT_EQ(reduced_groups, expected_reduced_groups);
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
  const auto num_reduced_terms =
          reducer.Reduce(live.size(), /*num_primitives=*/2, Convert(live));
  const auto reduced_live = reducer.GetReducedLive();
  const auto reduced_groups = reducer.GetReducedGroups();

  const std::vector<std::vector<int64_t>> expected_reduced_live =
      {{0      },
       {      2},
       {      2},
       {      2},
       {      2},
       {   1   }};
  const std::vector<absl::flat_hash_set<int64_t>> expected_reduced_groups =
      {{0, 1}};
  EXPECT_EQ(num_reduced_terms, 8);
  EXPECT_EQ(reduced_live, expected_reduced_live);
  EXPECT_EQ(reduced_groups, expected_reduced_groups);
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
  const auto num_reduced_terms =
          reducer.Reduce(live.size(), /*num_primitives=*/2, Convert(live));
  const auto reduced_live = reducer.GetReducedLive();
  const auto reduced_groups = reducer.GetReducedGroups();

  const std::vector<std::vector<int64_t>> expected_reduced_live =
      {{   1   },
       {      2},
       {      2},
       {      2},
       {      2},
       {0      }};
  const std::vector<absl::flat_hash_set<int64_t>> expected_reduced_groups =
      {{0, 1}};
  EXPECT_EQ(num_reduced_terms, 8);
  EXPECT_EQ(reduced_live, expected_reduced_live);
  EXPECT_EQ(reduced_groups, expected_reduced_groups);
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
  const auto num_reduced_terms =
          reducer.Reduce(live.size(), /*num_primitives=*/2, Convert(live));
  const auto reduced_live = reducer.GetReducedLive();
  const auto reduced_groups = reducer.GetReducedGroups();

  const std::vector<std::vector<int64_t>> expected_reduced_live =
      {{0   },
       {0, 1},
       {0, 1},
       {0, 1},
       {0, 1},
       {0   }};
  const std::vector<absl::flat_hash_set<int64_t>> expected_reduced_groups = {};
  EXPECT_EQ(num_reduced_terms, 10);
  EXPECT_EQ(reduced_live, expected_reduced_live);
  EXPECT_EQ(reduced_groups, expected_reduced_groups);
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
  const auto num_reduced_terms =
          reducer.Reduce(live.size(), /*num_primitives=*/2, Convert(live));
  const auto reduced_live = reducer.GetReducedLive();
  const auto reduced_groups = reducer.GetReducedGroups();

  const std::vector<std::vector<int64_t>> expected_reduced_live =
      {{0   },
       {   2},
       {   2},
       {   2},
       {   2},
       {   2}};
  const std::vector<absl::flat_hash_set<int64_t>> expected_reduced_groups =
      {{0, 1}};
  EXPECT_EQ(num_reduced_terms, 8);
  EXPECT_EQ(reduced_live, expected_reduced_live);
  EXPECT_EQ(reduced_groups, expected_reduced_groups);
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
  const auto num_reduced_terms =
          reducer.Reduce(live.size(), /*num_primitives=*/2, Convert(live));
  const auto reduced_live = reducer.GetReducedLive();
  const auto reduced_groups = reducer.GetReducedGroups();

  const std::vector<std::vector<int64_t>> expected_reduced_live =
      {{2},
       {2},
       {2},
       {2},
       {2},
       {2}};
  const std::vector<absl::flat_hash_set<int64_t>> expected_reduced_groups =
      {{0, 1}};
  EXPECT_EQ(num_reduced_terms, 8);
  EXPECT_EQ(reduced_live, expected_reduced_live);
  EXPECT_EQ(reduced_groups, expected_reduced_groups);
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
  const auto num_reduced_terms =
          reducer.Reduce(live.size(), /*num_primitives=*/3, Convert(live));
  const auto reduced_live = reducer.GetReducedLive();
  const auto reduced_groups = reducer.GetReducedGroups();

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
  const std::vector<absl::flat_hash_set<int64_t>> expected_reduced_groups =
      {{0, 1}};
  EXPECT_EQ(num_reduced_terms, 15);
  EXPECT_EQ(reduced_live, expected_reduced_live);
  EXPECT_EQ(reduced_groups, expected_reduced_groups);
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
  const auto num_reduced_terms =
          reducer.Reduce(live.size(), /*num_primitives=*/3, Convert(live));
  const auto reduced_live = reducer.GetReducedLive();
  const auto reduced_groups = reducer.GetReducedGroups();

  const std::vector<std::vector<int64_t>> expected_reduced_live =
      {{3},
       {3},
       {3},
       {4},
       {4},
       {4}};
  const std::vector<absl::flat_hash_set<int64_t>> expected_reduced_groups =
      {{0, 1}, {0, 2}};
  EXPECT_EQ(num_reduced_terms, 10);
  EXPECT_EQ(reduced_live, expected_reduced_live);
  EXPECT_EQ(reduced_groups, expected_reduced_groups);
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
  const auto num_reduced_terms =
          reducer.Reduce(live.size(), /*num_primitives=*/4, Convert(live));
  const auto reduced_live = reducer.GetReducedLive();
  const auto reduced_groups = reducer.GetReducedGroups();

  const std::vector<std::vector<int64_t>> expected_reduced_live =
      {{0      },
       {      4},
       {      4},
       {      4},
       {      5},
       {      5},
       {      5},
       {   3   }};
  const std::vector<absl::flat_hash_set<int64_t>> expected_reduced_groups =
      {{0, 1}, {2, 3}};
  EXPECT_EQ(num_reduced_terms, 12);
  EXPECT_EQ(reduced_live, expected_reduced_live);
  EXPECT_EQ(reduced_groups, expected_reduced_groups);
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
  const auto num_reduced_terms =
          reducer.Reduce(live.size(), /*num_primitives=*/2, Convert(live));
  const auto reduced_live = reducer.GetReducedLive();
  const auto reduced_groups = reducer.GetReducedGroups();

  const std::vector<std::vector<int64_t>> expected_reduced_live =
      {{0, 1},
       {0, 1}};
  const std::vector<absl::flat_hash_set<int64_t>> expected_reduced_groups = {};
  EXPECT_EQ(num_reduced_terms, 4);
  EXPECT_EQ(reduced_live, expected_reduced_live);
  EXPECT_EQ(reduced_groups, expected_reduced_groups);
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
  const auto num_reduced_terms =
          reducer.Reduce(live.size(), /*num_primitives=*/3, Convert(live));
  const auto reduced_live = reducer.GetReducedLive();
  const auto reduced_groups = reducer.GetReducedGroups();

  const std::vector<std::vector<int64_t>> expected_reduced_live =
      {{3},
       {3},
       {3},
       {3},
       {3},
       {3}};
  const std::vector<absl::flat_hash_set<int64_t>> expected_reduced_groups =
      {{0, 1, 2}};
  EXPECT_EQ(num_reduced_terms, 9);
  EXPECT_EQ(reduced_live, expected_reduced_live);
  EXPECT_EQ(reduced_groups, expected_reduced_groups);
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
  const auto num_reduced_terms =
          reducer.Reduce(live.size(), /*num_primitives=*/4, Convert(live));
  const auto reduced_live = reducer.GetReducedLive();
  const auto reduced_groups = reducer.GetReducedGroups();

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
  const std::vector<absl::flat_hash_set<int64_t>> expected_reduced_groups =
      {{0, 1}, {2, 3}};
  EXPECT_EQ(num_reduced_terms, 17);
  EXPECT_EQ(reduced_live, expected_reduced_live);
  EXPECT_EQ(reduced_groups, expected_reduced_groups);
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
  const auto num_reduced_terms =
          reducer.Reduce(live.size(), /*num_primitives=*/4, Convert(live));
  const auto reduced_live = reducer.GetReducedLive();
  const auto reduced_groups = reducer.GetReducedGroups();

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
  const std::vector<absl::flat_hash_set<int64_t>> expected_reduced_groups =
      {{0, 1}, {2, 3}, {0, 1, 2, 3}};
  EXPECT_EQ(num_reduced_terms, 22);
  EXPECT_EQ(reduced_live, expected_reduced_live);
  EXPECT_EQ(reduced_groups, expected_reduced_groups);
}

// clang-format on

}  // namespace
}  // namespace spmd
}  // namespace xla
