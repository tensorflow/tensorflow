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

// clang-format on

}  // namespace
}  // namespace spmd
}  // namespace xla
