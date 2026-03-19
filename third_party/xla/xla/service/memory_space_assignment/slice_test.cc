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

#include "xla/service/memory_space_assignment/slice.h"

#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "xla/service/time_utils.h"

namespace xla {
namespace memory_space_assignment {
namespace {

class SlicedPrefetchStartTimePickerTest : public ::testing::Test {
 protected:
  struct FakeInstructionData {
    float elapsed_time = 0.0;
    std::string computation;
  };

  std::vector<int64_t> Pick(
      const std::vector<FakeInstructionData>& schedule_data, int64_t num_slices,
      int64_t prefetch_start_time, int64_t prefetch_end_time) {
    return SlicedPrefetchStartTimePicker::Pick(
        num_slices, prefetch_start_time, prefetch_end_time,
        [&schedule_data](int64_t exclusive_start_time,
                         int64_t exclusive_end_time) {
          auto start_it = schedule_data.begin() +
                          ExclusiveToInclusiveStartTime(exclusive_start_time);
          auto end_it = (exclusive_end_time < schedule_data.size()
                             ? schedule_data.begin() + exclusive_end_time
                             : schedule_data.end());
          return std::accumulate(
              start_it, end_it, 0.0,
              [](float total, const FakeInstructionData& data) {
                return total + data.elapsed_time;
              });
        },
        [&schedule_data](int64_t lhs_time, int64_t rhs_time) {
          CHECK_GE(lhs_time, 0);
          CHECK_GE(rhs_time, 0);
          CHECK_LT(lhs_time, schedule_data.size());
          CHECK_LT(rhs_time, schedule_data.size());
          return schedule_data[lhs_time].computation ==
                 schedule_data[rhs_time].computation;
        });
  }
};

TEST_F(SlicedPrefetchStartTimePickerTest, Base1) {
  // The 2nd slice naturally should start after 1.5 time units have passed,
  // forcing us to start before t=1.
  EXPECT_THAT(Pick({
                       /*t=0*/ {1.0, "a"},
                       /*t=1*/ {1.0, "a"},
                       /*t=2*/ {1.0, "a"},
                   },
                   /*num_slices=*/2, /*prefetch_start_time=*/-1,
                   /*prefetch_end_time=*/3),
              ::testing::ElementsAre(-1, 0));
}

TEST_F(SlicedPrefetchStartTimePickerTest, Base2) {
  // The 2nd slice naturally should start after 6.0 time units have passed,
  // forcing us to start before t=0.
  EXPECT_THAT(Pick({
                       /*t=0*/ {10.0, "a"},
                       /*t=1*/ {1.0, "a"},
                       /*t=2*/ {1.0, "a"},
                   },
                   /*num_slices=*/2, /*prefetch_start_time=*/-1,
                   /*prefetch_end_time=*/3),
              ::testing::ElementsAre(-1, -1));
}

TEST_F(SlicedPrefetchStartTimePickerTest, Base3) {
  // The 2nd slice naturally should start after 1.0 time unit has passed.
  EXPECT_THAT(Pick({
                       /*t=0*/ {1.0, "a"},
                       /*t=1*/ {1.0, "a"},
                   },
                   /*num_slices=*/2, /*prefetch_start_time=*/-1,
                   /*prefetch_end_time=*/2),
              ::testing::ElementsAre(-1, 0));
}

TEST_F(SlicedPrefetchStartTimePickerTest, Zeros1) {
  // The 2nd slice naturally should start after 1.0 time unit has passed.
  // Make sure we don't add extra 0.0 cost instructions to the start time.
  EXPECT_THAT(Pick({
                       /*t=0*/ {1.0, "a"},
                       /*t=1*/ {0.0, "a"},
                       /*t=2*/ {0.0, "a"},
                       /*t=3*/ {0.0, "a"},
                       /*t=4*/ {1.0, "a"},
                   },
                   /*num_slices=*/2, /*prefetch_start_time=*/-1,
                   /*prefetch_end_time=*/5),
              ::testing::ElementsAre(-1, 0));
}

TEST_F(SlicedPrefetchStartTimePickerTest, Zeros2) {
  // The 2nd slice naturally should start after 2.0 time units have passed.
  // Make sure we don't add extra 0.0 cost instructions to the start time.
  EXPECT_THAT(Pick({
                       /*t=0*/ {1.0, "a"},
                       /*t=1*/ {0.0, "a"},
                       /*t=2*/ {1.0, "a"},
                       /*t=3*/ {0.0, "a"},
                       /*t=4*/ {1.0, "a"},
                       /*t=5*/ {0.0, "a"},
                       /*t=6*/ {1.0, "a"},
                   },
                   /*num_slices=*/2, /*prefetch_start_time=*/-1,
                   /*prefetch_end_time=*/7),
              ::testing::ElementsAre(-1, 2));
}

TEST_F(SlicedPrefetchStartTimePickerTest, Zeros3) {
  // The first slice always comes at prefetch_start_time. The 2nd slice
  // naturally should start after 1.5 time units have passed, causing us to
  // start after t=2. Make sure we don't add extra 0.0 cost instructions to the
  // start time.
  EXPECT_THAT(Pick({
                       /*t=0*/ {1.0, "a"},
                       /*t=1*/ {0.0, "a"},
                       /*t=2*/ {1.0, "a"},
                       /*t=3*/ {0.0, "a"},
                       /*t=4*/ {1.0, "a"},
                       /*t=5*/ {0.0, "a"},
                       /*t=6*/ {1.0, "a"},
                   },
                   /*num_slices=*/2, /*prefetch_start_time=*/1,
                   /*prefetch_end_time=*/7),
              ::testing::ElementsAre(1, 2));
}

TEST_F(SlicedPrefetchStartTimePickerTest, MidSchedule) {
  EXPECT_THAT(Pick({
                       /*t=0*/ {1.0, "a"},
                       /*t=1*/ {1.0, "a"},
                       /*t=3*/ {1.0, "a"},
                       /*t=4*/ {1.0, "a"},
                       /*t=5*/ {1.0, "a"},
                       /*t=6*/ {1.0, "a"},
                       /*t=7*/ {1.0, "a"},
                       /*t=8*/ {1.0, "a"},
                       /*t=9*/ {1.0, "a"},
                       /*t=10*/ {1.0, "a"},
                       /*t=11*/ {1.0, "a"},
                       /*t=12*/ {1.0, "a"},
                   },
                   /*num_slices=*/2, /*prefetch_start_time=*/5,
                   /*prefetch_end_time=*/10),
              ::testing::ElementsAre(5, 7));
}

TEST_F(SlicedPrefetchStartTimePickerTest, ManySlices) {
  EXPECT_THAT(Pick({
                       /*t=0*/ {1.0, "a"},
                       /*t=1*/ {1.0, "a"},
                       /*t=2*/ {1.0, "a"},
                       /*t=3*/ {1.0, "a"},
                       /*t=4*/ {1.0, "a"},
                       /*t=5*/ {1.0, "a"},
                       /*t=6*/ {1.0, "a"},
                       /*t=7*/ {1.0, "a"},
                       /*t=8*/ {1.0, "a"},
                       /*t=9*/ {1.0, "a"},
                       /*t=10*/ {1.0, "a"},
                       /*t=11*/ {1.0, "a"},
                       /*t=12*/ {1.0, "a"},
                       /*t=13*/ {1.0, "a"},
                       /*t=14*/ {1.0, "a"},
                       /*t=15*/ {1.0, "a"},
                       /*t=16*/ {1.0, "a"},
                       /*t=17*/ {1.0, "a"},
                       /*t=18*/ {1.0, "a"},
                       /*t=19*/ {1.0, "a"},
                   },
                   /*num_slices=*/5, /*prefetch_start_time=*/-1,
                   /*prefetch_end_time=*/20),
              ::testing::ElementsAre(-1, 3, 7, 11, 15));
}

TEST_F(SlicedPrefetchStartTimePickerTest, DifferentParents) {
  // The 2nd slice naturally should start after t=2, but we are forced to push
  // it after t=1, since the instruction at t=3 has parent "b", while the first
  // instruction has parent "a."
  EXPECT_THAT(Pick({
                       /*t=0*/ {1.0, "a"},
                       /*t=1*/ {1.0, "a"},
                       /*t=2*/ {1.0, "b"},
                       /*t=3*/ {1.0, "b"},
                       /*t=4*/ {1.0, "b"},
                       /*t=5*/ {1.0, "a"},
                   },
                   /*num_slices=*/2, /*prefetch_start_time=*/-1,
                   /*prefetch_end_time=*/6),
              ::testing::ElementsAre(-1, 1));
}

}  // namespace
}  // namespace memory_space_assignment
}  // namespace xla
