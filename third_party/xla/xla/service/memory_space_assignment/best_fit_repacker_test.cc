/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/memory_space_assignment/best_fit_repacker.h"

#include <cstdint>

#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "xla/comparison_util.h"
#include "xla/service/memory_space_assignment/repacking.h"
#include "tsl/platform/test.h"

namespace xla {

class MemorySpaceAssignmentBestFitRepackerTest : public ::testing::Test {
 protected:
  using AllocationBlock =
      memory_space_assignment::MemorySpaceAssignmentRepacker::AllocationBlock;
  using SlicedAllocationData = memory_space_assignment::
      MemorySpaceAssignmentRepacker::SlicedAllocationData;
  using Slice = memory_space_assignment::MemorySpaceAssignmentRepacker::Slice;

  MemorySpaceAssignmentBestFitRepackerTest() : repacker_(100, 1, options_) {}

  AllocationBlock* MakeAllocationBlock(int64_t start_time, int64_t end_time,
                                       int64_t size,
                                       int64_t initial_offset = -1) {
    allocation_blocks_.push_back(
        {start_time,
         end_time,
         size,
         -1,
         initial_offset,
         static_cast<int64_t>(allocation_blocks_.size()),
         {}});
    AllocationBlock* block = &allocation_blocks_.back();
    block->colocations.push_back(block);
    return block;
  }

  std::list<AllocationBlock> allocation_blocks_;
  memory_space_assignment::MemorySpaceAssignmentBestFitRepacker::
      BestFitRepackOptions options_{/*validate=*/true,
                                    /*buffer_interval_compare=*/nullptr};
  memory_space_assignment::MemorySpaceAssignmentBestFitRepacker repacker_;
};

TEST_F(MemorySpaceAssignmentBestFitRepackerTest, Simple) {
  std::vector<AllocationBlock*> allocation_blocks;
  allocation_blocks.push_back(MakeAllocationBlock(10, 20, 10));
  allocation_blocks.push_back(MakeAllocationBlock(5, 25, 15));
  EXPECT_TRUE(*repacker_.Repack(absl::MakeSpan(allocation_blocks)));

  EXPECT_EQ(allocation_blocks[0]->offset, 15);
  EXPECT_EQ(allocation_blocks[1]->offset, 0);
}

TEST_F(MemorySpaceAssignmentBestFitRepackerTest, Colocation) {
  std::vector<AllocationBlock*> allocation_blocks;
  allocation_blocks.push_back(MakeAllocationBlock(0, 2, 10));
  allocation_blocks.push_back(MakeAllocationBlock(10, 20, 10));
  // Allocation blocks 0 and 1 are colocated.
  allocation_blocks[0]->colocations.push_back(allocation_blocks[1]);
  allocation_blocks[1]->colocations.push_back(allocation_blocks[0]);
  allocation_blocks.push_back(MakeAllocationBlock(5, 25, 15));
  EXPECT_TRUE(*repacker_.Repack(absl::MakeSpan(allocation_blocks)));

  EXPECT_EQ(allocation_blocks[0]->offset, 15);
  EXPECT_EQ(allocation_blocks[1]->offset, 15);
  EXPECT_EQ(allocation_blocks[2]->offset, 0);
}

TEST_F(MemorySpaceAssignmentBestFitRepackerTest, TooLarge) {
  // Memory size is 100, total size of buffers is 105.
  std::vector<AllocationBlock*> allocation_blocks;
  allocation_blocks.push_back(MakeAllocationBlock(10, 20, 10));
  allocation_blocks.push_back(MakeAllocationBlock(5, 25, 15));
  allocation_blocks.push_back(MakeAllocationBlock(15, 20, 10));
  allocation_blocks.push_back(MakeAllocationBlock(12, 22, 50));
  allocation_blocks.push_back(MakeAllocationBlock(10, 18, 20));
  EXPECT_FALSE(*repacker_.Repack(absl::MakeSpan(allocation_blocks)));

  // Make sure the buffers didn't get offset assignments.
  EXPECT_EQ(allocation_blocks[0]->offset, -1);
  EXPECT_EQ(allocation_blocks[1]->offset, -1);
  EXPECT_EQ(allocation_blocks[2]->offset, -1);
  EXPECT_EQ(allocation_blocks[3]->offset, -1);
  EXPECT_EQ(allocation_blocks[4]->offset, -1);
}

TEST_F(MemorySpaceAssignmentBestFitRepackerTest, ColocationDifferentSizes) {
  std::vector<AllocationBlock*> allocation_blocks;
  allocation_blocks.push_back(MakeAllocationBlock(0, 2, 5));
  allocation_blocks.push_back(MakeAllocationBlock(10, 20, 10));
  // Allocation blocks 0 and 1 are colocated.
  allocation_blocks[0]->colocations.push_back(allocation_blocks[1]);
  allocation_blocks[1]->colocations.push_back(allocation_blocks[0]);
  allocation_blocks.push_back(MakeAllocationBlock(9, 11, 2));
  allocation_blocks.push_back(MakeAllocationBlock(1, 2, 2));
  EXPECT_TRUE(*repacker_.Repack(absl::MakeSpan(allocation_blocks)));

  EXPECT_EQ(allocation_blocks[0]->offset, 0);
  EXPECT_EQ(allocation_blocks[1]->offset, 0);
  EXPECT_EQ(allocation_blocks[2]->offset, 10);
  EXPECT_EQ(allocation_blocks[3]->offset, 5);
}

TEST_F(MemorySpaceAssignmentBestFitRepackerTest, RepackedSlicesFit) {
  // Expected repacking:
  //
  // space
  //   ^
  //  8 |
  //  7 |                  +-----+
  //  6 |                  |  E  |
  //  5 |          +-------+-++  |
  //  4 |          |    B    |+--++--++--+
  //  3 |          |         ||  || F||  |
  //  2 +----------+---++----++  |+--++  |
  //  1 |      A       ||    C   ||   D  |
  //  0 +--------------++--------++------+
  //    +----|----|----|----|----|----|----|--> time
  //    0    5    10   15   20   25   30   35

  std::vector<AllocationBlock*> allocation_blocks;
  // Block A
  allocation_blocks.push_back(MakeAllocationBlock(0, 15, 2));
  // Block B
  allocation_blocks.push_back(MakeAllocationBlock(11, 21, 3));
  // Block C
  allocation_blocks.push_back(MakeAllocationBlock(16, 25, 4));
  allocation_blocks.back()->original_slice_data =
      SlicedAllocationData({{Slice{2, -1, 16}, Slice{2, -1, 22}}});
  // Block D
  allocation_blocks.push_back(MakeAllocationBlock(26, 33, 4));
  allocation_blocks.back()->original_slice_data =
      SlicedAllocationData({{Slice{2, -1, 26}, Slice{2, -1, 30}}});
  // Block E
  allocation_blocks.push_back(MakeAllocationBlock(19, 25, 3));
  allocation_blocks.back()->original_slice_data =
      SlicedAllocationData({{Slice{1, -1, 19}, Slice{2, -1, 22}}});
  // Block F
  allocation_blocks.push_back(MakeAllocationBlock(26, 29, 2));

  // Specify the repacking sort order as the order in which blocks were added to
  // allocation_blocks.
  // - By placing C after B, we test that the repacker can place a sliced block
  //   around another block
  // - By placing F after D, we test that the repacker can fill in the extra
  //   space left behind by slicing.
  absl::flat_hash_map<AllocationBlock*, int> sort_keys;
  for (int i = 0; i < allocation_blocks.size(); ++i) {
    sort_keys[allocation_blocks[i]] = i;
  }
  options_.buffer_interval_compare = LessThanByKey(
      [sort_keys](const memory_space_assignment::
                      MemorySpaceAssignmentBestFitRepacker::BufferInterval& x) {
        return sort_keys.at(x.buffer);
      });
  repacker_ = memory_space_assignment::MemorySpaceAssignmentBestFitRepacker(
      100, 1, options_);

  EXPECT_TRUE(*repacker_.Repack(absl::MakeSpan(allocation_blocks)));

  // Check results
  // Block A
  EXPECT_EQ(allocation_blocks[0]->offset, 0);
  EXPECT_FALSE(allocation_blocks[0]->repacked_slice_data.has_value());
  // Block B
  EXPECT_EQ(allocation_blocks[1]->offset, 2);
  EXPECT_FALSE(allocation_blocks[1]->repacked_slice_data.has_value());
  // Block C
  EXPECT_EQ(allocation_blocks[2]->offset, 0);
  ASSERT_TRUE(allocation_blocks[2]->repacked_slice_data.has_value());
  EXPECT_EQ(*allocation_blocks[2]->repacked_slice_data,
            (SlicedAllocationData({{Slice{2, 0, 16}, Slice{2, 2, 22}}})));
  // Block D
  EXPECT_EQ(allocation_blocks[3]->offset, 0);
  ASSERT_TRUE(allocation_blocks[3]->repacked_slice_data.has_value());
  EXPECT_EQ(*allocation_blocks[3]->repacked_slice_data,
            (SlicedAllocationData({{Slice{2, 0, 26}, Slice{2, 2, 30}}})));
  // Block E
  EXPECT_EQ(allocation_blocks[4]->offset, 4);
  ASSERT_TRUE(allocation_blocks[4]->repacked_slice_data.has_value());
  EXPECT_EQ(*allocation_blocks[4]->repacked_slice_data,
            (SlicedAllocationData({{Slice{1, 4, 22}, Slice{2, 5, 19}}})));
  // Block F
  EXPECT_EQ(allocation_blocks[5]->offset, 2);
  EXPECT_FALSE(allocation_blocks[5]->repacked_slice_data.has_value());
}

TEST_F(MemorySpaceAssignmentBestFitRepackerTest, SlicedColocationsFit) {
  // Expected repacking:
  //
  // space
  //   ^
  //  9 |              +-+
  //  8 |              | |
  //  7 |              |F|+-+
  //  6 |    +-----++-+| || |
  //  5 |    |  C  || || || |
  //  4 +----+--++-++ |+-++ |
  //  3 |    B  || E  || D  |
  //  2 +-------++--+-++----+
  //  1 |     A     |
  //  0 +-----------+
  //    +----|----|----|----|----|----|----|--> time
  //    0    5    10   15   20   25   30   35

  // D is colocated with E; thus, they will be allocated together.
  std::vector<AllocationBlock*> allocation_blocks;
  // Block A
  allocation_blocks.push_back(MakeAllocationBlock(0, 12, 2));
  // Block B
  allocation_blocks.push_back(MakeAllocationBlock(0, 8, 2));
  // Block C
  allocation_blocks.push_back(MakeAllocationBlock(5, 11, 2));
  // Block D
  allocation_blocks.push_back(MakeAllocationBlock(15, 20, 5));
  // Note, below we put the later start time first in the original slice data.
  // This shouldn't make a difference to the repacker because it will map
  // sizes to times as it sees fit.
  allocation_blocks.back()->original_slice_data =
      SlicedAllocationData({{Slice{2, -1, 18}, Slice{3, -1, 15}}});
  // Block E
  allocation_blocks.push_back(MakeAllocationBlock(9, 14, 4));
  allocation_blocks.back()->original_slice_data =
      SlicedAllocationData({{Slice{2, -1, 9}, Slice{2, -1, 12}}});
  // Colocate E with D.
  allocation_blocks.back()->colocations.push_back(allocation_blocks[3]);
  // Block F
  allocation_blocks.push_back(MakeAllocationBlock(15, 17, 5));

  // Specify the repacking sort order as the order in which blocks were added to
  // allocation_blocks.
  // - By placing E after C, we test that the repacker can place a sliced block
  //   around another block
  // - By placing F after D, we test that the repacker can fill in the extra
  //   space left behind by slicing.
  absl::flat_hash_map<AllocationBlock*, int> sort_keys;
  for (int i = 0; i < allocation_blocks.size(); ++i) {
    sort_keys[allocation_blocks[i]] = i;
  }
  options_.buffer_interval_compare = LessThanByKey(
      [sort_keys](const memory_space_assignment::
                      MemorySpaceAssignmentBestFitRepacker::BufferInterval& x) {
        return sort_keys.at(x.buffer);
      });
  repacker_ = memory_space_assignment::MemorySpaceAssignmentBestFitRepacker(
      100, 1, options_);

  EXPECT_TRUE(*repacker_.Repack(absl::MakeSpan(allocation_blocks)));

  // Check results
  // Block A
  EXPECT_EQ(allocation_blocks[0]->offset, 0);
  EXPECT_FALSE(allocation_blocks[0]->repacked_slice_data.has_value());
  // Block B
  EXPECT_EQ(allocation_blocks[1]->offset, 2);
  EXPECT_FALSE(allocation_blocks[1]->repacked_slice_data.has_value());
  // Block C
  EXPECT_EQ(allocation_blocks[2]->offset, 4);
  ASSERT_FALSE(allocation_blocks[2]->repacked_slice_data.has_value());
  // Block D
  EXPECT_EQ(allocation_blocks[3]->offset, 2);
  ASSERT_TRUE(allocation_blocks[3]->repacked_slice_data.has_value());
  EXPECT_EQ(*allocation_blocks[3]->repacked_slice_data,
            (SlicedAllocationData({{Slice{2, 2, 15}, Slice{3, 4, 18}}})));
  // Block E
  EXPECT_EQ(allocation_blocks[4]->offset, 2);
  ASSERT_TRUE(allocation_blocks[4]->repacked_slice_data.has_value());
  EXPECT_EQ(*allocation_blocks[4]->repacked_slice_data,
            (SlicedAllocationData({{Slice{2, 2, 9}, Slice{2, 4, 12}}})));
  // Block F
  EXPECT_EQ(allocation_blocks[5]->offset, 4);
  EXPECT_FALSE(allocation_blocks[5]->repacked_slice_data.has_value());
}

}  // namespace xla
