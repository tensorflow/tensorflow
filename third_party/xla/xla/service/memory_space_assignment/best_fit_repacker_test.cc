/* Copyright 2020 The OpenXLA Authors.

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
#include <list>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "xla/comparison_util.h"
#include "xla/service/heap_simulator/allocation_block.h"
#include "xla/service/heap_simulator/heap_simulator.h"
#include "tsl/platform/test.h"

namespace xla {

class MemorySpaceAssignmentBestFitRepackerTest : public ::testing::Test {
 protected:
  MemorySpaceAssignmentBestFitRepackerTest()
      : repacker_(100, 1, SliceTimePermutationIterator::Ty::kAll, options_) {}

  AllocationBlock* MakeAllocationBlock(int64_t start_time, int64_t end_time,
                                       int64_t size,
                                       int64_t initial_offset = -1) {
    allocation_blocks_.push_back(
        {start_time, end_time, size, -1, initial_offset,
         static_cast<int64_t>(allocation_blocks_.size())});
    AllocationBlock* block = &allocation_blocks_.back();
    block->next_colocated = block;
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
  allocation_blocks[0]->next_colocated = allocation_blocks[1];
  allocation_blocks[1]->next_colocated = allocation_blocks[0];
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
  allocation_blocks[0]->next_colocated = allocation_blocks[1];
  allocation_blocks[1]->next_colocated = allocation_blocks[0];
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
  //    ^
  //  7 |
  //  6 |                  +-----+
  //  5 |          +-------+-++ E|
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
  allocation_blocks.back()->original_slice_data = SlicedAllocationData(
      {{AllocatedSlice{2, -1, 16}, AllocatedSlice{2, -1, 22}}});
  // Block D
  allocation_blocks.push_back(MakeAllocationBlock(26, 33, 4));
  allocation_blocks.back()->original_slice_data = SlicedAllocationData(
      {{AllocatedSlice{2, -1, 26}, AllocatedSlice{2, -1, 30}}});
  // Block E
  allocation_blocks.push_back(MakeAllocationBlock(19, 25, 2));
  allocation_blocks.back()->original_slice_data = SlicedAllocationData(
      {{AllocatedSlice{1, -1, 19}, AllocatedSlice{1, -1, 22}}});
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
      100, 1, SliceTimePermutationIterator::Ty::kAll, options_);

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
            (SlicedAllocationData(
                {{AllocatedSlice{2, 0, 16}, AllocatedSlice{2, 2, 22}}})));
  // Block D
  EXPECT_EQ(allocation_blocks[3]->offset, 0);
  ASSERT_TRUE(allocation_blocks[3]->repacked_slice_data.has_value());
  EXPECT_EQ(*allocation_blocks[3]->repacked_slice_data,
            (SlicedAllocationData(
                {{AllocatedSlice{2, 0, 26}, AllocatedSlice{2, 2, 30}}})));
  // Block E
  EXPECT_EQ(allocation_blocks[4]->offset, 4);
  ASSERT_TRUE(allocation_blocks[4]->repacked_slice_data.has_value());
  EXPECT_EQ(*allocation_blocks[4]->repacked_slice_data,
            (SlicedAllocationData(
                {{AllocatedSlice{1, 4, 22}, AllocatedSlice{1, 5, 19}}})));
  // Block F
  EXPECT_EQ(allocation_blocks[5]->offset, 2);
  EXPECT_FALSE(allocation_blocks[5]->repacked_slice_data.has_value());
}

// Test that we do not permute slice start times in a way that changes the
// original slice size-start time mappings. Doing so breaks assumptions that
// MSA uses to construct its internal state prior to repacking.
TEST_F(MemorySpaceAssignmentBestFitRepackerTest,
       SliceTimePermutationsMatchOriginalSizeTimeMapping) {
  // Original placement:               Ideal repacking, but unsupported:
  //
  //  space                            space
  //    ^                                ^
  //  7 |    +---------+               7 |
  //  6 |    |    C    |               6 |    +---------+
  //  5 |    +-----+---+               5 |    |    C    |
  //  4 |    +-----+   |               4 |    +---------+
  //  3 |    |    B    |               3 |    |    B    |
  //  2 +----+----+----+               2 +----+----++   |
  //  1 |    A    |                    1 |    A    |+---+
  //  0 +---------+                    0 +---------+
  //    +----|----|----|----> time       +----|----|----|----> time
  //    0    5    10   15                0    5    10   15

  std::vector<AllocationBlock*> allocation_blocks;
  // Block A
  allocation_blocks.push_back(MakeAllocationBlock(0, 10, 2, 0));
  // Block B
  allocation_blocks.push_back(MakeAllocationBlock(5, 15, 3, 2));
  allocation_blocks.back()->original_slice_data = SlicedAllocationData(
      {{AllocatedSlice{2, 2, 5}, AllocatedSlice{1, 4, 11}}});
  // Block C
  allocation_blocks.push_back(MakeAllocationBlock(5, 15, 2, 6));

  // Specify the repacking sort order as the order in which blocks were added to
  // allocation_blocks. We need to do this so that B is placed before C. If C
  // is placed before B, C will sit directly on top of A, and the repacker would
  // never try to permute B's slice size-start time mapping.
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
      100, 1, SliceTimePermutationIterator::Ty::kAll, options_);

  // The repacker returns true as long as the result fits in the max size,
  // regardless of whether it has actually changed anything.
  EXPECT_TRUE(*repacker_.Repack(absl::MakeSpan(allocation_blocks)));

  // Typically the heap_simulator would prefer to start Block B at a smaller
  // offset, i.e., offset 1 rather than offset 2. However, in order to do so,
  // the repacker would have to permute the original slice size-start time
  // mapping, which is not permitted. Thus, we ensure that the repacked B's
  // larger slice is assigned the smaller offset and earlier start time.
  ASSERT_TRUE(allocation_blocks[1]->repacked_slice_data.has_value());
  ASSERT_EQ(
      allocation_blocks[1]->repacked_slice_data->slices_sorted_by_offset.size(),
      2);
  const AllocatedSlice& slice_with_smaller_offset =
      allocation_blocks[1]->repacked_slice_data->slices_sorted_by_offset[0];
  const AllocatedSlice& slice_with_larger_offset =
      allocation_blocks[1]->repacked_slice_data->slices_sorted_by_offset[1];
  // The larger slice is assigned to the smaller offset.
  ASSERT_GT(slice_with_smaller_offset.size, slice_with_larger_offset.size);
  const AllocatedSlice& larger_slice = slice_with_smaller_offset;
  const AllocatedSlice& smaller_slice = slice_with_larger_offset;
  // The larger slice is assigned to the earlier start time.
  ASSERT_LT(larger_slice.inclusive_start_time,
            smaller_slice.inclusive_start_time);
}

// Test that we do not permute slice start times in a way that changes the
// original slice size-start time mappings. Doing so breaks assumptions that
// MSA uses to construct its internal state prior to repacking.
TEST_F(MemorySpaceAssignmentBestFitRepackerTest,
       SliceTimePermutationsMatchOriginalSizeTimeMapping2) {
  // Original placement:                New placement:
  //
  //  space                               space
  //    ^                                   ^
  //  7 |                                 7 |
  //  6 |          +--------+             6 |
  //  5 |          |    B   |             5 |    +---------+
  //  4 |    +-----+---+----+             4 |    |    C    |
  //  3 |    |    C    |                  3 |    +-----+   |
  //  2 +----+----++   |                  2 +---------++---+----+
  //  1 |    A    |+---+                  1 |    A    ||   B    |
  //  0 +---------+                       0 +---------++--------+
  //    +----|----|----|----|--> time       +----|----|----|----|--> time
  //    0    5    10   15   20              0    5    10   15   20

  std::vector<AllocationBlock*> allocation_blocks;
  // Block A
  allocation_blocks.push_back(MakeAllocationBlock(0, 10, 2, 0));
  // Block B
  allocation_blocks.push_back(MakeAllocationBlock(11, 20, 2, 4));
  // Block C
  allocation_blocks.push_back(MakeAllocationBlock(5, 15, 3, 1));
  allocation_blocks.back()->original_slice_data = SlicedAllocationData(
      {{AllocatedSlice{1, 1, 5}, AllocatedSlice{2, 2, 11}}});

  // Specify the repacking sort order as the order in which blocks were added to
  // allocation_blocks. We need to do this so that B is placed before C.
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
      100, 1, SliceTimePermutationIterator::Ty::kAll, options_);

  // The repacker returns true as long as the result fits in the max size,
  // regardless of whether it has actually changed anything.
  EXPECT_TRUE(*repacker_.Repack(absl::MakeSpan(allocation_blocks)));

  // Check results
  //
  // Typically the heap_simulator would prefer to start the first slice of
  // Block C at time 5 and the second block at time 11, but that is not allowed
  // because it permutes the original slice size-time mapping.
  //
  // Block A
  EXPECT_EQ(allocation_blocks[0]->offset, 0);
  EXPECT_FALSE(allocation_blocks[0]->repacked_slice_data.has_value());
  // Block B
  EXPECT_EQ(allocation_blocks[1]->offset, 0);
  EXPECT_FALSE(allocation_blocks[1]->repacked_slice_data.has_value());
  // Block C
  EXPECT_EQ(allocation_blocks[2]->offset, 2);
  ASSERT_TRUE(allocation_blocks[2]->repacked_slice_data.has_value());
  EXPECT_EQ(*allocation_blocks[2]->repacked_slice_data,
            (SlicedAllocationData(
                {{AllocatedSlice{1, 2, 5}, AllocatedSlice{2, 3, 11}}})));
}

TEST_F(MemorySpaceAssignmentBestFitRepackerTest, SlicedColocationsFit) {
  // Expected repacking:
  //
  // space
  //    ^
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
  allocation_blocks.back()->original_slice_data = SlicedAllocationData(
      {{AllocatedSlice{2, -1, 15}, AllocatedSlice{3, -1, 18}}});
  // Block E
  allocation_blocks.push_back(MakeAllocationBlock(9, 14, 4));
  allocation_blocks.back()->original_slice_data = SlicedAllocationData(
      {{AllocatedSlice{2, -1, 9}, AllocatedSlice{2, -1, 12}}});
  // Colocate E with D.
  allocation_blocks.back()->next_colocated = allocation_blocks[3];
  allocation_blocks[3]->next_colocated = allocation_blocks.back();
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
      100, 1, SliceTimePermutationIterator::Ty::kAll, options_);

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
            (SlicedAllocationData(
                {{AllocatedSlice{2, 2, 15}, AllocatedSlice{3, 4, 18}}})));
  // Block E
  EXPECT_EQ(allocation_blocks[4]->offset, 2);
  ASSERT_TRUE(allocation_blocks[4]->repacked_slice_data.has_value());
  EXPECT_EQ(*allocation_blocks[4]->repacked_slice_data,
            (SlicedAllocationData(
                {{AllocatedSlice{2, 2, 9}, AllocatedSlice{2, 4, 12}}})));
  // Block F
  EXPECT_EQ(allocation_blocks[5]->offset, 4);
  EXPECT_FALSE(allocation_blocks[5]->repacked_slice_data.has_value());
}

// Test that we do not permute slice start times in a way that changes the
// original slice size-start time mappings. Doing so breaks assumptions that
// MSA uses to construct its internal state prior to repacking.
TEST_F(MemorySpaceAssignmentBestFitRepackerTest,
       SlicedColocationsPermutationsMatchOriginalSizeTimeMapping) {
  //  Original placement:                Ideal repacking, but unsupported:
  //
  //  space                              space
  //    ^                                  ^
  //  8 |                                8 |
  //  7 |+--------+     +---+            7 |
  //  6 ||        |     |   |            6 |
  //  5 ||    C   |     | D |            5 |+--------++--------+
  //  4 |+----+   |+----+   |            4 ||        ||        |
  //  3 |     |   ||        |            3 ||    C   ||    D   |
  //  2 |+---++---++---+----+            2 |+---++   |+---++   |
  //  1 || A |     | B |                 1 || A ||   || B ||   |
  //  0 |+---+     +---+                 0 |+---++---++---++---+
  //    +----|----|----|----|----> time    +----|----|----|----|----> time
  //    0    5    10   15   20             0    5    10   15   20

  std::vector<AllocationBlock*> allocation_blocks;
  // Block A
  allocation_blocks.push_back(MakeAllocationBlock(1, 5, 2));
  // Block B
  allocation_blocks.push_back(MakeAllocationBlock(11, 15, 2));
  // Block C
  allocation_blocks.push_back(MakeAllocationBlock(1, 10, 5));
  allocation_blocks.back()->original_slice_data = SlicedAllocationData(
      {{AllocatedSlice{2, 2, 6}, AllocatedSlice{3, 4, 1}}});
  // Block D
  allocation_blocks.push_back(MakeAllocationBlock(15, 20, 5));
  allocation_blocks.back()->original_slice_data = SlicedAllocationData(
      {{AllocatedSlice{2, 2, 11}, AllocatedSlice{3, 4, 16}}});
  // Colocate D with C.
  allocation_blocks.back()->next_colocated = allocation_blocks[2];
  allocation_blocks[2]->next_colocated = allocation_blocks.back();

  // Specify the repacking sort order as the order in which blocks were added to
  // allocation_blocks. By placing A and B before C/D, the repacker will try
  // permutations of C/D's slices that fit around A and B.
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
      100, 1, SliceTimePermutationIterator::Ty::kAll, options_);

  EXPECT_TRUE(*repacker_.Repack(absl::MakeSpan(allocation_blocks)));

  // Check results
  //
  // Typically the heap simulator would like start C/D at offset 0, which is
  // lower than C/D's actual placement at offset 2. However, in order to place
  // C/D at offset 0, we would need to permute the slice-time mappings of the
  // colocation D, which is not permitted.
  //
  // Block A
  EXPECT_EQ(allocation_blocks[0]->offset, 0);
  EXPECT_FALSE(allocation_blocks[0]->repacked_slice_data.has_value());
  // Block B
  EXPECT_EQ(allocation_blocks[1]->offset, 0);
  EXPECT_FALSE(allocation_blocks[1]->repacked_slice_data.has_value());
  // Block C
  EXPECT_EQ(allocation_blocks[2]->offset, 2);
  ASSERT_TRUE(allocation_blocks[2]->repacked_slice_data.has_value());
  EXPECT_EQ(*allocation_blocks[3]->repacked_slice_data,
            (SlicedAllocationData(
                {{AllocatedSlice{2, 2, 11}, AllocatedSlice{3, 4, 16}}})));
  // Block D
  EXPECT_EQ(allocation_blocks[3]->offset, 2);
  ASSERT_TRUE(allocation_blocks[3]->repacked_slice_data.has_value());
  EXPECT_EQ(*allocation_blocks[3]->repacked_slice_data,
            (SlicedAllocationData(
                {{AllocatedSlice{2, 2, 11}, AllocatedSlice{3, 4, 16}}})));
}

}  // namespace xla
