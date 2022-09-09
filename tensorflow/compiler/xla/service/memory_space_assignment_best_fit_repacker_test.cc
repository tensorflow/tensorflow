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

#include "tensorflow/compiler/xla/service/memory_space_assignment_best_fit_repacker.h"

#include "tensorflow/tsl/platform/test.h"

namespace xla {

class MemorySpaceAssignmentBestFitRepackerTest : public ::testing::Test {
 protected:
  using AllocationBlock = MemorySpaceAssignmentRepacker::AllocationBlock;

  MemorySpaceAssignmentBestFitRepackerTest() : repacker_(100, 1) {}

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
  MemorySpaceAssignmentBestFitRepacker repacker_;
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

}  // namespace xla
