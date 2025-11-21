/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/hlo/transforms/simplifiers/hlo_rematerialization_data_structures.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace xla {
namespace {

using ::testing::AllOf;
using ::testing::Field;
using ::testing::status::IsOk;
using ::testing::status::IsOkAndHolds;

// Our classes do not actually dereference their HloInstruction pointers, so we
// just make some up for testing purposes.
HloInstruction* GenerateFakeInstructionPointer(int id) {
  // Add one so that id 0 does not map to nullptr.
  return reinterpret_cast<HloInstruction*>(id + 1);
}

TEST(SleatorDietzOrderMaintenanceTest, InsertSequenceAndCompare) {
  // Inserting 100 things triggers a relabeling of the base record.
  constexpr int kNumInstructions = 100;

  SleatorDietzOrderMaintenance ordering;
  for (int i = 0; i < kNumInstructions; ++i) {
    // We insert each new instruction at the end.
    ASSERT_OK(ordering.InsertBeforeInstruction(
        /*old_instruction=*/nullptr,
        /*new_instruction=*/GenerateFakeInstructionPointer(i)));
    ASSERT_OK(ordering.VerifyInvariantsForTesting());
  }

  for (int i = 0; i < kNumInstructions; ++i) {
    for (int j = 0; j < kNumInstructions; ++j) {
      EXPECT_THAT(ordering.CompareOrder(GenerateFakeInstructionPointer(i),
                                        GenerateFakeInstructionPointer(j)),
                  IsOkAndHolds(i < j));
    }
  }
}

TEST(SleatorDietzOrderMaintenanceTest, AlternatingUpdateAndQuery) {
  SleatorDietzOrderMaintenance ordering;
  ASSERT_OK(ordering.InsertBeforeInstruction(
      /*old_instruction=*/nullptr,
      /*new_instruction=*/GenerateFakeInstructionPointer(5)));

  // ordering = {5}
  EXPECT_THAT(ordering.CompareOrder(GenerateFakeInstructionPointer(5),
                                    GenerateFakeInstructionPointer(5)),
              IsOkAndHolds(false));
  ASSERT_OK(ordering.InsertAfterInstruction(
      /*old_instruction=*/nullptr,
      /*new_instruction=*/GenerateFakeInstructionPointer(2)));

  // ordering = {2, 5}
  EXPECT_THAT(ordering.CompareOrder(GenerateFakeInstructionPointer(2),
                                    GenerateFakeInstructionPointer(5)),
              IsOkAndHolds(true));
  ASSERT_OK(ordering.InsertAfterInstruction(
      /*old_instruction=*/GenerateFakeInstructionPointer(2),
      /*new_instruction=*/GenerateFakeInstructionPointer(7)));

  // ordering = {2, 7, 5}
  EXPECT_THAT(ordering.CompareOrder(GenerateFakeInstructionPointer(5),
                                    GenerateFakeInstructionPointer(7)),
              IsOkAndHolds(false));
  ASSERT_OK(ordering.DeleteInstruction(GenerateFakeInstructionPointer(5)));

  // ordering = {2, 7}
  EXPECT_THAT(ordering.CompareOrder(GenerateFakeInstructionPointer(2),
                                    GenerateFakeInstructionPointer(7)),
              IsOkAndHolds(true));
}

TEST(SleatorDietzOrderMaintenanceTest, ContainsInstruction) {
  SleatorDietzOrderMaintenance ordering;
  ASSERT_OK(ordering.InsertBeforeInstruction(
      /*old_instruction=*/nullptr,
      /*new_instruction=*/GenerateFakeInstructionPointer(5)));
  ASSERT_OK(ordering.InsertAfterInstruction(
      /*old_instruction=*/nullptr,
      /*new_instruction=*/GenerateFakeInstructionPointer(2)));
  // ordering = {2, 5}
  EXPECT_TRUE(ordering.ContainsInstruction(GenerateFakeInstructionPointer(2)));
  EXPECT_TRUE(ordering.ContainsInstruction(GenerateFakeInstructionPointer(5)));
  EXPECT_FALSE(ordering.ContainsInstruction(GenerateFakeInstructionPointer(7)));
}

TEST(SleatorDietzOrderMaintenanceTest, GetFirstInstruction) {
  SleatorDietzOrderMaintenance ordering;
  EXPECT_EQ(ordering.GetFirstInstruction(), nullptr);
  ASSERT_OK(ordering.InsertBeforeInstruction(
      /*old_instruction=*/nullptr,
      /*new_instruction=*/GenerateFakeInstructionPointer(5)));
  ASSERT_OK(ordering.InsertAfterInstruction(
      /*old_instruction=*/nullptr,
      /*new_instruction=*/GenerateFakeInstructionPointer(2)));
  // ordering = {2, 5}
  EXPECT_EQ(ordering.GetFirstInstruction(), GenerateFakeInstructionPointer(2));
}

TEST(SleatorDietzOrderMaintenanceTest, GetPreviousAndNextInstruction) {
  constexpr int kNumInstructions = 100;

  SleatorDietzOrderMaintenance ordering;
  for (int i = 0; i < kNumInstructions; ++i) {
    // We insert each new instruction at the end.
    ASSERT_OK(ordering.InsertBeforeInstruction(
        /*old_instruction=*/nullptr,
        /*new_instruction=*/GenerateFakeInstructionPointer(i)));
    ASSERT_OK(ordering.VerifyInvariantsForTesting());
  }

  for (int i = 0; i < kNumInstructions; ++i) {
    EXPECT_THAT(
        ordering.GetPreviousInstruction(GenerateFakeInstructionPointer(i)),
        IsOkAndHolds(i == 0 ? nullptr : GenerateFakeInstructionPointer(i - 1)));
    EXPECT_THAT(ordering.GetNextInstruction(GenerateFakeInstructionPointer(i)),
                IsOkAndHolds(i == kNumInstructions - 1
                                 ? nullptr
                                 : GenerateFakeInstructionPointer(i + 1)));
  }
}

// Generate simple input vectors for AVLLazySegmentTreeTest.
std::vector<MemoryUsageAndInstruction> GenerateInitialInstructions(int n) {
  std::vector<MemoryUsageAndInstruction> instructions;
  for (int i = 0; i < n; ++i) {
    instructions.push_back({.memory_usage = 101 * ((i % 2) + 1),
                            .instruction = GenerateFakeInstructionPointer(i)});
  }
  return instructions;
}

TEST(AVLLazySegmentTreeTest, EmptyInitialization) {
  AVLLazySegmentTree segtree({});
  ASSERT_OK(segtree.VerifyInvariantsForTesting());
}

TEST(AVLLazySegmentTreeTest, NonemptyInitialization) {
  AVLLazySegmentTree segtree(GenerateInitialInstructions(100));
  ASSERT_OK(segtree.VerifyInvariantsForTesting());
}

TEST(AVLLazySegmentTreeTest, PureQueries) {
  AVLLazySegmentTree segtree(GenerateInitialInstructions(10));
  ASSERT_OK(segtree.VerifyInvariantsForTesting());

  for (int i = 0; i < 10; ++i) {
    for (int j = i; j < 10; ++j) {
      // Without any updates, querying the range [i, j] results in the first odd
      // instruction winning (or if there is no odd instruction, then the first
      // even instruction).
      int winner = (i % 2 == 0 && j > i) ? i + 1 : i;
      EXPECT_THAT(
          segtree.Query(GenerateFakeInstructionPointer(i),
                        GenerateFakeInstructionPointer(j)),
          IsOkAndHolds(AllOf(Field(&MemoryUsageAndInstruction::memory_usage,
                                   101 * ((winner % 2) + 1)),
                             Field(&MemoryUsageAndInstruction::instruction,
                                   GenerateFakeInstructionPointer(winner)))));
      ASSERT_OK(segtree.VerifyInvariantsForTesting());
    }
  }
  EXPECT_THAT(
      segtree.Query(),
      IsOkAndHolds(AllOf(Field(&MemoryUsageAndInstruction::memory_usage, 202),
                         Field(&MemoryUsageAndInstruction::instruction,
                               GenerateFakeInstructionPointer(1)))));
}

TEST(AVLLazySegmentTreeTest, AlternatingInsertAndQuery) {
  AVLLazySegmentTree segtree(GenerateInitialInstructions(5));
  ASSERT_OK(segtree.VerifyInvariantsForTesting());
  // (0 -> 101) (1 -> 202) (2 -> 101) (3 -> 202) (4 -> 101)
  EXPECT_THAT(segtree.InsertBeforeInstruction(
                  GenerateFakeInstructionPointer(1),
                  {.memory_usage = 251,
                   .instruction = GenerateFakeInstructionPointer(5)}),
              IsOk());
  ASSERT_OK(segtree.VerifyInvariantsForTesting());
  // (0 -> 101) (5 -> 251) (1 -> 202) (2 -> 101) (3 -> 202) (4 -> 101)
  // |---------------------------------------------------------------|
  // Query Range
  EXPECT_THAT(
      segtree.Query(GenerateFakeInstructionPointer(0),
                    GenerateFakeInstructionPointer(4)),
      IsOkAndHolds(AllOf(Field(&MemoryUsageAndInstruction::memory_usage, 251),
                         Field(&MemoryUsageAndInstruction::instruction,
                               GenerateFakeInstructionPointer(5)))));
  ASSERT_OK(segtree.VerifyInvariantsForTesting());

  EXPECT_THAT(segtree.InsertAfterInstruction(
                  GenerateFakeInstructionPointer(4),
                  {.memory_usage = 1,
                   .instruction = GenerateFakeInstructionPointer(6)}),
              IsOk());
  ASSERT_OK(segtree.VerifyInvariantsForTesting());
  // (0 -> 101) (5 -> 251) (1 -> 202) (2 -> 101) (3 -> 202) (4 -> 101) (6 -> 1)
  //                                                       Query Range |------|
  EXPECT_THAT(
      segtree.Query(GenerateFakeInstructionPointer(6),
                    GenerateFakeInstructionPointer(6)),
      IsOkAndHolds(AllOf(Field(&MemoryUsageAndInstruction::memory_usage, 1),
                         Field(&MemoryUsageAndInstruction::instruction,
                               GenerateFakeInstructionPointer(6)))));
  ASSERT_OK(segtree.VerifyInvariantsForTesting());
}

TEST(AVLLazySegmentTreeTest, AlternatingDeleteAndQuery) {
  AVLLazySegmentTree segtree(GenerateInitialInstructions(5));
  ASSERT_OK(segtree.VerifyInvariantsForTesting());
  // (0 -> 101) (1 -> 202) (2 -> 101) (3 -> 202) (4 -> 101)
  EXPECT_THAT(segtree.Delete(GenerateFakeInstructionPointer(1)), IsOk());
  ASSERT_OK(segtree.VerifyInvariantsForTesting());
  // (0 -> 101) (2 -> 101) (3 -> 202) (4 -> 101)
  // |-----------------------------------------| Query Range
  EXPECT_THAT(
      segtree.Query(GenerateFakeInstructionPointer(0),
                    GenerateFakeInstructionPointer(4)),
      IsOkAndHolds(AllOf(Field(&MemoryUsageAndInstruction::memory_usage, 202),
                         Field(&MemoryUsageAndInstruction::instruction,
                               GenerateFakeInstructionPointer(3)))));
  ASSERT_OK(segtree.VerifyInvariantsForTesting());

  EXPECT_THAT(segtree.Delete(GenerateFakeInstructionPointer(3)), IsOk());
  ASSERT_OK(segtree.VerifyInvariantsForTesting());
  // (0 -> 101) (2 -> 101) (4 -> 101)
  // |------------------------------| Query Range
  EXPECT_THAT(
      segtree.Query(GenerateFakeInstructionPointer(0),
                    GenerateFakeInstructionPointer(4)),
      IsOkAndHolds(AllOf(Field(&MemoryUsageAndInstruction::memory_usage, 101),
                         Field(&MemoryUsageAndInstruction::instruction,
                               GenerateFakeInstructionPointer(0)))));
  ASSERT_OK(segtree.VerifyInvariantsForTesting());
}

TEST(AVLLazySegmentTreeTest, AlternatingUpdateAndQuery) {
  AVLLazySegmentTree segtree(GenerateInitialInstructions(5));
  ASSERT_OK(segtree.VerifyInvariantsForTesting());
  // (0 -> 101) (1 -> 202) (2 -> 101) (3 -> 202) (4 -> 101)
  //            |------------------------------| Update Range
  EXPECT_THAT(segtree.Update(GenerateFakeInstructionPointer(1),
                             GenerateFakeInstructionPointer(3), -101),
              IsOk());
  ASSERT_OK(segtree.VerifyInvariantsForTesting());
  // (0 -> 101) (1 -> 101) (2 -> 0) (3 -> 101) (4 -> 101)
  //            |---------------------------------------| Query Range
  EXPECT_THAT(
      segtree.Query(GenerateFakeInstructionPointer(1),
                    GenerateFakeInstructionPointer(4)),
      IsOkAndHolds(AllOf(Field(&MemoryUsageAndInstruction::memory_usage, 101),
                         Field(&MemoryUsageAndInstruction::instruction,
                               GenerateFakeInstructionPointer(1)))));
  ASSERT_OK(segtree.VerifyInvariantsForTesting());

  // (0 -> 101) (1 -> 101) (2 -> 0) (3 -> 101) (4 -> 101)
  //                       |-----------------| Update Range
  EXPECT_THAT(segtree.Update(GenerateFakeInstructionPointer(2),
                             GenerateFakeInstructionPointer(3), 10000),
              IsOk());
  ASSERT_OK(segtree.VerifyInvariantsForTesting());
  // (0 -> 101) (1 -> 101) (2 -> 10000) (3 -> 10101) (4 -> 101)
  // |--------------------------------------------------------| Query Range
  EXPECT_THAT(
      segtree.Query(GenerateFakeInstructionPointer(0),
                    GenerateFakeInstructionPointer(4)),
      IsOkAndHolds(AllOf(Field(&MemoryUsageAndInstruction::memory_usage, 10101),
                         Field(&MemoryUsageAndInstruction::instruction,
                               GenerateFakeInstructionPointer(3)))));
  ASSERT_OK(segtree.VerifyInvariantsForTesting());
}

}  // namespace
}  // namespace xla
