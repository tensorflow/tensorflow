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
#include "xla/service/heap_simulator/free_chunks_manager.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <random>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/tsl/platform/test_benchmark.h"

namespace xla {
namespace {

using ::testing::ElementsAre;

// The maximum bound in the artificial memory space. We subtract 3 because
// INT64_MAX - 1 is used as the start of a dummy chunk at the end and we want
// to have a gap between the dummy chunk and the real chunk.
constexpr int64_t kMaxBound = std::numeric_limits<int64_t>::max() - 3;

// The illustrations in tests below show the state of memory: '---' represents
// free chunks, while blank space represents allocated chunks. E.g.:
// |-|                  |-------|               |-------|            |-------
// 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
// means that memory chunks [0,1), [7,10), [15,18), [22, max) are free, and
// [1,7), [10,15), [18,22) are allocated.
TEST(FreeChunksTest, AllocateionMedium) {
  FreeChunksManager chunks_manager([](int64_t addr) { return addr; });
  EXPECT_THAT(chunks_manager.GetFreeChunks(),
              ElementsAre(FreeChunksManager::MemoryChunk{0, kMaxBound}));
  chunks_manager.Allocate(1, 4);
  // |-|         |-------------------------------------------------------------
  // 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
  EXPECT_THAT(chunks_manager.GetFreeChunks(),
              ElementsAre(FreeChunksManager::MemoryChunk{0, 1},
                          FreeChunksManager::MemoryChunk{4, kMaxBound}));
  chunks_manager.Allocate(4, 7);
  // |-|                  |----------------------------------------------------
  // 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
  EXPECT_THAT(chunks_manager.GetFreeChunks(),
              ElementsAre(FreeChunksManager::MemoryChunk{0, 1},
                          FreeChunksManager::MemoryChunk{7, kMaxBound}));
  chunks_manager.Allocate(18, 22);
  // |-|                  |-------------------------------|            |-------
  // 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
  EXPECT_THAT(chunks_manager.GetFreeChunks(),
              ElementsAre(FreeChunksManager::MemoryChunk{0, 1},
                          FreeChunksManager::MemoryChunk{7, 18},
                          FreeChunksManager::MemoryChunk{22, kMaxBound}));
  chunks_manager.Allocate(10, 15);
  // |-|                  |-------|               |-------|            |-------
  // 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
  EXPECT_THAT(chunks_manager.GetFreeChunks(),
              ElementsAre(FreeChunksManager::MemoryChunk{0, 1},
                          FreeChunksManager::MemoryChunk{7, 10},
                          FreeChunksManager::MemoryChunk{15, 18},
                          FreeChunksManager::MemoryChunk{22, kMaxBound}));
}

TEST(FreeChunksTest, RemovalMediumLeftAdjacent) {
  FreeChunksManager chunks_manager([](int64_t addr) { return addr; });
  chunks_manager.Allocate(1, 4);
  chunks_manager.Allocate(4, 7);
  chunks_manager.Allocate(10, 15);
  chunks_manager.Allocate(18, 22);
  // |-|                  |-------|               |-------|            |-------
  // 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
  EXPECT_THAT(chunks_manager.GetFreeChunks(),
              ElementsAre(FreeChunksManager::MemoryChunk{0, 1},
                          FreeChunksManager::MemoryChunk{7, 10},
                          FreeChunksManager::MemoryChunk{15, 18},
                          FreeChunksManager::MemoryChunk{22, kMaxBound}));
  // Removing a chunk that is only adjacent to one on the left.
  chunks_manager.Deallocate(4, 7);
  // |-|         |----------------|               |-------|            |-------
  // 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
  EXPECT_THAT(chunks_manager.GetFreeChunks(),
              ElementsAre(FreeChunksManager::MemoryChunk{0, 1},
                          FreeChunksManager::MemoryChunk{4, 10},
                          FreeChunksManager::MemoryChunk{15, 18},
                          FreeChunksManager::MemoryChunk{22, kMaxBound}));
}

TEST(FreeChunksTest, RemovalMediumRightAdjacent) {
  FreeChunksManager chunks_manager([](int64_t addr) { return addr; });
  chunks_manager.Allocate(1, 4);
  chunks_manager.Allocate(4, 7);
  chunks_manager.Allocate(14, 18);
  chunks_manager.Allocate(18, 22);
  // |-|                  |-------------------|                        |-------
  // 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
  EXPECT_THAT(chunks_manager.GetFreeChunks(),
              ElementsAre(FreeChunksManager::MemoryChunk{0, 1},
                          FreeChunksManager::MemoryChunk{7, 14},
                          FreeChunksManager::MemoryChunk{22, kMaxBound}));
  // Removing a chunk that is only adjacent to one on the right.
  chunks_manager.Deallocate(14, 18);
  // |-|                  |-------------------------------|            |-------
  // 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
  EXPECT_THAT(chunks_manager.GetFreeChunks(),
              ElementsAre(FreeChunksManager::MemoryChunk{0, 1},
                          FreeChunksManager::MemoryChunk{7, 18},
                          FreeChunksManager::MemoryChunk{22, kMaxBound}));
}

TEST(FreeChunksTest, RemovalMediumBothAdjacent) {
  FreeChunksManager chunks_manager([](int64_t addr) { return addr; });
  chunks_manager.Allocate(1, 4);
  chunks_manager.Allocate(4, 7);
  chunks_manager.Allocate(7, 18);
  chunks_manager.Allocate(18, 22);
  // |-|                                                               |-------
  // 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
  EXPECT_THAT(chunks_manager.GetFreeChunks(),
              ElementsAre(FreeChunksManager::MemoryChunk{0, 1},
                          FreeChunksManager::MemoryChunk{22, kMaxBound}));
  // Removing a chunk that is adjacent to both sides.
  chunks_manager.Deallocate(7, 18);
  // |-|                  |-------------------------------|            |-------
  // 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
  EXPECT_THAT(chunks_manager.GetFreeChunks(),
              ElementsAre(FreeChunksManager::MemoryChunk{0, 1},
                          FreeChunksManager::MemoryChunk{7, 18},
                          FreeChunksManager::MemoryChunk{22, kMaxBound}));
}

TEST(FreeChunksTest, FindJustLargeEnoughInitialChunk) {
  FreeChunksManager chunks_manager([](int64_t addr) { return addr; });
  EXPECT_THAT(chunks_manager.GetFreeChunks(),
              ElementsAre(FreeChunksManager::MemoryChunk{0, kMaxBound}));
  auto chunk = chunks_manager.FindJustLargeEnough(1);
  ASSERT_TRUE(chunk.has_value());
  EXPECT_EQ(chunk->offset(), 0);
  EXPECT_EQ(chunk->end(), kMaxBound);
}

TEST(FreeChunksTest, FindJustLargeEnoughSimple) {
  FreeChunksManager chunks_manager([](int64_t addr) { return addr; });
  chunks_manager.Allocate(1, 4);
  chunks_manager.Allocate(7, 10);
  chunks_manager.Allocate(15, 22);
  // |-|         |-------|         |-------------|                     |-------
  // 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
  EXPECT_THAT(chunks_manager.GetFreeChunks(),
              ElementsAre(FreeChunksManager::MemoryChunk{0, 1},
                          FreeChunksManager::MemoryChunk{4, 7},
                          FreeChunksManager::MemoryChunk{10, 15},
                          FreeChunksManager::MemoryChunk{22, kMaxBound}));

  auto size_gt_1 = chunks_manager.FindJustLargeEnough(1);
  // |-|         |-------|         |-------------|                     |-------
  // |~|
  // 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
  ASSERT_TRUE(size_gt_1.has_value());
  EXPECT_EQ(size_gt_1->offset(), 0);
  EXPECT_EQ(size_gt_1->end(), 1);

  auto size_gt_2 = chunks_manager.FindJustLargeEnough(2);
  // |-|         |-------|         |-------------|                     |-------
  //             |~~~~~~~|
  // 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
  ASSERT_TRUE(size_gt_2.has_value());
  EXPECT_EQ(size_gt_2->offset(), 4);
  EXPECT_EQ(size_gt_2->end(), 7);

  auto size_gt_3 = chunks_manager.FindJustLargeEnough(3);
  // |-|         |-------|         |-------------|                     |-------
  //             |~~~~~~~|
  // 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
  ASSERT_TRUE(size_gt_3.has_value());
  EXPECT_EQ(size_gt_3->offset(), 4);
  EXPECT_EQ(size_gt_3->end(), 7);

  auto size_gt_4 = chunks_manager.FindJustLargeEnough(4);
  // |-|         |-------|         |-------------|                     |-------
  //                               |~~~~~~~~~~~~~|
  // 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
  ASSERT_TRUE(size_gt_4.has_value());
  EXPECT_EQ(size_gt_4->offset(), 10);
  EXPECT_EQ(size_gt_4->end(), 15);
}

TEST(FreeChunksTest, FindJustLargeEnoughWithUnboundedChunk) {
  FreeChunksManager chunks_manager([](int64_t addr) { return addr; });
  chunks_manager.Allocate(1, 4);
  chunks_manager.Allocate(10, 15);
  chunks_manager.Allocate(17, 22);
  // |-|         |----------------|               |----|               |-------
  // 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
  EXPECT_THAT(chunks_manager.GetFreeChunks(),
              ElementsAre(FreeChunksManager::MemoryChunk{0, 1},
                          FreeChunksManager::MemoryChunk{4, 10},
                          FreeChunksManager::MemoryChunk{15, 17},
                          FreeChunksManager::MemoryChunk{22, kMaxBound}));

  auto size_gt_1 = chunks_manager.FindJustLargeEnough(1);
  // |-|         |----------------|               |----|               |-------
  // |~|
  // 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
  ASSERT_TRUE(size_gt_1.has_value());
  EXPECT_EQ(size_gt_1->offset(), 0);
  EXPECT_EQ(size_gt_1->end(), 1);

  auto size_gt_2 = chunks_manager.FindJustLargeEnough(2);
  // |-|         |----------------|               |----|               |-------
  //                                              |~~~~|
  // 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
  ASSERT_TRUE(size_gt_2.has_value());
  EXPECT_EQ(size_gt_2->offset(), 15);
  EXPECT_EQ(size_gt_2->end(), 17);

  auto size_gt_3 = chunks_manager.FindJustLargeEnough(3);
  // |-|         |----------------|               |----|               |-------
  //             |~~~~~~~~~~~~~~~~|
  // 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
  ASSERT_TRUE(size_gt_3.has_value());
  EXPECT_EQ(size_gt_3->offset(), 4);
  EXPECT_EQ(size_gt_3->end(), 10);

  auto size_gt_6 = chunks_manager.FindJustLargeEnough(6);
  // |-|         |----------------|               |----|               |-------
  //             |~~~~~~~~~~~~~~~~|
  // 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
  ASSERT_TRUE(size_gt_6.has_value());
  EXPECT_EQ(size_gt_6->offset(), 4);
  EXPECT_EQ(size_gt_6->end(), 10);

  auto size_gt_7 = chunks_manager.FindJustLargeEnough(7);
  // |-|         |----------------|               |----|               |-------
  //                                                                   |~~~~~~~
  // 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
  ASSERT_TRUE(size_gt_7.has_value());
  EXPECT_EQ(size_gt_7->offset(), 22);
  EXPECT_GT(size_gt_7->end(), 1000);  // Returns an unbounded chunk.
}

TEST(FreeChunksTest, FindJustLargeEnoughMediumNoGap) {
  FreeChunksManager chunks_manager([](int64_t addr) { return addr; });
  chunks_manager.Allocate(1, 4);
  chunks_manager.Allocate(10, 15);
  chunks_manager.Allocate(4, 10);
  chunks_manager.Allocate(17, 22);
  // |-|                                                               |-------
  // 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
  EXPECT_THAT(chunks_manager.GetFreeChunks(),
              ElementsAre(FreeChunksManager::MemoryChunk{0, 1},
                          FreeChunksManager::MemoryChunk{15, 17},
                          FreeChunksManager::MemoryChunk{22, kMaxBound}));

  auto size_gt_2 = chunks_manager.FindJustLargeEnough(2);
  // |-|                                                               |-------
  //                                              |~~~~|
  // 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
  ASSERT_TRUE(size_gt_2.has_value());
  EXPECT_EQ(size_gt_2->offset(), 15);
  EXPECT_EQ(size_gt_2->end(), 17);

  auto size_gt_3 = chunks_manager.FindJustLargeEnough(3);
  // |-|                                                               |-------
  //                                                                   |~~~~~~~
  // 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
  ASSERT_TRUE(size_gt_3.has_value());
  EXPECT_EQ(size_gt_3->offset(), 22);
  EXPECT_GT(size_gt_3->end(), 1000);  // Returns an unbounded chunk.
}

TEST(FreeChunksTest, FindJustLargeEnoughReturnsNullopt) {
  FreeChunksManager chunks_manager([](int64_t addr) { return addr; });
  chunks_manager.Allocate(0, kMaxBound);
  EXPECT_TRUE(chunks_manager.GetFreeChunks().empty());
  EXPECT_EQ(chunks_manager.FindJustLargeEnough(1), std::nullopt);
}

void BM_FreeChunksManagerStress(::testing::benchmark::State& state) {
  std::mt19937 generator;
  std::uniform_int_distribution<int64_t> size_dist(1, 100);
  std::bernoulli_distribution op_dist(0.5);  // 0.5 for insert/erase

  for (auto s : state) {
    state.PauseTiming();
    FreeChunksManager chunks_manager([](int64_t addr) { return addr; });
    std::vector<std::pair<int64_t, int64_t>> allocated_chunks;
    int64_t num_ops = state.range(0);
    state.ResumeTiming();

    for (int i = 0; i < num_ops; ++i) {
      if (allocated_chunks.empty() || op_dist(generator)) {
        // Allocate
        int64_t size = size_dist(generator);
        auto chunk = chunks_manager.FindJustLargeEnough(size);
        if (!chunk.has_value()) {
          continue;
        }
        chunks_manager.Allocate(chunk->aligned_chunk_offset(),
                                chunk->aligned_chunk_offset() + size);
        allocated_chunks.push_back({chunk->aligned_chunk_offset(),
                                    chunk->aligned_chunk_offset() + size});
      } else {
        // Deallocate
        std::uniform_int_distribution<size_t> erase_dist(
            0, allocated_chunks.size() - 1);
        size_t idx = erase_dist(generator);
        std::pair<int64_t, int64_t> chunk_to_erase = allocated_chunks[idx];
        chunks_manager.Deallocate(chunk_to_erase.first, chunk_to_erase.second);
        std::swap(allocated_chunks[idx], allocated_chunks.back());
        allocated_chunks.pop_back();
      }
    }
  }
}

BENCHMARK(BM_FreeChunksManagerStress)->Range(1'000, 1'000'000);

}  // namespace
}  // namespace xla
