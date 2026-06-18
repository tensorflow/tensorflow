/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/tsl/framework/bfc_allocator.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <random>
#include <vector>

#include "absl/base/casts.h"
#include "absl/synchronization/blocking_counter.h"
#include "xla/tsl/framework/allocator.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/tsl/platform/threadpool.h"
#include "tsl/platform/mem.h"

namespace tsl {
namespace {

// Minimal SubAllocator backed by port::AlignedMalloc for host memory.
class MallocSubAllocator : public SubAllocator {
 public:
  MallocSubAllocator() : SubAllocator({}, {}) {}

  void* Alloc(size_t alignment, size_t num_bytes,
              size_t* bytes_received) override {
    void* ptr = port::AlignedMalloc(num_bytes,
                                    static_cast<std::align_val_t>(alignment));
    *bytes_received = num_bytes;
    return ptr;
  }

  void Free(void* ptr, size_t num_bytes) override { port::AlignedFree(ptr); }

  bool SupportsCoalescing() const override { return false; }
};

// Helper to check pointer alignment.
bool IsAligned(const void* ptr, size_t alignment) {
  return (absl::bit_cast<uintptr_t>(ptr) & (alignment - 1)) == 0;
}

TEST(BFCAllocatorTest, AllocateAndFree) {
  BFCAllocator alloc(std::make_unique<MallocSubAllocator>(),
                     /*total_memory=*/1 << 20, /*name=*/"test",
                     BFCAllocator::Options{});

  void* ptr = alloc.AllocateRaw(64, 512);
  ASSERT_NE(ptr, nullptr);
  alloc.DeallocateRaw(ptr);
}

TEST(BFCAllocatorTest, DefaultAlignment) {
  BFCAllocator alloc(std::make_unique<MallocSubAllocator>(),
                     /*total_memory=*/1 << 20, /*name=*/"test",
                     BFCAllocator::Options{});

  // BFC always returns pointers aligned to at least kAllocatorAlignment (64).
  void* ptr = alloc.AllocateRaw(Allocator::kAllocatorAlignment, 1);
  ASSERT_NE(ptr, nullptr);
  EXPECT_TRUE(IsAligned(ptr, Allocator::kAllocatorAlignment));
  alloc.DeallocateRaw(ptr);
}

// Parameterized test that verifies alignment is respected for various
// power-of-two alignments from 32 bytes to 4096 bytes.
class BFCAllocatorAlignmentTest : public ::testing::TestWithParam<size_t> {};

TEST_P(BFCAllocatorAlignmentTest, RespectsRequestedAlignment) {
  const size_t alignment = GetParam();
  BFCAllocator alloc(std::make_unique<MallocSubAllocator>(),
                     /*total_memory=*/1 << 20, /*name=*/"test",
                     BFCAllocator::Options{});

  // Allocate a small block first to push the arena cursor off any "lucky"
  // alignment, then allocate with the requested alignment.
  void* filler = alloc.AllocateRaw(Allocator::kAllocatorAlignment, 256);
  ASSERT_NE(filler, nullptr);

  constexpr int kTrials = 8;
  void* ptrs[kTrials];

  for (int i = 0; i < kTrials; ++i) {
    ptrs[i] = alloc.AllocateRaw(alignment, 256);
    ASSERT_NE(ptrs[i], nullptr);
    EXPECT_TRUE(IsAligned(ptrs[i], alignment))
        << "Allocation " << i << " at " << ptrs[i] << " not aligned to "
        << alignment;
  }

  for (int i = 0; i < kTrials; ++i) {
    alloc.DeallocateRaw(ptrs[i]);
  }
  alloc.DeallocateRaw(filler);
}

INSTANTIATE_TEST_SUITE_P(Alignments, BFCAllocatorAlignmentTest,
                         ::testing::Values(32, 64, 128, 256, 512, 1024, 2048,
                                           4096));

// Stress test: allocate and free chunks of varying sizes and alignments in
// randomized order across multiple iterations. This exercises chunk splitting,
// alignment padding, coalescing on free, and reuse of freed chunks.
TEST(BFCAllocatorTest, StressAllocFree) {
  BFCAllocator alloc(std::make_unique<MallocSubAllocator>(),
                     /*total_memory=*/16 << 20, /*name=*/"stress",
                     BFCAllocator::Options{});

  constexpr std::array<size_t, 5> kAlignments = {64, 256, 512, 1024, 4096};
  constexpr std::array<size_t, 7> kSizes = {1, 128, 256, 700, 1024, 4096, 8192};
  constexpr int kNumAllocs = kAlignments.size() * kSizes.size();
  constexpr int kIterations = 20;

  struct AllocSpec {
    size_t alignment;
    size_t size;
  };

  // Build 10 copies of each (alignment, size) pair = 350 allocations.
  constexpr int kCopies = 10;
  std::vector<AllocSpec> specs;
  specs.reserve(kNumAllocs * kCopies);
  for (int c = 0; c < kCopies; ++c) {
    for (size_t align : kAlignments) {
      for (size_t size : kSizes) {
        specs.push_back({align, size});
      }
    }
  }

  std::mt19937 rng(42);

  for (int iter = 0; iter < kIterations; ++iter) {
    // Shuffle allocation order each iteration.
    std::shuffle(specs.begin(), specs.end(), rng);

    std::vector<void*> ptrs;
    ptrs.reserve(specs.size());

    // Allocate all.
    for (const auto& spec : specs) {
      void* ptr = alloc.AllocateRaw(spec.alignment, spec.size);
      ASSERT_NE(ptr, nullptr)
          << "Failed at iter=" << iter << " size=" << spec.size
          << " alignment=" << spec.alignment;
      EXPECT_TRUE(IsAligned(ptr, spec.alignment))
          << "iter=" << iter << " ptr=" << ptr
          << " alignment=" << spec.alignment;
      ptrs.push_back(ptr);
    }

    // Shuffle deallocation order so free/coalesce paths vary.
    std::shuffle(ptrs.begin(), ptrs.end(), rng);

    for (void* ptr : ptrs) {
      alloc.DeallocateRaw(ptr);
    }
  }
}

// SubAllocator that always returns 256-byte (kMinAllocationSize) aligned
// memory but ignores higher alignment requests. This simulates GPU allocators
// like DeviceMemAllocator where cudaMalloc returns 256-byte aligned memory
// regardless of the requested alignment.
class GpuLikeSubAllocator : public SubAllocator {
 public:
  GpuLikeSubAllocator() : SubAllocator({}, {}) {}

  void* Alloc(size_t /*alignment*/, size_t num_bytes,
              size_t* bytes_received) override {
    // Always align to 256 bytes, ignoring the requested alignment.
    void* ptr = port::AlignedMalloc(num_bytes, std::align_val_t{256});
    *bytes_received = num_bytes;
    return ptr;
  }

  void Free(void* ptr, size_t num_bytes) override { port::AlignedFree(ptr); }

  bool SupportsCoalescing() const override { return false; }
};

// Verify that BFC still respects alignment even when the sub-allocator only
// provides 256-byte aligned regions (as GPU sub-allocators do).
TEST(BFCAllocatorTest, AlignmentWithGpuLikeSubAllocator) {
  BFCAllocator alloc(std::make_unique<GpuLikeSubAllocator>(),
                     /*total_memory=*/1 << 20, /*name=*/"gpu_like",
                     BFCAllocator::Options{});

  // Push the cursor off any lucky alignment.
  void* filler = alloc.AllocateRaw(Allocator::kAllocatorAlignment, 256);
  ASSERT_NE(filler, nullptr);

  constexpr std::array<size_t, 4> kAlignments = {256, 512, 1024, 4096};
  constexpr int kTrials = 8;

  for (size_t alignment : kAlignments) {
    for (int i = 0; i < kTrials; ++i) {
      void* ptr = alloc.AllocateRaw(alignment, 256);
      ASSERT_NE(ptr, nullptr);
      EXPECT_TRUE(IsAligned(ptr, alignment))
          << "ptr=" << ptr << " alignment=" << alignment;
      alloc.DeallocateRaw(ptr);
    }
  }

  alloc.DeallocateRaw(filler);
}

//===----------------------------------------------------------------------===//
// Performance benchmarks.
//===----------------------------------------------------------------------===//

static constexpr size_t kBenchAllocSize = 1024;
static constexpr size_t kBenchAlignment = Allocator::kAllocatorAlignment;

static void BM_AllocAndFree(benchmark::State& state) {
  BFCAllocator alloc(std::make_unique<MallocSubAllocator>(),
                     /*total_memory=*/256 << 20, /*name=*/"bench",
                     BFCAllocator::Options{});

  for (auto _ : state) {
    void* ptr = alloc.AllocateRaw(kBenchAlignment, kBenchAllocSize);
    alloc.DeallocateRaw(ptr);
  }
  state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BM_AllocAndFree);

static void BM_AllocBatchThenFree(benchmark::State& state) {
  int batch = state.range(0);
  BFCAllocator alloc(std::make_unique<MallocSubAllocator>(),
                     /*total_memory=*/256 << 20, /*name=*/"bench",
                     BFCAllocator::Options{});

  std::vector<void*> ptrs(batch);
  for (auto _ : state) {
    for (int i = 0; i < batch; ++i) {
      ptrs[i] = alloc.AllocateRaw(kBenchAlignment, kBenchAllocSize);
    }
    for (int i = 0; i < batch; ++i) {
      alloc.DeallocateRaw(ptrs[i]);
    }
  }
  state.SetItemsProcessed(state.iterations() * batch);
}

BENCHMARK(BM_AllocBatchThenFree)->Arg(100)->Arg(1000);

static void BM_AllocAndFreeUnderContention(benchmark::State& state) {
  size_t num_threads = state.range(0);
  static constexpr int kItersPerThread = 10000;

  BFCAllocator alloc(std::make_unique<MallocSubAllocator>(),
                     /*total_memory=*/256 << 20, /*name=*/"bench",
                     BFCAllocator::Options{});
  tsl::thread::ThreadPool threads(tsl::Env::Default(), "bench", num_threads);

  for (auto _ : state) {
    absl::BlockingCounter counter(num_threads);
    for (int t = 0; t < num_threads; ++t) {
      threads.Schedule([&] {
        for (int i = 0; i < kItersPerThread; ++i) {
          void* ptr = alloc.AllocateRaw(kBenchAlignment, kBenchAllocSize);
          alloc.DeallocateRaw(ptr);
        }
        counter.DecrementCount();
      });
    }
    counter.Wait();
  }
  state.SetItemsProcessed(state.iterations() * num_threads * kItersPerThread);
}

BENCHMARK(BM_AllocAndFreeUnderContention)
    ->MeasureProcessCPUTime()
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->Arg(16);

}  // namespace
}  // namespace tsl
