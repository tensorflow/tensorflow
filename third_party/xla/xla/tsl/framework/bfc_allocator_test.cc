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
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <random>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/base/log_severity.h"
#include "absl/base/no_destructor.h"
#include "absl/log/scoped_mock_log.h"
#include "absl/synchronization/blocking_counter.h"
#include "xla/tsl/framework/allocator.h"
#include "xla/tsl/framework/scoped_allocation_trace.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/tsl/platform/threadpool.h"

namespace tsl {
namespace {

using ::testing::_;
using ::testing::AllOf;
using ::testing::AtLeast;
using ::testing::HasSubstr;

static constexpr size_t kAlignment = Allocator::kAllocatorAlignment;

static const absl::NoDestructor<AllocationAttributes> kUpper(
    /*retry_on_failure=*/false, /*allocation_will_be_logged=*/false,
    /*freed_by_func=*/nullptr, AllocationEnd::kUpper);

static const absl::NoDestructor<AllocationAttributes> kLower(
    /*retry_on_failure=*/false, /*allocation_will_be_logged=*/false,
    /*freed_by_func=*/nullptr, AllocationEnd::kLower);

// SubAllocator that hands out fake (non-dereferenceable) addresses without
// allocating any real memory. It bump-allocates from a large, fixed virtual
// base so addresses are unique, well-aligned, and consistent. This lets tests
// exercise huge pools and verify the exact addresses BFC returns without
// touching device memory.
class FakeSubAllocator : public SubAllocator {
 public:
  // kBase is a high, page-aligned constant so returned addresses look like
  // plausible device pointers and never collide with real ones.
  static constexpr uintptr_t kBase = uintptr_t{1} << 40;

  explicit FakeSubAllocator(
      std::optional<size_t> hardcoded_alignment = std::nullopt)
      : SubAllocator({}, {}), hardcoded_alignment_(hardcoded_alignment) {}

  void* Alloc(size_t alignment, size_t num_bytes,
              size_t* bytes_received) override {
    const size_t effective_alignment = hardcoded_alignment_.value_or(alignment);
    uintptr_t aligned =
        (next_ + (effective_alignment - 1)) & ~(effective_alignment - 1);
    next_ = aligned + num_bytes;
    *bytes_received = num_bytes;
    return absl::bit_cast<void*>(aligned);
  }

  void Free(void* ptr, size_t num_bytes) override {}

  bool SupportsCoalescing() const override { return false; }

 private:
  std::optional<size_t> hardcoded_alignment_;
  uintptr_t next_ = kBase;
};

// Helper to check pointer alignment.
bool IsAligned(const void* ptr, size_t alignment) {
  return (absl::bit_cast<uintptr_t>(ptr) & (alignment - 1)) == 0;
}

TEST(BFCAllocatorTest, AllocateAndFree) {
  BFCAllocator alloc(std::make_unique<FakeSubAllocator>(),
                     /*total_memory=*/1 << 20, /*name=*/"test",
                     BFCAllocator::Options{});

  void* ptr = alloc.AllocateRaw(64, 512);
  ASSERT_NE(ptr, nullptr);
  alloc.DeallocateRaw(ptr);
}

TEST(BFCAllocatorTest, DefaultAlignment) {
  BFCAllocator alloc(std::make_unique<FakeSubAllocator>(),
                     /*total_memory=*/1 << 20, /*name=*/"test",
                     BFCAllocator::Options{});

  // BFC always returns pointers aligned to at least kAllocatorAlignment (64).
  void* ptr = alloc.AllocateRaw(kAlignment, 1);
  ASSERT_NE(ptr, nullptr);
  EXPECT_TRUE(IsAligned(ptr, kAlignment));
  alloc.DeallocateRaw(ptr);
}

TEST(BFCAllocatorTest, OomLogsAllocationAnnotations) {
  BFCAllocator::Options opts;
  opts.allow_growth = false;
  opts.allow_retry_on_failure = false;
  BFCAllocator alloc(std::make_unique<FakeSubAllocator>(),
                     /*total_memory=*/1024, /*name=*/"annotated", opts);

  void* ptr = nullptr;
  {
    ScopedAllocationTrace exec_scope("xla.execute",
                                     {{"executable", "module"}, {"device", 7}});
    ScopedAllocationTrace buffer_scope(
        "xla.buffer", {{"kind", "live_out"}, {"allocation_index", 3}});
    ptr = alloc.AllocateRaw(kAlignment, 512);
  }
  ASSERT_NE(ptr, nullptr);

  absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
  EXPECT_CALL(
      log,
      Log(absl::LogSeverity::kInfo, _,
          AllOf(HasSubstr("InUse at"), HasSubstr("allocation_annotation"),
                HasSubstr("xla.execute{executable=module, device=7}"),
                HasSubstr("xla.buffer{kind=live_out, allocation_index=3}"))))
      .Times(AtLeast(1));
  log.StartCapturingLogs();

  EXPECT_EQ(alloc.AllocateRaw(kAlignment, 2048), nullptr);

  log.StopCapturingLogs();
  alloc.DeallocateRaw(ptr);
}

// Parameterized test that verifies alignment is respected for various
// power-of-two alignments from 32 bytes to 4096 bytes.
class BFCAllocatorAlignmentTest : public ::testing::TestWithParam<size_t> {};

TEST_P(BFCAllocatorAlignmentTest, RespectsRequestedAlignment) {
  const size_t alignment = GetParam();
  BFCAllocator alloc(std::make_unique<FakeSubAllocator>(),
                     /*total_memory=*/1 << 20, /*name=*/"test",
                     BFCAllocator::Options{});

  // Allocate a small block first to push the arena cursor off any "lucky"
  // alignment, then allocate with the requested alignment.
  void* filler = alloc.AllocateRaw(kAlignment, 256);
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
  BFCAllocator alloc(std::make_unique<FakeSubAllocator>(),
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

// Verify that BFC respects requested alignment even when the sub-allocator
// ignores it and returns addresses aligned above the required minimum.
TEST(BFCAllocatorTest, AlignmentWithHardcodedSubAllocatorAlignment) {
  constexpr size_t kHardcodedAlignment = 256;
  BFCAllocator alloc(std::make_unique<FakeSubAllocator>(kHardcodedAlignment),
                     /*total_memory=*/1 << 20,
                     /*name=*/"hardcoded_alignment", BFCAllocator::Options{});

  // Push the cursor off any lucky alignment.
  void* filler = alloc.AllocateRaw(kAlignment, 256);
  ASSERT_NE(filler, nullptr);

  constexpr std::array<size_t, 4> kAlignments = {kHardcodedAlignment, 512, 1024,
                                                 4096};
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
// Spatial partitioning tests.
//===----------------------------------------------------------------------===//

TEST(BFCAllocatorTest, SpatialAllocatesFromEnds) {
  BFCAllocator::Options opts;
  opts.allow_growth = false;
  opts.enable_spatial_partitioning = true;
  BFCAllocator alloc(std::make_unique<FakeSubAllocator>(),
                     /*total_memory=*/4096, /*name=*/"spatial", opts);

  void* lower = alloc.AllocateRaw(kAlignment, 256);
  ASSERT_NE(lower, nullptr);

  void* upper = alloc.AllocateRaw(kAlignment, 256, *kUpper);
  ASSERT_NE(upper, nullptr);

  EXPECT_EQ(absl::bit_cast<uintptr_t>(upper) - absl::bit_cast<uintptr_t>(lower),
            4096 - 256);

  alloc.DeallocateRaw(upper);
  alloc.DeallocateRaw(lower);
}

// Lower activity does not perturb upper offsets.
TEST(BFCAllocatorTest, SpatialKeepsUpperOffset) {
  BFCAllocator::Options opts;
  opts.allow_growth = false;
  opts.enable_spatial_partitioning = true;

  BFCAllocator alloc_a(std::make_unique<FakeSubAllocator>(),
                       /*total_memory=*/4096, /*name=*/"spatial_a", opts);
  BFCAllocator alloc_b(std::make_unique<FakeSubAllocator>(),
                       /*total_memory=*/4096, /*name=*/"spatial_b", opts);

  void* lower_a = alloc_a.AllocateRaw(kAlignment, 256);
  void* upper_a = alloc_a.AllocateRaw(kAlignment, 512, *kUpper);
  void* lower_b = alloc_b.AllocateRaw(kAlignment, 256);
  void* extra_lower_b = alloc_b.AllocateRaw(kAlignment, 1024);
  void* upper_b = alloc_b.AllocateRaw(kAlignment, 512, *kUpper);

  ASSERT_NE(lower_a, nullptr);
  ASSERT_NE(upper_a, nullptr);
  ASSERT_NE(lower_b, nullptr);
  ASSERT_NE(extra_lower_b, nullptr);
  ASSERT_NE(upper_b, nullptr);

  const uintptr_t upper_offset_a =
      absl::bit_cast<uintptr_t>(upper_a) - absl::bit_cast<uintptr_t>(lower_a);
  const uintptr_t upper_offset_b =
      absl::bit_cast<uintptr_t>(upper_b) - absl::bit_cast<uintptr_t>(lower_b);
  EXPECT_EQ(upper_offset_a, upper_offset_b);

  alloc_a.DeallocateRaw(upper_a);
  alloc_a.DeallocateRaw(lower_a);
  alloc_b.DeallocateRaw(upper_b);
  alloc_b.DeallocateRaw(extra_lower_b);
  alloc_b.DeallocateRaw(lower_b);
}

// Upper must not reuse a non-boundary lower hole.
TEST(BFCAllocatorTest, SpatialSkipsLowerHole) {
  BFCAllocator::Options opts;
  opts.allow_growth = false;
  opts.enable_spatial_partitioning = true;
  BFCAllocator alloc(std::make_unique<FakeSubAllocator>(),
                     /*total_memory=*/1024, /*name=*/"spatial", opts);

  // Fill the whole region: two lower chunks then one upper chunk, leaving no
  // central gap.
  void* lower_a = alloc.AllocateRaw(kAlignment, 256);
  ASSERT_NE(lower_a, nullptr);
  void* lower_b = alloc.AllocateRaw(kAlignment, 256);
  ASSERT_NE(lower_b, nullptr);
  void* upper = alloc.AllocateRaw(kAlignment, 512, *kUpper);
  ASSERT_NE(upper, nullptr);

  // lower_a is trapped below live lower_b.
  alloc.DeallocateRaw(lower_a);

  // Upper must not reuse the trapped lower hole.
  void* trapped = alloc.AllocateRaw(kAlignment, 256, *kUpper);
  EXPECT_EQ(trapped, nullptr);

  alloc.DeallocateRaw(upper);
  alloc.DeallocateRaw(lower_b);
}

// Boundary frees rejoin the central gap.
TEST(BFCAllocatorTest, SpatialLowerReclaimsGap) {
  BFCAllocator::Options opts;
  opts.allow_growth = false;
  opts.enable_spatial_partitioning = true;
  BFCAllocator alloc(std::make_unique<FakeSubAllocator>(),
                     /*total_memory=*/2048, /*name=*/"spatial", opts);

  void* upper = alloc.AllocateRaw(kAlignment, 1024, *kUpper);
  ASSERT_NE(upper, nullptr);
  alloc.DeallocateRaw(upper);

  void* lower = alloc.AllocateRaw(kAlignment, 2048);
  ASSERT_NE(lower, nullptr);
  alloc.DeallocateRaw(lower);
}

// The dynamic boundary moves with frees, but ownership is still enforced.
TEST(BFCAllocatorTest, SpatialReclaimsGap) {
  BFCAllocator::Options opts;
  opts.allow_growth = false;
  opts.enable_spatial_partitioning = true;
  BFCAllocator alloc(std::make_unique<FakeSubAllocator>(),
                     /*total_memory=*/4096, /*name=*/"spatial", opts);

  void* lower0 = alloc.AllocateRaw(kAlignment, 1024);
  ASSERT_NE(lower0, nullptr);

  void* upper0 = alloc.AllocateRaw(kAlignment, 512, *kUpper);
  ASSERT_NE(upper0, nullptr);
  alloc.DeallocateRaw(upper0);

  // Adjacent upper free space rejoins the central gap.
  void* lower1 = alloc.AllocateRaw(kAlignment, 512);
  ASSERT_NE(lower1, nullptr);
  void* upper1 = alloc.AllocateRaw(kAlignment, 512, *kUpper);
  ASSERT_NE(upper1, nullptr);
  EXPECT_EQ(upper1, upper0);
  alloc.DeallocateRaw(upper1);

  // Lower claims the remaining central gap.
  void* lower2 = alloc.AllocateRaw(kAlignment, 2560);
  ASSERT_NE(lower2, nullptr);
  EXPECT_LE(absl::bit_cast<uintptr_t>(lower2),
            absl::bit_cast<uintptr_t>(upper1));
  EXPECT_EQ(absl::bit_cast<uintptr_t>(lower2) + 2560,
            absl::bit_cast<uintptr_t>(upper1) + 512);

  // Upper must not cross back into lower-owned space.
  void* upper2 = alloc.AllocateRaw(kAlignment, 256, *kUpper);
  EXPECT_EQ(upper2, nullptr);

  alloc.DeallocateRaw(lower2);
  alloc.DeallocateRaw(lower1);
  alloc.DeallocateRaw(lower0);
}

TEST(BFCAllocatorTest, SpatialUpperAlignmentSuffix) {
  BFCAllocator::Options opts;
  opts.allow_growth = false;
  opts.enable_spatial_partitioning = true;
  BFCAllocator alloc(std::make_unique<FakeSubAllocator>(),
                     /*total_memory=*/4096, /*name=*/"spatial", opts);
  const uintptr_t base = FakeSubAllocator::kBase;

  void* upper = alloc.AllocateRaw(1024, 256, *kUpper);
  ASSERT_NE(upper, nullptr);
  EXPECT_EQ(absl::bit_cast<uintptr_t>(upper), base + 3072);

  // The alignment suffix above `upper` is upper-owned.
  void* lower = alloc.AllocateRaw(kAlignment, 3072);
  ASSERT_NE(lower, nullptr);
  EXPECT_EQ(absl::bit_cast<uintptr_t>(lower), base);

  void* crossed = alloc.AllocateRaw(kAlignment, 768, *kLower);
  EXPECT_EQ(crossed, nullptr);

  alloc.DeallocateRaw(lower);
  alloc.DeallocateRaw(upper);
}

TEST(BFCAllocatorTest, SpatialLowerAlignmentPrefix) {
  BFCAllocator::Options opts;
  opts.allow_growth = false;
  opts.enable_spatial_partitioning = true;
  BFCAllocator alloc(std::make_unique<FakeSubAllocator>(),
                     /*total_memory=*/4096, /*name=*/"spatial", opts);
  const uintptr_t base = FakeSubAllocator::kBase;

  void* lower0 = alloc.AllocateRaw(kAlignment, 256);
  ASSERT_NE(lower0, nullptr);
  EXPECT_EQ(absl::bit_cast<uintptr_t>(lower0), base);

  void* lower1 = alloc.AllocateRaw(1024, 256);
  ASSERT_NE(lower1, nullptr);
  EXPECT_EQ(absl::bit_cast<uintptr_t>(lower1), base + 1024);

  void* upper = alloc.AllocateRaw(kAlignment, 2816, *kUpper);
  ASSERT_NE(upper, nullptr);
  EXPECT_EQ(absl::bit_cast<uintptr_t>(upper), base + 1280);

  // The alignment prefix below lower1 is lower-owned.
  void* crossed = alloc.AllocateRaw(kAlignment, 768, *kUpper);
  EXPECT_EQ(crossed, nullptr);

  alloc.DeallocateRaw(upper);
  alloc.DeallocateRaw(lower1);
  alloc.DeallocateRaw(lower0);
}

// A fully freed lower range reforms the central gap for upper allocations.
TEST(BFCAllocatorTest, SpatialUpperReclaimsAfterLowerFill) {
  BFCAllocator::Options opts;
  opts.allow_growth = false;
  opts.enable_spatial_partitioning = true;
  constexpr size_t kPool = size_t{1} << 30;  // 1 GiB.
  BFCAllocator alloc(std::make_unique<FakeSubAllocator>(),
                     /*total_memory=*/kPool, /*name=*/"repro", opts);
  const uintptr_t base = FakeSubAllocator::kBase;

  // Lower fills the entire pool, then frees it.
  constexpr size_t kChunk = size_t{32} << 20;  // 32 MiB.
  constexpr int kNumChunks = kPool / kChunk;   // 32 chunks exactly fill 1 GiB.
  std::vector<void*> lower_ptrs;
  lower_ptrs.reserve(kNumChunks);
  for (int i = 0; i < kNumChunks; ++i) {
    void* p = alloc.AllocateRaw(kAlignment, kChunk);
    ASSERT_NE(p, nullptr) << "lower fill failed at chunk " << i;
    lower_ptrs.push_back(p);
  }

  // Boundary coalescing should reform one whole-pool gap.
  for (void* p : lower_ptrs) {
    alloc.DeallocateRaw(p);
  }

  // Upper should now allocate from the top of the reformed gap.
  constexpr size_t kUpperBytes = 18 << 20;
  void* upper = alloc.AllocateRaw(kAlignment, kUpperBytes, *kUpper);
  ASSERT_NE(upper, nullptr) << "upper should reclaim the freed pool";
  EXPECT_EQ(absl::bit_cast<uintptr_t>(upper) + kUpperBytes, base + kPool)
      << "upper allocation should be anchored at the top of the pool";
  alloc.DeallocateRaw(upper);
}

TEST(BFCAllocatorTest, SpatialReusesOwnHoles) {
  BFCAllocator::Options opts;
  opts.allow_growth = false;
  opts.enable_spatial_partitioning = true;
  BFCAllocator alloc(std::make_unique<FakeSubAllocator>(),
                     /*total_memory=*/4096, /*name=*/"spatial", opts);
  const uintptr_t base = FakeSubAllocator::kBase;

  void* lower0 = alloc.AllocateRaw(kAlignment, 256);
  void* lower_hole = alloc.AllocateRaw(kAlignment, 256);
  void* lower_guard = alloc.AllocateRaw(kAlignment, 256);
  ASSERT_NE(lower0, nullptr);
  ASSERT_NE(lower_hole, nullptr);
  ASSERT_NE(lower_guard, nullptr);
  EXPECT_EQ(absl::bit_cast<uintptr_t>(lower_hole), base + 256);

  void* upper0 = alloc.AllocateRaw(kAlignment, 256, *kUpper);
  void* upper_hole = alloc.AllocateRaw(kAlignment, 256, *kUpper);
  void* upper_guard = alloc.AllocateRaw(kAlignment, 256, *kUpper);
  ASSERT_NE(upper0, nullptr);
  ASSERT_NE(upper_hole, nullptr);
  ASSERT_NE(upper_guard, nullptr);
  EXPECT_EQ(absl::bit_cast<uintptr_t>(upper_hole), base + 3584);

  alloc.DeallocateRaw(lower_hole);
  alloc.DeallocateRaw(upper_hole);

  // Own binned holes are reused before the central gap.
  void* lower_reuse = alloc.AllocateRaw(kAlignment, 256);
  ASSERT_NE(lower_reuse, nullptr);
  EXPECT_EQ(lower_reuse, lower_hole);

  void* upper_reuse = alloc.AllocateRaw(kAlignment, 256, *kUpper);
  ASSERT_NE(upper_reuse, nullptr);
  EXPECT_EQ(upper_reuse, upper_hole);

  alloc.DeallocateRaw(upper_reuse);
  alloc.DeallocateRaw(upper_guard);
  alloc.DeallocateRaw(upper0);
  alloc.DeallocateRaw(lower_reuse);
  alloc.DeallocateRaw(lower_guard);
  alloc.DeallocateRaw(lower0);
}

// Identical upper allocation sequences should produce identical offsets.
TEST(BFCAllocatorTest, SpatialUpperOffsetsStable) {
  constexpr size_t kPool = size_t{512} << 20;
  constexpr size_t kUpperAlignment = 512;
  // Fixed upper sizes, identical across simulated ranks.
  const std::array<size_t, 8> kUpperSizes = {4 << 20,  16 << 20, 1 << 20,
                                             18 << 20, 2 << 20,  8 << 20,
                                             4 << 20,  32 << 20};

  // Run the fixed upper sequence with randomized lower churn.
  auto run = [&](uint32_t lower_seed) -> std::vector<uintptr_t> {
    BFCAllocator::Options opts;
    opts.allow_growth = false;
    opts.enable_spatial_partitioning = true;
    BFCAllocator alloc(std::make_unique<FakeSubAllocator>(), kPool, "sym",
                       opts);
    const uintptr_t base = FakeSubAllocator::kBase;

    std::mt19937 rng(lower_seed);
    std::vector<std::pair<void*, size_t>> live_lower;  // (ptr, bytes)
    size_t live_lower_bytes = 0;
    // Keep utilization away from true exhaustion.
    constexpr size_t kLowerCap = kPool / 2;
    const std::array<size_t, 5> kLowerSizes = {256, 1 << 20, 8 << 20, 32 << 20,
                                               64 << 20};
    auto churn_lower = [&] {
      // A random burst of lower allocations and frees, leaving some live.
      const int ops = std::uniform_int_distribution<int>(0, 6)(rng);
      for (int i = 0; i < ops; ++i) {
        if (!live_lower.empty() &&
            std::uniform_int_distribution<int>(0, 2)(rng) == 0) {
          size_t idx = std::uniform_int_distribution<size_t>(
              0, live_lower.size() - 1)(rng);
          alloc.DeallocateRaw(live_lower[idx].first);
          live_lower_bytes -= live_lower[idx].second;
          live_lower.erase(live_lower.begin() + idx);
        } else {
          size_t bytes = kLowerSizes[std::uniform_int_distribution<size_t>(
              0, kLowerSizes.size() - 1)(rng)];
          if (live_lower_bytes + bytes > kLowerCap) {
            continue;
          }
          void* p = alloc.AllocateRaw(kUpperAlignment, bytes);
          if (p) {
            live_lower.push_back({p, bytes});
            live_lower_bytes += bytes;
          }
        }
      }
    };

    std::vector<uintptr_t> offsets;
    std::vector<void*> live_upper;
    for (size_t bytes : kUpperSizes) {
      churn_lower();
      void* p = alloc.AllocateRaw(kUpperAlignment, bytes, *kUpper);
      EXPECT_NE(p, nullptr)
          << "upper alloc failed under lower churn (seed " << lower_seed << ")";
      offsets.push_back(p ? absl::bit_cast<uintptr_t>(p) - base
                          : std::numeric_limits<uintptr_t>::max());
      if (p) {
        live_upper.push_back(p);
      }
      // Occasionally free an earlier upper temp, mimicking short-lived S(1).
      if (live_upper.size() > 2) {
        alloc.DeallocateRaw(live_upper.front());
        live_upper.erase(live_upper.begin());
      }
    }
    return offsets;
  };

  const std::vector<uintptr_t> rank0 = run(/*lower_seed=*/1);
  for (uint32_t seed = 2; seed <= 32; ++seed) {
    EXPECT_EQ(run(seed), rank0)
        << "upper offsets diverged for lower_seed=" << seed;
  }
}

TEST(BFCAllocatorTest, SpatialUnderContention) {
  BFCAllocator::Options opts;
  opts.allow_growth = false;
  opts.enable_spatial_partitioning = true;
  BFCAllocator alloc(std::make_unique<FakeSubAllocator>(),
                     /*total_memory=*/64 << 20, /*name=*/"contention", opts);

  constexpr int kNumThreads = 8;
  constexpr int kItersPerThread = 1000;
  constexpr size_t kBytes = 1024;

  std::atomic<int> failures{0};
  tsl::thread::ThreadPool threads(tsl::Env::Default(), "spatial_contention",
                                  kNumThreads);
  absl::BlockingCounter counter(kNumThreads);
  for (int t = 0; t < kNumThreads; ++t) {
    threads.Schedule([&] {
      for (int i = 0; i < kItersPerThread; ++i) {
        void* lower = alloc.AllocateRaw(kAlignment, kBytes, *kLower);
        void* upper = alloc.AllocateRaw(kAlignment, kBytes, *kUpper);
        if (!lower || !upper || !IsAligned(lower, kAlignment) ||
            !IsAligned(upper, kAlignment)) {
          failures.fetch_add(1, std::memory_order_relaxed);
        }
        alloc.DeallocateRaw(lower);
        alloc.DeallocateRaw(upper);
      }
      counter.DecrementCount();
    });
  }
  counter.Wait();
  EXPECT_EQ(failures.load(std::memory_order_relaxed), 0);
}

//===----------------------------------------------------------------------===//
// Performance benchmarks.
//===----------------------------------------------------------------------===//

static constexpr size_t kBenchAllocSize = 1024;

static void BM_AllocAndFree(benchmark::State& state) {
  BFCAllocator alloc(std::make_unique<FakeSubAllocator>(),
                     /*total_memory=*/256 << 20, /*name=*/"bench",
                     BFCAllocator::Options{});

  for (auto _ : state) {
    void* ptr = alloc.AllocateRaw(kAlignment, kBenchAllocSize);
    alloc.DeallocateRaw(ptr);
  }
  state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BM_AllocAndFree);

static void BM_AllocBatchThenFree(benchmark::State& state) {
  int batch = state.range(0);
  BFCAllocator alloc(std::make_unique<FakeSubAllocator>(),
                     /*total_memory=*/256 << 20, /*name=*/"bench",
                     BFCAllocator::Options{});

  std::vector<void*> ptrs(batch);
  for (auto _ : state) {
    for (int i = 0; i < batch; ++i) {
      ptrs[i] = alloc.AllocateRaw(kAlignment, kBenchAllocSize);
    }
    for (int i = 0; i < batch; ++i) {
      alloc.DeallocateRaw(ptrs[i]);
    }
  }
  state.SetItemsProcessed(state.iterations() * batch);
}

BENCHMARK(BM_AllocBatchThenFree)->Arg(100)->Arg(1000);

//===----------------------------------------------------------------------===//
// Spatial allocation benchmarks.
//===----------------------------------------------------------------------===//

static void BM_SpatialAllocBatchThenFree(benchmark::State& state) {
  const int batch = state.range(0);
  BFCAllocator::Options opts;
  opts.allow_growth = false;
  opts.enable_spatial_partitioning = true;
  BFCAllocator alloc(std::make_unique<FakeSubAllocator>(),
                     /*total_memory=*/256 << 20, /*name=*/"bench", opts);

  std::vector<void*> lower_ptrs(batch);
  std::vector<void*> upper_ptrs(batch);
  for (auto _ : state) {
    for (int i = 0; i < batch; ++i) {
      lower_ptrs[i] = alloc.AllocateRaw(kAlignment, kBenchAllocSize);
      tsl::testing::DoNotOptimize(lower_ptrs[i]);
    }
    for (int i = 0; i < batch; ++i) {
      upper_ptrs[i] = alloc.AllocateRaw(kAlignment, kBenchAllocSize, *kUpper);
      tsl::testing::DoNotOptimize(upper_ptrs[i]);
    }
    for (int i = 0; i < batch; ++i) {
      alloc.DeallocateRaw(lower_ptrs[i]);
      alloc.DeallocateRaw(upper_ptrs[i]);
    }
  }
  state.SetItemsProcessed(state.iterations() * batch * 2);
}

BENCHMARK(BM_SpatialAllocBatchThenFree)->Arg(100)->Arg(1000);

//===----------------------------------------------------------------------===//
// Contention benchmarks.
//===----------------------------------------------------------------------===//

static void BM_AllocAndFreeUnderContention(benchmark::State& state) {
  size_t num_threads = state.range(0);
  static constexpr int kItersPerThread = 10000;

  BFCAllocator alloc(std::make_unique<FakeSubAllocator>(),
                     /*total_memory=*/256 << 20, /*name=*/"bench",
                     BFCAllocator::Options{});
  tsl::thread::ThreadPool threads(tsl::Env::Default(), "bench", num_threads);

  for (auto _ : state) {
    absl::BlockingCounter counter(num_threads);
    for (int t = 0; t < num_threads; ++t) {
      threads.Schedule([&] {
        for (int i = 0; i < kItersPerThread; ++i) {
          void* ptr = alloc.AllocateRaw(kAlignment, kBenchAllocSize);
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
