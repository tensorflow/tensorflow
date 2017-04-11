/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#if GOOGLE_CUDA

#include "tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.h"

#include <algorithm>
#include <vector>

#include "tensorflow/core/common_runtime/gpu/gpu_init.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/platform/types.h"

namespace gpu = ::perftools::gputools;

namespace tensorflow {
namespace {

static void CheckStats(Allocator* a, int64 num_allocs, int64 bytes_in_use,
                       int64 max_bytes_in_use, int64 max_alloc_size) {
  AllocatorStats stats;
  a->GetStats(&stats);
  LOG(INFO) << "Alloc stats: " << std::endl << stats.DebugString();
  EXPECT_EQ(stats.bytes_in_use, bytes_in_use);
  EXPECT_EQ(stats.max_bytes_in_use, max_bytes_in_use);
  EXPECT_EQ(stats.num_allocs, num_allocs);
  EXPECT_EQ(stats.max_alloc_size, max_alloc_size);
}

TEST(GPUBFCAllocatorTest, NoDups) {
  GPUBFCAllocator a(0, 1 << 30);
  CheckStats(&a, 0, 0, 0, 0);

  // Allocate a lot of raw pointers
  std::vector<void*> ptrs;
  for (int s = 1; s < 1024; s++) {
    void* raw = a.AllocateRaw(1, s);
    ptrs.push_back(raw);
  }
  CheckStats(&a, 1023, 654336, 654336, 1024);

  std::sort(ptrs.begin(), ptrs.end());

  // Make sure none of them are equal, and that none of them overlap.
  for (size_t i = 1; i < ptrs.size(); i++) {
    ASSERT_NE(ptrs[i], ptrs[i - 1]);  // No dups
    size_t req_size = a.RequestedSize(ptrs[i - 1]);
    ASSERT_GT(req_size, 0);
    ASSERT_GE(static_cast<char*>(ptrs[i]) - static_cast<char*>(ptrs[i - 1]),
              req_size);
  }

  for (size_t i = 0; i < ptrs.size(); i++) {
    a.DeallocateRaw(ptrs[i]);
  }
  CheckStats(&a, 1023, 0, 654336, 1024);
}

TEST(GPUBFCAllocatorTest, AllocationsAndDeallocations) {
  GPUBFCAllocator a(0, 1 << 30);
  // Allocate 256 raw pointers of sizes between 100 bytes and about
  // a meg
  random::PhiloxRandom philox(123, 17);
  random::SimplePhilox rand(&philox);

  std::vector<void*> initial_ptrs;
  for (int s = 1; s < 256; s++) {
    size_t size = std::min<size_t>(
        std::max<size_t>(rand.Rand32() % 1048576, 100), 1048576);
    void* raw = a.AllocateRaw(1, size);

    initial_ptrs.push_back(raw);
  }

  // Deallocate half of the memory, and keep track of the others.
  std::vector<void*> existing_ptrs;
  for (size_t i = 0; i < initial_ptrs.size(); i++) {
    if (i % 2 == 1) {
      a.DeallocateRaw(initial_ptrs[i]);
    } else {
      existing_ptrs.push_back(initial_ptrs[i]);
    }
  }

  // Allocate a lot of raw pointers
  for (int s = 1; s < 256; s++) {
    size_t size = std::min<size_t>(
        std::max<size_t>(rand.Rand32() % 1048576, 100), 1048576);
    void* raw = a.AllocateRaw(1, size);
    existing_ptrs.push_back(raw);
  }

  std::sort(existing_ptrs.begin(), existing_ptrs.end());
  // Make sure none of them are equal
  for (size_t i = 1; i < existing_ptrs.size(); i++) {
    CHECK_NE(existing_ptrs[i], existing_ptrs[i - 1]);  // No dups

    size_t req_size = a.RequestedSize(existing_ptrs[i - 1]);
    ASSERT_GT(req_size, 0);

    // Check that they don't overlap.
    ASSERT_GE(static_cast<char*>(existing_ptrs[i]) -
                  static_cast<char*>(existing_ptrs[i - 1]),
              req_size);
  }

  for (size_t i = 0; i < existing_ptrs.size(); i++) {
    a.DeallocateRaw(existing_ptrs[i]);
  }
}

TEST(GPUBFCAllocatorTest, ExerciseCoalescing) {
  GPUBFCAllocator a(0, 1 << 30);
  CheckStats(&a, 0, 0, 0, 0);

  float* first_ptr = a.Allocate<float>(1024);
  a.DeallocateRaw(first_ptr);
  CheckStats(&a, 1, 0, 4096, 4096);
  for (int i = 0; i < 1024; ++i) {
    // Allocate several buffers of different sizes, and then clean them
    // all up.  We should be able to repeat this endlessly without
    // causing fragmentation and growth.
    float* t1 = a.Allocate<float>(1024);

    int64* t2 = a.Allocate<int64>(1048576);
    double* t3 = a.Allocate<double>(2048);
    float* t4 = a.Allocate<float>(10485760);

    a.DeallocateRaw(t1);
    a.DeallocateRaw(t2);
    a.DeallocateRaw(t3);
    a.DeallocateRaw(t4);
  }
  CheckStats(&a, 4097, 0, 1024 * sizeof(float) + 1048576 * sizeof(int64) +
                              2048 * sizeof(double) + 10485760 * sizeof(float),
             10485760 * sizeof(float));

  // At the end, we should have coalesced all memory into one region
  // starting at the beginning, so validate that allocating a pointer
  // starts from this region.
  float* first_ptr_after = a.Allocate<float>(1024);
  EXPECT_EQ(first_ptr, first_ptr_after);
  a.DeallocateRaw(first_ptr_after);
}

TEST(GPUBFCAllocatorTest, AllocateZeroBufSize) {
  GPUBFCAllocator a(0, 1 << 30);
  float* ptr = a.Allocate<float>(0);
  EXPECT_EQ(nullptr, ptr);
}

TEST(GPUBFCAllocatorTest, TracksSizes) {
  GPUBFCAllocator a(0, 1 << 30);
  EXPECT_EQ(true, a.TracksAllocationSizes());
}

TEST(GPUBFCAllocatorTest, AllocatedVsRequested) {
  GPUBFCAllocator a(0, 1 << 30);
  float* t1 = a.Allocate<float>(1);
  EXPECT_EQ(4, a.RequestedSize(t1));
  EXPECT_EQ(256, a.AllocatedSize(t1));
  a.DeallocateRaw(t1);
}

TEST(GPUBFCAllocatorTest, TestCustomMemoryLimit) {
  // Configure a 1MiB byte limit
  GPUBFCAllocator a(0, 1 << 20);

  float* first_ptr = a.Allocate<float>(1 << 6);
  float* second_ptr = a.Allocate<float>(1 << 20);

  EXPECT_NE(nullptr, first_ptr);
  EXPECT_EQ(nullptr, second_ptr);
  a.DeallocateRaw(first_ptr);
}

TEST(GPUBFCAllocatorTest, AllocationsAndDeallocationsWithGrowth) {
  GPUOptions options;
  options.set_allow_growth(true);

  // Max of 2GiB, but starts out small.
  GPUBFCAllocator a(0, 1LL << 31, options);

  // Allocate 10 raw pointers of sizes between 100 bytes and about
  // 64 megs.
  random::PhiloxRandom philox(123, 17);
  random::SimplePhilox rand(&philox);

  const int32 max_mem = 1 << 27;

  std::vector<void*> initial_ptrs;
  for (int s = 1; s < 10; s++) {
    size_t size = std::min<size_t>(
        std::max<size_t>(rand.Rand32() % max_mem, 100), max_mem);
    void* raw = a.AllocateRaw(1, size);

    initial_ptrs.push_back(raw);
  }

  // Deallocate half of the memory, and keep track of the others.
  std::vector<void*> existing_ptrs;
  for (size_t i = 0; i < initial_ptrs.size(); i++) {
    if (i % 2 == 1) {
      a.DeallocateRaw(initial_ptrs[i]);
    } else {
      existing_ptrs.push_back(initial_ptrs[i]);
    }
  }

  const int32 max_mem_2 = 1 << 26;
  // Allocate a lot of raw pointers between 100 bytes and 64 megs.
  for (int s = 1; s < 10; s++) {
    size_t size = std::min<size_t>(
        std::max<size_t>(rand.Rand32() % max_mem_2, 100), max_mem_2);
    void* raw = a.AllocateRaw(1, size);
    existing_ptrs.push_back(raw);
  }

  std::sort(existing_ptrs.begin(), existing_ptrs.end());
  // Make sure none of them are equal
  for (size_t i = 1; i < existing_ptrs.size(); i++) {
    CHECK_NE(existing_ptrs[i], existing_ptrs[i - 1]);  // No dups

    size_t req_size = a.RequestedSize(existing_ptrs[i - 1]);
    ASSERT_GT(req_size, 0);

    // Check that they don't overlap.
    ASSERT_GE(static_cast<char*>(existing_ptrs[i]) -
                  static_cast<char*>(existing_ptrs[i - 1]),
              req_size);
  }

  for (size_t i = 0; i < existing_ptrs.size(); i++) {
    a.DeallocateRaw(existing_ptrs[i]);
  }

  AllocatorStats stats;
  a.GetStats(&stats);
  LOG(INFO) << "Alloc stats: \n" << stats.DebugString();
}

TEST(GPUBFCAllocatorTest, DISABLED_AllocatorReceivesZeroMemory) {
  GPUBFCAllocator a(0, 1UL << 60);
  GPUBFCAllocator b(0, 1UL << 60);
  void* amem = a.AllocateRaw(1, 1);
  void* bmem = b.AllocateRaw(1, 1 << 30);
  a.DeallocateRaw(amem);
  b.DeallocateRaw(bmem);
}

static void BM_Allocation(int iters) {
  GPUBFCAllocator a(0, 1uLL << 33);
  // Exercise a few different allocation sizes
  std::vector<size_t> sizes = {256,        4096,      16384,    524288,
                               512,        1048576,   10485760, 104857600,
                               1048576000, 2048576000};
  int size_index = 0;

  while (--iters > 0) {
    size_t bytes = sizes[size_index++ % sizes.size()];
    void* p = a.AllocateRaw(1, bytes);
    a.DeallocateRaw(p);
  }
}
BENCHMARK(BM_Allocation);

static void BM_AllocationThreaded(int iters, int num_threads) {
  GPUBFCAllocator a(0, 1uLL << 33);
  thread::ThreadPool pool(Env::Default(), "test", num_threads);
  std::atomic_int_fast32_t count(iters);
  mutex done_lock;
  condition_variable done;
  bool done_flag = false;

  for (int t = 0; t < num_threads; t++) {
    pool.Schedule([&a, &count, &done_lock, &done, &done_flag, iters]() {
      // Exercise a few different allocation sizes
      std::vector<int> sizes = {256, 4096,    16384,    524288,
                                512, 1048576, 10485760, 104857600};
      int size_index = 0;
      for (int i = 0; i < iters; i++) {
        int bytes = sizes[size_index++ % sizes.size()];
        void* p = a.AllocateRaw(1, bytes);
        a.DeallocateRaw(p);
        if (count.fetch_sub(1) == 1) {
          mutex_lock l(done_lock);
          done_flag = true;
          done.notify_all();
          break;
        }
      }
    });
  }
  mutex_lock l(done_lock);
  if (!done_flag) {
    done.wait(l);
  }
}
BENCHMARK(BM_AllocationThreaded)->Arg(1)->Arg(4)->Arg(16);

// A more complex benchmark that defers deallocation of an object for
// "delay" allocations.
static void BM_AllocationDelayed(int iters, int delay) {
  GPUBFCAllocator a(0, 1 << 30);
  // Exercise a few different allocation sizes
  std::vector<int> sizes = {256, 4096, 16384, 4096, 512, 1024, 1024};
  int size_index = 0;

  std::vector<void*> ptrs;
  for (int i = 0; i < delay; i++) {
    ptrs.push_back(nullptr);
  }
  int pindex = 0;
  while (--iters > 0) {
    if (ptrs[pindex] != nullptr) {
      a.DeallocateRaw(ptrs[pindex]);
      ptrs[pindex] = nullptr;
    }
    int bytes = sizes[size_index++ % sizes.size()];
    void* p = a.AllocateRaw(1, bytes);
    ptrs[pindex] = p;
    pindex = (pindex + 1) % ptrs.size();
  }
  for (int i = 0; i < ptrs.size(); i++) {
    if (ptrs[i] != nullptr) {
      a.DeallocateRaw(ptrs[i]);
    }
  }
}
BENCHMARK(BM_AllocationDelayed)->Arg(1)->Arg(10)->Arg(100)->Arg(1000);

}  // namespace
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
