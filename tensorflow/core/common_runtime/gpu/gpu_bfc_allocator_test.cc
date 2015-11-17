#if GOOGLE_CUDA

#include "tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.h"

#include <algorithm>
#include <vector>

#include "tensorflow/stream_executor/stream_executor.h"
#include <gtest/gtest.h>
#include "tensorflow/core/common_runtime/gpu/gpu_init.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/port.h"

namespace gpu = ::perftools::gputools;

namespace tensorflow {
namespace {

TEST(GPUBFCAllocatorTest, NoDups) {
  GPUBFCAllocator a(0, 1 << 30);
  // Allocate a lot of raw pointers
  std::vector<void*> ptrs;
  for (int s = 1; s < 1024; s++) {
    void* raw = a.AllocateRaw(1, s);
    ptrs.push_back(raw);
  }

  std::sort(ptrs.begin(), ptrs.end());

  // Make sure none of them are equal, and that none of them overlap.
  for (size_t i = 0; i < ptrs.size(); i++) {
    if (i > 0) {
      ASSERT_NE(ptrs[i], ptrs[i - 1]);  // No dups
      size_t req_size = a.RequestedSize(ptrs[i - 1]);
      ASSERT_GT(req_size, 0);
      ASSERT_GE(static_cast<char*>(ptrs[i]) - static_cast<char*>(ptrs[i - 1]),
                req_size);
    }
  }

  for (size_t i = 0; i < ptrs.size(); i++) {
    a.DeallocateRaw(ptrs[i]);
  }
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
  for (size_t i = 0; i < existing_ptrs.size(); i++) {
    if (i > 0) {
      CHECK_NE(existing_ptrs[i], existing_ptrs[i - 1]);  // No dups

      size_t req_size = a.RequestedSize(existing_ptrs[i - 1]);
      ASSERT_GT(req_size, 0);

      // Check that they don't overlap.
      ASSERT_GE(static_cast<char*>(existing_ptrs[i]) -
                    static_cast<char*>(existing_ptrs[i - 1]),
                req_size);
    }
  }

  for (size_t i = 0; i < existing_ptrs.size(); i++) {
    a.DeallocateRaw(existing_ptrs[i]);
  }
}

TEST(GPUBFCAllocatorTest, ExerciseCoalescing) {
  GPUBFCAllocator a(0, 1 << 30);

  float* first_ptr = a.Allocate<float>(1024);
  a.Deallocate(first_ptr);
  for (int i = 0; i < 1024; ++i) {
    // Allocate several buffers of different sizes, and then clean them
    // all up.  We should be able to repeat this endlessly without
    // causing fragmentation and growth.
    float* t1 = a.Allocate<float>(1024);

    int64* t2 = a.Allocate<int64>(1048576);
    double* t3 = a.Allocate<double>(2048);
    float* t4 = a.Allocate<float>(10485760);

    a.Deallocate(t1);
    a.Deallocate(t2);
    a.Deallocate(t3);
    a.Deallocate(t4);
  }

  // At the end, we should have coalesced all memory into one region
  // starting at the beginning, so validate that allocating a pointer
  // starts from this region.
  float* first_ptr_after = a.Allocate<float>(1024);
  EXPECT_EQ(first_ptr, first_ptr_after);
  a.Deallocate(first_ptr_after);
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
  a.Deallocate(t1);
}

TEST(GPUBFCAllocatorTest, TestCustomMemoryLimit) {
  // Configure a 1MiB byte limit
  GPUBFCAllocator a(0, 1 << 20);

  float* first_ptr = a.Allocate<float>(1 << 6);
  float* second_ptr = a.Allocate<float>(1 << 20);

  EXPECT_NE(nullptr, first_ptr);
  EXPECT_EQ(nullptr, second_ptr);
  a.Deallocate(first_ptr);
}

}  // namespace
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
