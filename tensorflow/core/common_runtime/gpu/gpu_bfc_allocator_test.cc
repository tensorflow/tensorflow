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

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)

#include "tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.h"

#include <algorithm>
#include <vector>

#include "tensorflow/core/common_runtime/device/device_id.h"
#include "tensorflow/core/common_runtime/device/device_id_utils.h"
#include "tensorflow/core/common_runtime/device/device_mem_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_init.h"
#include "tensorflow/core/common_runtime/gpu/gpu_virtual_mem_allocator.h"
#include "tensorflow/core/framework/typed_allocator.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/bfc_memory_map.pb.h"
#include "tensorflow/stream_executor/gpu/gpu_driver.h"

namespace tensorflow {
namespace {

void CheckStats(Allocator* a, int64 num_allocs, int64 bytes_in_use,
                int64 peak_bytes_in_use, int64 largest_alloc_size) {
  absl::optional<AllocatorStats> stats = a->GetStats();
  EXPECT_TRUE(stats);
  if (!stats) {
    return;
  }
  LOG(INFO) << "Alloc stats: " << std::endl << stats->DebugString();
  EXPECT_EQ(stats->bytes_in_use, bytes_in_use);
  EXPECT_EQ(stats->peak_bytes_in_use, peak_bytes_in_use);
  EXPECT_EQ(stats->num_allocs, num_allocs);
  EXPECT_EQ(stats->largest_alloc_size, largest_alloc_size);
}

class GPUBFCAllocatorTest
    : public ::testing::TestWithParam<SubAllocator* (*)(size_t)> {};

#if CUDA_VERSION >= 10020
SubAllocator* CreateVirtualMemorySubAllocator(
    size_t virtual_address_space_size = 1ull << 32) {
  PlatformDeviceId gpu_id(0);
  auto executor =
      DeviceIdUtil::ExecutorForPlatformDeviceId(GPUMachineManager(), gpu_id)
          .ValueOrDie();
  auto* gpu_context = reinterpret_cast<stream_executor::gpu::GpuContext*>(
      executor->implementation()->GpuContextHack());
  return GpuVirtualMemAllocator::Create({}, {}, *gpu_context, gpu_id,
                                        virtual_address_space_size, {})
      .ValueOrDie()
      .release();
}
#endif

SubAllocator* CreateGPUMemAllocator(size_t) {
  PlatformDeviceId gpu_id(0);
  return new DeviceMemAllocator(
      DeviceIdUtil::ExecutorForPlatformDeviceId(GPUMachineManager(), gpu_id)
          .ValueOrDie(),
      gpu_id,
      /*use_unified_memory=*/false, {}, {});
}

SubAllocator* CreateSubAllocator(size_t virtual_address_space_size = 1ull
                                                                     << 32) {
#if CUDA_VERSION >= 10020
  return CreateVirtualMemorySubAllocator(virtual_address_space_size);
#else
  return CreateGPUMemAllocator(virtual_address_space_size);
#endif
}

auto TestSuiteValues() {
#if CUDA_VERSION >= 10020
  return ::testing::Values(&CreateGPUMemAllocator,
                           &CreateVirtualMemorySubAllocator);
#else
  return ::testing::Values(&CreateGPUMemAllocator);
#endif
}

TEST_P(GPUBFCAllocatorTest, NoDups) {
  GPUBFCAllocator a(GetParam()(1ull << 32), 1 << 30, "GPU_0_bfc");
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

TEST_P(GPUBFCAllocatorTest, AllocationsAndDeallocations) {
  GPUBFCAllocator a(GetParam()(1ull << 32), 1 << 30, "GPU_0_bfc");
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

  // Ensure out of memory errors work and do not prevent future allocations from
  // working.
  void* out_of_memory_ptr = a.AllocateRaw(1, (1 << 30) + 1);
  CHECK_EQ(out_of_memory_ptr, nullptr);

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

TEST_P(GPUBFCAllocatorTest, ExerciseCoalescing) {
  GPUBFCAllocator a(GetParam()(1ull << 32), 1 << 30, "GPU_0_bfc");
  CheckStats(&a, 0, 0, 0, 0);

  float* first_ptr = TypedAllocator::Allocate<float>(&a, 1024, {});
  a.DeallocateRaw(first_ptr);
  CheckStats(&a, 1, 0, 4096, 4096);
  for (int i = 0; i < 1024; ++i) {
    // Allocate several buffers of different sizes, and then clean them
    // all up.  We should be able to repeat this endlessly without
    // causing fragmentation and growth.
    float* t1 = TypedAllocator::Allocate<float>(&a, 1024, {});

    int64* t2 = TypedAllocator::Allocate<int64>(&a, 1048576, {});
    double* t3 = TypedAllocator::Allocate<double>(&a, 2048, {});
    float* t4 = TypedAllocator::Allocate<float>(&a, 10485760, {});

    a.DeallocateRaw(t1);
    a.DeallocateRaw(t2);
    a.DeallocateRaw(t3);
    a.DeallocateRaw(t4);
  }
  CheckStats(&a, 4097, 0,
             1024 * sizeof(float) + 1048576 * sizeof(int64) +
                 2048 * sizeof(double) + 10485760 * sizeof(float),
             10485760 * sizeof(float));

  // At the end, we should have coalesced all memory into one region
  // starting at the beginning, so validate that allocating a pointer
  // starts from this region.
  float* first_ptr_after = TypedAllocator::Allocate<float>(&a, 1024, {});
  EXPECT_EQ(first_ptr, first_ptr_after);
  a.DeallocateRaw(first_ptr_after);
}

TEST_P(GPUBFCAllocatorTest, AllocateZeroBufSize) {
  GPUBFCAllocator a(GetParam()(1ull << 32), 1 << 30, "GPU_0_bfc");
  float* ptr = TypedAllocator::Allocate<float>(&a, 0, {});
  EXPECT_EQ(nullptr, ptr);
}

TEST_P(GPUBFCAllocatorTest, TracksSizes) {
  GPUBFCAllocator a(GetParam()(1ull << 32), 1 << 30, "GPU_0_bfc");
  EXPECT_EQ(true, a.TracksAllocationSizes());
}

TEST_P(GPUBFCAllocatorTest, AllocatedVsRequested) {
  GPUBFCAllocator a(GetParam()(1ull << 32), 1 << 30, "GPU_0_bfc");
  float* t1 = TypedAllocator::Allocate<float>(&a, 1, {});
  EXPECT_EQ(4, a.RequestedSize(t1));
  EXPECT_EQ(256, a.AllocatedSize(t1));
  a.DeallocateRaw(t1);
}

TEST_P(GPUBFCAllocatorTest, TestCustomMemoryLimit) {
  // Configure a 2MiB byte limit
  GPUBFCAllocator a(GetParam()(1ull << 32), 2 << 20, "GPU_0_bfc");

  float* first_ptr = TypedAllocator::Allocate<float>(&a, 1 << 6, {});
  float* second_ptr = TypedAllocator::Allocate<float>(&a, 2 << 20, {});

  EXPECT_NE(nullptr, first_ptr);
  EXPECT_EQ(nullptr, second_ptr);
  a.DeallocateRaw(first_ptr);
}

TEST_P(GPUBFCAllocatorTest, AllocationsAndDeallocationsWithGrowth) {
  GPUOptions options;
  options.set_allow_growth(true);

  // Max of 2GiB, but starts out small.
  GPUBFCAllocator a(GetParam()(1ull << 32), 1LL << 31, "GPU_0_bfc");

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

  absl::optional<AllocatorStats> stats = a.GetStats();
  if (stats) {
    LOG(INFO) << "Alloc stats: \n" << stats->DebugString();
  }
}

TEST_P(GPUBFCAllocatorTest, DISABLED_AllocatorReceivesZeroMemory) {
  GPUBFCAllocator a(GetParam()(1ul << 62), 1UL << 60, "GPU_0_bfc");
  GPUBFCAllocator b(GetParam()(1ul << 62), 1UL << 60, "GPU_0_bfc");
  void* amem = a.AllocateRaw(1, 1);
  void* bmem = b.AllocateRaw(1, 1 << 30);
  a.DeallocateRaw(amem);
  b.DeallocateRaw(bmem);
}

INSTANTIATE_TEST_SUITE_P(GPUBFCAllocatorTestSuite, GPUBFCAllocatorTest,
                         TestSuiteValues());

static void BM_Allocation(::testing::benchmark::State& state) {
  GPUBFCAllocator a(CreateSubAllocator(1ul << 36), 1uLL << 33, "GPU_0_bfc");
  // Exercise a few different allocation sizes
  std::vector<size_t> sizes = {256,        4096,      16384,    524288,
                               512,        1048576,   10485760, 104857600,
                               1048576000, 2048576000};
  int size_index = 0;

  for (auto s : state) {
    size_t bytes = sizes[size_index++ % sizes.size()];
    void* p = a.AllocateRaw(1, bytes);
    a.DeallocateRaw(p);
  }
}
BENCHMARK(BM_Allocation);

static void BM_AllocationThreaded(::testing::benchmark::State& state) {
  int num_threads = state.range(0);
  int sub_iters = 500;  // Pick a reasonably large number.

  for (auto s : state) {
    state.PauseTiming();
    GPUBFCAllocator a(CreateSubAllocator(1ul << 36), 1uLL << 33, "GPU_0_bfc");
    thread::ThreadPool pool(Env::Default(), "test", num_threads);

    std::atomic_int_fast32_t count(sub_iters);
    mutex done_lock;
    condition_variable done;
    bool done_flag = false;
    state.ResumeTiming();
    for (int t = 0; t < num_threads; t++) {
      pool.Schedule([&a, &count, &done_lock, &done, &done_flag, sub_iters]() {
        // Exercise a few different allocation sizes
        std::vector<int> sizes = {256, 4096,    16384,    524288,
                                  512, 1048576, 10485760, 104857600};
        int size_index = 0;
        for (int i = 0; i < sub_iters; i++) {
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
}

BENCHMARK(BM_AllocationThreaded)->Arg(1)->Arg(4)->Arg(16);

// A more complex benchmark that defers deallocation of an object for
// "delay" allocations.
static void BM_AllocationDelayed(::testing::benchmark::State& state) {
  int delay = state.range(0);
  GPUBFCAllocator a(CreateSubAllocator(1ull << 32), 1 << 30, "GPU_0_bfc");
  // Exercise a few different allocation sizes
  std::vector<int> sizes = {256, 4096, 16384, 4096, 512, 1024, 1024};
  int size_index = 0;

  std::vector<void*> ptrs;
  ptrs.reserve(delay);
  for (int i = 0; i < delay; i++) {
    ptrs.push_back(nullptr);
  }
  int pindex = 0;
  for (auto s : state) {
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

class GPUBFCAllocatorPrivateMethodsTest
    : public ::testing::TestWithParam<SubAllocator* (*)(size_t)> {
 protected:
  void SetUp() override { CHECK_EQ(unsetenv("TF_FORCE_GPU_ALLOW_GROWTH"), 0); }

  // The following test methods are called from tests. The reason for this is
  // that this class is a friend class to BFCAllocator, but tests are not, so
  // only methods inside this class can access private members of BFCAllocator.

  void TestBinDebugInfo() {
    GPUBFCAllocator a(GetParam()(1ull << 32), 1 << 30, "GPU_0_bfc");

    std::vector<void*> initial_ptrs;
    std::vector<size_t> initial_ptrs_allocated_sizes;
    const int kNumTestSizes = 5;
    const int kNumChunksPerSize = 2;
    for (int i = 0; i < kNumTestSizes; i++) {
      for (int j = 0; j < kNumChunksPerSize; j++) {
        size_t size = 256 << i;
        void* raw = a.AllocateRaw(1, size);
        ASSERT_NE(raw, nullptr);
        initial_ptrs.push_back(raw);
        initial_ptrs_allocated_sizes.push_back(a.AllocatedSize(raw));
      }
    }

    std::array<BFCAllocator::BinDebugInfo, BFCAllocator::kNumBins> bin_infos;
    {
      mutex_lock l(a.lock_);
      bin_infos = a.get_bin_debug_info();
    }

    {
      MemoryDump md = a.RecordMemoryMap();
      EXPECT_EQ(md.chunk_size(), 1 + (kNumTestSizes * kNumChunksPerSize));
      for (int i = 0; i < BFCAllocator::kNumBins; i++) {
        const BFCAllocator::BinDebugInfo& bin_info = bin_infos[i];
        const BinSummary& bin_summary = md.bin_summary(i);
        if (i < kNumTestSizes) {
          const size_t requested_size = 2 * (256 << i);
          EXPECT_EQ(requested_size,
                    a.RequestedSize(initial_ptrs[2 * i]) +
                        a.RequestedSize(initial_ptrs[2 * i + 1]));
          size_t allocated_size = initial_ptrs_allocated_sizes[2 * i] +
                                  initial_ptrs_allocated_sizes[2 * i + 1];
          EXPECT_EQ(bin_info.total_bytes_in_use, allocated_size);
          EXPECT_EQ(bin_summary.total_bytes_in_use(), allocated_size);
          EXPECT_EQ(bin_info.total_bytes_in_bin, allocated_size);
          EXPECT_EQ(bin_summary.total_bytes_in_bin(), allocated_size);
          EXPECT_EQ(bin_info.total_requested_bytes_in_use, requested_size);
          EXPECT_EQ(bin_info.total_chunks_in_use, kNumChunksPerSize);
          EXPECT_EQ(bin_summary.total_chunks_in_use(), kNumChunksPerSize);
          EXPECT_EQ(bin_info.total_chunks_in_bin, kNumChunksPerSize);
          EXPECT_EQ(bin_summary.total_chunks_in_bin(), kNumChunksPerSize);
        } else {
          EXPECT_EQ(bin_info.total_bytes_in_use, 0);
          EXPECT_EQ(bin_summary.total_bytes_in_use(), 0);
          EXPECT_EQ(bin_info.total_requested_bytes_in_use, 0);
          EXPECT_EQ(bin_info.total_chunks_in_use, 0);
          EXPECT_EQ(bin_summary.total_chunks_in_use(), 0);
          if (i == BFCAllocator::kNumBins - 1) {
            EXPECT_GT(bin_info.total_bytes_in_bin, 0);
            EXPECT_GT(bin_summary.total_bytes_in_bin(), 0);
            EXPECT_EQ(bin_info.total_chunks_in_bin, 1);
            EXPECT_EQ(bin_summary.total_chunks_in_bin(), 1);
          } else {
            EXPECT_EQ(bin_info.total_bytes_in_bin, 0);
            EXPECT_EQ(bin_summary.total_bytes_in_bin(), 0);
            EXPECT_EQ(bin_info.total_chunks_in_bin, 0);
            EXPECT_EQ(bin_summary.total_chunks_in_bin(), 0);
          }
        }
      }
    }

    for (size_t i = 1; i < initial_ptrs.size(); i += 2) {
      a.DeallocateRaw(initial_ptrs[i]);
      initial_ptrs[i] = nullptr;
    }
    {
      mutex_lock l(a.lock_);
      bin_infos = a.get_bin_debug_info();
    }
    for (int i = 0; i < BFCAllocator::kNumBins; i++) {
      const BFCAllocator::BinDebugInfo& bin_info = bin_infos[i];
      if (i < 5) {
        // We cannot assert the exact number of bytes or chunks in the bin,
        // because it depends on what chunks were coalesced.
        size_t requested_size = 256 << i;
        EXPECT_EQ(requested_size, a.RequestedSize(initial_ptrs[2 * i]));
        EXPECT_EQ(bin_info.total_bytes_in_use,
                  initial_ptrs_allocated_sizes[2 * i]);
        EXPECT_GE(bin_info.total_bytes_in_bin,
                  initial_ptrs_allocated_sizes[2 * i]);
        EXPECT_EQ(bin_info.total_requested_bytes_in_use, requested_size);
        EXPECT_EQ(bin_info.total_chunks_in_use, 1);
        EXPECT_GE(bin_info.total_chunks_in_bin, 1);
      } else {
        EXPECT_EQ(bin_info.total_bytes_in_use, 0);
        EXPECT_EQ(bin_info.total_requested_bytes_in_use, 0);
        EXPECT_EQ(bin_info.total_chunks_in_use, 0);
      }
    }
  }

  void TestLog2FloorNonZeroSlow() {
    GPUBFCAllocator a(GetParam()(1ull << 32), 1 /* total_memory */,
                      "GPU_0_bfc");
    EXPECT_EQ(-1, a.Log2FloorNonZeroSlow(0));
    EXPECT_EQ(0, a.Log2FloorNonZeroSlow(1));
    EXPECT_EQ(1, a.Log2FloorNonZeroSlow(2));
    EXPECT_EQ(1, a.Log2FloorNonZeroSlow(3));
    EXPECT_EQ(9, a.Log2FloorNonZeroSlow(1023));
    EXPECT_EQ(10, a.Log2FloorNonZeroSlow(1024));
    EXPECT_EQ(10, a.Log2FloorNonZeroSlow(1025));
  }

  void TestForceAllowGrowth() {
    GPUOptions options;
    // Unset flag value uses provided option.
    unsetenv("TF_FORCE_GPU_ALLOW_GROWTH");
    options.set_allow_growth(true);
    GPUBFCAllocator unset_flag_allocator(GetParam()(1ull << 32), 1LL << 31,
                                         options, "GPU_0_bfc");
    EXPECT_EQ(GPUBFCAllocator::RoundedBytes(size_t{2 << 20}),
              unset_flag_allocator.curr_region_allocation_bytes_);

    // Unparseable flag value uses provided option.
    setenv("TF_FORCE_GPU_ALLOW_GROWTH", "unparseable", 1);
    options.set_allow_growth(true);
    GPUBFCAllocator unparsable_flag_allocator(GetParam()(1ull << 32), 1LL << 31,
                                              options, "GPU_1_bfc");
    EXPECT_EQ(GPUBFCAllocator::RoundedBytes(size_t{2 << 20}),
              unparsable_flag_allocator.curr_region_allocation_bytes_);

    // Max of 2GiB total memory. Env variable set forces allow_growth, which
    // does an initial allocation of 1MiB.
    setenv("TF_FORCE_GPU_ALLOW_GROWTH", "true", 1);
    options.set_allow_growth(false);
    GPUBFCAllocator force_allow_growth_allocator(
        GetParam()(1ull << 32), 1LL << 31, options, "GPU_2_bfc");
    EXPECT_EQ(GPUBFCAllocator::RoundedBytes(size_t{2 << 20}),
              force_allow_growth_allocator.curr_region_allocation_bytes_);

    // If env variable forces allow_growth disabled, all available memory is
    // allocated.
    setenv("TF_FORCE_GPU_ALLOW_GROWTH", "false", 1);
    options.set_allow_growth(true);
    GPUBFCAllocator force_no_allow_growth_allocator(
        GetParam()(1ull << 32), 1LL << 31, options, "GPU_3_bfc");
    EXPECT_EQ(GPUBFCAllocator::RoundedBytes(1LL << 31),
              force_no_allow_growth_allocator.curr_region_allocation_bytes_);
  }
};

TEST_P(GPUBFCAllocatorPrivateMethodsTest, BinDebugInfo) { TestBinDebugInfo(); }

TEST_P(GPUBFCAllocatorPrivateMethodsTest, Log2FloorNonZeroSlow) {
  TestLog2FloorNonZeroSlow();
}

TEST_P(GPUBFCAllocatorPrivateMethodsTest, ForceAllowGrowth) {
  TestForceAllowGrowth();
}

INSTANTIATE_TEST_SUITE_P(GPUBFCAllocatorPrivateMethodTestSuite,
                         GPUBFCAllocatorPrivateMethodsTest, TestSuiteValues());

// Tests that cannot be trivially parameterized for both suballocator types.
class GPUBFCAllocatorTest_SubAllocatorSpecific : public ::testing::Test {};

#if CUDA_VERSION >= 10020
// Benchmark for measuring "high water mark" for BFCAllocator owned memory.
TEST_F(GPUBFCAllocatorTest_SubAllocatorSpecific,
       VirtualAllocatorPromotesReuse) {
  GPUOptions options;
  options.set_allow_growth(true);

  constexpr size_t k512MiB = 512ull << 20;

  // 512 MiB allocator.
  GPUBFCAllocator a(CreateVirtualMemorySubAllocator(1ull << 32), k512MiB,
                    options, "GPU_0_bfc");
  // Allocate 128 raw pointers of 4 megs.
  const size_t size = 1LL << 22;
  std::vector<void*> initial_ptrs;
  for (size_t s = 0; s < 128; s++) {
    void* raw = a.AllocateRaw(1, size);
    initial_ptrs.push_back(raw);
  }
  // Deallocate all but the last one so the big chunk cannot be GC'd
  for (int i = 0; i < 127; ++i) {
    a.DeallocateRaw(initial_ptrs[i]);
  }
  void* big_alloc = a.AllocateRaw(1, k512MiB - size);
  EXPECT_NE(big_alloc, nullptr);
}
#endif

TEST_F(GPUBFCAllocatorTest_SubAllocatorSpecific,
       PhysicalAllocatorOomsFragmentation) {
  GPUOptions options;
  options.set_allow_growth(true);
  constexpr size_t k512MiB = 512ull << 20;

  // 512 MiB allocator. Garbage Collection turned off to simulate a situation
  // where there is memory pressure.
  GPUBFCAllocator a(CreateGPUMemAllocator(/*ignored*/ 0), k512MiB, options,
                    "GPU_0_bfc");
  // Allocate 128 raw pointers of 4 megs.
  const size_t size = 1LL << 22;
  std::vector<void*> initial_ptrs;
  for (size_t s = 0; s < 128; s++) {
    void* raw = a.AllocateRaw(1, size);
    initial_ptrs.push_back(raw);
  }
  // Deallocate all but the last one so the big chunk cannot be GC'd
  for (int i = 0; i < 127; ++i) {
    a.DeallocateRaw(initial_ptrs[i]);
  }
  void* big_alloc = a.AllocateRaw(1, k512MiB - size);
  EXPECT_EQ(big_alloc, nullptr);
}

// Tests that use private functions and cannot be trivially parameterized for
// both suballocator types.
class GPUBFCAllocatorPrivateMethodsTest_SubAllocatorSpecific
    : public ::testing::Test {
 protected:
  void SetUp() override { CHECK_EQ(unsetenv("TF_FORCE_GPU_ALLOW_GROWTH"), 0); }

  void TestRegionDeallocation() {
    GPUOptions options;
    options.set_allow_growth(true);

    // Max of 2GiB, but starts out small.
    GPUBFCAllocator a(CreateGPUMemAllocator(/*ignored*/ 0), 1LL << 31, options,
                      "GPU_0_bfc");

    // Allocate 128 raw pointers of 4 megs.
    const size_t size = 1LL << 22;
    std::vector<void*> initial_ptrs;
    for (size_t s = 0; s < 128; s++) {
      void* raw = a.AllocateRaw(1, size);
      initial_ptrs.push_back(raw);
    }

    {
      mutex_lock l(a.lock_);
      // Make sure there are more than 1 regions in preparation for the test.
      EXPECT_LT(1, a.region_manager_.regions().size());
    }

    // Deallocate all the memories except the last one.
    for (size_t i = 0; i < initial_ptrs.size() - 1; i++) {
      a.DeallocateRaw(initial_ptrs[i]);
    }

    // Deallocate free regions and there shall be only one region left.
    EXPECT_EQ(true, a.DeallocateFreeRegions(/*rounded_bytes=*/0));
    {
      mutex_lock l(a.lock_);
      EXPECT_EQ(1, a.region_manager_.regions().size());
    }

    // There should be only one chunk left in bins.
    size_t num_chunks_in_bins = 0;
    for (int i = 0; i < BFCAllocator::kNumBins; i++) {
      BFCAllocator::Bin* bin = a.BinFromIndex(i);
      num_chunks_in_bins += bin->free_chunks.size();
    }
    EXPECT_EQ(1, num_chunks_in_bins);
  }

#if CUDA_VERSION >= 10020
  // Counterpart to the GPUMemAllocator test suite TestRegionDeallocation tests.
  // Here we expect no deallocations because all allocations are coalesced into
  // a single region.
  void TestNoRegionDeallocation() {
    GPUOptions options;
    options.set_allow_growth(true);

    // Max of 2GiB, but starts out small.
    GPUBFCAllocator a(CreateVirtualMemorySubAllocator(1uLL << 32), 1LL << 31,
                      options, "GPU_0_bfc");

    // Allocate 128 raw pointers of 4 megs.
    const size_t size = 1LL << 22;
    std::vector<void*> initial_ptrs;
    for (size_t s = 0; s < 128; s++) {
      void* raw = a.AllocateRaw(1, size);
      initial_ptrs.push_back(raw);
    }

    {
      mutex_lock l(a.lock_);
      EXPECT_EQ(1, a.region_manager_.regions().size());
    }

    // Deallocate all the memories except the last one.
    for (size_t i = 0; i < initial_ptrs.size() - 1; i++) {
      a.DeallocateRaw(initial_ptrs[i]);
    }

    // Deallocate free regions and there should still be only one.
    EXPECT_EQ(false, a.DeallocateFreeRegions(/*rounded_bytes=*/0));
    {
      mutex_lock l(a.lock_);
      EXPECT_EQ(1, a.region_manager_.regions().size());
    }
  }
#endif
};

TEST_F(GPUBFCAllocatorPrivateMethodsTest_SubAllocatorSpecific,
       TestRegionDeallocation) {
  TestRegionDeallocation();
}

#if CUDA_VERSION >= 10020
TEST_F(GPUBFCAllocatorPrivateMethodsTest_SubAllocatorSpecific,
       TestNoRegionDeallocation) {
  TestNoRegionDeallocation();
}
#endif

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
