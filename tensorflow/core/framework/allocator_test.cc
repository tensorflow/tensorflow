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

#include "tensorflow/core/framework/allocator.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "benchmark/benchmark.h"  // from @com_google_benchmark
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"
#include "tensorflow/core/framework/typed_allocator.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/profiler/protobuf/memory_profile.pb.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_visitor.h"
#include "tsl/platform/mem.h"
#include "tsl/profiler/lib/profiler_session.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tensorflow {

static void CheckStats(Allocator* a, int64_t num_allocs, int64_t bytes_in_use,
                       int64_t peak_bytes_in_use, int64_t largest_alloc_size) {
  std::optional<AllocatorStats> stats = a->GetStats();
  EXPECT_TRUE(stats);
  if (!stats) {
    return;
  }
  LOG(INFO) << "Alloc stats: \n" << stats->DebugString();
#if defined(PLATFORM_GOOGLE) && defined(NDEBUG)
  // NOTE: allocator stats expectation depends on the system malloc,
  // and can vary as that changes.
  static const int64 kSlop = 50 * 1024;
  EXPECT_GT(stats->bytes_in_use, bytes_in_use - kSlop);
  EXPECT_LT(stats->bytes_in_use, bytes_in_use + kSlop);
  EXPECT_GT(stats->peak_bytes_in_use, peak_bytes_in_use - kSlop);
  EXPECT_LT(stats->peak_bytes_in_use, peak_bytes_in_use + kSlop);
  EXPECT_EQ(stats->num_allocs, num_allocs);
  EXPECT_EQ(stats->largest_alloc_size, largest_alloc_size);
#endif
}

TEST(AllocatorAttributesTest, AllCombos) {
  for (bool on_host : {false, true}) {
    for (bool nic_compatible : {false, true}) {
      for (bool gpu_compatible : {false, true}) {
        AllocatorAttributes aa;
        aa.set_on_host(on_host);
        aa.set_nic_compatible(nic_compatible);
        aa.set_gpu_compatible(gpu_compatible);
        EXPECT_EQ(on_host, aa.on_host());
        EXPECT_EQ(nic_compatible, aa.nic_compatible());
        EXPECT_EQ(gpu_compatible, aa.gpu_compatible());
      }
    }
  }
}

TEST(AllocatorAttributesTest, IsEqualOrLessRestrictiveThan) {
  AllocatorAttributes a, b;
  EXPECT_TRUE(a.IsEqualOrLessRestrictiveThan(b));
  EXPECT_TRUE(a.IsEqualOrLessRestrictiveThan(a));
  EXPECT_TRUE(b.IsEqualOrLessRestrictiveThan(b));

  b.set_gpu_compatible(true);
  // The set of flags in b is not a subset of those in a.
  EXPECT_TRUE(a.IsEqualOrLessRestrictiveThan(b));
  EXPECT_FALSE(b.IsEqualOrLessRestrictiveThan(a));
  EXPECT_TRUE(a.IsEqualOrLessRestrictiveThan(a));
  EXPECT_TRUE(b.IsEqualOrLessRestrictiveThan(b));

  a.set_nic_compatible(true);
  // Neither a nor b is a subset of the other.
  EXPECT_FALSE(a.IsEqualOrLessRestrictiveThan(b));
  EXPECT_FALSE(b.IsEqualOrLessRestrictiveThan(a));

  a.set_gpu_compatible(true);
  // The set of flags in b is a proper subset of those in a.
  EXPECT_TRUE(b.IsEqualOrLessRestrictiveThan(a));
  EXPECT_FALSE(a.IsEqualOrLessRestrictiveThan(b));
}

TEST(AllocatorAttributesTest, Merge) {
  AllocatorAttributes a, b;

  // Merging nic_compatible=True and nic_compatible=False results in
  // nic_compatible=True.
  EXPECT_EQ(a.value, 0);
  EXPECT_EQ(b.value, 0);
  EXPECT_FALSE(a.nic_compatible());
  EXPECT_FALSE(b.nic_compatible());
  b.set_nic_compatible(true);
  a.Merge(b);
  EXPECT_TRUE(a.nic_compatible());
  EXPECT_TRUE(b.nic_compatible());

  // a.Merge(b) does not change b.
  EXPECT_EQ(a.scope_id, 0);
  EXPECT_EQ(b.scope_id, 0);
  a.scope_id = 1;
  a.Merge(b);
  EXPECT_EQ(a.scope_id, 1);
  EXPECT_EQ(b.scope_id, 0);

  // If a.scope_id=1 and b.scope_id=0, then b.Merge(a) results in b.scope_id=1.
  a.scope_id = 1;
  b.scope_id = 0;
  b.Merge(a);
  EXPECT_EQ(a.scope_id, 1);
  EXPECT_EQ(b.scope_id, 1);

  // If a.scope_id and b.scope_id are same, then merge leaves them unchanged.
  a.scope_id = 2;
  b.scope_id = 2;
  a.Merge(b);
  EXPECT_EQ(a.scope_id, 2);
  EXPECT_EQ(b.scope_id, 2);
}

TEST(AllocatorAttributesDeathTest, MergeDifferentScopeIds) {
  AllocatorAttributes a, b;
  // If a.scope_id and b.scope_id are both positive but different, then
  // a.Merge(b) should cause a CHECK failure.
  a.scope_id = 3;
  b.scope_id = 4;
  EXPECT_DEATH({ a.Merge(b); }, "");
}

TEST(CPUAllocatorTest, Simple) {
  EnableCPUAllocatorStats();
  Allocator* a = cpu_allocator();
  std::vector<void*> ptrs;
  for (int s = 1; s < 1024; s++) {
    void* raw = a->AllocateRaw(1, s);
    ptrs.push_back(raw);
  }
  std::sort(ptrs.begin(), ptrs.end());
  CheckStats(a, 1023, 552640, 552640, 1024);
  for (size_t i = 0; i < ptrs.size(); i++) {
    if (i > 0) {
      CHECK_NE(ptrs[i], ptrs[i - 1]);  // No dups
    }
    a->DeallocateRaw(ptrs[i]);
  }
  CheckStats(a, 1023, 0, 552640, 1024);
  float* t1 = TypedAllocator::Allocate<float>(a, 1024, {});
  double* t2 = TypedAllocator::Allocate<double>(a, 1048576, {});
  CheckStats(a, 1025, 1048576 * sizeof(double) + 1024 * sizeof(float),
             1048576 * sizeof(double) + 1024 * sizeof(float),
             1048576 * sizeof(double));

  TypedAllocator::Deallocate(a, t1, 1024);
  TypedAllocator::Deallocate(a, t2, 1048576);

  CheckStats(a, 1025, 0, 1048576 * sizeof(double) + 1024 * sizeof(float),
             1048576 * sizeof(double));
  CHECK(a->ClearStats());
  CheckStats(a, 0, 0, 0, 0);
  DisableCPUAllocatorStats();
}

// Define a struct that we will use to observe behavior in the unit tests
struct TestStruct {
  int x;  // not used just want to make sure sizeof(TestStruct) > 1
};

TEST(CPUAllocatorTest, CheckStructSize) { CHECK_GT(sizeof(TestStruct), 1); }

TEST(CPUAllocatorTest, AllocateOverflowMaxSizeT) {
  Allocator* a = cpu_allocator();

  // The maximum size_t value will definitely overflow.
  size_t count_to_allocate = std::numeric_limits<size_t>::max();
  TestStruct* const test_pointer =
      TypedAllocator::Allocate<TestStruct>(a, count_to_allocate, {});

  CHECK_EQ(test_pointer, reinterpret_cast<TestStruct*>(NULL));
}

TEST(CPUAllocatorTest, AllocateOverflowSmallest) {
  Allocator* a = cpu_allocator();

  // count_to_allocate is the smallest count that will cause overflow.
  const size_t count_to_allocate =
      (std::numeric_limits<size_t>::max() / sizeof(TestStruct)) + 1;
  TestStruct* const test_pointer =
      TypedAllocator::Allocate<TestStruct>(a, count_to_allocate, {});

  CHECK_EQ(test_pointer, reinterpret_cast<TestStruct*>(NULL));
}

TEST(CPUAllocatorTest, Sizes) {
  Allocator* a = cpu_allocator();

  EXPECT_EQ(false, a->TracksAllocationSizes());
}

TEST(CPUAllocatorTest, AllocationBytesPadding) {
  // Types smaller than 32 bits (4 bytes) must be padded to a 4-byte boundary
  // so GPU atomicCAS can safely operate on the enclosing 32-bit word.
  EXPECT_EQ(TypedAllocator::AllocationBytes<int16_t>(0), 0);
  EXPECT_EQ(TypedAllocator::AllocationBytes<int16_t>(1), 4);
  EXPECT_EQ(TypedAllocator::AllocationBytes<int16_t>(2), 4);
  EXPECT_EQ(TypedAllocator::AllocationBytes<int16_t>(3), 8);
  EXPECT_EQ(TypedAllocator::AllocationBytes<int16_t>(4), 8);

  EXPECT_EQ(TypedAllocator::AllocationBytes<int8_t>(0), 0);
  EXPECT_EQ(TypedAllocator::AllocationBytes<int8_t>(1), 4);
  EXPECT_EQ(TypedAllocator::AllocationBytes<int8_t>(2), 4);
  EXPECT_EQ(TypedAllocator::AllocationBytes<int8_t>(3), 4);
  EXPECT_EQ(TypedAllocator::AllocationBytes<int8_t>(4), 4);
  EXPECT_EQ(TypedAllocator::AllocationBytes<int8_t>(5), 8);

  // Types >= 4 bytes are not padded.
  EXPECT_EQ(TypedAllocator::AllocationBytes<float>(0), 0);
  EXPECT_EQ(TypedAllocator::AllocationBytes<float>(1), sizeof(float));
  EXPECT_EQ(TypedAllocator::AllocationBytes<float>(3), 3 * sizeof(float));
  EXPECT_EQ(TypedAllocator::AllocationBytes<double>(3), 3 * sizeof(double));

  // Overflow checks on both sides of the boundary for sub-32-bit types.
  constexpr size_t kMax = std::numeric_limits<size_t>::max();
  // For int8_t: kMax % 4 == 3, so kMax - 3 is divisible by 4.
  // kMax - 3 does not overflow when padded.
  constexpr size_t int8_max_valid = kMax - 3;
  EXPECT_EQ(TypedAllocator::AllocationBytes<int8_t>(int8_max_valid),
            int8_max_valid);
  // kMax - 2, kMax - 1, and kMax overflow when padded to 4 bytes.
  EXPECT_EQ(TypedAllocator::AllocationBytes<int8_t>(int8_max_valid + 1), 0);
  EXPECT_EQ(TypedAllocator::AllocationBytes<int8_t>(int8_max_valid + 2), 0);
  EXPECT_EQ(TypedAllocator::AllocationBytes<int8_t>(kMax), 0);

  // For int16_t: multiplication or padding overflows at and beyond kMax / 2.
  constexpr size_t int16_max_mult = kMax / sizeof(int16_t);
  // (kMax / 2 - 1) * 2 == kMax - 3, which is 4-byte aligned and valid.
  EXPECT_EQ(TypedAllocator::AllocationBytes<int16_t>(int16_max_mult - 1),
            kMax - 3);
  // kMax / 2 * 2 == kMax - 1, which requires 2 pad bytes and overflows.
  EXPECT_EQ(TypedAllocator::AllocationBytes<int16_t>(int16_max_mult), 0);
  // Greater than kMax / 2 overflows multiplication.
  EXPECT_EQ(TypedAllocator::AllocationBytes<int16_t>(int16_max_mult + 1), 0);
  EXPECT_EQ(TypedAllocator::AllocationBytes<int16_t>(kMax), 0);

  // Verify that TypedAllocator::Allocate<T> returns nullptr on overflow.
  EXPECT_EQ(
      TypedAllocator::Allocate<int8_t>(cpu_allocator(), int8_max_valid + 1, {}),
      nullptr);
  EXPECT_EQ(TypedAllocator::Allocate<int8_t>(cpu_allocator(), kMax, {}),
            nullptr);
  EXPECT_EQ(
      TypedAllocator::Allocate<int16_t>(cpu_allocator(), int16_max_mult, {}),
      nullptr);
  EXPECT_EQ(TypedAllocator::Allocate<int16_t>(cpu_allocator(), kMax, {}),
            nullptr);

  // Test Allocate and Deallocate directly with a sub-32-bit type.
  int16_t* ptr_odd = TypedAllocator::Allocate<int16_t>(cpu_allocator(), 3, {});
  ASSERT_NE(ptr_odd, nullptr);
  ptr_odd[0] = 10;
  ptr_odd[1] = 20;
  ptr_odd[2] = 30;
  TypedAllocator::Deallocate(cpu_allocator(), ptr_odd, 3);

  int16_t* ptr_even = TypedAllocator::Allocate<int16_t>(cpu_allocator(), 4, {});
  ASSERT_NE(ptr_even, nullptr);
  TypedAllocator::Deallocate(cpu_allocator(), ptr_even, 4);

  int8_t* ptr_odd8 = TypedAllocator::Allocate<int8_t>(cpu_allocator(), 5, {});
  ASSERT_NE(ptr_odd8, nullptr);
  TypedAllocator::Deallocate(cpu_allocator(), ptr_odd8, 5);

  // Zero-element allocation does not return nullptr due to overflow check.
  int16_t* ptr_zero = TypedAllocator::Allocate<int16_t>(cpu_allocator(), 0, {});
  TypedAllocator::Deallocate(cpu_allocator(), ptr_zero, 0);

  // Verify exact byte counts passed to non-opaque (padded) and opaque-handle
  // (unpadded) allocators on both Allocate and Deallocate.
  class RecordingAllocator : public Allocator {
   public:
    explicit RecordingAllocator(bool opaque) : opaque_(opaque) {}
    std::string Name() override { return "RecordingAllocator"; }
    void* AllocateRaw(size_t alignment, size_t num_bytes) override {
      allocated_bytes = num_bytes;
      return dummy_;
    }
    void DeallocateRaw(void* ptr) override { deallocated_bytes = 0; }
    void DeallocateRawAlignedDelete(void* ptr, size_t alignment,
                                    size_t num_bytes) override {
      deallocated_bytes = num_bytes;
    }
    bool AllocatesOpaqueHandle() const override { return opaque_; }

    size_t allocated_bytes = 0;
    size_t deallocated_bytes = 0;

   private:
    bool opaque_;
    alignas(Allocator::kAllocatorAlignment) char
        dummy_[Allocator::kAllocatorAlignment] = {};
  };

  RecordingAllocator padded_alloc(/*opaque=*/false);
  int16_t* padded_ptr =
      TypedAllocator::Allocate<int16_t>(&padded_alloc, 3, {});
  EXPECT_EQ(padded_alloc.allocated_bytes, 8);
  TypedAllocator::Deallocate(&padded_alloc, padded_ptr, 3);
  EXPECT_EQ(padded_alloc.deallocated_bytes, 8);

  RecordingAllocator opaque_alloc(/*opaque=*/true);
  int16_t* opaque_ptr =
      TypedAllocator::Allocate<int16_t>(&opaque_alloc, 3, {});
  EXPECT_EQ(opaque_alloc.allocated_bytes, 3 * sizeof(int16_t));
  TypedAllocator::Deallocate(&opaque_alloc, opaque_ptr, 3);
  EXPECT_EQ(opaque_alloc.deallocated_bytes, 3 * sizeof(int16_t));
}

TEST(CPUAllocatorTest, ProfilerReporting) {
  // TODO(b/196611863): Make debugging work even without GetAllocatedSize.
  void* p = tsl::port::AlignedMalloc(8, static_cast<std::align_val_t>(1));
  const std::size_t alloc_size = port::MallocExtension_GetAllocatedSize(p);
  port::AlignedFree(p);
  if (alloc_size == 0) {
    LOG(WARNING) << "Skipping Memory Debugging test. It requires "
                 << "port::MallocExtension_GetAllocatedSize to work.";
    return;
  }

  EnableCPUAllocatorStats();
  Allocator* a = cpu_allocator();

  // Allocate something before profiling starts
  void* p1 = a->AllocateRaw(1, 16);

  // Start profiling
  std::unique_ptr<tsl::ProfilerSession> profiler =
      tsl::ProfilerSession::Create(tsl::ProfilerSession::DefaultOptions());

  // Profiled allocations
  void* p2 = a->AllocateRaw(1, 32);
  a->DeallocateRaw(p1);

  // Get profiling results
  tensorflow::profiler::XSpace xspace;
  EXPECT_EQ(absl::OkStatus(), profiler->CollectData(&xspace));

  // Validate the output
  const auto plane = ::tsl::profiler::FindPlaneWithName(
      xspace, ::tensorflow::profiler::kHostThreadsPlaneName);
  ::tensorflow::profiler::XPlaneVisitor xplane(plane);

  ASSERT_EQ(plane->name(), ::tensorflow::profiler::kHostThreadsPlaneName)
      << "XSpace: " << xspace.DebugString();
  ASSERT_EQ(plane->event_metadata_size(), 2)
      << "XSpace: " << xspace.DebugString();

  const auto& line = plane->lines(0);
  ASSERT_EQ(line.events_size(), 2) << "XSpace: " << xspace.DebugString();
  const auto& events = line.events();

  ::tensorflow::profiler::XEventVisitor e0(&xplane, &line, &events[0]);
  EXPECT_EQ(e0.Name(), "MemoryAllocation")
      << "XSpace: " << xspace.DebugString();
  {
    std::optional<std::string> bytes_allocated, peak_bytes_in_use,
        requested_bytes, allocation_bytes;
    e0.ForEachStat([&](const ::tensorflow::profiler::XStatVisitor& stat) {
      LOG(ERROR) << "STAT " << stat.Name() << ": " << stat.ToString();
      if (stat.Name() == "bytes_allocated") {
        bytes_allocated = stat.ToString();
      } else if (stat.Name() == "peak_bytes_in_use") {
        peak_bytes_in_use = stat.ToString();
      } else if (stat.Name() == "requested_bytes") {
        requested_bytes = stat.ToString();
      } else if (stat.Name() == "allocation_bytes") {
        allocation_bytes = stat.ToString();
      }
    });
    ASSERT_TRUE(bytes_allocated && peak_bytes_in_use && requested_bytes &&
                allocation_bytes)
        << "XSpace: " << xspace.DebugString();
    EXPECT_EQ(*bytes_allocated, "48") << "XSpace: " << xspace.DebugString();
    EXPECT_EQ(*peak_bytes_in_use, "48") << "XSpace: " << xspace.DebugString();
    EXPECT_EQ(*requested_bytes, "32") << "XSpace: " << xspace.DebugString();
    EXPECT_EQ(*allocation_bytes, "32") << "XSpace: " << xspace.DebugString();
  }

  ::tensorflow::profiler::XEventVisitor e1(&xplane, &line, &events[1]);
  EXPECT_EQ(e1.Name(), "MemoryDeallocation")
      << "XSpace: " << xspace.DebugString();
  {
    std::optional<std::string> bytes_allocated, peak_bytes_in_use,
        allocation_bytes;
    e1.ForEachStat([&](const ::tensorflow::profiler::XStatVisitor& stat) {
      if (stat.Name() == "bytes_allocated") {
        bytes_allocated = stat.ToString();
      } else if (stat.Name() == "peak_bytes_in_use") {
        peak_bytes_in_use = stat.ToString();
      } else if (stat.Name() == "allocation_bytes") {
        allocation_bytes = stat.ToString();
      }
    });
    ASSERT_TRUE(bytes_allocated && peak_bytes_in_use && allocation_bytes)
        << "XSpace: " << xspace.DebugString();
    EXPECT_EQ(*bytes_allocated, "32") << "XSpace: " << xspace.DebugString();
    EXPECT_EQ(*peak_bytes_in_use, "48") << "XSpace: " << xspace.DebugString();
    EXPECT_EQ(*allocation_bytes, "16") << "XSpace: " << xspace.DebugString();
  }

  // Cleanup
  a->DeallocateRaw(p2);
  DisableCPUAllocatorStats();
}

namespace {

AllocatorAttributes DeviceAllocatorAttribute() {
  AllocatorAttributes attr;
  attr.value |= (0x1 << 24);
  return attr;
}

bool HasDeviceAllocatorAttribute(const AllocatorAttributes& attr) {
  return attr.value & (0x1 << 24);
}

}  // namespace

TEST(CustomAllocatorAttributes, TestSetterAndGetter) {
  AllocatorAttributes attr = DeviceAllocatorAttribute();
  EXPECT_TRUE(HasDeviceAllocatorAttribute(attr));
  EXPECT_FALSE(HasDeviceAllocatorAttribute(AllocatorAttributes()));
}

static void BM_Allocation(::testing::benchmark::State& state) {
  const int arg = state.range(0);

  Allocator* a = cpu_allocator();
  // Exercise a few different allocation sizes
  std::vector<int> sizes = {256, 4096, 16384, 524288, 512, 1048576};
  int size_index = 0;

  if (arg) EnableCPUAllocatorStats();
  for (auto s : state) {
    int bytes = sizes[size_index++ % sizes.size()];
    void* p = a->AllocateRaw(1, bytes);
    a->DeallocateRaw(p);
  }
  if (arg) DisableCPUAllocatorStats();
}
BENCHMARK(BM_Allocation)->Arg(0)->Arg(1);

}  // namespace tensorflow
