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

#include "tensorflow/core/framework/tracking_allocator.h"

#include <unordered_map>

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

class TestableSizeTrackingAllocator : public Allocator {
 public:
  string Name() override { return "test"; }
  void* AllocateRaw(size_t /*alignment*/, size_t num_bytes) override {
    void* ptr = port::Malloc(num_bytes);
    size_map_[ptr] = num_bytes;
    return ptr;
  }
  void DeallocateRaw(void* ptr) override {
    const auto& iter = size_map_.find(ptr);
    EXPECT_NE(size_map_.end(), iter);
    size_map_.erase(iter);
    port::Free(ptr);
  }
  bool TracksAllocationSizes() override { return true; }
  size_t RequestedSize(const void* ptr) override {
    const auto& iter = size_map_.find(ptr);
    EXPECT_NE(size_map_.end(), iter);
    return iter->second;
  }
  void GetStats(AllocatorStats* stats) override { stats->Clear(); }

 private:
  std::unordered_map<const void*, size_t> size_map_;
};

class NoMemoryAllocator : public Allocator {
 public:
  string Name() override { return "test"; }
  void* AllocateRaw(size_t /*alignment*/, size_t num_bytes) override {
    return nullptr;
  }
  void DeallocateRaw(void* ptr) override {}
  bool TracksAllocationSizes() override { return true; }
  void GetStats(AllocatorStats* stats) override { stats->Clear(); }
};

TEST(TrackingAllocatorTest, SimpleNoTracking) {
  Allocator* a = cpu_allocator();

  EXPECT_FALSE(a->TracksAllocationSizes());

  // Don't enable the tracking inside the tracking allocator. Since
  // the cpu_allocator doesn't track allocations itself the tracking
  // will be partial
  TrackingAllocator* ta = new TrackingAllocator(a, false);

  void* p1 = ta->AllocateRaw(4, 4);
  ta->DeallocateRaw(p1);
  void* p2 = ta->AllocateRaw(4, 12);

  std::tuple<size_t, size_t, size_t> sizes = ta->GetSizes();

  EXPECT_EQ(16, std::get<0>(sizes));
  EXPECT_EQ(0, std::get<1>(sizes));
  EXPECT_EQ(0, std::get<2>(sizes));

  ta->DeallocateRaw(p2);
  auto records = ta->GetRecordsAndUnRef();
  EXPECT_EQ(4, records[0].alloc_bytes);
  EXPECT_EQ(12, records[1].alloc_bytes);

  // This time enable the tracking inside the tracking allocator
  ta = new TrackingAllocator(a, true);
  p1 = ta->AllocateRaw(4, 4);
  EXPECT_EQ(4, ta->RequestedSize(p1));
  EXPECT_LE(4, ta->AllocatedSize(p1));
  EXPECT_EQ(1, ta->AllocationId(p1));

  ta->DeallocateRaw(p1);
  p2 = ta->AllocateRaw(4, 12);
  EXPECT_EQ(12, ta->RequestedSize(p2));
  EXPECT_LE(12, ta->AllocatedSize(p2));
  EXPECT_EQ(2, ta->AllocationId(p2));

  sizes = ta->GetSizes();

  EXPECT_LE(16, std::get<0>(sizes));
  EXPECT_LE(12, std::get<1>(sizes));
  EXPECT_LE(12, std::get<2>(sizes));

  ta->DeallocateRaw(p2);
  records = ta->GetRecordsAndUnRef();
  EXPECT_LE(4, records[0].alloc_bytes);
  EXPECT_GE(-4, records[1].alloc_bytes);
  EXPECT_LE(12, records[2].alloc_bytes);
  EXPECT_GE(-12, records[3].alloc_bytes);
}

TEST(TrackingAllocatorTest, SimpleTracking) {
  TestableSizeTrackingAllocator a = TestableSizeTrackingAllocator();

  EXPECT_TRUE(a.TracksAllocationSizes());

  TrackingAllocator* ta = new TrackingAllocator(&a, false);

  void* p1 = ta->AllocateRaw(4, 12);
  ta->DeallocateRaw(p1);
  void* p2 = ta->AllocateRaw(4, 4);

  std::tuple<size_t, size_t, size_t> sizes = ta->GetSizes();

  EXPECT_EQ(16, std::get<0>(sizes));
  EXPECT_EQ(12, std::get<1>(sizes));
  EXPECT_EQ(4, std::get<2>(sizes));

  ta->DeallocateRaw(p2);

  auto records = ta->GetRecordsAndUnRef();
  EXPECT_EQ(12, records[0].alloc_bytes);
  EXPECT_EQ(-12, records[1].alloc_bytes);
  EXPECT_EQ(4, records[2].alloc_bytes);
  EXPECT_EQ(-4, records[3].alloc_bytes);
}

TEST(TrackingAllocatorTest, OutOfMemory) {
  NoMemoryAllocator a;

  EXPECT_TRUE(a.TracksAllocationSizes());

  TrackingAllocator* ta = new TrackingAllocator(&a, false);

  void* p1 = ta->AllocateRaw(4, 12);
  EXPECT_EQ(nullptr, p1);

  std::tuple<size_t, size_t, size_t> sizes = ta->GetSizes();

  EXPECT_EQ(0, std::get<0>(sizes));
  EXPECT_EQ(0, std::get<1>(sizes));
  EXPECT_EQ(0, std::get<2>(sizes));

  EXPECT_EQ(0, ta->GetRecordsAndUnRef().size());
}

TEST(TrackingAllocatorTest, FreeNullPtr) {
  NoMemoryAllocator a;

  EXPECT_TRUE(a.TracksAllocationSizes());

  TrackingAllocator* ta = new TrackingAllocator(&a, false);

  ta->DeallocateRaw(nullptr);

  std::tuple<size_t, size_t, size_t> sizes = ta->GetSizes();

  EXPECT_EQ(0, std::get<0>(sizes));
  EXPECT_EQ(0, std::get<1>(sizes));
  EXPECT_EQ(0, std::get<2>(sizes));

  EXPECT_EQ(0, ta->GetRecordsAndUnRef().size());
}

}  // namespace tensorflow
