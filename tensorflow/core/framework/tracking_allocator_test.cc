#include "tensorflow/core/framework/tracking_allocator.h"

#include <unordered_map>

#include <gtest/gtest.h>
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

class TestableSizeTrackingAllocator : public Allocator {
 public:
  string Name() override { return "test"; }
  void* AllocateRaw(size_t /*alignment*/, size_t num_bytes) override {
    void* ptr = malloc(num_bytes);
    size_map_[ptr] = num_bytes;
    return ptr;
  }
  void DeallocateRaw(void* ptr) override {
    const auto& iter = size_map_.find(ptr);
    EXPECT_NE(size_map_.end(), iter);
    size_map_.erase(iter);
    free(ptr);
  }
  bool TracksAllocationSizes() override { return true; }
  size_t RequestedSize(void* ptr) override {
    const auto& iter = size_map_.find(ptr);
    EXPECT_NE(size_map_.end(), iter);
    return iter->second;
  }

 private:
  std::unordered_map<void*, size_t> size_map_;
};

class NoMemoryAllocator : public Allocator {
 public:
  string Name() override { return "test"; }
  void* AllocateRaw(size_t /*alignment*/, size_t num_bytes) override {
    return nullptr;
  }
  void DeallocateRaw(void* ptr) override {}
  bool TracksAllocationSizes() override { return true; }
};

TEST(TrackingAllocatorTest, SimpleNoTracking) {
  Allocator* a = cpu_allocator();

  EXPECT_FALSE(a->TracksAllocationSizes());

  TrackingAllocator* ta = new TrackingAllocator(a);

  void* p1 = ta->AllocateRaw(4, 4);
  ta->Deallocate(p1);
  void* p2 = ta->AllocateRaw(4, 12);

  std::pair<size_t, size_t> sizes = ta->GetSizesAndUnRef();

  EXPECT_EQ(16, sizes.first);
  EXPECT_EQ(0, sizes.second);

  ta->Deallocate(p2);
}

TEST(TrackingAllocatorTest, SimpleTracking) {
  TestableSizeTrackingAllocator a = TestableSizeTrackingAllocator();

  EXPECT_TRUE(a.TracksAllocationSizes());

  TrackingAllocator* ta = new TrackingAllocator(&a);

  void* p1 = ta->AllocateRaw(4, 12);
  ta->Deallocate(p1);
  void* p2 = ta->AllocateRaw(4, 4);

  std::pair<size_t, size_t> sizes = ta->GetSizesAndUnRef();

  EXPECT_EQ(16, sizes.first);
  EXPECT_EQ(12, sizes.second);

  ta->Deallocate(p2);
}

TEST(TrackingAllocatorTest, OutOfMemory) {
  NoMemoryAllocator a;

  EXPECT_TRUE(a.TracksAllocationSizes());

  TrackingAllocator* ta = new TrackingAllocator(&a);

  void* p1 = ta->AllocateRaw(4, 12);
  EXPECT_EQ(nullptr, p1);

  std::pair<size_t, size_t> sizes = ta->GetSizesAndUnRef();

  EXPECT_EQ(0, sizes.first);
  EXPECT_EQ(0, sizes.second);
}

TEST(TrackingAllocatorTest, FreeNullPtr) {
  NoMemoryAllocator a;

  EXPECT_TRUE(a.TracksAllocationSizes());

  TrackingAllocator* ta = new TrackingAllocator(&a);

  ta->DeallocateRaw(nullptr);

  std::pair<size_t, size_t> sizes = ta->GetSizesAndUnRef();

  EXPECT_EQ(0, sizes.first);
  EXPECT_EQ(0, sizes.second);
}

}  // namespace tensorflow
