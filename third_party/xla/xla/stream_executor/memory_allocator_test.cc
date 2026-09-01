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

#include "xla/stream_executor/memory_allocator.h"

#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/synchronization/notification.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/tsl/platform/env.h"

namespace stream_executor {
namespace {

class DummyMemoryAllocation : public MemoryAllocation {
 public:
  explicit DummyMemoryAllocation(void* opaque) : opaque_(opaque) {}
  DeviceAddressBase address() const override {
    return DeviceAddressBase(opaque_, 100);
  }

 private:
  void* opaque_;
};

TEST(AllocationTrackerTest, TrackAndFree) {
  MemoryAllocator::AllocationTracker tracker;
  void* ptr = reinterpret_cast<void*>(0x1234);
  auto alloc = std::make_unique<DummyMemoryAllocation>(ptr);

  ASSERT_OK_AND_ASSIGN(DeviceAddressBase addr, tracker.Track(std::move(alloc)));
  EXPECT_TRUE(tracker.IsTracked(addr));

  EXPECT_OK(tracker.Free(addr));
  EXPECT_FALSE(tracker.IsTracked(addr));
}

TEST(AllocationTrackerTest, FreeAllowsBaseAddressWithSmallerSize) {
  MemoryAllocator::AllocationTracker tracker;

  void* ptr0 = reinterpret_cast<void*>(0x1234);
  auto alloc0 = std::make_unique<DummyMemoryAllocation>(ptr0);
  ASSERT_OK_AND_ASSIGN(DeviceAddressBase addr0,
                       tracker.Track(std::move(alloc0)));
  EXPECT_OK(tracker.Free(DeviceAddressBase(addr0.opaque(), 8)));
  EXPECT_FALSE(tracker.IsTracked(addr0));

  void* ptr1 = reinterpret_cast<void*>(0x5678);
  auto alloc1 = std::make_unique<DummyMemoryAllocation>(ptr1);
  ASSERT_OK_AND_ASSIGN(DeviceAddressBase addr1,
                       tracker.Track(std::move(alloc1)));
  EXPECT_OK(tracker.Free(DeviceAddressBase(addr1.opaque(), 0)));
  EXPECT_FALSE(tracker.IsTracked(addr1));
}

TEST(AllocationTrackerTest, FreeAllowsPayloadHandleWithSmallerSize) {
  MemoryAllocator::AllocationTracker tracker;
  void* ptr = reinterpret_cast<void*>(0x1234);
  auto alloc = std::make_unique<DummyMemoryAllocation>(ptr);

  ASSERT_OK_AND_ASSIGN(DeviceAddressBase addr, tracker.Track(std::move(alloc)));
  DeviceAddressBase mismatched_size(addr.opaque(), 8);
  mismatched_size.SetPayload(addr.payload());

  EXPECT_OK(tracker.Free(mismatched_size));
  EXPECT_FALSE(tracker.IsTracked(addr));
}

TEST(AllocationTrackerTest, FreeRejectsPayloadHandleWithOversizedRange) {
  MemoryAllocator::AllocationTracker tracker;
  void* ptr = reinterpret_cast<void*>(0x1234);
  auto alloc = std::make_unique<DummyMemoryAllocation>(ptr);

  ASSERT_OK_AND_ASSIGN(DeviceAddressBase addr, tracker.Track(std::move(alloc)));
  DeviceAddressBase oversized(addr.opaque(), 101);
  oversized.SetPayload(addr.payload());

  EXPECT_FALSE(tracker.Free(oversized).ok());
  EXPECT_TRUE(tracker.IsTracked(addr));

  EXPECT_OK(tracker.Free(addr));
}

TEST(AllocationTrackerTest, FreeRejectsBaseAddressWithOversizedRange) {
  MemoryAllocator::AllocationTracker tracker;
  void* ptr = reinterpret_cast<void*>(0x1234);
  auto alloc = std::make_unique<DummyMemoryAllocation>(ptr);

  ASSERT_OK_AND_ASSIGN(DeviceAddressBase addr, tracker.Track(std::move(alloc)));
  EXPECT_FALSE(tracker.Free(DeviceAddressBase(addr.opaque(), 101)).ok());
  EXPECT_TRUE(tracker.IsTracked(addr));

  EXPECT_OK(tracker.Free(addr));
}

// A MemoryAllocation that interacts with the AllocationTracker in its
// destructor. This is used to test that the tracker does not hold its lock when
// destroying the allocation, which would otherwise cause a deadlock.
class DeadlockingMemoryAllocation : public MemoryAllocation {
 public:
  explicit DeadlockingMemoryAllocation(
      void* opaque, MemoryAllocator::AllocationTracker* tracker)
      : opaque_(opaque), tracker_(tracker) {}

  ~DeadlockingMemoryAllocation() override {
    // In a real scenario, this might wait for a GPU stream to finish,
    // and that stream might have a host callback that calls into the tracker.
    // We simulate this by spawning a thread that calls into the tracker,
    // and waiting for it to finish. If the tracker's lock is held during
    // this destructor, the thread will block forever, and this wait will
    // deadlock.
    absl::Notification thread_started;
    absl::Notification thread_finished;
    std::unique_ptr<tsl::Thread> t(tsl::Env::Default()->StartThread(
        tsl::ThreadOptions(), "tracker_test_thread", [&]() {
          thread_started.Notify();
          // Try to track a new allocation. This requires the tracker's lock.
          auto new_alloc = std::make_unique<DummyMemoryAllocation>(
              reinterpret_cast<void*>(0x5678));
          auto statusor = tracker_->Track(std::move(new_alloc));
          EXPECT_OK(statusor.status());
          thread_finished.Notify();
        }));

    thread_started.WaitForNotification();
    // Wait for the thread to finish. If the lock is held, this will deadlock.
    thread_finished.WaitForNotification();
    t.reset();
  }

  DeviceAddressBase address() const override {
    return DeviceAddressBase(opaque_, 100);
  }

 private:
  void* opaque_;
  MemoryAllocator::AllocationTracker* tracker_;
};

TEST(AllocationTrackerTest, FreeDoesNotHoldLockDuringDestruction) {
  MemoryAllocator::AllocationTracker tracker;
  void* ptr = reinterpret_cast<void*>(0x1234);
  auto alloc = std::make_unique<DeadlockingMemoryAllocation>(ptr, &tracker);
  ASSERT_OK_AND_ASSIGN(DeviceAddressBase addr, tracker.Track(std::move(alloc)));
  // Freeing the allocation will trigger its destructor.
  // The destructor will spawn a thread that tries to acquire the tracker's
  // lock. If Free() holds the lock while destroying the allocation, this will
  // deadlock.
  EXPECT_OK(tracker.Free(addr));
}

}  // namespace
}  // namespace stream_executor
