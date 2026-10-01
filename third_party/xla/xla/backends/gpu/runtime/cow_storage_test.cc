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

#include "xla/backends/gpu/runtime/cow_storage.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/optimization.h"
#include "absl/container/flat_hash_set.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "xla/backends/gpu/runtime/device_slot.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"

namespace xla::gpu {
namespace {

using ::testing::Contains;
using ::testing::Each;
using ::testing::Not;

// Counts constructions and destructions. Neither copyable nor movable, so a
// storage that copies or moves its states fails to compile.
struct Tracked {
  static constexpr int kInitialValue = 42;

  Tracked() { constructed.fetch_add(1); }
  ~Tracked() { destroyed.fetch_add(1); }
  Tracked(const Tracked&) = delete;
  Tracked& operator=(const Tracked&) = delete;

  static void Reset() {
    constructed.store(0);
    destroyed.store(0);
  }

  static inline std::atomic<int> constructed{0};
  static inline std::atomic<int> destroyed{0};

  int value = kInitialValue;
};
static_assert(!std::is_copy_constructible_v<Tracked>);
static_assert(!std::is_copy_assignable_v<Tracked>);
static_assert(!std::is_move_constructible_v<Tracked>);
static_assert(!std::is_move_assignable_v<Tracked>);

// Notifies `entered`, then waits for `gate`, for whichever is set. Lets a test
// keep a CowStorage insert, and so its mutex, held.
struct GatedState {
  GatedState() {
    if (absl::Notification* notification = entered.load()) {
      notification->Notify();
    }
    if (absl::Notification* notification = gate.load()) {
      notification->WaitForNotification();
    }
  }

  static void Reset() {
    entered.store(nullptr);
    gate.store(nullptr);
  }

  static inline std::atomic<absl::Notification*> entered{nullptr};
  static inline std::atomic<absl::Notification*> gate{nullptr};
};

bool IsCacheLineAligned(const void* ptr) {
  return reinterpret_cast<uintptr_t>(ptr) % ABSL_CACHELINE_SIZE == 0;
}

template <typename T>
size_t CountDistinct(const std::vector<T*>& ptrs) {
  return absl::flat_hash_set<T*>(ptrs.begin(), ptrs.end()).size();
}

class CowStorageTest : public ::testing::Test {
 protected:
  void SetUp() override {
    Tracked::Reset();
    GatedState::Reset();
  }
  void TearDown() override { GatedState::Reset(); }
};

TEST_F(CowStorageTest, FindReturnsNullForOrdinalsNotCreated) {
  CowStorage storage;
  EXPECT_EQ(storage.Find(-1), nullptr);
  EXPECT_EQ(storage.GetOrCreate(-1, &DeviceSlot::Create<Tracked>), nullptr);
  EXPECT_EQ(storage.Find(0), nullptr);
  EXPECT_NE(storage.GetOrCreate(5, &DeviceSlot::Create<Tracked>), nullptr);
  EXPECT_EQ(storage.Find(-1), nullptr);
  EXPECT_EQ(storage.GetOrCreate(-1, &DeviceSlot::Create<Tracked>), nullptr);
  EXPECT_EQ(storage.Find(4), nullptr);
  EXPECT_EQ(storage.Find(6), nullptr);
  EXPECT_EQ(Tracked::constructed.load(), 1);
}

TEST_F(CowStorageTest, SparseOrdinals) {
  CowStorage storage;
  DeviceSlot* s0 = storage.GetOrCreate(0, &DeviceSlot::Create<Tracked>);
  DeviceSlot* s7 = storage.GetOrCreate(7, &DeviceSlot::Create<Tracked>);
  DeviceSlot* s31 = storage.GetOrCreate(31, &DeviceSlot::Create<Tracked>);

  EXPECT_NE(s0, nullptr);
  EXPECT_NE(s7, nullptr);
  EXPECT_NE(s31, nullptr);
  EXPECT_EQ(storage.Find(0), s0);
  EXPECT_EQ(storage.Find(7), s7);
  EXPECT_EQ(storage.Find(31), s31);
  for (int gap : {1, 6, 8, 30, 32}) {
    EXPECT_EQ(storage.Find(gap), nullptr) << "ordinal " << gap;
  }
  EXPECT_EQ(Tracked::constructed.load(), 3);
}

TEST_F(CowStorageTest, GetOrCreateBuildsOnce) {
  CowStorage storage;
  DeviceSlot* slot = storage.GetOrCreate(3, &DeviceSlot::Create<Tracked>);
  EXPECT_NE(slot, nullptr);
  EXPECT_EQ(storage.GetOrCreate(3, &DeviceSlot::Create<Tracked>), slot);
  EXPECT_EQ(storage.Find(3), slot);
  EXPECT_EQ(Tracked::constructed.load(), 1);
}

TEST_F(CowStorageTest, StatesKeepAddressAcrossInserts) {
  // More ordinals than a snapshot stores inline.
  constexpr int kNumOrdinals = 16;
  CowStorage storage;
  std::vector<Tracked*> states;
  for (int ordinal = 0; ordinal < kNumOrdinals; ++ordinal) {
    states.push_back(DeviceSlot::Unwrap<Tracked>(
        storage.GetOrCreate(ordinal, &DeviceSlot::Create<Tracked>)));
  }
  EXPECT_THAT(states, Not(Contains(nullptr)));
  EXPECT_EQ(CountDistinct(states), kNumOrdinals);
  for (int ordinal = 0; ordinal < kNumOrdinals; ++ordinal) {
    DeviceSlot* slot = storage.Find(ordinal);
    EXPECT_EQ(DeviceSlot::Unwrap<Tracked>(slot), states[ordinal])
        << "ordinal " << ordinal;
    EXPECT_TRUE(IsCacheLineAligned(slot)) << "ordinal " << ordinal;
  }
  EXPECT_EQ(Tracked::constructed.load(), kNumOrdinals);
}

TEST_F(CowStorageTest, DestroysEveryState) {
  {
    CowStorage storage;
    for (int ordinal = 0; ordinal < 3; ++ordinal) {
      storage.GetOrCreate(ordinal, &DeviceSlot::Create<Tracked>);
    }
    EXPECT_EQ(Tracked::destroyed.load(), 0);
  }
  EXPECT_EQ(Tracked::constructed.load(), 3);
  EXPECT_EQ(Tracked::destroyed.load(), 3);
}

TEST_F(CowStorageTest, ConcurrentGetOrCreateBuildsOneState) {
  constexpr int kNumThreads = 8;
  CowStorage storage;
  std::vector<Tracked*> states(kNumThreads, nullptr);
  absl::Notification start;
  {
    tsl::thread::ThreadPool pool(tsl::Env::Default(), "cow_storage_test",
                                 kNumThreads);
    for (int i = 0; i < kNumThreads; ++i) {
      pool.Schedule([&, i] {
        start.WaitForNotification();
        states[i] = DeviceSlot::Unwrap<Tracked>(
            storage.GetOrCreate(0, &DeviceSlot::Create<Tracked>));
      });
    }
    start.Notify();
  }  // Joins the pool.
  EXPECT_EQ(Tracked::constructed.load(), 1);
  EXPECT_NE(states[0], nullptr);
  EXPECT_THAT(states, Each(states[0]));
}

TEST_F(CowStorageTest, ConcurrentReadersSeeStableFullyBuiltStates) {
  constexpr int kNumOrdinals = 16;
  constexpr int kNumReaders = 4;
  CowStorage storage;
  std::vector<Tracked*> created(kNumOrdinals, nullptr);
  std::vector<std::vector<Tracked*>> seen(
      kNumReaders, std::vector<Tracked*>(kNumOrdinals, nullptr));
  absl::Notification start;
  std::atomic<bool> writers_done{false};
  std::atomic<int> errors{0};
  {
    tsl::thread::ThreadPool readers(tsl::Env::Default(), "readers",
                                    kNumReaders);
    for (int reader = 0; reader < kNumReaders; ++reader) {
      readers.Schedule([&, reader] {
        start.WaitForNotification();
        std::vector<Tracked*>& last_seen = seen[reader];
        bool done = false;
        while (!done) {
          // Loaded before the scan, so the last scan starts after all writers
          // have finished.
          done = writers_done.load();
          for (int ordinal = 0; ordinal < kNumOrdinals; ++ordinal) {
            Tracked* state = DeviceSlot::Unwrap<Tracked>(storage.Find(ordinal));
            if (state == nullptr) {
              continue;
            }
            // A published state is fully built and never replaced.
            if (state->value != Tracked::kInitialValue ||
                (last_seen[ordinal] != nullptr &&
                 last_seen[ordinal] != state)) {
              errors.fetch_add(1);
            }
            last_seen[ordinal] = state;
          }
        }
      });
    }
    {
      tsl::thread::ThreadPool writers(tsl::Env::Default(), "writers",
                                      kNumOrdinals);
      for (int ordinal = 0; ordinal < kNumOrdinals; ++ordinal) {
        writers.Schedule([&, ordinal] {
          start.WaitForNotification();
          created[ordinal] = DeviceSlot::Unwrap<Tracked>(
              storage.GetOrCreate(ordinal, &DeviceSlot::Create<Tracked>));
        });
      }
      start.Notify();
    }  // Joins the writers.
    writers_done.store(true);
  }  // Joins the readers.
  EXPECT_EQ(errors.load(), 0);
  EXPECT_EQ(Tracked::constructed.load(), kNumOrdinals);
  EXPECT_THAT(created, Not(Contains(nullptr)));
  for (int reader = 0; reader < kNumReaders; ++reader) {
    EXPECT_EQ(seen[reader], created) << "reader " << reader;
  }
}

// An insert holds the mutex while it builds the state. `Find` must not wait
// for it.
TEST_F(CowStorageTest, FindDoesNotWaitForInsert) {
  CowStorage storage;
  DeviceSlot* existing =
      storage.GetOrCreate(0, &DeviceSlot::Create<GatedState>);
  ASSERT_NE(existing, nullptr);

  absl::Notification entered;
  absl::Notification release;
  absl::Notification read_done;
  GatedState::entered.store(&entered);
  GatedState::gate.store(&release);

  bool insert_in_progress = false;
  bool read_finished = false;
  DeviceSlot* found = nullptr;
  DeviceSlot* uncommitted = nullptr;
  {
    tsl::thread::ThreadPool pool(tsl::Env::Default(), "cow_storage_test", 2);
    pool.Schedule(
        [&] { storage.GetOrCreate(1, &DeviceSlot::Create<GatedState>); });
    insert_in_progress =
        entered.WaitForNotificationWithTimeout(absl::Seconds(10));
    if (insert_in_progress) {
      pool.Schedule([&] {
        found = storage.Find(0);
        uncommitted = storage.Find(1);
        read_done.Notify();
      });
      read_finished =
          read_done.WaitForNotificationWithTimeout(absl::Seconds(10));
    }
    // Always unblock the insert, so the pool can join.
    release.Notify();
  }  // Joins the pool.
  GatedState::Reset();

  EXPECT_TRUE(insert_in_progress);
  EXPECT_TRUE(read_finished);
  EXPECT_EQ(found, existing);
  EXPECT_EQ(uncommitted, nullptr);
}

}  // namespace
}  // namespace xla::gpu
