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

#include "xla/backends/gpu/runtime/per_device_state.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/optimization.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/notification.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"

namespace xla::gpu {
namespace {

using absl_testing::IsOkAndHolds;
using absl_testing::StatusIs;
using ::testing::Contains;
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

template <typename T>
size_t CountDistinct(const std::vector<T*>& ptrs) {
  return absl::flat_hash_set<T*>(ptrs.begin(), ptrs.end()).size();
}

template <typename T>
size_t CountDistinctCacheLines(const std::vector<T*>& ptrs) {
  absl::flat_hash_set<uintptr_t> lines;
  for (const T* ptr : ptrs) {
    lines.insert(reinterpret_cast<uintptr_t>(ptr) / ABSL_CACHELINE_SIZE);
  }
  return lines.size();
}

class PerDeviceStateTest : public ::testing::Test {
 protected:
  void SetUp() override { Tracked::Reset(); }
};

TEST_F(PerDeviceStateTest, OrdinalsBelowNumDevicesExistAfterConstruction) {
  PerDeviceState<Tracked> storage(4);
  EXPECT_EQ(storage.num_device_slots(), 4);
  EXPECT_EQ(Tracked::constructed.load(), 4);
  for (int ordinal = 0; ordinal < 4; ++ordinal) {
    Tracked* state = storage.Find(ordinal);
    EXPECT_NE(state, nullptr) << "ordinal " << ordinal;
    EXPECT_THAT(storage.GetOrCreate(ordinal), IsOkAndHolds(state))
        << "ordinal " << ordinal;
  }
  EXPECT_EQ(Tracked::constructed.load(), 4);
}

TEST_F(PerDeviceStateTest, WithoutDevicesStatesAreBuiltOnFirstTouch) {
  for (int num_devices : {0, -1}) {
    SCOPED_TRACE(::testing::Message() << "num_devices " << num_devices);
    Tracked::Reset();
    PerDeviceState<Tracked> storage(num_devices);
    EXPECT_EQ(storage.num_device_slots(), 0);
    EXPECT_EQ(Tracked::constructed.load(), 0);
    EXPECT_EQ(storage.Find(0), nullptr);
    ASSERT_OK_AND_ASSIGN(Tracked * state, storage.GetOrCreate(0));
    EXPECT_NE(state, nullptr);
    EXPECT_EQ(storage.Find(0), state);
    EXPECT_EQ(Tracked::constructed.load(), 1);
  }
}

TEST_F(PerDeviceStateTest, OrdinalAtOrAboveNumDevicesIsBuiltOnFirstTouch) {
  PerDeviceState<Tracked> storage(2);
  EXPECT_EQ(storage.Find(2), nullptr);
  EXPECT_EQ(storage.Find(5), nullptr);
  ASSERT_OK_AND_ASSIGN(Tracked * state, storage.GetOrCreate(5));
  EXPECT_NE(state, nullptr);
  EXPECT_EQ(storage.Find(5), state);
  EXPECT_EQ(storage.Find(2), nullptr);
  EXPECT_EQ(Tracked::constructed.load(), 3);
}

TEST_F(PerDeviceStateTest, NegativeOrdinalIsRejected) {
  PerDeviceState<Tracked> storage(2);
  EXPECT_EQ(storage.Find(-1), nullptr);
  EXPECT_THAT(storage.GetOrCreate(-1),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_EQ(Tracked::constructed.load(), 2);
}

TEST_F(PerDeviceStateTest, StatesAreDistinctAcrossStorages) {
  // Ordinals 0 and 1 are built in the constructor, 2 and 3 on first touch.
  PerDeviceState<Tracked> storage(2);
  std::vector<Tracked*> states;
  for (int ordinal = 0; ordinal < 4; ++ordinal) {
    ASSERT_OK_AND_ASSIGN(Tracked * state, storage.GetOrCreate(ordinal));
    states.push_back(state);
  }
  EXPECT_THAT(states, Not(Contains(nullptr)));
  EXPECT_EQ(CountDistinct(states), 4);
  EXPECT_EQ(CountDistinctCacheLines(states), 4);
  for (int ordinal = 0; ordinal < 4; ++ordinal) {
    EXPECT_EQ(storage.Find(ordinal), states[ordinal]) << "ordinal " << ordinal;
  }
  EXPECT_EQ(Tracked::constructed.load(), 4);
}

TEST_F(PerDeviceStateTest, ConcurrentGetOrCreateAcrossStorages) {
  constexpr int kNumDevices = 4;
  constexpr int kNumOrdinals = 8;
  constexpr int kThreadsPerOrdinal = 4;
  constexpr int kNumThreads = kNumOrdinals * kThreadsPerOrdinal;
  PerDeviceState<Tracked> storage(kNumDevices);
  std::vector<absl::StatusOr<Tracked*>> results(kNumThreads);
  absl::Notification start;
  {
    tsl::thread::ThreadPool pool(tsl::Env::Default(), "per_device_state_test",
                                 kNumThreads);
    for (int i = 0; i < kNumThreads; ++i) {
      pool.Schedule([&, i] {
        start.WaitForNotification();
        results[i] = storage.GetOrCreate(i % kNumOrdinals);
      });
    }
    start.Notify();
  }  // Joins the pool.
  EXPECT_EQ(Tracked::constructed.load(), kNumOrdinals);
  for (int i = 0; i < kNumThreads; ++i) {
    Tracked* state = storage.Find(i % kNumOrdinals);
    EXPECT_NE(state, nullptr) << "thread " << i;
    EXPECT_THAT(results[i], IsOkAndHolds(state)) << "thread " << i;
  }
}

TEST_F(PerDeviceStateTest, GetOrCreateAndInitializeRunsOncePerOrdinal) {
  constexpr int kNumDevices = 2;
  constexpr int kNumOrdinals = 4;
  constexpr int kThreadsPerOrdinal = 4;
  constexpr int kNumThreads = kNumOrdinals * kThreadsPerOrdinal;
  PerDeviceState<Tracked> storage(kNumDevices);
  std::atomic<int> init_calls{0};
  std::vector<absl::Status> statuses(kNumThreads);
  absl::Notification start;
  {
    tsl::thread::ThreadPool pool(tsl::Env::Default(), "per_device_state_test",
                                 kNumThreads);
    for (int i = 0; i < kNumThreads; ++i) {
      pool.Schedule([&, i] {
        start.WaitForNotification();
        int ordinal = i % kNumOrdinals;
        statuses[i] =
            storage.GetOrCreateAndInitialize(ordinal, [&](Tracked* state) {
              init_calls.fetch_add(1);
              state->value = 100 + ordinal;
              return absl::OkStatus();
            });
      });
    }
    start.Notify();
  }  // Joins the pool.
  EXPECT_EQ(init_calls.load(), kNumOrdinals);
  for (int i = 0; i < kNumThreads; ++i) {
    EXPECT_THAT(statuses[i], absl_testing::IsOk()) << "thread " << i;
  }
  for (int ordinal = 0; ordinal < kNumOrdinals; ++ordinal) {
    Tracked* state = storage.Find(ordinal);
    ASSERT_NE(state, nullptr) << "ordinal " << ordinal;
    EXPECT_EQ(state->value, 100 + ordinal) << "ordinal " << ordinal;
  }
}

TEST_F(PerDeviceStateTest, GetOrCreateAndInitializeCachesErrorStatus) {
  PerDeviceState<Tracked> storage(2);
  int init_calls = 0;
  auto failing_init = [&](Tracked*) {
    ++init_calls;
    return absl::InternalError("init failed");
  };

  EXPECT_THAT(storage.GetOrCreateAndInitialize(0, failing_init),
              StatusIs(absl::StatusCode::kInternal));
  EXPECT_THAT(storage.GetOrCreateAndInitialize(0,
                                               [&](Tracked*) {
                                                 ++init_calls;
                                                 return absl::OkStatus();
                                               }),
              StatusIs(absl::StatusCode::kInternal));
  EXPECT_EQ(init_calls, 1);

  EXPECT_THAT(storage.GetOrCreateAndInitialize(-1, failing_init),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_EQ(init_calls, 1);
}

}  // namespace
}  // namespace xla::gpu
