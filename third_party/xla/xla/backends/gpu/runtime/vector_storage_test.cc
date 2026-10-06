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

#include "xla/backends/gpu/runtime/vector_storage.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/optimization.h"
#include "absl/container/flat_hash_set.h"
#include "xla/backends/gpu/runtime/device_slot.h"

namespace xla::gpu {
namespace {

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
std::vector<std::unique_ptr<DeviceSlot>> MakeSlots(int size) {
  std::vector<std::unique_ptr<DeviceSlot>> slots;
  slots.reserve(size);
  for (int i = 0; i < size; ++i) {
    slots.push_back(DeviceSlot::Create<T>());
  }
  return slots;
}

bool IsCacheLineAligned(const void* ptr) {
  return reinterpret_cast<uintptr_t>(ptr) % ABSL_CACHELINE_SIZE == 0;
}

template <typename T>
size_t CountDistinct(const std::vector<T*>& ptrs) {
  return absl::flat_hash_set<T*>(ptrs.begin(), ptrs.end()).size();
}

class VectorStorageTest : public ::testing::Test {
 protected:
  void SetUp() override { Tracked::Reset(); }
};

TEST_F(VectorStorageTest, StatesExistAfterConstruction) {
  {
    VectorStorage storage(MakeSlots<Tracked>(4));
    EXPECT_EQ(storage.size(), 4);
    EXPECT_EQ(Tracked::constructed.load(), 4);
    for (int ordinal = 0; ordinal < 4; ++ordinal) {
      Tracked* state = DeviceSlot::Unwrap<Tracked>(storage.Find(ordinal));
      ASSERT_NE(state, nullptr) << "ordinal " << ordinal;
      EXPECT_EQ(state->value, Tracked::kInitialValue) << "ordinal " << ordinal;
    }
  }
  EXPECT_EQ(Tracked::destroyed.load(), 4);
}

TEST_F(VectorStorageTest, FindOutsideRangeReturnsNull) {
  VectorStorage storage(MakeSlots<Tracked>(4));
  EXPECT_EQ(storage.Find(-1), nullptr);
  EXPECT_EQ(storage.Find(4), nullptr);
}

TEST_F(VectorStorageTest, EmptyStorageFindsNothing) {
  VectorStorage empty({});
  EXPECT_EQ(empty.size(), 0);
  EXPECT_EQ(empty.Find(0), nullptr);
  EXPECT_EQ(Tracked::constructed.load(), 0);
}

TEST_F(VectorStorageTest, StatesAreDistinctAndStable) {
  VectorStorage storage(MakeSlots<Tracked>(4));
  std::vector<Tracked*> states;
  for (int ordinal = 0; ordinal < 4; ++ordinal) {
    states.push_back(DeviceSlot::Unwrap<Tracked>(storage.Find(ordinal)));
  }
  EXPECT_THAT(states, Not(Contains(nullptr)));
  EXPECT_EQ(CountDistinct(states), 4);
  for (int ordinal = 0; ordinal < 4; ++ordinal) {
    EXPECT_EQ(DeviceSlot::Unwrap<Tracked>(storage.Find(ordinal)),
              states[ordinal])
        << "ordinal " << ordinal;
  }
}

TEST_F(VectorStorageTest, StatesAreCacheLineAligned) {
  VectorStorage storage(MakeSlots<Tracked>(4));
  for (int ordinal = 0; ordinal < 4; ++ordinal) {
    DeviceSlot* slot = storage.Find(ordinal);
    EXPECT_NE(slot, nullptr) << "ordinal " << ordinal;
    EXPECT_TRUE(IsCacheLineAligned(slot)) << "ordinal " << ordinal;
  }
}

TEST_F(VectorStorageTest, ForEachVisitsAllSlots) {
  VectorStorage empty({});
  int empty_visits = 0;
  empty.ForEach([&](DeviceSlot&) { ++empty_visits; });
  EXPECT_EQ(empty_visits, 0);

  VectorStorage storage(MakeSlots<Tracked>(4));
  std::vector<DeviceSlot*> visited;
  storage.ForEach([&](DeviceSlot& slot) { visited.push_back(&slot); });
  ASSERT_EQ(visited.size(), 4);
  for (int ordinal = 0; ordinal < 4; ++ordinal) {
    EXPECT_EQ(visited[ordinal], storage.Find(ordinal));
  }
}

}  // namespace
}  // namespace xla::gpu
