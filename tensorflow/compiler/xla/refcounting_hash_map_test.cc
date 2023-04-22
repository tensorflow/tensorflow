/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/refcounting_hash_map.h"

#include <functional>
#include <memory>

#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/errors.h"

namespace xla {
namespace {

struct DeleteNotifier {
  DeleteNotifier() = default;
  DeleteNotifier(const DeleteNotifier&) = delete;
  DeleteNotifier& operator=(const DeleteNotifier&) = delete;
  DeleteNotifier(DeleteNotifier&& o) noexcept : fn(std::move(o.fn)) {
    o.fn = nullptr;
  }
  DeleteNotifier& operator=(DeleteNotifier&& o) noexcept {
    fn = o.fn;
    o.fn = nullptr;
    return *this;
  }

  ~DeleteNotifier() {
    if (fn) {
      fn();
    }
  }

  std::function<void()> fn;
};

TEST(RefcountingHashMapTest, PointerIdentity) {
  RefcountingHashMap<int, int> m;
  auto factory = [](const int) { return absl::make_unique<int>(); };
  std::shared_ptr<int> a = m.GetOrCreateIfAbsent(0, factory);
  std::shared_ptr<int> b = m.GetOrCreateIfAbsent(0, factory);
  std::shared_ptr<int> c = m.GetOrCreateIfAbsent(1, factory);
  EXPECT_EQ(a.get(), b.get());
  EXPECT_NE(a.get(), c.get());
}

TEST(RefcountingHashMapTest, DefaultInitialized) {
  RefcountingHashMap<int, int> m;
  auto factory = [](const int) { return absl::make_unique<int>(); };
  EXPECT_EQ(*m.GetOrCreateIfAbsent(42, factory), 0);
}

TEST(RefcountingHashMapTest, DeletesEagerly) {
  RefcountingHashMap<int, DeleteNotifier> m;
  bool deleted = false;
  auto factory = [](const int) { return absl::make_unique<DeleteNotifier>(); };
  auto handle = m.GetOrCreateIfAbsent(0, factory);
  handle->fn = [&] { deleted = true; };
  EXPECT_FALSE(deleted);
  handle = nullptr;
  EXPECT_TRUE(deleted);
}

TEST(RefcountingHashMapTest, CustomFactory) {
  RefcountingHashMap<int, int> m;
  auto factory = [](const int x) { return absl::make_unique<int>(x + 1); };
  EXPECT_EQ(*m.GetOrCreateIfAbsent(0, factory), 1);
  EXPECT_EQ(*m.GetOrCreateIfAbsent(100, factory), 101);
}

}  // anonymous namespace
}  // namespace xla
