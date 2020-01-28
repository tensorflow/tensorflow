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

#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/types.h"

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
  std::shared_ptr<int> a = m[0];
  std::shared_ptr<int> b = m[0];
  std::shared_ptr<int> c = m[1];
  EXPECT_EQ(a.get(), b.get());
  EXPECT_NE(a.get(), c.get());
}

TEST(RefcountingHashMapTest, DefaultInitialized) {
  RefcountingHashMap<int, int> m;
  EXPECT_EQ(*m[42], 0);
}

TEST(RefcountingHashMapTest, DeletesEagerly) {
  RefcountingHashMap<int, DeleteNotifier> m;
  bool deleted = false;
  auto handle = m[0];
  handle->fn = [&] { deleted = true; };
  EXPECT_FALSE(deleted);
  handle = nullptr;
  EXPECT_TRUE(deleted);
}

TEST(RefcountingHashMapTest, CustomFactory) {
  RefcountingHashMap<int, int> m(
      [](const int& x) { return absl::make_unique<int>(x + 1); });
  EXPECT_EQ(*m[0], 1);
  EXPECT_EQ(*m[100], 101);
}

TEST(RefcountingHashMapTest, ForEachEmpty) {
  RefcountingHashMap<int, int> m;
  int64 count = 0;
  m.ForEach([&](const int&, std::shared_ptr<int>) { ++count; });
  EXPECT_EQ(count, 0);
}

TEST(RefcountingHashMapTest, ForEachNonempty) {
  RefcountingHashMap<int, int> m;
  auto a = m[0];
  auto b = m[1];

  std::vector<int> seen_keys;
  std::vector<int*> seen_values;
  m.ForEach([&](const int& k, std::shared_ptr<int> v) {
    seen_keys.push_back(k);
    seen_values.push_back(v.get());
  });
  EXPECT_THAT(seen_keys, testing::UnorderedElementsAre(0, 1));
  EXPECT_THAT(seen_values, testing::UnorderedElementsAre(a.get(), b.get()));
}

}  // anonymous namespace
}  // namespace xla
