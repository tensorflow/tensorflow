/* Copyright 2023 The OpenXLA Authors.

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
#include "xla/hlo/ir/ptrvec.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <utility>
#include <vector>

#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/test_benchmark.h"

namespace xla {
namespace {

class PtrVecTest : public testing::Test {
 public:
  int* NewInt(int v) {
    ints_.push_back(std::make_unique<int>(v));
    return ints_.back().get();
  }

  void Fill(PtrVec<int*>& dst, absl::Span<const int> src) {
    for (int v : src) {
      dst.push_back(NewInt(v));
    }
  }

  std::vector<int> Pointees(const PtrVec<int*>& src) {
    std::vector<int> result;
    result.reserve(src.size());
    for (int* ptr : src) {
      result.push_back(*ptr);
    }
    return result;
  }

 private:
  // Underlying storage for pointers stored in PtrVec<>.
  std::vector<std::unique_ptr<int>> ints_;
};

// Some useful vectors to test with.
std::vector<std::vector<int>> TestCases() {
  return std::vector<std::vector<int>>{
      {},
      {100},
      {200, 300},
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
  };
}

TEST_F(PtrVecTest, Accessors) {
  for (const auto& c : TestCases()) {
    SCOPED_TRACE(c.size());
    PtrVec<int*> v;
    Fill(v, c);
    ASSERT_EQ(v.empty(), c.empty());
    ASSERT_EQ(v.size(), c.size());
    if (!c.empty()) {
      ASSERT_EQ(*v.front(), c.front());
      ASSERT_EQ(*v.back(), c.back());
    }
  }
}

TEST_F(PtrVecTest, ConstIteration) {
  for (const auto& c : TestCases()) {
    SCOPED_TRACE(c.size());
    PtrVec<int*> v;
    Fill(v, c);
    int i = 0;
    const PtrVec<int*>& const_v = v;
    for (int* ptr : const_v) {
      ASSERT_EQ(*ptr, c[i]);
      i++;
    }
  }
}

TEST_F(PtrVecTest, NonConstIteration) {
  int other_value = -1;
  int* other_ptr = &other_value;
  for (const auto& c : TestCases()) {
    SCOPED_TRACE(c.size());
    PtrVec<int*> v;
    Fill(v, c);
    int i = 0;
    for (int*& slot : v) {
      ASSERT_EQ(*slot, c[i]);
      slot = other_ptr;
      i++;
    }

    for (int* ptr : v) {
      ASSERT_EQ(ptr, other_ptr);
      ASSERT_EQ(*ptr, other_value);
    }
  }
}

TEST_F(PtrVecTest, ReverseIteration) {
  for (const auto& c : TestCases()) {
    SCOPED_TRACE(c.size());
    PtrVec<int*> v;
    Fill(v, c);
    int i = c.size();
    for (auto it = v.rbegin(); it != v.rend(); ++it) {
      i--;
      int* ptr = *it;
      ASSERT_EQ(*ptr, c[i]);
    }
  }
}

TEST_F(PtrVecTest, Indexing) {
  for (const auto& c : TestCases()) {
    SCOPED_TRACE(c.size());
    PtrVec<int*> v;
    Fill(v, c);
    for (int i = 0; i < c.size(); i++) {
      ASSERT_EQ(*v[i], c[i]);
      ASSERT_EQ(*v.at(i), c[i]);
    }
  }
}

TEST_F(PtrVecTest, Data) {
  for (const auto& c : TestCases()) {
    SCOPED_TRACE(c.size());
    PtrVec<int*> v;
    Fill(v, c);
    int** data = v.data();
    for (int i = 0; i < c.size(); i++) {
      ASSERT_EQ(*data[i], c[i]);
    }
  }
}

TEST_F(PtrVecTest, ConversionToVector) {
  for (const auto& c : TestCases()) {
    SCOPED_TRACE(c.size());
    PtrVec<int*> v;
    Fill(v, c);
    std::vector<int*> vec = v;
    ASSERT_EQ(vec.size(), c.size());
    for (int i = 0; i < c.size(); i++) {
      ASSERT_EQ(*vec[i], c[i]);
    }
  }
}

TEST_F(PtrVecTest, Clear) {
  for (const auto& c : TestCases()) {
    SCOPED_TRACE(c.size());
    PtrVec<int*> v;
    Fill(v, c);
    v.clear();
    EXPECT_EQ(Pointees(v), std::vector<int>{});
  }
}

TEST_F(PtrVecTest, PopBack) {
  for (const auto& c : TestCases()) {
    SCOPED_TRACE(c.size());
    PtrVec<int*> v;
    Fill(v, c);
    auto model = c;
    while (!model.empty()) {
      model.pop_back();
      v.pop_back();
      EXPECT_EQ(Pointees(v), model);
    }
  }
}

TEST_F(PtrVecTest, Erase) {
  for (const auto& c : TestCases()) {
    if (c.empty()) {
      continue;
    }
    SCOPED_TRACE(c.size());
    PtrVec<int*> v;
    Fill(v, c);
    auto model = c;
    int offset = c.size() / 2;
    model.erase(model.begin() + offset);
    v.erase(v.begin() + offset);
    EXPECT_EQ(Pointees(v), model);
  }
}

TEST_F(PtrVecTest, EraseToEmpty) {
  for (const auto& c : TestCases()) {
    SCOPED_TRACE(c.size());
    PtrVec<int*> v;
    Fill(v, c);
    for (int i = 0; i < c.size(); i++) {
      v.erase(v.begin());
    }
    EXPECT_TRUE(v.empty());
    EXPECT_EQ(v.size(), 0);
  }
}

TEST_F(PtrVecTest, MoveAndCopy) {
  const auto cases = TestCases();
  for (const auto& x : cases) {
    for (const auto& y : cases) {
      SCOPED_TRACE(absl::StrFormat("from %d to %d", x.size(), y.size()));

      // Copy construct
      {
        PtrVec<int*> b;
        Fill(b, y);
        PtrVec<int*> a = b;
        ASSERT_EQ(Pointees(a), y);
      }

      // Move construct
      {
        PtrVec<int*> b;
        Fill(b, y);
        PtrVec<int*> a = std::move(b);
        ASSERT_EQ(Pointees(a), y);
        // NOLINTNEXTLINE(bugprone-use-after-move)
        ASSERT_EQ(Pointees(b), std::vector<int>{});
      }

      // Copy
      {
        PtrVec<int*> a;
        Fill(a, x);
        ASSERT_EQ(Pointees(a), x);
        PtrVec<int*> b;
        Fill(b, y);
        a = b;
        ASSERT_EQ(Pointees(a), y);
      }

      // Move
      {
        PtrVec<int*> a;
        Fill(a, x);
        PtrVec<int*> b;
        Fill(b, y);
        a = std::move(b);
        ASSERT_EQ(Pointees(a), y);
        // NOLINTNEXTLINE(bugprone-use-after-move)
        ASSERT_EQ(Pointees(b), std::vector<int>{});
      }
    }
  }
}

TEST_F(PtrVecTest, SelfAssign) {
  for (const auto& c : TestCases()) {
    SCOPED_TRACE(c.size());
    PtrVec<int*> a;
    Fill(a, c);
    a = a;
    ASSERT_EQ(Pointees(a), c);
  }
}

TEST_F(PtrVecTest, InitializerList) {
  PtrVec<int*> src;
  Fill(src, {0, 1});

  EXPECT_EQ(Pointees(PtrVec<int*>({})), std::vector<int>({}));
  EXPECT_EQ(Pointees(PtrVec<int*>({src[0]})), std::vector<int>({0}));
  EXPECT_EQ(Pointees(PtrVec<int*>({src[0], src[1]})), std::vector<int>({0, 1}));
}

TEST_F(PtrVecTest, IterConstruct) {
  for (const auto& c : TestCases()) {
    SCOPED_TRACE(c.size());
    PtrVec<int*> a;
    Fill(a, c);
    PtrVec<int*> b(a.begin(), a.end());
    EXPECT_EQ(Pointees(b), c);
  }
}

TEST_F(PtrVecTest, IterAssign) {
  for (const auto& x : TestCases()) {
    for (const auto& y : TestCases()) {
      SCOPED_TRACE(absl::StrFormat("from %d to %d", x.size(), y.size()));
      PtrVec<int*> a;
      Fill(a, x);
      PtrVec<int*> b;
      Fill(b, y);

      b.assign(a.begin(), a.end());
      EXPECT_EQ(Pointees(b), x);
    }
  }
}

TEST_F(PtrVecTest, Capacity) {
  const auto cases = TestCases();
  for (const auto& c : cases) {
    SCOPED_TRACE(c.size());
    PtrVec<int*> v;
    Fill(v, c);
    ASSERT_GE(v.capacity(), v.size());

    // Must have inlined rep for 0 or 1
    if (c.size() <= 1) {
      ASSERT_EQ(v.capacity(), 1);
    }
  }
}

TEST_F(PtrVecTest, Reserve) {
  const auto cases = TestCases();
  for (const auto& c : cases) {
    SCOPED_TRACE(c.size());
    PtrVec<int*> v;
    Fill(v, c);

    // Growing within current capacity should leave data() unchanged.
    const auto* data = v.data();
    for (size_t new_capacity = 0; new_capacity <= v.capacity();
         new_capacity++) {
      v.reserve(new_capacity);
      ASSERT_EQ(data, v.data()) << new_capacity;
    }

    // Now try growing over existing capacity.
    for (size_t increment : std::vector<size_t>({1, 100})) {
      v.reserve(v.capacity() + increment);
      ASSERT_NE(data, v.data()) << increment;
      data = v.data();
    }

    // Check that contents are unchanged.
    int i = 0;
    for (int* ptr : v) {
      ASSERT_EQ(*ptr, c[i]);
      i++;
    }
  }
}

TEST_F(PtrVecTest, ShrinkToEmpty) {
  const auto cases = TestCases();
  for (const auto& c : cases) {
    SCOPED_TRACE(c.size());
    PtrVec<int*> v;
    Fill(v, c);
    v.resize(0);
    EXPECT_EQ(Pointees(v), std::vector<int>{});
  }
}

TEST_F(PtrVecTest, ShrinkViaResize) {
  const auto cases = TestCases();
  for (const auto& c : cases) {
    SCOPED_TRACE(c.size());
    PtrVec<int*> v;
    Fill(v, c);

    if (v.empty()) {
      continue;  // Cannot shrink empty vector
    }

    v.resize(v.size() - 1);
    EXPECT_EQ(v.size(), c.size() - 1);

    auto model = c;
    model.pop_back();
    EXPECT_EQ(Pointees(v), model);
  }
}

TEST_F(PtrVecTest, Grow) {
  int other_value = -1;
  int* other_ptr = &other_value;

  const auto cases = TestCases();
  for (const auto& c : cases) {
    for (size_t growth : std::vector<size_t>({0, 1, 100, c.size() * 2})) {
      SCOPED_TRACE(absl::StrFormat("grow from %d by %d", c.size(), growth));
      PtrVec<int*> v;
      Fill(v, c);

      v.resize(v.size() + growth, other_ptr);
      ASSERT_EQ(v.size(), c.size() + growth);

      // Should have old values up to original size and other_ptr afterwards.
      for (size_t i = 0; i < c.size(); i++) {
        EXPECT_EQ(*v[i], c[i]);
      }
      for (size_t i = 0; i < growth; i++) {
        EXPECT_EQ(*v[c.size() + i], other_value);
      }
    }
  }
}

TEST_F(PtrVecTest, ReducedAlignment) {
  const char* str = "hello world";
  for (int i = 0; i < 11; i++) {
    PtrVec<const char*> vec;
    vec.push_back(&str[i]);
    EXPECT_EQ(vec.size(), 1);
    EXPECT_EQ(vec[0], &str[i]);

    PtrVec<const char*> copy;
    copy = vec;
    EXPECT_EQ(copy.size(), 1);
    EXPECT_EQ(copy[0], &str[i]);
  }
}

TEST_F(PtrVecTest, ReducedAlignmentOverwrite) {
  // Get a char pointer with bottom two bits zero.
  char buf[100];
  char* ptr = buf;
  while ((reinterpret_cast<uintptr_t>(ptr) & 0x3) != 0) {
    ptr++;
  }

  PtrVec<char*> vec;
  vec.push_back(ptr);
  EXPECT_EQ(vec.size(), 1);

  // Try overwriting with differently aligned pointers.
  for (int addition = 0; addition <= 3; addition++) {
    vec[0] = ptr + addition;
    EXPECT_EQ(vec.size(), 1);
    EXPECT_EQ(vec[0], ptr + addition);
  }
}

struct Elem {
  int64_t number;
};

void BM_PtrVecIter(::testing::benchmark::State& state) {
  const int n = state.range(0);
  std::vector<Elem> storage(n);
  PtrVec<Elem*> vec;
  for (int i = 0; i < n; i++) {
    storage[i].number = i;
    vec.push_back(&storage[i]);
  }

  uintptr_t sum = 0;
  for (auto s : state) {
    for (int i = 0; i < vec.size(); i++) {
      sum += reinterpret_cast<uintptr_t>(vec[i]);
    }
  }
  VLOG(1) << sum;
}
BENCHMARK(BM_PtrVecIter)->Arg(0)->Arg(1)->Arg(2)->Arg(4)->Arg(8)->Arg(1024);

void BM_StdVecIter(::testing::benchmark::State& state) {
  const int n = state.range(0);
  std::vector<Elem> storage(n);
  std::vector<Elem*> vec;
  for (int i = 0; i < n; i++) {
    storage[i].number = i;
    vec.push_back(&storage[i]);
  }

  uintptr_t sum = 0;
  for (auto s : state) {
    for (int i = 0; i < vec.size(); i++) {
      sum += reinterpret_cast<uintptr_t>(vec[i]);
    }
  }
  VLOG(1) << sum;
}
BENCHMARK(BM_StdVecIter)->Arg(0)->Arg(1)->Arg(2)->Arg(4)->Arg(8)->Arg(1024);

}  // namespace
}  // namespace xla
