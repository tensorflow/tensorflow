/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/tfrt/mlrt/bytecode/bytecode.h"

#include <array>
#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"

namespace mlrt {
namespace bc {
namespace {

TEST(ByteCodeTest, VectorOfTrivial) {
  Buffer buffer;
  Allocator alloc(&buffer);

  auto ctor = New<Vector<uint32_t>>(&alloc, /*size=*/4);

  for (int i = 0; i < 4; ++i) {
    ctor.ConstructAt(i, i);
  }

  Vector<uint32_t> view(buffer.Get(ctor.address()));

  ASSERT_EQ(view.size(), 4);
  EXPECT_EQ(view[0], 0);
  EXPECT_EQ(view[1], 1);
  EXPECT_EQ(view[2], 2);
  EXPECT_EQ(view[3], 3);

  EXPECT_THAT(view, ::testing::ElementsAreArray({0, 1, 2, 3}));

  Vector<uint32_t> empty;
  ASSERT_TRUE(empty.empty());
}

TEST(ByteCodeTest, VectorOfVector) {
  Buffer buffer;
  Allocator alloc(&buffer);

  using T = Vector<uint32_t>;
  using V = Vector<T>;

  auto vctor = New<V>(&alloc, 3);

  {
    auto tctor = vctor.ConstructAt(0, 2);
    tctor.ConstructAt(0, 0);
    tctor.ConstructAt(1, 1);
  }

  {
    auto tctor = vctor.ConstructAt(1, 1);
    tctor.ConstructAt(0, 2);
  }

  vctor.ConstructAt(2, 0);

  V v(buffer.Get(vctor.address()));

  auto t0 = v[0];
  ASSERT_EQ(t0.size(), 2);
  EXPECT_EQ(t0[0], 0);
  EXPECT_EQ(t0[1], 1);
  EXPECT_THAT(t0, testing::ElementsAreArray({0, 1}));

  auto t1 = v[1];
  ASSERT_EQ(t1.size(), 1);
  EXPECT_EQ(t1[0], 2);
  EXPECT_THAT(t1, testing::ElementsAreArray({2}));

  auto t2 = v[2];
  ASSERT_EQ(t2.size(), 0);

  Vector<Vector<uint32_t>> empty;
  ASSERT_TRUE(empty.empty());
}

TEST(ByteCodeTest, String) {
  Buffer buffer;
  Allocator alloc(&buffer);

  auto ctor = New<String>(&alloc, "bytecode string");

  String view(buffer.Get(ctor.address()));

  EXPECT_EQ(view.str(), "bytecode string");
  EXPECT_EQ(view.Get(), "bytecode string");
  EXPECT_EQ(absl::string_view(view), "bytecode string");
}

TEST(ByteCodeTest, PlaceVectorOfTrivial) {
  Buffer buffer;
  Allocator alloc(&buffer);

  auto ctor = New<Vector<uint32_t>>(&alloc, /*size=*/4);

  std::array<uint32_t, 4> data = {0, 1, 2, 3};

  ctor.Place(reinterpret_cast<const char*>(data.data()),
             data.size() * sizeof(uint32_t));

  Vector<uint32_t> view(buffer.Get(ctor.address()));

  ASSERT_EQ(view.size(), 4);
  EXPECT_EQ(view[0], 0);
  EXPECT_EQ(view[1], 1);
  EXPECT_EQ(view[2], 2);
  EXPECT_EQ(view[3], 3);

  EXPECT_THAT(view, ::testing::ElementsAreArray({0, 1, 2, 3}));
}

TEST(ByteCodeTest, ReadIteratorDistance) {
  Buffer buffer;
  Allocator alloc(&buffer);

  auto ctor = New<Vector<uint32_t>>(&alloc, /*size=*/4);

  for (int i = 0; i < 4; ++i) {
    ctor.ConstructAt(i, i);
  }

  Vector<uint32_t> view(buffer.Get(ctor.address()));

  EXPECT_EQ(view.end() - view.begin(), 4);
}

TEST(ByteCodeTest, ReadIteratorCompare) {
  Buffer buffer;
  Allocator alloc(&buffer);

  auto ctor = New<Vector<uint32_t>>(&alloc, /*size=*/4);

  for (int i = 0; i < 4; ++i) {
    ctor.ConstructAt(i, i);
  }

  Vector<uint32_t> view(buffer.Get(ctor.address()));

  EXPECT_GE(view.end(), view.begin());
  EXPECT_GT(view.end(), view.begin());
  EXPECT_LE(view.begin(), view.end());
  EXPECT_LT(view.begin(), view.end());
}

}  // namespace
}  // namespace bc
}  // namespace mlrt
