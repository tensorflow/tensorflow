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
#include "tensorflow/core/tfrt/mlrt/interpreter/value.h"

#include <utility>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"

namespace mlrt {
namespace {

TEST(ValueTest, SmallCopyable) {
  struct SmallCopyable {
    int v;
  };

  Value value(SmallCopyable{100});
  EXPECT_EQ(value.Get<SmallCopyable>().v, 100);

  Value value_copy(value);
  EXPECT_EQ(value_copy.Get<SmallCopyable>().v, 100);
  EXPECT_EQ(value.Get<SmallCopyable>().v, 100);

  Value value_move = std::move(value);
  EXPECT_EQ(value_move.Get<SmallCopyable>().v, 100);
  EXPECT_FALSE(value.HasValue());  // NOLINT

  ASSERT_TRUE(value_move.HasValue());
  value_move.Destroy<SmallCopyable>();
  EXPECT_FALSE(value_move.HasValue());

  value_move = SmallCopyable{100};
  EXPECT_EQ(value_move.Get<SmallCopyable>().v, 100);
}

TEST(ValueTest, LargeCopyable) {
  constexpr char kData[] =
      "<<This line contains 32 bytes>>\n"
      "<<This line contains 32 bytes>>\n"
      "<<This line contains 32 bytes>>\n"
      "<<This line contains 32 bytes>>";

  static_assert(sizeof(kData) == 128);

  struct LargeCopyable {
    char data[128] =
        "<<This line contains 32 bytes>>\n"
        "<<This line contains 32 bytes>>\n"
        "<<This line contains 32 bytes>>\n"
        "<<This line contains 32 bytes>>";
  };

  Value value(LargeCopyable{});
  EXPECT_EQ(absl::string_view(value.Get<LargeCopyable>().data), kData);

  Value value_copy = value;
  EXPECT_EQ(absl::string_view(value_copy.Get<LargeCopyable>().data), kData);
  EXPECT_EQ(absl::string_view(value.Get<LargeCopyable>().data), kData);

  Value value_move = std::move(value);
  EXPECT_EQ(absl::string_view(value_move.Get<LargeCopyable>().data), kData);
  EXPECT_FALSE(value.HasValue());  // NOLINT

  ASSERT_TRUE(value_move.HasValue());
  value_move.Destroy<LargeCopyable>();
  EXPECT_FALSE(value_move.HasValue());

  value_move = LargeCopyable{};
  EXPECT_EQ(absl::string_view(value_move.Get<LargeCopyable>().data), kData);
}

TEST(ValueTest, SmallMoveOnly) {
  struct SmallMoveOnly {
    int v;

    explicit SmallMoveOnly(int v) : v(v) {}
    SmallMoveOnly(const SmallMoveOnly&) = delete;
    SmallMoveOnly& operator=(const SmallMoveOnly&) = delete;
    SmallMoveOnly(SmallMoveOnly&&) = default;
    SmallMoveOnly& operator=(SmallMoveOnly&&) = default;
  };

  Value value(SmallMoveOnly(100));
  EXPECT_EQ(value.Get<SmallMoveOnly>().v, 100);

  Value value_move = std::move(value);
  EXPECT_EQ(value_move.Get<SmallMoveOnly>().v, 100);
  EXPECT_FALSE(value.HasValue());  // NOLINT
}

TEST(ValueTest, LargeMoveOnly) {
  constexpr char kData[] =
      "<<This line contains 32 bytes>>\n"
      "<<This line contains 32 bytes>>\n"
      "<<This line contains 32 bytes>>\n"
      "<<This line contains 32 bytes>>";

  static_assert(sizeof(kData) == 128);

  struct LargeMoveOnly {
    char data[128] =
        "<<This line contains 32 bytes>>\n"
        "<<This line contains 32 bytes>>\n"
        "<<This line contains 32 bytes>>\n"
        "<<This line contains 32 bytes>>";

    LargeMoveOnly() = default;
    LargeMoveOnly(const LargeMoveOnly&) = delete;
    LargeMoveOnly& operator=(const LargeMoveOnly&) = delete;
    LargeMoveOnly(LargeMoveOnly&&) = default;
    LargeMoveOnly& operator=(LargeMoveOnly&&) = default;
  };

  Value value(LargeMoveOnly{});
  EXPECT_EQ(absl::string_view(value.Get<LargeMoveOnly>().data), kData);

  Value value_move = std::move(value);
  EXPECT_EQ(absl::string_view(value_move.Get<LargeMoveOnly>().data), kData);
  EXPECT_FALSE(value.HasValue());  // NOLINT
}

TEST(ValueTest, Error) {
  Value arg(100);

  arg.HandleError(arg);

  EXPECT_EQ(arg.Get<int>(), 100);

  struct Small {
    int* v = nullptr;

    void HandleError(Value* arg) { *v = arg->Get<int>(); }
  };

  int v = 0;

  Value value(Small{&v});

  value.HandleError(arg);

  EXPECT_EQ(v, 100);
  EXPECT_EQ(*value.Get<Small>().v, 100);

  struct Large {
    int* v = nullptr;
    char data[128];

    void HandleError(Value* arg) { *v = arg->Get<int>(); }
  };

  v = 0;

  value = Value(Large{&v});

  value.HandleError(arg);

  EXPECT_EQ(v, 100);
  EXPECT_EQ(*value.Get<Large>().v, 100);
}

}  // namespace
}  // namespace mlrt
