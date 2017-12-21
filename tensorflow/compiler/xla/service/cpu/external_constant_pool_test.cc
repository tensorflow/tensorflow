/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/cpu/external_constant_pool.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace cpu {
namespace {
class ExternalConstantPoolTest : public ::testing::Test {};

template <typename T>
T GetFromBuffer(const uint8* buffer, int64 index) {
  T result;
  std::memcpy(&result, buffer + index * sizeof(T), sizeof(T));
  return result;
}

TEST(ExternalConstantPoolTest, Basic) {
  ExternalConstantPool constant_pool;
  EXPECT_EQ(constant_pool.Find("name-0"), nullptr);
  const auto literal = Literal::CreateR2({{1, 2}, {3, 4}});
  constant_pool.Insert("name-0", *literal, 4);
  const uint8* constant = constant_pool.Find("name-0");
  ASSERT_NE(constant, nullptr);

  EXPECT_EQ(GetFromBuffer<int32>(constant, 0), 1);
  EXPECT_EQ(GetFromBuffer<int32>(constant, 1), 2);
  EXPECT_EQ(GetFromBuffer<int32>(constant, 2), 3);
  EXPECT_EQ(GetFromBuffer<int32>(constant, 3), 4);

  EXPECT_EQ(constant_pool.Find("name-1"), nullptr);
}

TEST(ExternalConstantPoolTest, RowMinorLayout) {
  ExternalConstantPool constant_pool;
  EXPECT_EQ(constant_pool.Find("name-0"), nullptr);
  const auto literal = Literal::CreateR2WithLayout(
      {{1, 2}, {3, 4}}, LayoutUtil::MakeLayout({0, 1}));
  constant_pool.Insert("name-0", *literal, 4);
  const uint8* constant = constant_pool.Find("name-0");
  ASSERT_NE(constant, nullptr);

  EXPECT_EQ(GetFromBuffer<int32>(constant, 0), 1);
  EXPECT_EQ(GetFromBuffer<int32>(constant, 1), 3);
  EXPECT_EQ(GetFromBuffer<int32>(constant, 2), 2);
  EXPECT_EQ(GetFromBuffer<int32>(constant, 3), 4);
}

TEST(ExternalConstantPoolTest, Alignment) {
  ExternalConstantPool constant_pool;
  EXPECT_EQ(constant_pool.Find("name-0"), nullptr);

  for (int i = 0; i < 8; i++) {
    int64 alignment = 1 << i;
    string name = tensorflow::strings::StrCat("name-", i);

    const auto literal = Literal::CreateR2({{1, 2}, {3, 4}});
    constant_pool.Insert(name, *literal, alignment);

    const uint8* constant = constant_pool.Find(name);
    ASSERT_NE(constant, nullptr);
    EXPECT_EQ(reinterpret_cast<intptr_t>(constant) % alignment, 0);
  }
}

}  // namespace
}  // namespace cpu
}  // namespace xla
