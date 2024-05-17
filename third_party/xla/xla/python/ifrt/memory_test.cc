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

#include "xla/python/ifrt/memory.h"

#include <memory>
#include <optional>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using ::testing::Optional;

namespace xla {
namespace ifrt {
namespace {

TEST(MemoryKindTest, EqualityForUnspecified) {
  MemoryKind memory_kind1;
  MemoryKind memory_kind2;
  EXPECT_EQ(memory_kind1, memory_kind2);
}

TEST(MemoryKindTest, EqualityForSameString) {
  MemoryKind memory_kind1("abc");
  MemoryKind memory_kind2("abc");
  EXPECT_EQ(memory_kind1, memory_kind2);
}

TEST(MemoryKindTest, EqualityForSameStringContent) {
  MemoryKind memory_kind1("abc");
  MemoryKind memory_kind2(absl::StrCat("ab", "c"));
  EXPECT_EQ(memory_kind1, memory_kind2);
}

TEST(MemoryKindTest, InequalityForDifferentStringContent) {
  MemoryKind memory_kind1("abc");
  MemoryKind memory_kind2("def");
  EXPECT_NE(memory_kind1, memory_kind2);
}

TEST(MemoryKindTest, InequalityBetweenSpecifiedAndUnspecified) {
  {
    MemoryKind memory_kind1("abc");
    MemoryKind memory_kind2;
    EXPECT_NE(memory_kind1, memory_kind2);
  }
  {
    MemoryKind memory_kind1;
    MemoryKind memory_kind2("abc");
    EXPECT_NE(memory_kind1, memory_kind2);
  }
}

TEST(MemoryKindTest, MemorySafety) {
  auto memory_kind_str = std::make_unique<std::string>("abc");
  MemoryKind memory_kind(*memory_kind_str);

  memory_kind_str.reset();
  EXPECT_THAT(memory_kind.memory_kind(), Optional(absl::string_view("abc")));
}

TEST(MemoryKindTest, EqualityForUnspecifiedAndNullopt) {
  MemoryKind memory_kind1;
  MemoryKind memory_kind2(std::nullopt);
  EXPECT_EQ(memory_kind1, memory_kind2);
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
