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
#include "absl/hash/hash.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

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

TEST(MemoryKindTest, DefaultMemoryKindIsDevice) {
  MemoryKind default_memory_kind;
  MemoryKind device_memory_kind("device");
  MemoryKind nullopt_memory_kind(std::nullopt);

  EXPECT_TRUE(default_memory_kind.is_default());
  EXPECT_TRUE(device_memory_kind.is_default());
  EXPECT_TRUE(nullopt_memory_kind.is_default());

  EXPECT_EQ(default_memory_kind, device_memory_kind);
  EXPECT_EQ(nullopt_memory_kind, device_memory_kind);

  EXPECT_EQ(default_memory_kind.memory_kind(), "device");
  EXPECT_EQ(absl::StrCat(default_memory_kind), "device");
  EXPECT_EQ(absl::Hash<MemoryKind>()(default_memory_kind),
            absl::Hash<MemoryKind>()(device_memory_kind));
}

TEST(MemoryKindTest, EmptyStringIsNotDefault) {
  MemoryKind empty_memory_kind("");
  MemoryKind default_memory_kind;
  EXPECT_FALSE(empty_memory_kind.is_default());
  EXPECT_NE(empty_memory_kind, default_memory_kind);
}

TEST(MemoryKindTest, MemoryKindValue) {
  MemoryKind memory_kind("abc");
  EXPECT_EQ(memory_kind.value(), "abc");
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
