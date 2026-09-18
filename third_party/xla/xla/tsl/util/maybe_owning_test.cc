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

#include "xla/tsl/util/maybe_owning.h"

#include <memory>

#include <gtest/gtest.h>

namespace tsl {
namespace {

TEST(MaybeOwningTest, Null) {
  tsl::MaybeOwning<char> m(nullptr);
  EXPECT_EQ(m.get(), nullptr);
  EXPECT_EQ(m.get_mutable(), nullptr);
}

TEST(MaybeOwningTest, Owning) {
  tsl::MaybeOwning<char> m(std::make_unique<char>());
  *m.get_mutable() = 'a';
  EXPECT_EQ(*m, 'a');
}

TEST(MaybeOwningTest, Shared) {
  auto owner = std::make_unique<char>();
  *owner = 'x';
  tsl::MaybeOwning<char> c1(owner.get());
  tsl::MaybeOwning<char> c2(owner.get());

  EXPECT_EQ(*c1, 'x');
  EXPECT_EQ(*c2, 'x');
  EXPECT_EQ(c1.get(), c2.get());
}

TEST(MaybeOwningTest, ReleaseOwning) {
  MaybeOwning<int> ptr(std::make_unique<int>(42));
  EXPECT_TRUE(ptr.OwnsPtr());
  EXPECT_TRUE(ptr);
  EXPECT_NE(ptr.get(), nullptr);
  EXPECT_EQ(*ptr, 42);

  std::unique_ptr<int> released = ptr.ReleaseOwning();
  EXPECT_NE(released, nullptr);
  EXPECT_EQ(*released, 42);
  EXPECT_FALSE(ptr.OwnsPtr());
  EXPECT_TRUE(ptr);
  EXPECT_EQ(ptr.get(), released.get());
}

TEST(MaybeOwningTest, ReleaseNonOwning) {
  int value = 100;
  MaybeOwning<int> non_owning(&value);
  EXPECT_FALSE(non_owning.OwnsPtr());
  EXPECT_TRUE(non_owning);
  EXPECT_EQ(non_owning.get(), &value);
  EXPECT_EQ(*non_owning, 100);

  std::unique_ptr<int> released = non_owning.ReleaseOwning();
  EXPECT_EQ(released, nullptr);
  EXPECT_FALSE(non_owning.OwnsPtr());
  EXPECT_TRUE(non_owning);
  EXPECT_NE(non_owning.get(), nullptr);
  EXPECT_EQ(*non_owning, 100);
}

}  // namespace
}  // namespace tsl
