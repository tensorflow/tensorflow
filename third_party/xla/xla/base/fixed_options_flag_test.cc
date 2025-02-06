/* Copyright 2025 The OpenXLA Authors.

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

// Unit tests for FixedOptionsFlag.

#include "xla/base/fixed_options_flag.h"

#include <string>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"

namespace xla {
namespace {

enum class Foo {
  kBar,
  kBaz,
};

static const FixedOptionsFlagParser<Foo>& GetFooParser() {
  static const auto& parser = GetFixedOptionsFlagParser<Foo>({
      {"bar", Foo::kBar, "the first option"},
      {"baz", Foo::kBaz},
  });
  return parser;
};

bool AbslParseFlag(absl::string_view text, Foo* foo, std::string* error) {
  return GetFooParser().Parse(text, foo, error);
}

std::string AbslUnparseFlag(Foo foo) { return GetFooParser().Unparse(foo); }

TEST(FixedOptionsFlag, ParseSucceedsForValidOptions) {
  Foo foo;
  std::string error;
  ASSERT_TRUE(AbslParseFlag("bar", &foo, &error));
  EXPECT_EQ(foo, Foo::kBar);
  ASSERT_TRUE(AbslParseFlag("baz", &foo, &error));
  EXPECT_EQ(foo, Foo::kBaz);
}

TEST(FixedOptionsFlag, ParseFailsForInvalidOptions) {
  Foo foo;
  std::string error;
  ASSERT_FALSE(AbslParseFlag("foo", &foo, &error));
  EXPECT_EQ(error,
            "Unrecognized flag option: foo. Valid options are: bar (the first "
            "option), baz.");
}

TEST(FixedOptionsFlag, UnparseSucceedsForValidOptions) {
  EXPECT_EQ(AbslUnparseFlag(Foo::kBar), "bar");
  EXPECT_EQ(AbslUnparseFlag(Foo::kBaz), "baz");
}

TEST(FixedOptionsFlag, UnparseFailsForInvalidOptions) {
  EXPECT_EQ(AbslUnparseFlag(static_cast<Foo>(123)), "123");
}

}  // namespace
}  // namespace xla
