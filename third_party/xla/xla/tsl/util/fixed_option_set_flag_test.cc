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

// Unit tests for FixedOptionSetFlag.

#include "xla/tsl/util/fixed_option_set_flag.h"

#include <string>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"

namespace xla {
namespace {

enum class Foo {
  kBar,
  kBaz,
};

static const FixedOptionSetFlagParser<Foo>& GetFooParser() {
  static const auto& parser = GetFixedOptionSetFlagParser<Foo>({
      {"bar", Foo::kBar, "the first option"},
      {"baz", Foo::kBaz},
  });
  return parser;
};

bool AbslParseFlag(absl::string_view text, Foo* foo, std::string* error) {
  return GetFooParser().Parse(text, foo, error);
}

std::string AbslUnparseFlag(Foo foo) { return GetFooParser().Unparse(foo); }

TEST(FixedOptionSetFlag, ParseSucceedsForValidOptions) {
  Foo foo;
  std::string error;
  ASSERT_TRUE(AbslParseFlag("bar", &foo, &error));
  EXPECT_EQ(foo, Foo::kBar);
  ASSERT_TRUE(AbslParseFlag("baz", &foo, &error));
  EXPECT_EQ(foo, Foo::kBaz);
}

TEST(FixedOptionSetFlag, ParseFailsForInvalidOptions) {
  Foo foo;
  std::string error;
  ASSERT_FALSE(AbslParseFlag("foo", &foo, &error));
  EXPECT_EQ(error,
            "Unrecognized flag option: foo. Valid options are: bar (the first "
            "option), baz.");
}

TEST(FixedOptionSetFlag, UnparseSucceedsForValidOptions) {
  EXPECT_EQ(AbslUnparseFlag(Foo::kBar), "bar");
  EXPECT_EQ(AbslUnparseFlag(Foo::kBaz), "baz");
}

TEST(FixedOptionSetFlag, UnparseFailsForInvalidOptions) {
  EXPECT_EQ(AbslUnparseFlag(static_cast<Foo>(123)), "123");
}

enum class FooWithAliases {
  kBar,
  kBaz,
};

static const FixedOptionSetFlagParser<FooWithAliases>&
GetFooWithAliasesParser() {
  static const auto& parser = GetFixedOptionSetFlagParser<FooWithAliases>(
      {
          {"bar", FooWithAliases::kBar, "the first option"},
          // "baz" and "baz2" are aliases for the same option. The first one
          // listed takes precedence when unparsing.
          {"baz", FooWithAliases::kBaz},
          {"baz2", FooWithAliases::kBaz},
      },
      // Cannot use designated initializers here because tensorflow needs to
      // support C++17.
      {/*allow_aliases=*/true});
  return parser;
}

bool AbslParseFlag(absl::string_view text, FooWithAliases* foo,
                   std::string* error) {
  return GetFooWithAliasesParser().Parse(text, foo, error);
}

std::string AbslUnparseFlag(FooWithAliases foo) {
  return GetFooWithAliasesParser().Unparse(foo);
}

TEST(FixedOptionSetFlag, ParseSucceedsForValidOptionsWithAliases) {
  FooWithAliases foo;
  std::string error;
  ASSERT_TRUE(AbslParseFlag("bar", &foo, &error));
  EXPECT_EQ(foo, FooWithAliases::kBar);
  ASSERT_TRUE(AbslParseFlag("baz", &foo, &error));
  EXPECT_EQ(foo, FooWithAliases::kBaz);
  ASSERT_TRUE(AbslParseFlag("baz2", &foo, &error));
  EXPECT_EQ(foo, FooWithAliases::kBaz);
}

TEST(FixedOptionSetFlag, UnparseSucceedsForValidOptionsWithAliases) {
  EXPECT_EQ(AbslUnparseFlag(FooWithAliases::kBar), "bar");
  EXPECT_EQ(AbslUnparseFlag(FooWithAliases::kBaz), "baz");
}

TEST(FixedOptionSetFlag, ParseFailsForInvalidOptionsWithAliases) {
  FooWithAliases foo;
  std::string error;
  ASSERT_FALSE(AbslParseFlag("baz3", &foo, &error));
  EXPECT_EQ(error,
            "Unrecognized flag option: baz3. Valid options are: bar (the first "
            "option), baz, baz2.");
}

enum class FooCaseInsensitive {
  kBar,
  kBaz,
};

static const FixedOptionSetFlagParser<FooCaseInsensitive>&
GetFooCaseInsensitiveParser() {
  static const auto& parser = GetFixedOptionSetFlagParser<FooCaseInsensitive>(
      {
          {"bar", FooCaseInsensitive::kBar, "the first option"},
          {"baz", FooCaseInsensitive::kBaz},
      },
      // Cannot use designated initializers here because tensorflow needs to
      // support C++17.
      {/*allow_aliases=*/false,
       /*case_sensitive_do_not_use_in_new_code=*/false});
  return parser;
}

bool AbslParseFlag(absl::string_view text, FooCaseInsensitive* foo,
                   std::string* error) {
  return GetFooCaseInsensitiveParser().Parse(text, foo, error);
}

std::string AbslUnparseFlag(FooCaseInsensitive foo) {
  return GetFooCaseInsensitiveParser().Unparse(foo);
}

TEST(FixedOptionSetFlag, ParseSucceedsForValidOptionsCaseInsensitive) {
  FooCaseInsensitive foo;
  std::string error;
  ASSERT_TRUE(AbslParseFlag("BaR", &foo, &error));
  EXPECT_EQ(foo, FooCaseInsensitive::kBar);
  ASSERT_TRUE(AbslParseFlag("bAz", &foo, &error));
  EXPECT_EQ(foo, FooCaseInsensitive::kBaz);
}

TEST(FixedOptionSetFlag, UnparseSucceedsForValidOptionsCaseInsensitive) {
  EXPECT_EQ(AbslUnparseFlag(FooCaseInsensitive::kBar), "bar");
  EXPECT_EQ(AbslUnparseFlag(FooCaseInsensitive::kBaz), "baz");
}

TEST(FixedOptionSetFlag, ParseFailsForInvalidOptionsCaseInsensitive) {
  FooCaseInsensitive foo;
  std::string error;
  ASSERT_FALSE(AbslParseFlag("foo", &foo, &error));
  EXPECT_EQ(error,
            "Unrecognized flag option: foo. Valid options are: bar (the first "
            "option), baz.");
}

}  // namespace
}  // namespace xla
