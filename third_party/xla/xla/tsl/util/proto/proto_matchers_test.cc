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

#include "xla/tsl/util/proto/proto_matchers.h"

#include <sstream>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/util/proto/proto_matchers_test_protos.pb.h"
#include "tsl/platform/protobuf.h"

namespace tsl {
namespace proto_testing {
namespace {

using ::testing::Matcher;
using ::testing::MatchesRegex;
using ::testing::Not;

Foo MakeFoo(absl::string_view sv) {
  const std::string s = std::string(sv);
  Foo foo;
  EXPECT_TRUE(::tsl::protobuf::TextFormat::ParseFromString(s, &foo));
  return foo;
}

// Returns the description of the given matcher.
std::string Describe(const Matcher<const Foo&>& matcher) {
  std::stringstream ss;
  matcher.DescribeTo(&ss);
  return ss.str();
}

TEST(EqualsProto, DescribesSelfWhenGivenProto) {
  const Matcher<const Foo&> matcher =
      EqualsProto(MakeFoo(R"pb(
        s1: "foo" r3: "a" r3: "b" r3: "c" s4: "t"
      )pb"));

  EXPECT_THAT(Describe(matcher),
              MatchesRegex("equals (.|\n)*s1: \"foo\"(.|\n)*r3: \"a\"(.|\n)r3: "
                           "\"b\"(.|\n)r3: \"c\"(.|\n)s4: \"t\"(.|\n)"));
  EXPECT_THAT(
      Describe(Not(matcher)),
      MatchesRegex("not equals (.|\n)*s1: \"foo\"(.|\n)*r3: "
                   "\"a\"(.|\n)r3: \"b\"(.|\n)r3: \"c\"(.|\n)s4: \"t\"(.|\n)"));
}

TEST(EqualsProto, DescribesSelfWhenGivenString) {
  const std::string s = R"pb(s1: "foo" r3: "a" r3: "b" r3: "c" s4: "t")pb";
  const Matcher<const Foo&> matcher =
      EqualsProto(R"pb(s1: "foo" r3: "a" r3: "b" r3: "c" s4: "t")pb");

  EXPECT_EQ(Describe(matcher), absl::StrCat("equals ", s));
  EXPECT_EQ(Describe(Not(matcher)), absl::StrCat("not equals ", s));
}

TEST(EqualsProto, WorksWithProtoArgument) {
  const Foo foo = MakeFoo(R"(
      s1: "foo"
      r3: "a"
      r3: "b"
      r3: "c"
      s4: "t"
    )");
  EXPECT_THAT(foo,
              EqualsProto(MakeFoo(R"pb(
                s1: "foo" r3: "a" r3: "b" r3: "c" s4: "t"
              )pb")));
  EXPECT_THAT(foo, Not(EqualsProto(MakeFoo(R"pb(
                s1: "bar"
                r3: "a"
                r3: "b"
                r3: "c"
                s4: "t"
              )pb"))));
}

TEST(EqualsProto, WorksWithStringArgument) {
  const Foo foo = MakeFoo(R"(
      s1: "foo"
      r3: "a"
      r3: "b"
      r3: "c"
      s4: "t"
    )");
  EXPECT_THAT(foo, EqualsProto(R"pb(
                s1: "foo" r3: "a" r3: "b" r3: "c" s4: "t"
              )pb"));
  EXPECT_THAT(foo, EqualsProto(R"pb(
                r3: "a" r3: "b" s1: "foo" r3: "c" s4: "t"
              )pb"));
  EXPECT_THAT(foo,
              Not(EqualsProto(R"pb(
                s1: "foobar" r3: "a" r3: "b" r3: "c" s4: "t"
              )pb")));
  EXPECT_THAT(foo, Not(EqualsProto(R"pb(
                !garbage ^ &*
              )pb")));
  EXPECT_THAT(
      foo,
      Not(EqualsProto(R"pb(
        r3: "a" r3: "b" s1: "foo" r3: "c" r3: "d" s4: "t"
      )pb")));
  EXPECT_THAT(foo,
              Not(EqualsProto(R"pb(
                s1: "foo" r3: "b" r3: "c" r3: "a" s4: "t"
              )pb")));
  EXPECT_THAT(
      foo, Not(EqualsProto(R"pb(
        s1: "foo" i2: 32 r3: "a" r3: "b" r3: "c" s4: "t"
      )pb")));
}

TEST(EqualsProto, WorksWithPartially) {
  const Foo foo = MakeFoo(R"(
      s1: "foo"
      i2: 32
      r3: "a"
      r3: "b"
      r3: "c"
      s4: "t"
    )");
  EXPECT_THAT(
      foo, Partially(EqualsProto(R"pb(
        s1: "foo" r3: "a" r3: "b" r3: "c" s4: "t"
      )pb")));
  EXPECT_THAT(foo, Partially(EqualsProto(R"pb(
                r3: "a" r3: "b" r3: "c" s4: "t"
              )pb")));
  EXPECT_THAT(foo, Partially(EqualsProto(R"pb(
                s1: "foo"
              )pb")));
  EXPECT_THAT(foo, Partially(EqualsProto(R"pb(
                r3: "a" r3: "b" r3: "c"
              )pb")));
  // bad order
  EXPECT_THAT(foo,
              Not(Partially(EqualsProto(R"pb(
                s1: "foo" r3: "b" r3: "c" r3: "a"
              )pb"))));
  // new value
  EXPECT_THAT(foo, Not(Partially(EqualsProto(R"pb(
                s1: "foo"
                i2: 10
                r3: "a"
                r3: "b"
                r3: "c"
              )pb"))));
}

TEST(EqualsProto, WorksWithIgnoringRepeatedFieldOrdering) {
  const Foo foo = MakeFoo(R"(
      s1: "foo"
      r3: "a"
      r3: "b"
      r3: "c"
      s4: "t"
    )");
  EXPECT_THAT(foo, IgnoringRepeatedFieldOrdering(EqualsProto(R"pb(
                s1: "foo"
                r3: "a"
                r3: "c"
                r3: "b"
                s4: "t"
              )pb")));
  EXPECT_THAT(foo, IgnoringRepeatedFieldOrdering(EqualsProto(R"pb(
                r3: "a"
                r3: "c"
                s1: "foo"
                r3: "b"
                s4: "t"
              )pb")));
  EXPECT_THAT(foo, Not(IgnoringRepeatedFieldOrdering(EqualsProto(R"pb(
                s1: "foobar"
                r3: "b"
                r3: "a"
                r3: "c"
                s4: "t"
              )pb"))));
  EXPECT_THAT(foo, Not(IgnoringRepeatedFieldOrdering(EqualsProto(R"pb(
                r3: "b"
                r3: "a"
                s1: "foo"
                r3: "c"
                r3: "d"
                s4: "t"
              )pb"))));
  EXPECT_THAT(foo, Not(IgnoringRepeatedFieldOrdering(EqualsProto(R"pb(
                s1: "foo"
                i2: 32
                r3: "a"
                r3: "b"
                r3: "c"
                s4: "t"
              )pb"))));
}

TEST(EqualsProto, WorksWithPartiallyAndIgnoringOrder) {
  const Foo foo = MakeFoo(R"(
      s1: "foo"
      i2: 32
      r3: "a"
      r3: "b"
      r3: "c"
      s4: "t"
    )");
  EXPECT_THAT(foo, Partially(IgnoringRepeatedFieldOrdering(EqualsProto(R"pb(
                s1: "foo"
                r3: "a"
                r3: "b"
                r3: "c"
                s4: "t"
              )pb"))));
  EXPECT_THAT(foo, Partially(IgnoringRepeatedFieldOrdering(EqualsProto(R"pb(
                r3: "b"
                r3: "a"
                r3: "c"
                s4: "t"
              )pb"))));
  EXPECT_THAT(foo, Partially(IgnoringRepeatedFieldOrdering(EqualsProto(R"pb(
                s1: "foo"
              )pb"))));
  // bad order
  EXPECT_THAT(foo, Not(Partially(IgnoringRepeatedFieldOrdering(EqualsProto(R"pb(
                s1: "bar"
                r3: "b"
                r3: "c"
                r3: "a"
              )pb")))));
  // new value
  EXPECT_THAT(foo, Not(Partially(IgnoringRepeatedFieldOrdering(EqualsProto(R"pb(
                s1: "foo"
                i2: 10
                r3: "b"
                r3: "a"
                r3: "c"
              )pb")))));
}

TEST(EqualsProto, DoesNotMatchWhenGivenSameValueAsDefault) {
  const Foo foo = MakeFoo(R"(
      s1: "foo"
      r3: "a"
      r3: "b"
      r3: "c"
      s4: "s"
    )");
  EXPECT_THAT(foo, Not(EqualsProto(R"pb(
                s1: "foo" r3: "a" r3: "b" r3: "c"
              )pb")));
}

TEST(EquivToProto, DescribesSelfWhenGivenProto) {
  const Matcher<const Foo&> matcher =
      EquivToProto(MakeFoo(R"pb(
        s1: "foo" r3: "a" r3: "b" r3: "c" s4: "t"
      )pb"));

  EXPECT_THAT(
      Describe(matcher),
      MatchesRegex(
          "is equivalent to (.|\n)*s1: \"foo\"(.|\n)*r3: \"a\"(.|\n)r3: "
          "\"b\"(.|\n)r3: \"c\"(.|\n)s4: \"t\"(.|\n)"));
  EXPECT_THAT(
      Describe(Not(matcher)),
      MatchesRegex("is not equivalent to (.|\n)*s1: \"foo\"(.|\n)*r3: "
                   "\"a\"(.|\n)r3: \"b\"(.|\n)r3: \"c\"(.|\n)s4: \"t\"(.|\n)"));
}

TEST(EquivToProto, DescribesSelfWhenGivenString) {
  const std::string s = R"pb(s1: "foo" r3: "a" r3: "b" r3: "c" s4: "t")pb";
  const Matcher<const Foo&> matcher = EquivToProto(s);

  EXPECT_EQ(Describe(matcher), absl::StrCat("is equivalent to ", s));
  EXPECT_EQ(Describe(Not(matcher)), absl::StrCat("is not equivalent to ", s));
}

TEST(EquivToProto, WorksWithProtoArgument) {
  const Foo foo = MakeFoo(R"(
      s1: "foo"
      r3: "a"
      r3: "b"
      r3: "c"
      s4: "t"
    )");
  EXPECT_THAT(
      foo, EquivToProto(MakeFoo(R"pb(
        s1: "foo" r3: "a" r3: "b" r3: "c" s4: "t"
      )pb")));
  EXPECT_THAT(foo, Not(EquivToProto(MakeFoo(R"pb(
                s1: "bar"
                r3: "a"
                r3: "b"
                r3: "c"
                s4: "t"
              )pb"))));
}

TEST(EquivToProto, WorksWithStringArgument) {
  const Foo foo = MakeFoo(R"(
      s1: "foo"
      r3: "a"
      r3: "b"
      r3: "c"
      s4: "t"
    )");
  EXPECT_THAT(foo, EquivToProto(R"pb(
                s1: "foo" r3: "a" r3: "b" r3: "c" s4: "t"
              )pb"));
  EXPECT_THAT(foo, EquivToProto(R"pb(
                r3: "a" r3: "b" s1: "foo" r3: "c" s4: "t"
              )pb"));
  EXPECT_THAT(foo,
              Not(EquivToProto(R"pb(
                s1: "foobar" r3: "a" r3: "b" r3: "c" s4: "t"
              )pb")));
  EXPECT_THAT(foo, Not(EquivToProto(R"pb(
                !garbage ^ &*
              )pb")));
  EXPECT_THAT(
      foo,
      Not(EquivToProto(R"pb(
        r3: "a" r3: "b" s1: "foo" r3: "c" r3: "d" s4: "t"
      )pb")));
  EXPECT_THAT(foo,
              Not(EquivToProto(R"pb(
                s1: "foo" r3: "b" r3: "c" r3: "a" s4: "t"
              )pb")));
  EXPECT_THAT(
      foo,
      Not(EquivToProto(R"pb(
        s1: "foo" i2: 32 r3: "a" r3: "b" r3: "c" s4: "t"
      )pb")));
}

TEST(EquivToProto, MatchesWhenGivenSameValueAsDefault) {
  const Foo foo = MakeFoo(R"(
      s1: "foo"
      r3: "a"
      r3: "b"
      r3: "c"
      s4: "s"
    )");
  EXPECT_THAT(foo, EquivToProto(R"pb(
                s1: "foo" r3: "a" r3: "b" r3: "c"
              )pb"));
}

TEST(EquivToProto, WorksWithPartially) {
  const Foo foo = MakeFoo(R"(
      s1: "foo"
      i2: 32
      r3: "a"
      r3: "b"
      r3: "c"
      s4: "s"
    )");
  EXPECT_THAT(foo, Partially(EquivToProto(R"pb(
                s1: "foo" r3: "a" r3: "b" r3: "c"
              )pb")));
  EXPECT_THAT(foo, Partially(EquivToProto(R"pb(
                r3: "a" r3: "b" r3: "c"
              )pb")));
  EXPECT_THAT(foo, Partially(EquivToProto(R"pb(
                s1: "foo"
              )pb")));
  EXPECT_THAT(foo, Partially(EquivToProto(R"pb(
                r3: "a" r3: "b" r3: "c"
              )pb")));
  // bad order
  EXPECT_THAT(foo,
              Not(Partially(EquivToProto(R"pb(
                s1: "foo" r3: "b" r3: "c" r3: "a"
              )pb"))));
  // new value
  EXPECT_THAT(foo, Not(Partially(EquivToProto(R"pb(
                s1: "foo"
                i2: 10
                r3: "a"
                r3: "b"
                r3: "c"
              )pb"))));
}

TEST(EquivToProto, WorksWithIgnoringRepeatedFieldOrdering) {
  const Foo foo = MakeFoo(R"(
      s1: "foo"
      r3: "a"
      r3: "b"
      r3: "c"
    )");
  EXPECT_THAT(foo, IgnoringRepeatedFieldOrdering(EquivToProto(R"pb(
                s1: "foo"
                r3: "a"
                r3: "c"
                r3: "b"
                s4: "s"
              )pb")));
  EXPECT_THAT(foo, IgnoringRepeatedFieldOrdering(EquivToProto(R"pb(
                r3: "a"
                r3: "c"
                s1: "foo"
                r3: "b"
                s4: "s"
              )pb")));
  EXPECT_THAT(foo, Not(IgnoringRepeatedFieldOrdering(EquivToProto(R"pb(
                s1: "foobar"
                r3: "b"
                r3: "a"
                r3: "c"
                s4: "s"
              )pb"))));
  EXPECT_THAT(foo, Not(IgnoringRepeatedFieldOrdering(EquivToProto(R"pb(
                r3: "b"
                r3: "a"
                s1: "foo"
                r3: "c"
                r3: "d"
                s4: "s"
              )pb"))));
  EXPECT_THAT(foo, Not(IgnoringRepeatedFieldOrdering(EquivToProto(R"pb(
                s1: "foo"
                i2: 32
                r3: "a"
                r3: "b"
                r3: "c"
                s4: "s"
              )pb"))));
}

TEST(EquivToProto, WorksWithPartiallyAndIgnoringOrder) {
  const Foo foo = MakeFoo(R"(
      s1: "foo"
      i2: 32
      r3: "a"
      r3: "b"
      r3: "c"
      s4: "s"
    )");
  EXPECT_THAT(foo, Partially(IgnoringRepeatedFieldOrdering(EquivToProto(R"pb(
                s1: "foo"
                r3: "a"
                r3: "b"
                r3: "c"
              )pb"))));
  EXPECT_THAT(foo, Partially(IgnoringRepeatedFieldOrdering(EquivToProto(R"pb(
                r3: "b"
                r3: "a"
                r3: "c"
              )pb"))));
  EXPECT_THAT(foo, Partially(IgnoringRepeatedFieldOrdering(EquivToProto(R"pb(
                s1: "foo"
              )pb"))));
  // bad order
  EXPECT_THAT(foo,
              Not(Partially(IgnoringRepeatedFieldOrdering(EquivToProto(R"pb(
                s1: "bar"
                r3: "b"
                r3: "c"
                r3: "a"
              )pb")))));
  // new value
  EXPECT_THAT(foo, Not(Partially(IgnoringRepeatedFieldOrdering(EquivToProto(
                       R"pb(
                         s1: "foo" i2: 10 r3: "b" r3: "a" r3: "c"
                       )pb")))));
}

TEST(EquivToProto, DescribesSelfWithPartiallyAndIgnoringOrder) {
  const std::string s = R"pb(s1: "foo" r3: "a" r3: "b" r3: "c" s4: "t")pb";
  const Matcher<const Foo&> matcher =
      Partially(IgnoringRepeatedFieldOrdering(EquivToProto(s)));

  EXPECT_EQ(Describe(matcher),
            absl::StrCat("is equivalent to (ignoring extra fields) (ignoring "
                         "repeated field order) ",
                         s));
  EXPECT_EQ(Describe(Not(matcher)),
            absl::StrCat("is not equivalent to (ignoring extra fields) "
                         "(ignoring repeated field order) ",
                         s));
}

}  // namespace
}  // namespace proto_testing
}  // namespace tsl
