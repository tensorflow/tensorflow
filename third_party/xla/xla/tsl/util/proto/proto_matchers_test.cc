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
        s1: "foo" r3: "a" r3: "b" r3: "c"
      )pb"));

  EXPECT_THAT(Describe(matcher),
              MatchesRegex("equals (.|\n)*s1: \"foo\"(.|\n)*r3: \"a\"(.|\n)r3: "
                           "\"b\"(.|\n)r3: \"c\"(.|\n)"));
  EXPECT_THAT(Describe(Not(matcher)),
              MatchesRegex("not equals (.|\n)*s1: \"foo\"(.|\n)*r3: "
                           "\"a\"(.|\n)r3: \"b\"(.|\n)r3: \"c\"(.|\n)"));
}

TEST(EqualsProto, DescribesSelfWhenGivenString) {
  const Matcher<const Foo&> matcher =
      EqualsProto(R"pb(s1: "foo" r3: "a" r3: "b" r3: "c")pb");

  EXPECT_EQ(Describe(matcher),
            R"pb(equals s1: "foo" r3: "a" r3: "b" r3: "c")pb");
  EXPECT_EQ(Describe(Not(matcher)),
            R"pb(not equals s1: "foo" r3: "a" r3: "b" r3: "c")pb");
}

TEST(EqualsProto, WorksWithProtoArgument) {
  const Foo foo = MakeFoo(R"(
      s1: "foo"
      r3: "a"
      r3: "b"
      r3: "c"
    )");
  EXPECT_THAT(foo, EqualsProto(MakeFoo(R"pb(
                s1: "foo" r3: "a" r3: "b" r3: "c"
              )pb")));
  EXPECT_THAT(foo,
              Not(EqualsProto(MakeFoo(R"pb(
                s1: "bar" r3: "a" r3: "b" r3: "c"
              )pb"))));
}

TEST(EqualsProto, WorksWithStringArgument) {
  const Foo foo = MakeFoo(R"(
      s1: "foo"
      r3: "a"
      r3: "b"
      r3: "c"
    )");
  EXPECT_THAT(foo, EqualsProto(R"pb(
                s1: "foo" r3: "a" r3: "b" r3: "c"
              )pb"));
  EXPECT_THAT(foo, EqualsProto(R"pb(
                r3: "a" r3: "b" s1: "foo" r3: "c"
              )pb"));
  EXPECT_THAT(foo, Not(EqualsProto(R"pb(
                s1: "foobar" r3: "a" r3: "b" r3: "c"
              )pb")));
  EXPECT_THAT(foo, Not(EqualsProto(R"pb(
                !garbage ^ &*
              )pb")));
  EXPECT_THAT(foo,
              Not(EqualsProto(R"pb(
                r3: "a" r3: "b" s1: "foo" r3: "c" r3: "d"
              )pb")));
  EXPECT_THAT(foo, Not(EqualsProto(R"pb(
                s1: "foo" r3: "b" r3: "c" r3: "a"
              )pb")));
  EXPECT_THAT(foo, Not(EqualsProto(R"pb(
                s1: "foo" i2: 32 r3: "a" r3: "b" r3: "c"
              )pb")));
}

TEST(EqualsProto, WorksWithPartially) {
  const Foo foo = MakeFoo(R"(
      s1: "foo"
      i2: 32
      r3: "a"
      r3: "b"
      r3: "c"
    )");
  EXPECT_THAT(foo, Partially(EqualsProto(R"pb(
                s1: "foo" r3: "a" r3: "b" r3: "c"
              )pb")));
  EXPECT_THAT(foo, Partially(EqualsProto(R"pb(
                r3: "a" r3: "b" r3: "c"
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
    )");
  EXPECT_THAT(foo, IgnoringRepeatedFieldOrdering(EqualsProto(R"pb(
                s1: "foo"
                r3: "a"
                r3: "c"
                r3: "b"
              )pb")));
  EXPECT_THAT(foo, IgnoringRepeatedFieldOrdering(EqualsProto(R"pb(
                r3: "a"
                r3: "c"
                s1: "foo"
                r3: "b"
              )pb")));
  EXPECT_THAT(foo, Not(IgnoringRepeatedFieldOrdering(EqualsProto(R"pb(
                s1: "foobar"
                r3: "b"
                r3: "a"
                r3: "c"
              )pb"))));
  EXPECT_THAT(foo, Not(IgnoringRepeatedFieldOrdering(EqualsProto(R"pb(
                r3: "b"
                r3: "a"
                s1: "foo"
                r3: "c"
                r3: "d"
              )pb"))));
  EXPECT_THAT(foo, Not(IgnoringRepeatedFieldOrdering(EqualsProto(R"pb(
                s1: "foo"
                i2: 32
                r3: "a"
                r3: "b"
                r3: "c"
              )pb"))));
}

TEST(EqualsProto, WorksWithPartiallyAndIgnoringOrder) {
  const Foo foo = MakeFoo(R"(
      s1: "foo"
      i2: 32
      r3: "a"
      r3: "b"
      r3: "c"
    )");
  EXPECT_THAT(foo, Partially(IgnoringRepeatedFieldOrdering(EqualsProto(R"pb(
                s1: "foo"
                r3: "a"
                r3: "b"
                r3: "c"
              )pb"))));
  EXPECT_THAT(foo, Partially(IgnoringRepeatedFieldOrdering(EqualsProto(R"pb(
                r3: "b"
                r3: "a"
                r3: "c"
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

}  // namespace
}  // namespace proto_testing
}  // namespace tsl
