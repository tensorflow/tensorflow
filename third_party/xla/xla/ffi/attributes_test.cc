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

#include "xla/ffi/attributes.h"

#include <cstdint>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/ffi/api/api.h"
#include "xla/ffi/attribute_map.h"

namespace xla {

// User-defined types, decoded exactly the way a runtime FFI handler would.
enum class Mode : int32_t { kFast = 0, kSlow = 1 };

struct Range {
  int32_t lo;
  int32_t hi;
};

}  // namespace xla

// User-defined decodings like these are the reason `Attributes` materializes a
// real `XLA_FFI_Attrs`: `AttrDecoding` specializations are written against the
// XLA FFI C API and are registered outside of XLA, so an attribute decoder that
// bypassed the C API would silently fail to find them.
XLA_FFI_REGISTER_ENUM_ATTR_DECODING(::xla::Mode);
XLA_FFI_REGISTER_STRUCT_ATTR_DECODING(::xla::Range, StructMember<int32_t>("lo"),
                                      StructMember<int32_t>("hi"));

namespace xla::ffi {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;

TEST(AttributesTest, GetScalars) {
  Attributes attrs = Attributes::Create({
      {"pred", true},
      {"i32", 42},
      {"i64", int64_t{43}},
      {"f32", 42.0f},
      {"f64", 43.0},
  });

  EXPECT_THAT(attrs.Get<bool>("pred"), IsOkAndHolds(true));
  EXPECT_THAT(attrs.Get<int32_t>("i32"), IsOkAndHolds(42));
  EXPECT_THAT(attrs.Get<int64_t>("i64"), IsOkAndHolds(43));
  EXPECT_THAT(attrs.Get<float>("f32"), IsOkAndHolds(42.0f));
  EXPECT_THAT(attrs.Get<double>("f64"), IsOkAndHolds(43.0));
}

TEST(AttributesTest, GetString) {
  Attributes attrs = Attributes::Create({{"str", "hello"}});
  EXPECT_THAT(attrs.Get<absl::string_view>("str"), IsOkAndHolds("hello"));
}

TEST(AttributesTest, GetArray) {
  Attributes attrs = Attributes::Create({{"arr", {int32_t{1}, 2, 3}}});

  ASSERT_OK_AND_ASSIGN(absl::Span<const int32_t> arr,
                       attrs.Get<absl::Span<const int32_t>>("arr"));
  EXPECT_THAT(arr, ElementsAre(1, 2, 3));
}

TEST(AttributesTest, GetNestedDictionary) {
  Attributes attrs = Attributes::Create({{"nested", {{"i32", 1}}}});

  ASSERT_OK_AND_ASSIGN(Dictionary nested, attrs.Get<Dictionary>("nested"));
  EXPECT_THAT(nested.get<int32_t>("i32"), IsOkAndHolds(1));
}

// Types registered with `XLA_FFI_REGISTER_ENUM_ATTR_DECODING` work here exactly
// as they do in runtime FFI handlers.
TEST(AttributesTest, GetRegisteredEnum) {
  Attributes attrs = Attributes::Create({{"mode", int32_t{1}}});

  EXPECT_THAT(attrs.Get<Mode>("mode"), IsOkAndHolds(Mode::kSlow));
}

// Same for types registered with `XLA_FFI_REGISTER_STRUCT_ATTR_DECODING`, which
// decode from a nested dictionary.
TEST(AttributesTest, GetRegisteredStruct) {
  Attributes attrs = Attributes::Create({{"range", {{"lo", 1}, {"hi", 2}}}});

  ASSERT_OK_AND_ASSIGN(Range range, attrs.Get<Range>("range"));
  EXPECT_EQ(range.lo, 1);
  EXPECT_EQ(range.hi, 2);
}

// `AttributesDictionary` is an aggregate, so a hand-built one has a null
// `attrs`. Like `AttributesDictionary::ToProto` and `operator==`, we treat that
// as an empty dictionary instead of dereferencing null.
TEST(AttributesTest, GetHandBuiltNullDictionaryDecodesAsEmpty) {
  AttributesMap map;
  map.emplace("nested", Attribute(AttributesDictionary()));

  Attributes attrs = Attributes::Create(map);

  ASSERT_OK_AND_ASSIGN(Dictionary nested, attrs.Get<Dictionary>("nested"));
  EXPECT_EQ(nested.size(), 0);
}

TEST(AttributesTest, GetMissingAttributeFails) {
  Attributes attrs = Attributes::Create({{"i32", 42}});

  EXPECT_THAT(attrs.Get<int32_t>("nope"),
              StatusIs(absl::StatusCode::kInternal, HasSubstr("nope")));
}

// Decoding is strict about integer widths, matching runtime FFI handlers.
TEST(AttributesTest, GetWithWrongWidthFails) {
  Attributes attrs = Attributes::Create({{"i32", 42}});

  EXPECT_THAT(attrs.Get<int64_t>("i32"), StatusIs(absl::StatusCode::kInternal));
  EXPECT_THAT(attrs.Get<float>("i32"), StatusIs(absl::StatusCode::kInternal));
}

TEST(AttributesTest, Contains) {
  Attributes attrs = Attributes::Create({{"i32", 42}, {"str", "hello"}});

  EXPECT_TRUE(attrs.Contains("i32"));
  EXPECT_TRUE(attrs.Contains("str"));
  EXPECT_FALSE(attrs.Contains("nope"));

  EXPECT_TRUE(attrs.Contains<int32_t>("i32"));
  EXPECT_FALSE(attrs.Contains<int64_t>("i32"));
  EXPECT_FALSE(attrs.Contains<int32_t>("nope"));
}

TEST(AttributesTest, SizeAndEmpty) {
  Attributes empty = Attributes::Create(AttributesMap());
  EXPECT_EQ(empty.size(), 0);
  EXPECT_TRUE(empty.empty());

  Attributes attrs = Attributes::Create({{"a", 1}, {"b", 2}});
  EXPECT_EQ(attrs.size(), 2);
  EXPECT_FALSE(attrs.empty());
}

// Values that alias the underlying storage must survive a move of the owner.
TEST(AttributesTest, MoveKeepsStorageAlive) {
  Attributes attrs = Attributes::Create({{"str", "hello"}, {"arr", {1, 2, 3}}});

  Attributes moved = std::move(attrs);

  EXPECT_THAT(moved.Get<absl::string_view>("str"), IsOkAndHolds("hello"));
  ASSERT_OK_AND_ASSIGN(absl::Span<const int32_t> arr,
                       moved.Get<absl::Span<const int32_t>>("arr"));
  EXPECT_THAT(arr, ElementsAre(1, 2, 3));

  Attributes move_assigned = Attributes::Create(AttributesMap());
  move_assigned = std::move(moved);
  EXPECT_THAT(move_assigned.Get<absl::string_view>("str"),
              IsOkAndHolds("hello"));
}

}  // namespace
}  // namespace xla::ffi
