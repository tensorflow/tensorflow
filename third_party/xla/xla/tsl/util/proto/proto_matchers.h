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

// This library defines some proto matchers in the ::tsl::proto_testing
// namespace for use in tests.
//
// The matchers are:
//
//   EqualsProto(Proto)
//   EqualsProto(string)
//
// The EqualsProto(Proto) matcher matches a proto that equals the given proto.
// The EqualsProto(string) matcher matches a proto that equals the given proto
// (represented as a text string).
//
// It also defines a few transformers for proto matchers:
//
//   Partially(m)
//   IgnoringRepeatedFieldOrdering(m)
//
// Partially(m) is like m, but ignores any fields that are not set in the
// expected proto.
//
// IgnoringRepeatedFieldOrdering(m) is like m, but ignores the order of elements
// in repeated fields.
//
// Partially() and IgnoringRepeatedFieldOrdering() can be nested, e.g.
//
//   Partially(IgnoringRepeatedFieldOrdering(EqualsProto(R"pb(
//     s1: "foo"
//     r3: "a"
//     r3: "b"
//     r3: "c"
//   )pb"))));
//
// will match a proto that has the same elements in r3, but in any order, and
// will ignore any extra fields that are set.

#ifndef XLA_TSL_UTIL_PROTO_PROTO_MATCHERS_H_
#define XLA_TSL_UTIL_PROTO_PROTO_MATCHERS_H_

#include <memory>
#include <ostream>
#include <string>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "tsl/platform/protobuf.h"

namespace tsl {
namespace proto_testing {

// Marks the proto-matcher as allowing for extra fields to be set in the result.
// NB that extra elements of repeated fields are still considered a failure.
template <typename InnerMatcher,
          typename = typename InnerMatcher::is_proto_matcher>
inline InnerMatcher Partially(InnerMatcher i) {
  i.SetPartial();
  return i;
}

// Marks the proto matcher as allowing for repeated fields to be in any order.
template <typename InnerMatcher,
          typename = typename InnerMatcher::is_proto_matcher>
inline InnerMatcher IgnoringRepeatedFieldOrdering(InnerMatcher i) {
  i.SetUnorderedRepeatedFields();
  return i;
}

namespace internal {

// A wrapper around a unique_ptr that can implicitly convert to either a raw
// pointer or a unique_ptr.
template <typename T>
class UniquePtrWrapper {
 public:
  // Creates a wrapper around the given unique_ptr.
  explicit UniquePtrWrapper(std::unique_ptr<T> ptr) : ptr_(std::move(ptr)) {}

  // Implicitly converts to a raw pointer of a compatible type, transferring
  // ownership to the caller.
  template <typename U>
  operator U*() {  // NOLINT(google-explicit-constructor)
    return ptr_.release();
  }

  // Implicitly converts to a unique_ptr of a compatible type.
  template <typename U>
  operator std::unique_ptr<U>() {  // NOLINT(google-explicit-constructor)
    return std::move(ptr_);
  }

 private:
  std::unique_ptr<T> ptr_;
};

// Matcher ignore-checker that makes fields not in 'gold' ignored.
class PartialIgnore final
    : public ::tsl::protobuf::util::MessageDifferencer::IgnoreCriteria {
  using SpecificField =
      ::tsl::protobuf::util::MessageDifferencer::SpecificField;

 public:
  PartialIgnore() = default;

  bool IsIgnored(const ::tsl::protobuf::Message& gold,
                 const ::tsl::protobuf::Message& test,
                 const ::tsl::protobuf::FieldDescriptor* field,
                 const std ::vector<SpecificField>& specific_field) final {
    // Ignore any field fully absent from the gold proto.
    if (field->is_repeated()) {
      return gold.GetReflection()->FieldSize(gold, field) == 0;
    }
    return !gold.GetReflection()->HasField(gold, field);
  }

  bool IsUnknownFieldIgnored(const ::tsl::protobuf::Message&,
                             const ::tsl::protobuf::Message&,
                             const SpecificField&,
                             const std ::vector<SpecificField>&) final {
    return true;
  }
};

// Matcher that checks for proto equality. It can be modified by the Partially
// and IgnoreRepeatedFieldOrdering to adjust the values. ExpectedProto can be
// either a proto or string.
template <typename ExpectedProto>
class EqualsProtoMatcher {
 public:
  static_assert(std::is_base_of_v<::tsl::protobuf::Message, ExpectedProto> ||
                    std::is_same_v<ExpectedProto, std::string>,
                "EqualsProto(p) requires p to be a proto or a string.");
  using is_gtest_matcher = void;
  using is_proto_matcher = void;

  explicit EqualsProtoMatcher(ExpectedProto expected_proto)
      : expected_proto_(std::move(expected_proto)) {}

  // Matches a proto against the expected proto.
  template <typename ActualProto>
  bool MatchAndExplain(ActualProto actual_proto,
                       ::testing::MatchResultListener* listener) const {
    using Actual = std::remove_reference_t<ActualProto>;
    Actual expected;
    const Actual* expected_ptr = &expected;
    if constexpr (std::is_same_v<ExpectedProto, std::string>) {
      const bool parsed = ::tsl::protobuf::TextFormat::ParseFromString(
          expected_proto_, &expected);
      if (!parsed) {
        *listener << "Unable to parse \"" << expected_proto_ << "\" as "
                  << expected.GetTypeName();
        return false;
      }
    } else {
      expected_ptr = &expected_proto_;
    }

    ::tsl::protobuf::util::MessageDifferencer diff;
    diff.set_report_ignores(false);
    if (partial_) {
      diff.AddIgnoreCriteria(std::make_unique<PartialIgnore>());
    }
    std::string str_report;
    if (unordered_repeated_fields_) {
      diff.set_repeated_field_comparison(
          ::tsl::protobuf::util::MessageDifferencer::AS_SET);
    }
    diff.ReportDifferencesToString(&str_report);
    bool same_message = diff.Compare(*expected_ptr, actual_proto);
    if (same_message) {
      return true;
    }
    *listener << str_report;
    return false;
  }

  // Describes this matcher to an ostream.
  void DescribeTo(std::ostream* os) const {
    *os << absl::StreamFormat(
        "equals%s%s ", partial_ ? " (ignoring extra fields)" : "",
        unordered_repeated_fields_ ? " (ignoring repeated field order)" : "");
    // StreamFormat() doesn't work with some versions of protobuf, so we need
    // to convert expected_proto_ to a string manually.
    std::string expected_proto_str;
    if constexpr (std::is_same_v<ExpectedProto, std::string>) {
      *os << expected_proto_;
    } else {
      *os << expected_proto_.DebugString();
    }
  }

  // Describes the negation of this matcher to an ostream.
  void DescribeNegationTo(std::ostream* os) const {
    *os << "not ";
    DescribeTo(os);
  }

  void SetPartial() { partial_ = true; }
  void SetUnorderedRepeatedFields() { unordered_repeated_fields_ = true; }

 private:
  ExpectedProto expected_proto_;
  bool partial_ = false;
  bool unordered_repeated_fields_ = false;
};

}  // namespace internal

// Returns a matcher that matches a proto that equals the given proto.
template <typename Proto, typename = std::enable_if_t<std::is_base_of_v<
                              ::tsl::protobuf::Message, Proto>>>
inline auto EqualsProto(Proto proto) {
  return internal::EqualsProtoMatcher<Proto>(std::move(proto));
}

// Returns a matcher that matches a proto that equals the given proto
// (represented as a text string).
inline auto EqualsProto(absl::string_view proto) {
  return internal::EqualsProtoMatcher<std::string>(std::string(proto));
}

}  // namespace proto_testing
}  // namespace tsl

#endif  // XLA_TSL_UTIL_PROTO_PROTO_MATCHERS_H_
