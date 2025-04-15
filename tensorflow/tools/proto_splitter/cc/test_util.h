/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_TOOLS_PROTO_SPLITTER_CC_TEST_UTIL_H_
#define TENSORFLOW_TOOLS_PROTO_SPLITTER_CC_TEST_UTIL_H_

#include <ostream>
#include <string>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "tensorflow/core/platform/test.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/protobuf.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {
namespace tools::proto_splitter {

// Ensures that all Messages are less than the max size. std::string chunks are
// not limited by the max size, so they are ignored in this check.
#define EXPECT_CHUNK_SIZES(chunks, max_size)                                \
  do {                                                                      \
    for (auto chunk : *chunks) {                                            \
      if (std::holds_alternative<std::shared_ptr<tsl::protobuf::Message>>(  \
              chunk)) {                                                     \
        EXPECT_LE(std::get<std::shared_ptr<tsl::protobuf::Message>>(chunk)  \
                      ->ByteSizeLong(),                                     \
                  max_size);                                                \
      } else if (std::holds_alternative<tsl::protobuf::Message*>(chunk)) {  \
        EXPECT_LE(std::get<tsl::protobuf::Message*>(chunk)->ByteSizeLong(), \
                  max_size);                                                \
      }                                                                     \
    }                                                                       \
  } while (0)

inline std::string SerializeAsString(const tsl::protobuf::Message& message) {
  std::string result;
  {
    // Use a nested block to guarantee triggering coded_stream's destructor
    // before `result` is used. Due to copy elision, this code works without
    // the nested block, but small, innocent looking changes can break it.
    tsl::protobuf::io::StringOutputStream string_stream(&result);
    tsl::protobuf::io::CodedOutputStream coded_stream(&string_stream);
    coded_stream.SetSerializationDeterministic(true);
    message.SerializeToCodedStream(&coded_stream);
  }
  return result;
}

template <typename MessageType>
tsl::StatusOr<MessageType> ParseTextProto(const char* text_proto) {
  tsl::protobuf::TextFormat::Parser parser;
  MessageType parsed_proto;
  bool success =
      tsl::protobuf::TextFormat::ParseFromString(text_proto, &parsed_proto);
  if (!success) {
    return absl::InvalidArgumentError(
        "Input text could not be parsed to proto.");
  }
  return parsed_proto;
}

// TODO(b/229726259): EqualsProto is not available in OSS.

// Simple implementation of a proto matcher comparing string representations.
// This will fail if the string representations are not deterministic.
class ProtoStringMatcher {
 public:
  explicit ProtoStringMatcher(const tsl::protobuf::Message& expected)
      : expected_(SerializeAsString(expected)) {}

  explicit ProtoStringMatcher(const std::string& expected)
      : text_format_(expected), use_text_format_(true) {}

  template <typename Message>
  bool MatchAndExplain(const Message& p,
                       ::testing::MatchResultListener*) const {
    if (use_text_format_) {
      Message* m = p.New();
      CHECK(tsl::protobuf::TextFormat::ParseFromString(text_format_, m))
          << "Failed to parse proto text: " << text_format_;
      std::string expected = SerializeAsString(*m);
      delete m;
      return SerializeAsString(p) == expected;
    } else {
      return SerializeAsString(p) == expected_;
    }
  }

  void DescribeTo(::std::ostream* os) const {
    if (use_text_format_)
      *os << text_format_;
    else
      *os << expected_;
  }
  void DescribeNegationTo(::std::ostream* os) const {
    *os << "not equal to expected message: ";
    auto out_message = use_text_format_ ? &text_format_ : &expected_;
    if (out_message->size() < 1e5)
      *os << *out_message;
    else
      *os << "(too large to print) \n";
  }

 private:
  const std::string expected_;
  const std::string text_format_;
  const bool use_text_format_ = false;
};

inline ::testing::PolymorphicMatcher<ProtoStringMatcher> EqualsProto(
    const tsl::protobuf::Message& x) {
  return ::testing::MakePolymorphicMatcher(ProtoStringMatcher(x));
}

inline ::testing::PolymorphicMatcher<ProtoStringMatcher> EqualsProto(
    const std::string& x) {
  return ::testing::MakePolymorphicMatcher(ProtoStringMatcher(x));
}

}  // namespace tools::proto_splitter
}  // namespace tensorflow

#endif  // TENSORFLOW_TOOLS_PROTO_SPLITTER_CC_TEST_UTIL_H_
