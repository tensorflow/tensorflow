/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/util/proto/proto_utils.h"

#include <gmock/gmock.h>
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

using proto_utils::ParseTextFormatFromString;
using proto_utils::StringErrorCollector;
using ::testing::ContainsRegex;

TEST(ParseTextFormatFromStringTest, Success) {
  protobuf::DescriptorProto output;
  TF_ASSERT_OK(ParseTextFormatFromString("name: \"foo\"", &output));
  EXPECT_EQ(output.name(), "foo");
}

TEST(ParseTextFormatFromStringTest, ErrorOnInvalidSyntax) {
  protobuf::DescriptorProto output;
  Status status = ParseTextFormatFromString("name: foo", &output);
  EXPECT_EQ(status.code(), error::INVALID_ARGUMENT);
  EXPECT_THAT(status.error_message(), ContainsRegex("foo"));
  EXPECT_FALSE(output.has_name());
}

TEST(ParseTextFormatFromStringTest, ErrorOnUnknownFieldName) {
  protobuf::DescriptorProto output;
  Status status = ParseTextFormatFromString("badname: \"foo\"", &output);
  EXPECT_EQ(status.code(), error::INVALID_ARGUMENT);
  EXPECT_THAT(status.error_message(), ContainsRegex("badname"));
  EXPECT_FALSE(output.has_name());
}

TEST(ParseTextFormatFromStringTest, DiesOnNullOutputPointer) {
#ifndef NDEBUG
  ASSERT_DEATH(ParseTextFormatFromString("foo", nullptr).IgnoreError(),
               "output.*non NULL");
#else
  // Under NDEBUG we don't die but should still return an error status.
  Status status = ParseTextFormatFromString("foo", nullptr);
  EXPECT_EQ(status.code(), error::INVALID_ARGUMENT);
  EXPECT_THAT(status.error_message(), ContainsRegex("output.*non NULL"));
#endif
}

TEST(StringErrorCollectorTest, AppendsError) {
  string err;
  StringErrorCollector collector(&err);
  collector.AddError(1, 2, "foo");
  EXPECT_EQ("1(2): foo\n", err);
}

TEST(StringErrorCollectorTest, AppendsWarning) {
  string err;
  StringErrorCollector collector(&err);
  collector.AddWarning(1, 2, "foo");
  EXPECT_EQ("1(2): foo\n", err);
}

TEST(StringErrorCollectorTest, AppendsMultipleError) {
  string err;
  StringErrorCollector collector(&err);
  collector.AddError(1, 2, "foo");
  collector.AddError(3, 4, "bar");
  EXPECT_EQ("1(2): foo\n3(4): bar\n", err);
}

TEST(StringErrorCollectorTest, AppendsMultipleWarning) {
  string err;
  StringErrorCollector collector(&err);
  collector.AddWarning(1, 2, "foo");
  collector.AddWarning(3, 4, "bar");
  EXPECT_EQ("1(2): foo\n3(4): bar\n", err);
}

TEST(StringErrorCollectorTest, OffsetWorks) {
  string err;
  StringErrorCollector collector(&err, true);
  collector.AddError(1, 2, "foo");
  collector.AddWarning(3, 4, "bar");
  EXPECT_EQ("2(3): foo\n4(5): bar\n", err);
}

TEST(StringErrorCollectorTest, DiesOnNullErrorText) {
#ifndef NDEBUG
  ASSERT_DEATH(StringErrorCollector(nullptr), "error_text.*non NULL");
#else
  // Under NDEBUG we don't die and instead AddError/AddWarning just do nothing.
  StringErrorCollector collector(nullptr);
  collector.AddError(1, 2, "foo");
  collector.AddWarning(3, 4, "bar");
#endif
}

}  // namespace tensorflow
