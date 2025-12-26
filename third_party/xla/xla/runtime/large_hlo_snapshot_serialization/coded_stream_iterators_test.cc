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

#include "xla/runtime/large_hlo_snapshot_serialization/coded_stream_iterators.h"

#include <string>

#include <gtest/gtest.h>
#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"
#include "tsl/platform/protobuf.h"

namespace xla {
namespace {

TEST(CodedStreamIteratorsTest, CodedStreamInputIteratorEof) {
  std::string data = "";
  tsl::protobuf::io::ArrayInputStream input_stream(data.data(), data.size());
  tsl::protobuf::io::CodedInputStream coded_input_stream(&input_stream);
  CodedStreamInputIterator iterator(&coded_input_stream);
  EXPECT_TRUE(iterator == iterator);
  EXPECT_FALSE(iterator != iterator);
  EXPECT_TRUE(iterator == CodedStreamInputIterator());
}

TEST(CodedStreamIteratorsTest, CodedStreamInputIteratorRead) {
  std::string data = "hello";
  tsl::protobuf::io::ArrayInputStream input_stream(data.data(), data.size());
  tsl::protobuf::io::CodedInputStream coded_input_stream(&input_stream);
  CodedStreamInputIterator iterator(&coded_input_stream);
  EXPECT_EQ(*iterator, 'h');
  ++iterator;
  EXPECT_EQ(*iterator, 'e');
  iterator++;  // Test postfix operator
  EXPECT_EQ(*iterator, 'l');
  ++iterator;
}

TEST(CodedStreamIteratorsTest, CodedStreamInputIteratorReadLimit) {
  std::string data = "hello";
  tsl::protobuf::io::ArrayInputStream input_stream(data.data(), data.size());
  tsl::protobuf::io::CodedInputStream coded_input_stream(&input_stream);
  CodedStreamInputIterator iterator(&coded_input_stream, 2);
  EXPECT_EQ(*iterator, 'h');
  ++iterator;
  EXPECT_EQ(*iterator, 'e');
  ++iterator;
  EXPECT_TRUE(iterator == CodedStreamInputIterator());
}

TEST(CodedStreamIteratorsTest, CodedStreamOutputIteratorWrite) {
  std::string data = "";
  tsl::protobuf::io::StringOutputStream output_stream(&data);
  tsl::protobuf::io::CodedOutputStream coded_output_stream(&output_stream);
  CodedStreamOutputIterator iterator(&coded_output_stream);
  *iterator = 'h';
  *iterator = 'e';
  *iterator = 'l';
  *iterator = 'l';
  *iterator = 'o';
  coded_output_stream.Trim();
  EXPECT_EQ(data, "hello");
}

}  // namespace
}  // namespace xla
