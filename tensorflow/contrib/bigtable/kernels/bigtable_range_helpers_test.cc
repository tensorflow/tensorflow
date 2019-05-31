/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/contrib/bigtable/kernels/bigtable_range_helpers.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

TEST(MultiModeKeyRangeTest, SimplePrefix) {
  MultiModeKeyRange r = MultiModeKeyRange::FromPrefix("prefix");
  EXPECT_EQ("prefix", r.begin_key());
  EXPECT_EQ("prefiy", r.end_key());
  EXPECT_TRUE(r.contains_key("prefixed_key"));
  EXPECT_FALSE(r.contains_key("not-prefixed-key"));
  EXPECT_FALSE(r.contains_key("prefi"));
  EXPECT_FALSE(r.contains_key("prefiy"));
  EXPECT_FALSE(r.contains_key("early"));
  EXPECT_FALSE(r.contains_key(""));
}

TEST(MultiModeKeyRangeTest, Range) {
  MultiModeKeyRange r = MultiModeKeyRange::FromRange("a", "b");
  EXPECT_EQ("a", r.begin_key());
  EXPECT_EQ("b", r.end_key());
  EXPECT_TRUE(r.contains_key("a"));
  EXPECT_TRUE(r.contains_key("ab"));
  EXPECT_FALSE(r.contains_key("b"));
  EXPECT_FALSE(r.contains_key("bc"));
  EXPECT_FALSE(r.contains_key("A"));
  EXPECT_FALSE(r.contains_key("B"));
  EXPECT_FALSE(r.contains_key(""));
}

TEST(MultiModeKeyRangeTest, InvertedRange) {
  MultiModeKeyRange r = MultiModeKeyRange::FromRange("b", "a");
  EXPECT_FALSE(r.contains_key("a"));
  EXPECT_FALSE(r.contains_key("b"));
  EXPECT_FALSE(r.contains_key(""));
}

TEST(MultiModeKeyRangeTest, EmptyPrefix) {
  MultiModeKeyRange r = MultiModeKeyRange::FromPrefix("");
  EXPECT_EQ("", r.begin_key());
  EXPECT_EQ("", r.end_key());
  EXPECT_TRUE(r.contains_key(""));
  EXPECT_TRUE(r.contains_key("a"));
  EXPECT_TRUE(r.contains_key("z"));
  EXPECT_TRUE(r.contains_key("A"));
  EXPECT_TRUE(r.contains_key("ZZZZZZ"));
}

TEST(MultiModeKeyRangeTest, HalfRange) {
  MultiModeKeyRange r = MultiModeKeyRange::FromRange("start", "");
  EXPECT_EQ("start", r.begin_key());
  EXPECT_EQ("", r.end_key());
  EXPECT_TRUE(r.contains_key("start"));
  EXPECT_TRUE(r.contains_key("starting"));
  EXPECT_TRUE(r.contains_key("z-end"));
  EXPECT_FALSE(r.contains_key(""));
  EXPECT_FALSE(r.contains_key("early"));
}

TEST(MultiModeKeyRangeTest, PrefixWrapAround) {
  string prefix = "abc\xff";
  MultiModeKeyRange r = MultiModeKeyRange::FromPrefix(prefix);
  EXPECT_EQ(prefix, r.begin_key());
  EXPECT_EQ("abd", r.end_key());

  EXPECT_TRUE(r.contains_key("abc\xff\x07"));
  EXPECT_TRUE(r.contains_key("abc\xff\x15"));
  EXPECT_TRUE(r.contains_key("abc\xff\x61"));
  EXPECT_TRUE(r.contains_key("abc\xff\xff"));
  EXPECT_FALSE(r.contains_key("abc\0"));
  EXPECT_FALSE(r.contains_key("abd"));
}

TEST(MultiModeKeyRangeTest, PrefixSignedWrapAround) {
  string prefix = "abc\x7f";
  MultiModeKeyRange r = MultiModeKeyRange::FromPrefix(prefix);
  EXPECT_EQ(prefix, r.begin_key());
  EXPECT_EQ("abc\x80", r.end_key());

  EXPECT_TRUE(r.contains_key("abc\x7f\x07"));
  EXPECT_TRUE(r.contains_key("abc\x7f\x15"));
  EXPECT_TRUE(r.contains_key("abc\x7f\x61"));
  EXPECT_TRUE(r.contains_key("abc\x7f\xff"));
  EXPECT_FALSE(r.contains_key("abc\0"));
  EXPECT_FALSE(r.contains_key("abc\x01"));
  EXPECT_FALSE(r.contains_key("abd"));
  EXPECT_FALSE(r.contains_key("ab\x80"));
}

}  // namespace
}  // namespace tensorflow
