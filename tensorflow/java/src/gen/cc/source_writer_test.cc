/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/java/src/gen/cc/source_writer.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

TEST(AppendTest, SingleLineText) {
  SourceBufferWriter writer;
  writer.Append("You say goodbye and I say hello!");

  const char* expected = "You say goodbye and I say hello!";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(AppendTest, MultiLineText) {
  SourceBufferWriter writer;
  writer.Append("You say goodbye\nand I say hello!");

  const char* expected = "You say goodbye\nand I say hello!";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(AppendTest, MultiLineTextWithIndent) {
  SourceBufferWriter writer;
  writer.Indent(2).Append("You say goodbye\nand I say hello!");

  const char* expected = "  You say goodbye\nand I say hello!";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(AppendTest, MultiLineTextWithPrefix) {
  SourceBufferWriter writer;
  writer.Prefix("--").Append("You say goodbye\nand I say hello!");

  const char* expected = "--You say goodbye\nand I say hello!";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(AppendTest, MultiLineTextWithIndentAndPrefix) {
  SourceBufferWriter writer;
  writer.Indent(2).Prefix("--").Append("You say goodbye\nand I say hello!");

  const char* expected = "  --You say goodbye\nand I say hello!";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(WriteTest, SingleLineText) {
  SourceBufferWriter writer;
  writer.Write("You say goodbye and I say hello!");

  const char* expected = "You say goodbye and I say hello!";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(WriteTest, MultiLineText) {
  SourceBufferWriter writer;
  writer.Write("You say goodbye\nand I say hello!");

  const char* expected = "You say goodbye\nand I say hello!";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(WriteTest, MultiLineTextWithIndent) {
  SourceBufferWriter writer;
  writer.Indent(2).Write("You say goodbye\nand I say hello!");

  const char* expected = "  You say goodbye\n  and I say hello!";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(WriteTest, MultiLineTextWithPrefix) {
  SourceBufferWriter writer;
  writer.Prefix("--").Write("You say goodbye\nand I say hello!");

  const char* expected = "--You say goodbye\n--and I say hello!";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(WriteTest, MultiLineTextWithIndentAndPrefix) {
  SourceBufferWriter writer;
  writer.Indent(2).Prefix("--").Write("You say goodbye\nand I say hello!");

  const char* expected = "  --You say goodbye\n  --and I say hello!";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(MarginTest, Basic) {
  SourceBufferWriter writer;
  writer.Append("You say goodbye").EndLine().Append("and I say hello!");

  const char* expected = "You say goodbye\nand I say hello!";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(MarginTest, Indent) {
  SourceBufferWriter writer;
  writer.Append("You say goodbye")
      .EndLine()
      .Indent(2)
      .Append("and I say hello!");

  const char* expected = "You say goodbye\n  and I say hello!";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(MarginTest, IndentAndOutdent) {
  SourceBufferWriter writer;
  writer.Append("You say goodbye")
      .EndLine()
      .Indent(2)
      .Append("and I say hello!")
      .EndLine()
      .Indent(-2)
      .Append("Hello, hello!");

  const char* expected = "You say goodbye\n  and I say hello!\nHello, hello!";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(MarginTest, Prefix) {
  SourceBufferWriter writer;
  writer.Append("You say goodbye")
      .EndLine()
      .Prefix("--")
      .Append("and I say hello!");

  const char* expected = "You say goodbye\n--and I say hello!";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(MarginTest, PrefixAndRemovePrefix) {
  SourceBufferWriter writer;
  writer.Append("You say goodbye")
      .EndLine()
      .Prefix("--")
      .Append("and I say hello!")
      .EndLine()
      .Prefix("")
      .Append("Hello, hello!");

  const char* expected = "You say goodbye\n--and I say hello!\nHello, hello!";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(MarginTest, IndentAndPrefixAndOutdentAndRemovePrefix) {
  SourceBufferWriter writer;
  writer.Append("You say goodbye")
      .EndLine()
      .Indent(2)
      .Prefix("--")
      .Append("and I say hello!")
      .EndLine()
      .Indent(-2)
      .Prefix("")
      .Append("Hello, hello!");

  const char* expected = "You say goodbye\n  --and I say hello!\nHello, hello!";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(MarginTest, NegativeIndent) {
  SourceBufferWriter writer;
  writer.Append("You say goodbye")
      .EndLine()
      .Indent(-10)
      .Append("and I say hello!");

  const char* expected = "You say goodbye\nand I say hello!";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(MarginTest, CumulativeIndent) {
  SourceBufferWriter writer;
  writer.Append("You say goodbye")
      .EndLine()
      .Indent(2)
      .Append("and I say hello!")
      .EndLine()
      .Indent(2)
      .Append("Hello, hello!");

  const char* expected =
      "You say goodbye\n  and I say hello!\n    Hello, hello!";
  ASSERT_STREQ(expected, writer.str().data());
}

TEST(MarginTest, EmptyPrefix) {
  SourceBufferWriter writer;
  writer.Append("You say goodbye")
      .EndLine()
      .Prefix("")
      .Append("and I say hello!");

  const char* expected = "You say goodbye\nand I say hello!";
  ASSERT_STREQ(expected, writer.str().data());
}

}  // namespace
}  // namespace tensorflow
