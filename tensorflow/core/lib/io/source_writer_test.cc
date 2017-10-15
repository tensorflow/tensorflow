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

#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/lib/io/source_writer.h"

namespace tensorflow {
namespace io {

TEST(WriteTest, SingleLineText) {
  SourceBufferWriter writer;
  writer.Write("You say goodbye and I say hello!");

  const char* expected = "You say goodbye and I say hello!";
  ASSERT_STREQ(expected, writer.ToString().data());
}

TEST(WriteTest, MultiLineText) {
  SourceBufferWriter writer;
  writer.Write("You say goodbye\nand I say hello!");

  const char* expected = "You say goodbye\nand I say hello!";
  ASSERT_STREQ(expected, writer.ToString().data());
}

TEST(WriteTest, MultiLineTextWithIndent) {
  SourceBufferWriter writer;
  writer.Indent(2)->Write("You say goodbye\nand I say hello!");

  const char* expected = "  You say goodbye\nand I say hello!";
  ASSERT_STREQ(expected, writer.ToString().data());
}

TEST(WriteTest, MultiLineTextWithPrefix) {
  SourceBufferWriter writer;
  writer.LinePrefix("--")->Write("You say goodbye\nand I say hello!");

  const char* expected = "--You say goodbye\nand I say hello!";
  ASSERT_STREQ(expected, writer.ToString().data());
}

TEST(WriteTest, MultiLineTextWithIndentAndPrefix) {
  SourceBufferWriter writer;
  writer.Indent(2)
      ->LinePrefix("--")
      ->Write("You say goodbye\nand I say hello!");

  const char* expected = "  --You say goodbye\nand I say hello!";
  ASSERT_STREQ(expected, writer.ToString().data());
}

TEST(InlineTest, SingleLineText) {
  SourceBufferWriter writer;
  writer.Inline("You say goodbye and I say hello!");

  const char* expected = "You say goodbye and I say hello!";
  ASSERT_STREQ(expected, writer.ToString().data());
}

TEST(InlineTest, MultiLineText) {
  SourceBufferWriter writer;
  writer.Inline("You say goodbye\nand I say hello!");

  const char* expected = "You say goodbye\nand I say hello!";
  ASSERT_STREQ(expected, writer.ToString().data());
}

TEST(InlineTest, MultiLineTextWithIndent) {
  SourceBufferWriter writer;
  writer.Indent(2)->Inline("You say goodbye\nand I say hello!");

  const char* expected = "  You say goodbye\n  and I say hello!";
  ASSERT_STREQ(expected, writer.ToString().data());
}

TEST(InlineTest, MultiLineTextWithPrefix) {
  SourceBufferWriter writer;
  writer.LinePrefix("--")->Inline("You say goodbye\nand I say hello!");

  const char* expected = "--You say goodbye\n--and I say hello!";
  ASSERT_STREQ(expected, writer.ToString().data());
}

TEST(InlineTest, MultiLineTextWithIndentAndPrefix) {
  SourceBufferWriter writer;
  writer.Indent(2)
      ->LinePrefix("--")
      ->Inline("You say goodbye\nand I say hello!");

  const char* expected = "  --You say goodbye\n  --and I say hello!";
  ASSERT_STREQ(expected, writer.ToString().data());
}

TEST(MarginTest, Basic) {
  SourceBufferWriter writer;
  writer.Write("You say goodbye")
      ->EndOfLine()
      ->Write("and I say hello!");

  const char* expected = "You say goodbye\nand I say hello!";
  ASSERT_STREQ(expected, writer.ToString().data());
}

TEST(MarginTest, Indent) {
  SourceBufferWriter writer;
  writer.Write("You say goodbye")
      ->EndOfLine()
      ->Indent(2)
      ->Write("and I say hello!");

  const char* expected = "You say goodbye\n  and I say hello!";
  ASSERT_STREQ(expected, writer.ToString().data());
}

TEST(MarginTest, IndentAndOutdent) {
  SourceBufferWriter writer;
  writer.Write("You say goodbye")
      ->EndOfLine()
      ->Indent(2)
      ->Write("and I say hello!")
      ->EndOfLine()
      ->Indent(-2)
      ->Write("Hello, hello!");

  const char* expected = "You say goodbye\n  and I say hello!\nHello, hello!";
  ASSERT_STREQ(expected, writer.ToString().data());
}

TEST(MarginTest, LinePrefix) {
  SourceBufferWriter writer;
  writer.Write("You say goodbye")
      ->EndOfLine()
      ->LinePrefix("--")
      ->Write("and I say hello!");

  const char* expected = "You say goodbye\n--and I say hello!";
  ASSERT_STREQ(expected, writer.ToString().data());
}

TEST(MarginTest, PrefixAndRemovePrefix) {
  SourceBufferWriter writer;
  writer.Write("You say goodbye")
      ->EndOfLine()
      ->LinePrefix("--")
      ->Write("and I say hello!")
      ->EndOfLine()
      ->RemoveLinePrefix()
      ->Write("Hello, hello!");

  const char* expected = "You say goodbye\n--and I say hello!\nHello, hello!";
  ASSERT_STREQ(expected, writer.ToString().data());
}

TEST(MarginTest, IndentAndPrefixAndOutdentAndRemovePrefix) {
  SourceBufferWriter writer;
  writer.Write("You say goodbye")
      ->EndOfLine()
      ->Indent(2)
      ->LinePrefix("--")
      ->Write("and I say hello!")
      ->EndOfLine()
      ->Indent(-2)
      ->RemoveLinePrefix()
      ->Write("Hello, hello!");

  const char* expected = "You say goodbye\n  --and I say hello!\nHello, hello!";
  ASSERT_STREQ(expected, writer.ToString().data());
}

TEST(MarginTest, NegativeIndent) {
  SourceBufferWriter writer;
  writer.Write("You say goodbye")
      ->EndOfLine()
      ->Indent(-10)
      ->Write("and I say hello!");

  const char* expected = "You say goodbye\nand I say hello!";
  ASSERT_STREQ(expected, writer.ToString().data());
}

TEST(MarginTest, CumulativeIndent) {
  SourceBufferWriter writer;
  writer.Write("You say goodbye")
      ->EndOfLine()
      ->Indent(2)
      ->Write("and I say hello!")
      ->EndOfLine()
      ->Indent(2)
      ->Write("Hello, hello!");

  const char* expected =
      "You say goodbye\n  and I say hello!\n    Hello, hello!";
  ASSERT_STREQ(expected, writer.ToString().data());
}

TEST(MarginTest, EmptyPrefix) {
  SourceBufferWriter writer;
  writer.Write("You say goodbye")
      ->EndOfLine()
      ->LinePrefix("")
      ->Write("and I say hello!");

  const char* expected = "You say goodbye\nand I say hello!";
  ASSERT_STREQ(expected, writer.ToString().data());
}


}  // namespace io
}  // namespace tensorflow
