/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/experimental/support/codegen/utils.h"

#include <gtest/gtest.h>

namespace tflite {
namespace support {
namespace codegen {
namespace {

TEST(ErrorReporterTest, TestReportError) {
  ErrorReporter err;
  err.Error("some text");
  EXPECT_EQ(err.GetMessage(), "[ERROR] some text\n");
  EXPECT_EQ(err.GetMessage(), "");
}

TEST(CodeGeneratorTest, TestExample) {
  ErrorReporter err;
  CodeWriter writer(&err);
  writer.SetTokenValue("NAME", "Foo");
  const std::string text = R"(void {{NAME}}() { printf("%s", "{{NAME}}"); })";
  writer.Append(text);
  writer.SetTokenValue("NAME", "Bar");
  writer.Append(text);
  EXPECT_EQ(
      "void Foo() { printf(\"%s\", \"Foo\"); }\n"
      "void Bar() { printf(\"%s\", \"Bar\"); }\n",
      writer.ToString());
}

TEST(CodeGeneratorTest, TestInexistentToken) {
  ErrorReporter err;
  CodeWriter writer(&err);
  writer.SetTokenValue("NAME", "Foo");
  const std::string text = R"(void {{name}}() {})";
  writer.Append(text);
  EXPECT_EQ(err.GetMessage(),
            "[ERROR] Internal: Cannot find value with token 'name'\n");
}

TEST(CodeGeneratorTest, TestUnclosedToken) {
  ErrorReporter err;
  CodeWriter writer(&err);
  writer.SetTokenValue("NAME", "Foo");
  const std::string text = R"(void {{NAME}() {})";
  writer.Append(text);
  EXPECT_EQ(err.GetMessage(),
            "[ERROR] Internal: Invalid template: {{token}} is not closed.\n");
}

TEST(CodeGeneratorTest, TestIndentControl) {
  ErrorReporter err;
  CodeWriter writer(&err);
  writer.SetIndentString("  ");
  writer.Indent();
  writer.AppendNoNewLine("abcde");  // Will indent
  EXPECT_EQ("  abcde", writer.ToString());
  writer.Clear();
  writer.Indent();
  writer.AppendNoNewLine("abc\n\nde");
  // The blank line will not indent
  EXPECT_EQ("  abc\n\n  de", writer.ToString());
  writer.Clear();
  writer.Indent();
  writer.Append("abc");
  writer.Outdent();
  writer.AppendNoNewLine("def");
  EXPECT_EQ("  abc\ndef", writer.ToString());
}

TEST(CaseConversionTest, TestSnakeToCamel) {
  EXPECT_EQ("imACamel", SnakeCaseToCamelCase("im_a_camel"));
  EXPECT_EQ("imACamel", SnakeCaseToCamelCase("im_a_camel_"));
  EXPECT_EQ("ImACamel", SnakeCaseToCamelCase("_im_a_camel"));
  EXPECT_EQ("", SnakeCaseToCamelCase("_"));
  EXPECT_EQ("camel", SnakeCaseToCamelCase("camel"));
}

}  // namespace
}  // namespace codegen
}  // namespace support
}  // namespace tflite
