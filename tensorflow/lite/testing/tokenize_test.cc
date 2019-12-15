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
#include "tensorflow/lite/testing/tokenize.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace tflite {
namespace testing {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

class TokenCollector : public TokenProcessor {
 public:
  void ConsumeToken(std::string* token) override { tokens_.push_back(*token); }
  const std::vector<std::string>& Tokens() { return tokens_; }

 private:
  std::vector<std::string> tokens_;
};

std::vector<std::string> TokenizeString(const std::string& s) {
  std::stringstream ss(s);
  TokenCollector collector;
  Tokenize(&ss, &collector);
  return collector.Tokens();
}

TEST(TokenizeTest, TokenDetection) {
  EXPECT_THAT(TokenizeString("x :1"), ElementsAre("x", ":", "1"));
  EXPECT_THAT(TokenizeString("x:1"), ElementsAre("x", ":", "1"));
  EXPECT_THAT(TokenizeString("x {1"), ElementsAre("x", "{", "1"));
  EXPECT_THAT(TokenizeString("x{1"), ElementsAre("x", "{", "1"));
  EXPECT_THAT(TokenizeString("x }1"), ElementsAre("x", "}", "1"));
  EXPECT_THAT(TokenizeString("x}1"), ElementsAre("x", "}", "1"));
  EXPECT_THAT(TokenizeString("x \"1"), ElementsAre("x", "1"));
  EXPECT_THAT(TokenizeString("x\"1"), ElementsAre("x", "1"));
}

TEST(TokenizeTest, QuotedTokenDetection) {
  EXPECT_THAT(TokenizeString("\"w:x{y}z\"1"), ElementsAre("w:x{y}z", "1"));
  EXPECT_THAT(TokenizeString("\"w:x{y}z\"\"1\""), ElementsAre("w:x{y}z", "1"));
}

TEST(TokenizeTest, Delimiters) {
  EXPECT_THAT(TokenizeString("}"), ElementsAre("}"));
  EXPECT_THAT(TokenizeString("}}"), ElementsAre("}", "}"));
  EXPECT_THAT(TokenizeString("{"), ElementsAre("{"));
  EXPECT_THAT(TokenizeString("{{"), ElementsAre("{", "{"));
  EXPECT_THAT(TokenizeString(":"), ElementsAre(":"));
  EXPECT_THAT(TokenizeString("::"), ElementsAre(":", ":"));
}

TEST(TokenizeTest, CornerCases) {
  EXPECT_THAT(TokenizeString("  i { b:a } "),
              ElementsAre("i", "{", "b", ":", "a", "}"));
  EXPECT_THAT(TokenizeString(" }"), ElementsAre("}"));
  EXPECT_THAT(TokenizeString(" }  "), ElementsAre("}"));
  EXPECT_THAT(TokenizeString(" {}  "), ElementsAre("{", "}"));
  EXPECT_THAT(TokenizeString(" x{}  y{} "),
              ElementsAre("x", "{", "}", "y", "{", "}"));
  EXPECT_THAT(TokenizeString("x:1 y:2 "),
              ElementsAre("x", ":", "1", "y", ":", "2"));
  EXPECT_THAT(TokenizeString("x:\"1\" y:2 "),
              ElementsAre("x", ":", "1", "y", ":", "2"));
  EXPECT_THAT(TokenizeString("x:\"1, 2\" y:\"\" "),
              ElementsAre("x", ":", "1, 2", "y", ":", ""));
}

TEST(TokenizeTest, NewLines) {
  EXPECT_THAT(TokenizeString("x:\n1,\n 2 \n  y :\n3 \n"),
              ElementsAre("x", ":", "1,", "2", "y", ":", "3"));
}

TEST(TokenizeTest, LongString) {
  EXPECT_THAT(
      TokenizeString("   i { b:a } input {"
                     "a: \"1e-1, 2,3\" b:\"1,2,3\"\n c{ "
                     "id:1 x{d{a:"
                     "1}}} f:2 "
                     "\n}\n t:1"),
      ElementsAreArray({"i",  "{", "b",         ":", "a", "}",     "input", "{",
                        "a",  ":", "1e-1, 2,3", "b", ":", "1,2,3", "c",     "{",
                        "id", ":", "1",         "x", "{", "d",     "{",     "a",
                        ":",  "1", "}",         "}", "}", "f",     ":",     "2",
                        "}",  "t", ":",         "1"}));
}

}  // namespace
}  // namespace testing
}  // namespace tflite
