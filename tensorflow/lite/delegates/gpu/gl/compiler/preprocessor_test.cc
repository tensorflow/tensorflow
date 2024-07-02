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

#include "tensorflow/lite/delegates/gpu/gl/compiler/preprocessor.h"

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

class AccuInlineRewrite : public InlineRewrite {
 public:
  explicit AccuInlineRewrite(std::vector<std::string>* blocks)
      : blocks_(blocks) {}

  RewriteStatus Rewrite(absl::string_view input, std::string* output) final {
    blocks_->push_back(std::string(input.data(), input.size()));
    output->append("r:");
    output->append(input.data(), input.size());
    return RewriteStatus::SUCCESS;
  }

  std::vector<std::string>* blocks_;
};

std::vector<std::string> ParseInlines(const std::string& text) {
  std::vector<std::string> blocks;
  TextPreprocessor preprocessor('$', false);
  AccuInlineRewrite rewrite(&blocks);
  preprocessor.AddRewrite(&rewrite);
  std::string discard;
  preprocessor.Rewrite(text, &discard).IgnoreError();
  return blocks;
}

TEST(Preprocessor, CornerCases) {
  EXPECT_THAT(ParseInlines(""), testing::ElementsAre());
  EXPECT_THAT(ParseInlines("text text"), testing::ElementsAre());
  EXPECT_THAT(ParseInlines("$$"), testing::ElementsAre(""));
}

TEST(Preprocessor, One) {
  EXPECT_THAT(ParseInlines("$text$"), testing::ElementsAre("text"));
  EXPECT_THAT(ParseInlines(" $text$ "), testing::ElementsAre("text"));
}

TEST(Preprocessor, More) {
  EXPECT_THAT(ParseInlines("Test $inline1$\n$inline2$ test $inline3$ "),
              testing::ElementsAre("inline1", "inline2", "inline3"));
}

std::string RewriteInlines(const std::string& text) {
  std::vector<std::string> blocks;
  TextPreprocessor preprocessor('$', false);
  AccuInlineRewrite rewrite(&blocks);
  preprocessor.AddRewrite(&rewrite);
  std::string out;
  preprocessor.Rewrite(text, &out).IgnoreError();
  return out;
}

TEST(Preprocessor, RewriteCornerCases) {
  EXPECT_EQ(RewriteInlines(""), "");
  EXPECT_EQ(RewriteInlines("text text"), "text text");
  EXPECT_EQ(RewriteInlines("$$"), "r:");
}

TEST(Preprocessor, RewriteOne) {
  EXPECT_EQ(RewriteInlines("$text$"), "r:text");
  EXPECT_EQ(RewriteInlines(" $text$ "), " r:text ");
}

TEST(Preprocessor, RewriteMore) {
  EXPECT_EQ(RewriteInlines("Test $inline1$\n$inline2$ test $inline3$ "),
            "Test r:inline1\nr:inline2 test r:inline3 ");
}

class SingleRewrite : public InlineRewrite {
 public:
  RewriteStatus Rewrite(absl::string_view input, std::string* output) final {
    if (input == "foo") {
      output->append("bla");
      return RewriteStatus::SUCCESS;
    }
    return RewriteStatus::NOT_RECOGNIZED;
  }

  std::vector<std::string>* blocks_;
};

TEST(Preprocessor, KeepUnknownRewrites) {
  TextPreprocessor preprocessor('$', true);
  SingleRewrite rewrite;
  preprocessor.AddRewrite(&rewrite);
  std::string out;
  ASSERT_TRUE(preprocessor.Rewrite("Good morning, $name$! $foo$", &out).ok());
  EXPECT_EQ("Good morning, $name$! bla", out);
}

TEST(Preprocessor, KeepUnknownRewrites_Fail) {
  TextPreprocessor preprocessor('$', false);
  SingleRewrite rewrite;
  preprocessor.AddRewrite(&rewrite);
  std::string out;
  EXPECT_FALSE(preprocessor.Rewrite("Good morning, $name$! $foo$", &out).ok());
}

}  // namespace
}  // namespace gl
}  // namespace gpu
}  // namespace tflite
