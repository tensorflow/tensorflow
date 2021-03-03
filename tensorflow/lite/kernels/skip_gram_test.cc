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

#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/string_type.h"
#include "tensorflow/lite/string_util.h"

namespace tflite {
namespace {

using ::testing::ElementsAre;

static char kSentence[] = "The quick\t brown fox\n jumps over\n the lazy dog!";

class SkipGramOp : public SingleOpModel {
 public:
  SkipGramOp(int ngram_size, int max_skip_size, bool include_all_ngrams) {
    input_ = AddInput(TensorType_STRING);
    output_ = AddOutput(TensorType_STRING);

    SetBuiltinOp(BuiltinOperator_SKIP_GRAM, BuiltinOptions_SkipGramOptions,
                 CreateSkipGramOptions(builder_, ngram_size, max_skip_size,
                                       include_all_ngrams)
                     .Union());
    BuildInterpreter({{1}});
  }
  void SetInput(const string& content) {
    PopulateStringTensor(input_, {content});
  }

  std::vector<string> GetOutput() {
    std::vector<string> ans;
    TfLiteTensor* tensor = interpreter_->tensor(output_);

    int num = GetStringCount(tensor);
    for (int i = 0; i < num; i++) {
      StringRef strref = GetString(tensor, i);
      ans.push_back(string(strref.str, strref.len));
    }
    return ans;
  }

 private:
  int input_;
  int output_;
};

TEST(SkipGramTest, TestUnigram) {
  SkipGramOp m(1, 0, false);

  m.SetInput(kSentence);
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), testing::UnorderedElementsAreArray(
                                 {"The", "quick", "brown", "fox", "jumps",
                                  "over", "the", "lazy", "dog!"}));
}

TEST(SkipGramTest, TestBigram) {
  SkipGramOp m(2, 0, false);
  m.SetInput(kSentence);
  m.Invoke();
  EXPECT_THAT(m.GetOutput(),
              testing::UnorderedElementsAreArray(
                  {"The quick", "quick brown", "brown fox", "fox jumps",
                   "jumps over", "over the", "the lazy", "lazy dog!"}));
}

TEST(SkipGramTest, TestAllBigram) {
  SkipGramOp m(2, 0, true);
  m.SetInput(kSentence);
  m.Invoke();
  EXPECT_THAT(m.GetOutput(),
              testing::UnorderedElementsAreArray(
                  {// Unigram
                   "The", "quick", "brown", "fox", "jumps", "over", "the",
                   "lazy", "dog!",
                   //  Bigram
                   "The quick", "quick brown", "brown fox", "fox jumps",
                   "jumps over", "over the", "the lazy", "lazy dog!"}));
}

TEST(SkipGramTest, TestAllTrigram) {
  SkipGramOp m(3, 0, true);
  m.SetInput(kSentence);
  m.Invoke();
  EXPECT_THAT(m.GetOutput(),
              testing::UnorderedElementsAreArray(
                  {// Unigram
                   "The", "quick", "brown", "fox", "jumps", "over", "the",
                   "lazy", "dog!",
                   // Bigram
                   "The quick", "quick brown", "brown fox", "fox jumps",
                   "jumps over", "over the", "the lazy", "lazy dog!",
                   // Trigram
                   "The quick brown", "quick brown fox", "brown fox jumps",
                   "fox jumps over", "jumps over the", "over the lazy",
                   "the lazy dog!"}));
}

TEST(SkipGramTest, TestSkip1Bigram) {
  SkipGramOp m(2, 1, false);
  m.SetInput(kSentence);
  m.Invoke();
  EXPECT_THAT(
      m.GetOutput(),
      testing::UnorderedElementsAreArray(
          {"The quick", "The brown", "quick brown", "quick fox", "brown fox",
           "brown jumps", "fox jumps", "fox over", "jumps over", "jumps the",
           "over the", "over lazy", "the lazy", "the dog!", "lazy dog!"}));
}

TEST(SkipGramTest, TestSkip2Bigram) {
  SkipGramOp m(2, 2, false);
  m.SetInput(kSentence);
  m.Invoke();
  EXPECT_THAT(m.GetOutput(),
              testing::UnorderedElementsAreArray(
                  {"The quick",  "The brown",   "The fox",    "quick brown",
                   "quick fox",  "quick jumps", "brown fox",  "brown jumps",
                   "brown over", "fox jumps",   "fox over",   "fox the",
                   "jumps over", "jumps the",   "jumps lazy", "over the",
                   "over lazy",  "over dog!",   "the lazy",   "the dog!",
                   "lazy dog!"}));
}

TEST(SkipGramTest, TestSkip1Trigram) {
  SkipGramOp m(3, 1, false);
  m.SetInput(kSentence);
  m.Invoke();
  EXPECT_THAT(m.GetOutput(),
              testing::UnorderedElementsAreArray(
                  {"The quick brown", "The quick fox",    "The brown fox",
                   "The brown jumps", "quick brown fox",  "quick brown jumps",
                   "quick fox jumps", "quick fox over",   "brown fox jumps",
                   "brown fox over",  "brown jumps over", "brown jumps the",
                   "fox jumps over",  "fox jumps the",    "fox over the",
                   "fox over lazy",   "jumps over the",   "jumps over lazy",
                   "jumps the lazy",  "jumps the dog!",   "over the lazy",
                   "over the dog!",   "over lazy dog!",   "the lazy dog!"}));
}

TEST(SkipGramTest, TestSkip2Trigram) {
  SkipGramOp m(3, 2, false);
  m.SetInput(kSentence);
  m.Invoke();
  EXPECT_THAT(m.GetOutput(),
              testing::UnorderedElementsAreArray(
                  {"The quick brown",  "The quick fox",     "The quick jumps",
                   "The brown fox",    "The brown jumps",   "The brown over",
                   "The fox jumps",    "The fox over",      "The fox the",
                   "quick brown fox",  "quick brown jumps", "quick brown over",
                   "quick fox jumps",  "quick fox over",    "quick fox the",
                   "quick jumps over", "quick jumps the",   "quick jumps lazy",
                   "brown fox jumps",  "brown fox over",    "brown fox the",
                   "brown jumps over", "brown jumps the",   "brown jumps lazy",
                   "brown over the",   "brown over lazy",   "brown over dog!",
                   "fox jumps over",   "fox jumps the",     "fox jumps lazy",
                   "fox over the",     "fox over lazy",     "fox over dog!",
                   "fox the lazy",     "fox the dog!",      "jumps over the",
                   "jumps over lazy",  "jumps over dog!",   "jumps the lazy",
                   "jumps the dog!",   "jumps lazy dog!",   "over the lazy",
                   "over the dog!",    "over lazy dog!",    "the lazy dog!"}));
}

TEST(SkipGramTest, TestAllSkip2Trigram) {
  SkipGramOp m(3, 2, true);
  m.SetInput(kSentence);
  m.Invoke();
  EXPECT_THAT(
      m.GetOutput(),
      testing::UnorderedElementsAreArray(
          {// Unigram
           "The", "quick", "brown", "fox", "jumps", "over", "the", "lazy",
           "dog!",
           // Bigram
           "The quick", "The brown", "The fox", "quick brown", "quick fox",
           "quick jumps", "brown fox", "brown jumps", "brown over", "fox jumps",
           "fox over", "fox the", "jumps over", "jumps the", "jumps lazy",
           "over the", "over lazy", "over dog!", "the lazy", "the dog!",
           "lazy dog!",
           // Trigram
           "The quick brown", "The quick fox", "The quick jumps",
           "The brown fox", "The brown jumps", "The brown over",
           "The fox jumps", "The fox over", "The fox the", "quick brown fox",
           "quick brown jumps", "quick brown over", "quick fox jumps",
           "quick fox over", "quick fox the", "quick jumps over",
           "quick jumps the", "quick jumps lazy", "brown fox jumps",
           "brown fox over", "brown fox the", "brown jumps over",
           "brown jumps the", "brown jumps lazy", "brown over the",
           "brown over lazy", "brown over dog!", "fox jumps over",
           "fox jumps the", "fox jumps lazy", "fox over the", "fox over lazy",
           "fox over dog!", "fox the lazy", "fox the dog!", "jumps over the",
           "jumps over lazy", "jumps over dog!", "jumps the lazy",
           "jumps the dog!", "jumps lazy dog!", "over the lazy",
           "over the dog!", "over lazy dog!", "the lazy dog!"}));
}

TEST(SkipGramTest, TestSingleWord) {
  SkipGramOp m(1, 1, false);
  m.SetInput("Hi");
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAre("Hi"));
}

TEST(SkipGramTest, TestWordsLessThanGram) {
  SkipGramOp m(3, 1, false);
  m.SetInput("Hi hi");
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), std::vector<string>());
}

TEST(SkipGramTest, TestEmptyInput) {
  SkipGramOp m(1, 1, false);
  m.SetInput("");
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAre());
}

TEST(SkipGramTest, TestWhitespaceInput) {
  SkipGramOp m(1, 1, false);
  m.SetInput("    ");
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAre());
}

TEST(SkipGramTest, TestInputWithExtraSpace) {
  SkipGramOp m(1, 1, false);
  m.SetInput("   Hello   world    !  ");
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAre("Hello", "world", "!"));
}

}  // namespace
}  // namespace tflite
