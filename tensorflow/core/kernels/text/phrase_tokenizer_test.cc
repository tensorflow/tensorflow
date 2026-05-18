// Copyright 2025 TF.Text Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/core/kernels/text/phrase_tokenizer.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/tsl/platform/statusor.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/resource_loader.h"

namespace tensorflow {
namespace text {
namespace {

using ::testing::ElementsAre;

/* With the following vocab:
<UNK>
I
heard
the
news
today
have
heard news today
the news today
*/
constexpr char kTestConfigPath[] =
    "tensorflow/core/kernels/text/test_data/"
    "phrase_tokenizer_model.fb";

TEST(PhraseTokenizerTest, Tokenize) {
  absl::string_view input("I heard the news      today");
  std::vector<std::string> output_tokens;
  std::vector<int> output_token_ids;

  std::string config_flatbuffer;
  auto status = tensorflow::ReadFileToString(
      tensorflow::Env::Default(),
      tensorflow::GetDataDependencyFilepath(kTestConfigPath),
      &config_flatbuffer);
  ASSERT_TRUE(status.ok());

  TF_ASSERT_OK_AND_ASSIGN(auto tokenizer,
                          PhraseTokenizer::Create(config_flatbuffer.data()));

  tokenizer.Tokenize(input, &output_tokens, &output_token_ids);
  EXPECT_THAT(output_tokens, ElementsAre("I", "heard", "the news today"));
  EXPECT_THAT(output_token_ids, ElementsAre(1, 2, 8));
}

TEST(PhraseTokenizerTest, TokenizeLonger) {
  absl::string_view input("I heard the news      today I heard");
  std::vector<std::string> output_tokens;
  std::vector<int> output_token_ids;

  std::string config_flatbuffer;
  auto status = tensorflow::ReadFileToString(
      tensorflow::Env::Default(),
      tensorflow::GetDataDependencyFilepath(kTestConfigPath),
      &config_flatbuffer);
  ASSERT_TRUE(status.ok());

  TF_ASSERT_OK_AND_ASSIGN(auto tokenizer,
                          PhraseTokenizer::Create(config_flatbuffer.data()));

  tokenizer.Tokenize(input, &output_tokens, &output_token_ids);
  EXPECT_THAT(output_tokens,
              ElementsAre("I", "heard", "the news today", "I", "heard"));
  EXPECT_THAT(output_token_ids, ElementsAre(1, 2, 8, 1, 2));
}

TEST(PhraseTokenizerTest, DeTokenize) {
  std::vector<int> input({1, 2, 8});

  std::string config_flatbuffer;
  auto status = tensorflow::ReadFileToString(
      tensorflow::Env::Default(),
      tensorflow::GetDataDependencyFilepath(kTestConfigPath),
      &config_flatbuffer);
  ASSERT_TRUE(status.ok());

  TF_ASSERT_OK_AND_ASSIGN(auto tokenizer,
                          PhraseTokenizer::Create(config_flatbuffer.data()));

  auto output_string = tokenizer.Detokenize(input);
  EXPECT_EQ(output_string.value(), "I heard the news today");
}

}  // namespace
}  // namespace text
}  // namespace tensorflow
