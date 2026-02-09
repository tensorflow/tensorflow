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

#include "tensorflow_text/core/kernels/fast_wordpiece_tokenizer_utils.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace tensorflow {
namespace text {
namespace fast_wordpiece_tokenizer_utils {
namespace {

// Testing spec struct for token encoding / decoding.
struct TokenSpec {
  friend std::ostream& operator<<(std::ostream& os, const TokenSpec& s) {
    return os << "token_id:" << s.token_id << ", "
              << "token_length:" << s.token_length << ", "
              << "is_suffix_token:" << s.is_suffix_token << std::endl;
  }

  int token_id;
  int token_length;
  bool is_suffix_token;
};

// Parameterized tests specs for token encoding / decoding.
const std::vector<TokenSpec>& GetTokenSpecs() {
  static const std::vector<TokenSpec>& kSpecs = *new std::vector<TokenSpec>{
      // Test 0.
      {
          .token_id = 0,
          .token_length = 1,
          .is_suffix_token = false,
      },
      // Test 1.
      {
          .token_id = 1,
          .token_length = 1,
          .is_suffix_token = false,
      },
      // Test 2.
      {
          .token_id = 2,
          .token_length = 1,
          .is_suffix_token = true,
      },
      // Test 3.
      {
          .token_id = 3,
          .token_length = 10,
          .is_suffix_token = false,
      },
      // Test 4.
      {
          .token_id = 4,
          .token_length = 10,
          .is_suffix_token = true,
      },
      // Test 5.
      {
          .token_id = kMaxSupportedVocabSize - 1,
          .token_length = kMaxVocabTokenLengthInUTF8Bytes,
          .is_suffix_token = true,
      },
  };
  return kSpecs;
}

using TokenEncodingDecodingTest = testing::TestWithParam<TokenSpec>;

TEST_P(TokenEncodingDecodingTest, GeneralTest) {
  const TokenSpec& spec = GetParam();
  ASSERT_OK_AND_ASSIGN(
      auto encoded_value,
      EncodeToken(spec.token_id, spec.token_length, spec.is_suffix_token));
  EXPECT_THAT(GetTokenId(encoded_value), spec.token_id);
  EXPECT_THAT(GetTokenLength(encoded_value), spec.token_length);
  EXPECT_THAT(IsSuffixToken(encoded_value), spec.is_suffix_token);
}

INSTANTIATE_TEST_SUITE_P(TestTokenEncodingDecoding, TokenEncodingDecodingTest,
                         testing::ValuesIn(GetTokenSpecs()));

struct FailurePopListSpec {
  friend std::ostream& operator<<(std::ostream& os,
                                  const FailurePopListSpec& s) {
    return os << "offset:" << s.offset << ", "
              << "length:" << s.length << std::endl;
  }

  int offset;
  int length;
};

// Parameterized tests specs for failure pop list encoding and decoding.
const std::vector<FailurePopListSpec>& GetFailurePopListSpecs() {
  static const std::vector<FailurePopListSpec>& kSpecs =
      *new std::vector<FailurePopListSpec>{
          // Test 0.
          {
              .offset = 0,
              .length = 1,
          },
          // Test 1.
          {
              .offset = 0,
              .length = 3,
          },
          // Test 2.
          {
              .offset = 11,
              .length = 10,
          },
          // Test 3.
          {
              .offset = kMaxSupportedFailurePoolOffset,
              .length = kMaxFailurePopsListSize,
          },
      };
  return kSpecs;
}

using FailurePopListEncodingDecodingTest =
    testing::TestWithParam<FailurePopListSpec>;

TEST_P(FailurePopListEncodingDecodingTest, GeneralTest) {
  const FailurePopListSpec& spec = GetParam();
  auto offset_and_length = EncodeFailurePopList(spec.offset, spec.length);
  int offset, length;
  GetFailurePopsOffsetAndLength(offset_and_length, offset, length);
  EXPECT_THAT(offset, spec.offset);
  EXPECT_THAT(length, spec.length);
}

INSTANTIATE_TEST_SUITE_P(TestFailurePopListEncodingDecoding,
                         FailurePopListEncodingDecodingTest,
                         testing::ValuesIn(GetFailurePopListSpecs()));

}  // namespace
}  // namespace fast_wordpiece_tokenizer_utils
}  // namespace text
}  // namespace tensorflow
