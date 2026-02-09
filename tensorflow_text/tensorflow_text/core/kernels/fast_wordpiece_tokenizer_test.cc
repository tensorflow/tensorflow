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

#include "tensorflow_text/core/kernels/fast_wordpiece_tokenizer.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/flags/flag.h"
#include "icu4c/source/common/unicode/uchar.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow_text/core/kernels/fast_wordpiece_tokenizer_model_builder.h"

namespace tensorflow {
namespace text {
namespace {

using ::testing::AnyOf;
using ::testing::ElementsAre;

constexpr char kTestConfigPath[] =
    "tensorflow_text/python/ops/test_data/"
    "fast_wordpiece_tokenizer_model.fb";

TEST(FastWordpieceTokenizerTest, LoadAndTokenize) {
  std::string config_flatbuffer;
  auto status = tensorflow::ReadFileToString(
      tensorflow::Env::Default(), kTestConfigPath, &config_flatbuffer);
  ASSERT_TRUE(status.ok());

  // The config_flatbuffer used here is built from the following config:
  //  * vocab = {"a", "abc", "abcdefghi", "##de", "##defgxy", "##deh", "##f",
  //             "##ghz", "<unk>"}
  //  * unk_token = "<unk>"
  //  * suffix_indicator = "##"
  //  * max_bytes_per_token = 100
  ASSERT_OK_AND_ASSIGN(
      auto tokenizer, FastWordpieceTokenizer::Create(config_flatbuffer.data()));

  std::string input = "abcdefghz";
  std::vector<std::string> output_tokens;
  std::vector<int> output_ids;
  std::vector<int> output_start_offsets;
  std::vector<int> output_end_offsets;
  tokenizer.Tokenize(input, &output_tokens, &output_ids, &output_start_offsets,
                     &output_end_offsets);
  EXPECT_THAT(output_tokens, ElementsAre("abc", "##de", "##f", "##ghz"));
  EXPECT_THAT(output_ids, ElementsAre(1, 3, 6, 7));
  EXPECT_THAT(output_start_offsets, ElementsAre(0, 3, 5, 6));
  EXPECT_THAT(output_end_offsets, ElementsAre(3, 5, 6, 9));
}

using TestPunctuationVersionMismatch = testing::TestWithParam<std::string>;

TEST_P(TestPunctuationVersionMismatch, Test) {
  // The config_flatbuffer used here is built from the following config:
  //  * vocab = {"a", "abc", "abcdefghi", "##de", "##defgxy", "##deh", "##f",
  //             "##ghz", "<unk>"}
  //  * unk_token = "<unk>"
  //  * suffix_indicator = "##"
  //  * max_bytes_per_token = 100
  //  * end_to_end = True

  const std::string kTestConfigUnicodePath = GetParam();

  // We test the new punctuation symbol: \341\255\277, which was available in
  // Unicode 16: https://www.fileformat.info/info/unicode/char//1b7f/index.htm,
  // but not in 15.1.
  // We also test an existing punctuation symbol ">".
  std::string input = "abc>abc\341\255\277abc";

  std::string config_flatbuffer;
  auto status = tensorflow::ReadFileToString(
      tensorflow::Env::Default(), kTestConfigUnicodePath, &config_flatbuffer);
  ASSERT_TRUE(status.ok());

  ASSERT_OK_AND_ASSIGN(
      auto tokenizer, FastWordpieceTokenizer::Create(config_flatbuffer.data()));

  std::vector<std::string> output_tokens;
  std::vector<int> output_ids;
  std::vector<int> output_start_offsets;
  std::vector<int> output_end_offsets;
  tokenizer.Tokenize(input, &output_tokens, &output_ids, &output_start_offsets,
                     &output_end_offsets);

  // If the runtime environment has unicode <=15.1, "\341\255\277" is not a
  // punctuation, so "abc\341\255\277abc" is one token.
  // If the runtime environment has unicode >=16.0, "\341\255\277" is a
  // punctuation, so tokens are "abc", "<unk>", "abc"
  EXPECT_THAT(output_tokens.size(), AnyOf(3, 5));
  if (!u_ispunct(0x1b7f)) {
    // We have a runtime environment of unicode <= 15.1.
    EXPECT_THAT(output_tokens, ElementsAre("abc", "<unk>", "<unk>"));
    EXPECT_THAT(output_ids, ElementsAre(1, 8, 8));
    EXPECT_THAT(output_start_offsets, ElementsAre(0, 3, 4));
    EXPECT_THAT(output_end_offsets, ElementsAre(3, 4, 13));
  } else {
    // We have a runtime environment of unicode >= 16.0.
    EXPECT_THAT(output_tokens,
                ElementsAre("abc", "<unk>", "abc", "<unk>", "abc"));
    EXPECT_THAT(output_ids, ElementsAre(1, 8, 1, 8, 1));
    EXPECT_THAT(output_start_offsets, ElementsAre(0, 3, 4, 7, 10));
    EXPECT_THAT(output_end_offsets, ElementsAre(3, 4, 7, 10, 13));
  }
}

INSTANTIATE_TEST_SUITE_P(FastWordpieceTokenizerPunctuationTest,
                         TestPunctuationVersionMismatch,
                         testing::Values(
                             // Unicode v 15.1 config
                             "tensorflow_text/python/ops/test_data/"
                             "fast_wordpiece_tokenizer_model_ver_15_1.fb",
                             // Unicode v 16.0 config
                             "tensorflow_text/python/ops/test_data/"
                             "fast_wordpiece_tokenizer_model_ver_16_0.fb"));

template <typename T>
std::string ListToString(const std::vector<T>& list) {
  return absl::StrCat("[", absl::StrJoin(list, ", "), "]");
}

// Testing spec struct for parameterized tests.
struct Spec {
  friend std::ostream& operator<<(std::ostream& os, const Spec& s) {
    return os << "vocab: " << ListToString(s.vocab) << ", "
              << "unk_token:" << s.unk_token << ", "
              << "suffix_indicator:" << s.suffix_indicator << ", "
              << "max_bytes_per_token:" << s.max_bytes_per_token << ", "
              << "input:" << s.input << ", "
              << "expected_tokens:" << ListToString(s.expected_tokens) << ", "
              << "expected_token_ids:" << ListToString(s.expected_token_ids)
              << ", "
              << "expected_token_start_offsets:"
              << ListToString(s.expected_token_start_offsets) << ", "
              << "expected_token_end_offsets:"
              << ListToString(s.expected_token_end_offsets) << std::endl;
  }

  std::vector<std::string> vocab;
  std::string unk_token;
  std::string suffix_indicator;
  int max_bytes_per_token;
  std::string input;
  std::vector<std::string> expected_tokens;
  std::vector<int> expected_token_ids;
  std::vector<int> expected_token_start_offsets = {};
  std::vector<int> expected_token_end_offsets = {};
  // Only used when detokenizing the tokenized ids back to text.
  std::string expected_detokenized_text;
};

// Parameterized tests specs for Tokenize() when input is a single word.
const std::vector<Spec>& GetTestSpecsForTokenizeSingleWord() {
  static const std::vector<Spec>& v = *new std::vector<Spec>{
      // Test suite 1, normal vocabulary.
      // Test 0: Empty input.
      {
          .vocab = {"a", "abc", "abcdefghi", "##de", "##defgxy", "##deh", "##f",
                    "##ghz", "<unk>"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "",
          .expected_tokens = {},
          .expected_token_ids = {},
          .expected_token_start_offsets = {},
          .expected_token_end_offsets = {},
      },
      // Test 1: Basic.
      {
          .vocab = {"a", "abc", "abcdefghi", "##de", "##defgxy", "##deh", "##f",
                    "##ghz", "<unk>"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "abcdefghz",
          .expected_tokens = {"abc", "##de", "##f", "##ghz"},
          .expected_token_ids = {1, 3, 6, 7},
          .expected_token_start_offsets = {0, 3, 5, 6},
          .expected_token_end_offsets = {3, 5, 6, 9},
      },
      // Test 2: Collect more tokens at the end.
      {
          .vocab = {"a", "abc", "abcdefghi", "##de", "##defgxy", "##deh", "##f",
                    "##ghz", "<unk>"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "abcdef",
          .expected_tokens = {"abc", "##de", "##f"},
          .expected_token_ids = {1, 3, 6},
          .expected_token_start_offsets = {0, 3, 5},
          .expected_token_end_offsets = {3, 5, 6},
      },
      // Test 3: Unseen character alone. Result is <unk>.
      {
          .vocab = {"a", "abc", "abcdefghi", "##de", "##defgxy", "##deh", "##f",
                    "##ghz", "<unk>"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "X",
          .expected_tokens = {"<unk>"},
          .expected_token_ids = {8},
          .expected_token_start_offsets = {0},
          .expected_token_end_offsets = {1},
      },
      // Test 4: Unseen character at the beginning. Result is <unk>.
      {
          .vocab = {"a", "abc", "abcdefghi", "##de", "##defgxy", "##deh", "##f",
                    "##ghz", "<unk>"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "Xde",
          .expected_tokens = {"<unk>"},
          .expected_token_ids = {8},
          .expected_token_start_offsets = {0},
          .expected_token_end_offsets = {3},
      },
      // Test 5: Unseen character in the middle. Result is <unk>.
      {
          .vocab = {"a", "abc", "abcdefghi", "##de", "##defgxy", "##deh", "##f",
                    "##ghz", "<unk>"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "abcXde",
          .expected_tokens = {"<unk>"},
          .expected_token_ids = {8},
          .expected_token_start_offsets = {0},
          .expected_token_end_offsets = {6},
      },
      // Test 6: Unseen character at the end. Result is <unk>.
      {
          .vocab = {"a", "abc", "abcdefghi", "##de", "##defgxy", "##deh", "##f",
                    "##ghz", "<unk>"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "abcX",
          .expected_tokens = {"<unk>"},
          .expected_token_ids = {8},
          .expected_token_start_offsets = {0},
          .expected_token_end_offsets = {4},
      },
      // Test 7: Input has leading suffix indicator. Result is normal.
      {
          .vocab = {"a", "abc", "abcdefghi", "##de", "##defgxy", "##deh", "##f",
                    "##ghz", "<unk>"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "##deh",
          .expected_tokens = {"##deh"},
          .expected_token_ids = {5},
          .expected_token_start_offsets = {0},
          .expected_token_end_offsets = {5},
      },
      // Test 8: Input has the leading suffix indicator. Vocab has "#" and
      // "###". Result is normal.
      {
          .vocab = {"a", "abc", "abcdefghi", "##de", "##defgxy", "##deh", "##f",
                    "##ghz", "#", "###", "<unk>"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "##deh",
          .expected_tokens = {"##deh"},
          .expected_token_ids = {5},
          .expected_token_start_offsets = {0},
          .expected_token_end_offsets = {5},
      },
      // Test 9: Input is the suffix indicator itself. Result is <unk>.
      {
          .vocab = {"a", "abc", "abcdefghi", "##de", "##defgxy", "##deh", "##f",
                    "##ghz", "<unk>"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "##",
          .expected_tokens = {"<unk>"},
          .expected_token_ids = {8},
          .expected_token_start_offsets = {0},
          .expected_token_end_offsets = {2},
      },
      // Test 10: [PAD] is in the vocabulary. Input is [PAD].
      {
          .vocab = {"[pad]", "a", "abc", "abcdefghi", "##de", "##defgxy",
                    "##deh", "##f", "##ghz", "#", "###", "<unk>"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "[pad]",
          .expected_tokens = {"[pad]"},
          .expected_token_ids = {0},
          .expected_token_start_offsets = {0},
          .expected_token_end_offsets = {5},
      },
      // Test 11: [PAD] is not in the vocabulary. Input is [PAD].
      {
          .vocab = {"a", "abc", "abcdefghi", "##de", "##defgxy", "##deh", "##f",
                    "##ghz", "#", "###", "<unk>"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "[pad]",
          .expected_tokens = {"<unk>"},
          .expected_token_ids = {10},
          .expected_token_start_offsets = {0},
          .expected_token_end_offsets = {5},
      },

      // Test suite 2, input contains #.
      // Test 12: Input is #. Result is <unk>.
      {
          .vocab = {"a", "abc", "abcdefghi", "##de", "##defgxy", "##deh", "##f",
                    "##ghz", "<unk>"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "#",
          .expected_tokens = {"<unk>"},
          .expected_token_ids = {8},
          .expected_token_start_offsets = {0},
          .expected_token_end_offsets = {1},
      },
      // Test 13: Input is #. Result is not <unk>.
      {
          .vocab = {"a", "abc", "abcdefghi", "##de", "##defgxy", "##deh", "##f",
                    "##ghz", "#", "###", "<unk>"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "#",
          .expected_tokens = {"#"},
          .expected_token_ids = {8},
          .expected_token_start_offsets = {0},
          .expected_token_end_offsets = {1},
      },
      // Test 14: Input is #. The suffix indicator is in the vocab. Result is
      // not <unk>.
      {
          .vocab = {"a", "abc", "abcdefghi", "##de", "##defgxy", "##deh", "##f",
                    "##ghz", "#", "###", "##", "<unk>"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "#",
          .expected_tokens = {"#"},
          .expected_token_ids = {8},
          .expected_token_start_offsets = {0},
          .expected_token_end_offsets = {1},
      },
      // Test 15: Input is the suffix indicator itself. Result is <unk>.
      {
          .vocab = {"a", "abc", "abcdefghi", "##de", "##defgxy", "##deh", "##f",
                    "##ghz", "#", "<unk>"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "##",
          .expected_tokens = {"<unk>"},
          .expected_token_ids = {9},
          .expected_token_start_offsets = {0},
          .expected_token_end_offsets = {2},
      },
      // Test 16: Input is the suffix indicator itself. Result is not <unk>.
      {
          .vocab = {"a", "abc", "abcdefghi", "##de", "##defgxy", "##deh", "##f",
                    "##ghz", "#", "###", "<unk>"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "##",
          .expected_tokens = {"#", "###"},
          .expected_token_ids = {8, 9},
          .expected_token_start_offsets = {0, 1},
          .expected_token_end_offsets = {1, 2},
      },
      // Test 17: Input is the suffix indicator itself. The suffix indicator is
      // in the vocab. Result is not <unk>.
      {
          .vocab = {"a", "abc", "abcdefghi", "##de", "##defgxy", "##deh", "##f",
                    "##ghz", "#", "###", "##", "<unk>"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "##",
          .expected_tokens = {"##"},
          .expected_token_ids = {10},
          .expected_token_start_offsets = {0},
          .expected_token_end_offsets = {2},
      },
      // Test 18: Input is ###. Result is <unk>.
      {
          .vocab = {"a", "abc", "abcdefghi", "##de", "##defgxy", "##deh", "##f",
                    "##ghz", "#", "<unk>"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "###",
          .expected_tokens = {"<unk>"},
          .expected_token_ids = {9},
          .expected_token_start_offsets = {0},
          .expected_token_end_offsets = {3},
      },
      // Test 19: Input is ###. Result is not <unk>.
      {
          .vocab = {"a", "abc", "abcdefghi", "##de", "##defgxy", "##deh", "##f",
                    "##ghz", "#", "###", "<unk>"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "###",
          .expected_tokens = {"###"},
          .expected_token_ids = {9},
          .expected_token_start_offsets = {0},
          .expected_token_end_offsets = {3},
      },
      // Test 20: Input is ###. The suffix indicator is in the vocab. Result is
      // not <unk>.
      {
          .vocab = {"a", "abc", "abcdefghi", "##de", "##defgxy", "##deh", "##f",
                    "##ghz", "#", "###", "##", "<unk>"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "###",
          .expected_tokens = {"###"},
          .expected_token_ids = {9},
          .expected_token_start_offsets = {0},
          .expected_token_end_offsets = {3},
      },
      // Test 21: Input is ####. Result is not <unk>.
      {
          .vocab = {"a", "abc", "abcdefghi", "##de", "##defgxy", "##deh", "##f",
                    "##ghz", "#", "###", "<unk>"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "####",
          .expected_tokens = {"###", "###"},
          .expected_token_ids = {9, 9},
          .expected_token_start_offsets = {0, 3},
          .expected_token_end_offsets = {3, 4},
      },
      // Test 22: Input is ####. The suffix indicator is in the vocab. Result
      // is not <unk>.
      {
          .vocab = {"a", "abc", "abcdefghi", "##de", "##defgxy", "##deh", "##f",
                    "##ghz", "#", "###", "##", "<unk>"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "####",
          .expected_tokens = {"###", "###"},
          .expected_token_ids = {9, 9},
          .expected_token_start_offsets = {0, 3},
          .expected_token_end_offsets = {3, 4},
      },

      // Test suite 3, the vocabulary contains empty tokens ("", "##").
      // Test 23: The empty prefix token ("") and the empty suffix token ("##")
      // are in the vocabulary.
      {
          .vocab = {"a", "abc", "abcdefghi", "##de", "##defgxy", "##deh", "##f",
                    "##ghz", "", "##", "<unk>"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "abcdefghz",
          .expected_tokens = {"abc", "##de", "##f", "##ghz"},
          .expected_token_ids = {1, 3, 6, 7},
          .expected_token_start_offsets = {0, 3, 5, 6},
          .expected_token_end_offsets = {3, 5, 6, 9},
      },
      // Test 24: The empty prefix token ("") and the empty suffix ("##") token
      // are in the vocabulary. Input is empty.
      {
          .vocab = {"a", "abc", "abcdefghi", "##de", "##defgxy", "##deh", "##f",
                    "##ghz", "", "##", "<unk>"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "",
          .expected_tokens = {},
          .expected_token_ids = {},
          .expected_token_start_offsets = {},
          .expected_token_end_offsets = {},
      },
      // Test 25: The empty prefix token ("") and the empty suffix token ("##")
      // are in the vocabulary. Input is the suffix indicator.
      {
          .vocab = {"a", "abc", "abcdefghi", "##de", "##defgxy", "##deh", "##f",
                    "##ghz", "", "##", "<unk>"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "##",
          .expected_tokens = {"##"},
          .expected_token_ids = {9},
          .expected_token_start_offsets = {0},
          .expected_token_end_offsets = {2},
      },
      // Test 26: The empty prefix token ("") and the empty suffix token ("##")
      // are in the vocabulary. There are vocab tokens after the empty vocab
      // tokens in the vocab. Result is one vocab token.
      {
          .vocab = {"a", "abc", "abcdefghi", "##de", "##defgxy", "##deh", "##f",
                    "##ghz", "", "##", "xyz", "<unk>"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "xyz",
          .expected_tokens = {"xyz"},
          .expected_token_ids = {10},
          .expected_token_start_offsets = {0},
          .expected_token_end_offsets = {3},
      },
      // Test 27: The empty prefix token ("") and the empty suffix ("##") token
      // are in the vocabulary. There are vocab tokens after the empty vocab
      // tokens in the vocab. Result has multiple tokens.
      {
          .vocab = {"a", "abc", "abcdefghi", "##de", "##defgxy", "##deh", "##f",
                    "##ghz", "", "##", "xy", "##z", "<unk>"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "xyz",
          .expected_tokens = {"xy", "##z"},
          .expected_token_ids = {10, 11},
          .expected_token_start_offsets = {0, 2},
          .expected_token_end_offsets = {2, 3},
      },
      // Test 28: The empty prefix token ("") and the empty suffix token ("##")
      // are in the vocabulary. Input has the leading suffix indicator.
      {
          .vocab = {"a", "abc", "abcdefghi", "##de", "##defgxy", "##deh", "##f",
                    "##ghz", "", "##", "<unk>"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "##deh",
          .expected_tokens = {"##deh"},
          .expected_token_ids = {5},
          .expected_token_start_offsets = {0},
          .expected_token_end_offsets = {5},
      },

      // Test suite 4, No suffix tokens in the vocabulary.
      // Test 29: No suffix tokens in the vocabulary. Result is normal.
      {
          .vocab = {"a", "abc", "abcdefghi", "<unk>"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "abc",
          .expected_tokens = {"abc"},
          .expected_token_ids = {1},
          .expected_token_start_offsets = {0},
          .expected_token_end_offsets = {3},
      },
      // Test 30: No suffix tokens in the vocabulary. Result is <unk>.
      {
          .vocab = {"a", "abc", "de", "abcdefghi", "<unk>"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "abcde",
          .expected_tokens = {"<unk>"},
          .expected_token_ids = {4},
          .expected_token_start_offsets = {0},
          .expected_token_end_offsets = {5},
      },
      // Test 31: No suffix tokens in the vocabulary. A different input. Result
      // is <unk>.
      {
          .vocab = {"a", "abc", "de", "abcdefghi", "<unk>"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "abcdz",
          .expected_tokens = {"<unk>"},
          .expected_token_ids = {4},
          .expected_token_start_offsets = {0},
          .expected_token_end_offsets = {5},
      },
      // Test 32: No suffix tokens in the vocabulary. Input is #. Result is
      // <unk>
      {
          .vocab = {"a", "abc", "de", "abcdefghi", "<unk>"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "#",
          .expected_tokens = {"<unk>"},
          .expected_token_ids = {4},
          .expected_token_start_offsets = {0},
          .expected_token_end_offsets = {1},
      },
      // Test 33: No suffix tokens in the vocabulary. Input is #. Result is not
      // <unk>.
      {
          .vocab = {"a", "abc", "de", "abcdefghi", "<unk>", "#"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "#",
          .expected_tokens = {"#"},
          .expected_token_ids = {5},
          .expected_token_start_offsets = {0},
          .expected_token_end_offsets = {1},
      },
      // Test 34: No suffix tokens in the vocabulary. Vocab has the suffix
      // indicator. Input is #.
      {
          .vocab = {"a", "abc", "de", "abcdefghi", "<unk>", "##"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "#",
          .expected_tokens = {"<unk>"},
          .expected_token_ids = {4},
          .expected_token_start_offsets = {0},
          .expected_token_end_offsets = {1},
      },
      // Test 35: No suffix tokens in the vocabulary. Input is ##. Result is
      // <unk>.
      {
          .vocab = {"a", "abc", "de", "abcdefghi", "<unk>"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "##",
          .expected_tokens = {"<unk>"},
          .expected_token_ids = {4},
          .expected_token_start_offsets = {0},
          .expected_token_end_offsets = {2},
      },
      // Test 36: No suffix tokens in the vocabulary. Vocab has the suffix
      // indicator. Input is #. Result is <unk>.
      {
          .vocab = {"a", "abc", "de", "abcdefghi", "<unk>", "##"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "#",
          .expected_tokens = {"<unk>"},
          .expected_token_ids = {4},
          .expected_token_start_offsets = {0},
          .expected_token_end_offsets = {1},
      },
      // Test 37: No suffix tokens in the vocabulary. Vocab has the suffix
      // indicator. Input is ##.
      {
          .vocab = {"a", "abc", "de", "abcdefghi", "<unk>", "##"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "##",
          .expected_tokens = {"##"},
          .expected_token_ids = {5},
          .expected_token_start_offsets = {0},
          .expected_token_end_offsets = {2},
      },
      // Test 38: No suffix tokens in the vocabulary. Vocab has '#'. Input is
      // ##. Result is <unk>.
      {
          .vocab = {"a", "abc", "de", "abcdefghi", "<unk>", "#"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "##",
          .expected_tokens = {"<unk>"},
          .expected_token_ids = {4},
          .expected_token_start_offsets = {0},
          .expected_token_end_offsets = {2},
      },
      // Test 39: No suffix tokens in the vocabulary. Vocab has the suffix
      // indicator and "#". Input is ##.
      {
          .vocab = {"a", "abc", "de", "abcdefghi", "<unk>", "##", "#"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "##",
          .expected_tokens = {"##"},
          .expected_token_ids = {5},
          .expected_token_start_offsets = {0},
          .expected_token_end_offsets = {2},
      },
      // Test 40: No suffix tokens in the vocabulary. Input is ###. Result is
      // <unk>.
      {
          .vocab = {"a", "abc", "de", "abcdefghi", "<unk>"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "###",
          .expected_tokens = {"<unk>"},
          .expected_token_ids = {4},
          .expected_token_start_offsets = {0},
          .expected_token_end_offsets = {3},
      },
      // Test 41: No suffix tokens in the vocabulary. Vocab has '#'. Input is
      // ###. Result is <unk>.
      {
          .vocab = {"a", "abc", "de", "abcdefghi", "<unk>", "#"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "###",
          .expected_tokens = {"<unk>"},
          .expected_token_ids = {4},
          .expected_token_start_offsets = {0},
          .expected_token_end_offsets = {3},
      },
      // Test 42: No suffix tokens in the vocabulary. Vocab has the suffix
      // indicator. Input is ###.
      {
          .vocab = {"a", "abc", "de", "abcdefghi", "<unk>", "##"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "###",
          .expected_tokens = {"<unk>"},
          .expected_token_ids = {4},
          .expected_token_start_offsets = {0},
          .expected_token_end_offsets = {3},
      },
      // Test 43: There is only one suffix tokens "###" in the vocabulary.
      // Input is ###.
      {
          .vocab = {"a", "abc", "de", "abcdefghi", "<unk>", "###"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "###",
          .expected_tokens = {"###"},
          .expected_token_ids = {5},
          .expected_token_start_offsets = {0},
          .expected_token_end_offsets = {3},
      },

      // Test suite 5, No prefix tokens in the vocabulary.
      // Test 44: No prefix tokens in the vocabulary. Input is a prefix token.
      {
          .vocab = {"##a", "##abc", "<unk>"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "abc",
          .expected_tokens = {"<unk>"},
          .expected_token_ids = {2},
          .expected_token_start_offsets = {0},
          .expected_token_end_offsets = {3},
      },
      // Test 45: No prefix tokens in the vocabulary. Input is a suffix token.
      {
          .vocab = {"##a", "##abc", "<unk>"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "##abc",
          .expected_tokens = {"##abc"},
          .expected_token_ids = {1},
          .expected_token_start_offsets = {0},
          .expected_token_end_offsets = {5},
      },

      // Test suite 6, more tests.
      // Test 46: Input is empty.
      {
          .vocab = {"<pad>", "<unk>", "<s>", "want", "##want", "##ed", "wa",
                    "un", "runn", "##ing"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "",
          .expected_tokens = {},
          .expected_token_ids = {},
          .expected_token_start_offsets = {},
          .expected_token_end_offsets = {},
      },
      // Test 47: Normal input.
      {
          .vocab = {"<pad>", "<unk>", "<s>", "want", "##want", "##ed", "wa",
                    "un", "runn", "##ing"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "unwanted",
          .expected_tokens = {"un", "##want", "##ed"},
          .expected_token_ids = {7, 4, 5},
          .expected_token_start_offsets = {0, 2, 6},
          .expected_token_end_offsets = {2, 6, 8},
      },
      // Test 48: Unseen character.
      {
          .vocab = {"<pad>", "<unk>", "<s>", "want", "##want", "##ed", "wa",
                    "un", "runn", "##ing"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "unwantedX",
          .expected_tokens = {"<unk>"},
          .expected_token_ids = {1},
          .expected_token_start_offsets = {0},
          .expected_token_end_offsets = {9},
      },

      // Test suite 7. Testing on long inputs (kMaxInputCharPerWord = 100). The
      // word length below means the number of utf-8 bytes.
      // Test 49: Word length = 99 (i.e., kMaxInputCharPerWord-1).
      {
          .vocab = {"<unk>", "0123456789", "##0123456789", "##012345678"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "01234567890123456789012345678901234567890123456789012345678"
                   "9012345678901234567890123456789012345678",
          .expected_tokens = {"0123456789", "##0123456789", "##0123456789",
                              "##0123456789", "##0123456789", "##0123456789",
                              "##0123456789", "##0123456789", "##0123456789",
                              "##012345678"},
          .expected_token_ids = {1, 2, 2, 2, 2, 2, 2, 2, 2, 3},
          .expected_token_start_offsets = {0, 10, 20, 30, 40, 50, 60, 70, 80,
                                           90},
          .expected_token_end_offsets = {10, 20, 30, 40, 50, 60, 70, 80, 90,
                                         99},
      },
      // Test 50: Word length = 100 (i.e., kMaxInputCharPerWord). Contains a
      // multi-bytes Unicode char.
      {
          .vocab = {"<unk>", "0123456789", "##0123456789", "##01234567",
                    /*U+05C3*/ "##\xD7\x83", "##a"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "01234567890123456789012345678901234567890123456789012345678"
                   "901234567890123456789012345678901234567\xD7\x83",
          .expected_tokens = {"0123456789", "##0123456789", "##0123456789",
                              "##0123456789", "##0123456789", "##0123456789",
                              "##0123456789", "##0123456789", "##0123456789",
                              "##01234567", "##\xD7\x83"},
          .expected_token_ids = {1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 4},
          .expected_token_start_offsets = {0, 10, 20, 30, 40, 50, 60, 70, 80,
                                           90, 98},
          .expected_token_end_offsets = {10, 20, 30, 40, 50, 60, 70, 80, 90, 98,
                                         100},
      },
      // Test 51: Word length = 101 (i.e., kMaxInputCharPerWord+1). Contains a
      // multi-bytes Unicode char.
      {
          .vocab = {"<unk>", "0123456789", "##0123456789", "##012345678",
                    /*U+05C3*/ "##\xD7\x83", "##a"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "01234567890123456789012345678901234567890123456789012345678"
                   "9012345678901234567890123456789012345678\xD7\x83",
          .expected_tokens = {"<unk>"},
          .expected_token_ids = {0},
          .expected_token_start_offsets = {0},
          .expected_token_end_offsets = {101},
      },
      // Test 52: Word length = 101 (i.e., kMaxInputCharPerWord+1).
      {
          .vocab = {"<unk>", "0123456789", "##0123456789", "##012345678",
                    "##a"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "01234567890123456789012345678901234567890123456789012345678"
                   "90123456789012345678901234567890123456789a",
          .expected_tokens = {"<unk>"},
          .expected_token_ids = {0},
          .expected_token_start_offsets = {0},
          .expected_token_end_offsets = {101},
      },
      // Test 53: Word length = 99 (i.e., kMaxInputCharPerWord-1). The word is
      // not tokenizable.
      {
          .vocab = {"<unk>", "0123456789", "##0123456789",
                    "##012345678\xe2\x80\x8B", "##\xe2\x80\x8B"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "01234567890123456789012345678901234567890123456789012345678"
                   "9012345678901234567890123456789012345678",
          .expected_tokens = {"<unk>"},
          .expected_token_ids = {0},
          .expected_token_start_offsets = {0},
          .expected_token_end_offsets = {99},
      },

      // Test suite 8. Normal vocab and inputs.
      // Test 54.
      {
          .vocab = {"<unk>", "play", "see", "##ing", "##ed", "##es", "##ly",
                    "##on", "##s", "##able"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "play",
          .expected_tokens = {"play"},
          .expected_token_ids = {1},
          .expected_token_start_offsets = {0},
          .expected_token_end_offsets = {4},
      },
      // Test 55.
      {
          .vocab = {"<unk>", "play", "see", "##ing", "##ed", "##es", "##ly",
                    "##on", "##s", "##able"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "playing",
          .expected_tokens = {"play", "##ing"},
          .expected_token_ids = {1, 3},
          .expected_token_start_offsets = {0, 4},
          .expected_token_end_offsets = {4, 7},
      },
      // Test 56.
      {
          .vocab = {"<unk>", "play", "see", "##ing", "##ed", "##es", "##ly",
                    "##on", "##s", "##able"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "sees",
          .expected_tokens = {"see", "##s"},
          .expected_token_ids = {2, 8},
          .expected_token_start_offsets = {0, 3},
          .expected_token_end_offsets = {3, 4},
      },
      // Test 57.
      {
          .vocab = {"<unk>", "play", "see", "##ing", "##ed", "##es", "##ly",
                    "##on", "##s", "##able", "u", "un", "##de", "##deni"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "undeniable",
          .expected_tokens = {"un", "##deni", "##able"},
          .expected_token_ids = {11, 13, 9},
          .expected_token_start_offsets = {0, 2, 6},
          .expected_token_end_offsets = {2, 6, 10},
      },
      // Test 58.
      {
          .vocab = {"<unk>", "play", "see", "##ing", "##ed", "##es", "##ly",
                    "##on", "##s", "##able", "u", "un", "##de", "##deni",
                    "undeniable"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "undeniable",
          .expected_tokens = {"undeniable"},
          .expected_token_ids = {14},
          .expected_token_start_offsets = {0},
          .expected_token_end_offsets = {10},
      },
      // Test 59.
      {
          .vocab = {"<unk>",  "s",       "su",    "super",   "##per", "##ca",
                    "##cali", "##f",     "##fra", "##g",     "##gil", "##i",
                    "##is",   "##istic", "##e",   "##ex",    "##pi",  "##pia",
                    "##li",   "##lido",  "##ci",  "##cious", "##ous"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "supercalifragilisticexpialidocious",
          .expected_tokens = {"super", "##cali", "##fra", "##gil", "##istic",
                              "##ex", "##pia", "##lido", "##cious"},
          .expected_token_ids = {3, 6, 8, 10, 13, 15, 17, 19, 21},
          .expected_token_start_offsets = {0, 5, 9, 12, 15, 20, 22, 25, 29},
          .expected_token_end_offsets = {5, 9, 12, 15, 20, 22, 25, 29, 34},
      },

      // Test suite 9. Different unk_tokens.
      // Test 60: Basic with a different unk_token.
      {
          .vocab = {"a", "abc", "abcdefghi", "##de", "##defgxy", "##deh", "##f",
                    "##ghz", "[unk]"},
          .unk_token = "[unk]",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "abcdefghz",
          .expected_tokens = {"abc", "##de", "##f", "##ghz"},
          .expected_token_ids = {1, 3, 6, 7},
          .expected_token_start_offsets = {0, 3, 5, 6},
          .expected_token_end_offsets = {3, 5, 6, 9},
      },
      // Test 61: Untokenizable with a different unk_token.
      {
          .vocab = {"a", "abc", "abcdefghi", "##de", "##defgxy", "##deh", "##f",
                    "##ghz", "[unk]"},
          .unk_token = "[unk]",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "abcdefghzX",
          .expected_tokens = {"[unk]"},
          .expected_token_ids = {8},
          .expected_token_start_offsets = {0},
          .expected_token_end_offsets = {10},
      },

      // Test suite 10. Input is the unk_token.
      // Test 62: Input is the unk_token.
      {
          .vocab = {"a", "abc", "abcdefghi", "##de", "##defgxy", "##deh", "##f",
                    "##ghz", "[unk]"},
          .unk_token = "[unk]",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "[unk]",
          .expected_tokens = {"[unk]"},
          .expected_token_ids = {8},
          .expected_token_start_offsets = {0},
          .expected_token_end_offsets = {5},
      },

      // Test suite 11. Input is the suffix indicator itself.
      // Test 63: Suffix indicator is "##" and is tokenizable.
      {
          .vocab = {"#", "###", "a", "abc", "abcdefghi", "##de", "##defgxy",
                    "##deh", "##f", "##ghz", "[unk]"},
          .unk_token = "[unk]",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "##",
          .expected_tokens = {"#", "###"},
          .expected_token_ids = {0, 1},
          .expected_token_start_offsets = {0, 1},
          .expected_token_end_offsets = {1, 2},
      },
      // Test 64: Suffix indicator is "##" but not tokenizable.
      {
          .vocab = {"a", "abc", "abcdefghi", "##de", "##defgxy", "##deh", "##f",
                    "##ghz", "[unk]"},
          .unk_token = "[unk]",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "##",
          .expected_tokens = {"[unk]"},
          .expected_token_ids = {8},
          .expected_token_start_offsets = {0},
          .expected_token_end_offsets = {2},
      },
      // Test 65: Suffix indicator is "##" and "##" is in the vocabulary.
      {
          .vocab = {"#", "###", "##", "a", "abc", "abcdefghi", "##de",
                    "##defgxy", "##deh", "##f", "##ghz", "[unk]"},
          .unk_token = "[unk]",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "##",
          .expected_tokens = {"##"},
          .expected_token_ids = {2},
          .expected_token_start_offsets = {0},
          .expected_token_end_offsets = {2},
      },
      // Test 66: Suffix indicator is "###" and is tokenizable.
      {
          .vocab = {"#", "####", "[unk]"},
          .unk_token = "[unk]",
          .suffix_indicator = "###",
          .max_bytes_per_token = 100,
          .input = "###",
          .expected_tokens = {"#", "####", "####"},
          .expected_token_ids = {0, 1, 1},
          .expected_token_start_offsets = {0, 1, 2},
          .expected_token_end_offsets = {1, 2, 3},
      },
      // Test 67: Suffix indicator is "###" and is tokenizable. A different
      // vocab.
      {
          .vocab = {"#", "####", "##", "[unk]"},
          .unk_token = "[unk]",
          .suffix_indicator = "###",
          .max_bytes_per_token = 100,
          .input = "###",
          .expected_tokens = {"##", "####"},
          .expected_token_ids = {2, 1},
          .expected_token_start_offsets = {0, 2},
          .expected_token_end_offsets = {2, 3},
      },

      // Test suite 12, different suffix indicators.
      // Test 68: A different suffix indicator.
      {
          .vocab = {"a", "abc", "abcdefghi", "<suffix>de", "<suffix>defgxy",
                    "<suffix>deh", "<suffix>f", "<suffix>ghz", "<unk>"},
          .unk_token = "<unk>",
          .suffix_indicator = "<suffix>",
          .max_bytes_per_token = 100,
          .input = "abcdefghz",
          .expected_tokens = {"abc", "<suffix>de", "<suffix>f", "<suffix>ghz"},
          .expected_token_ids = {1, 3, 6, 7},
          .expected_token_start_offsets = {0, 3, 5, 6},
          .expected_token_end_offsets = {3, 5, 6, 9},
      },
      // Test 69: The suffix indicator is empty.
      {
          .vocab = {"a", "abc", "abcdefghi", "de", "defgxy", "deh", "f", "ghz",
                    "<unk>"},
          .unk_token = "<unk>",
          .suffix_indicator = "",
          .max_bytes_per_token = 100,
          .input = "abcdefghz",
          .expected_tokens = {"abc", "de", "f", "ghz"},
          .expected_token_ids = {1, 3, 6, 7},
          .expected_token_start_offsets = {0, 3, 5, 6},
          .expected_token_end_offsets = {3, 5, 6, 9},
      },
      // Test 70: The suffix indicator is empty. Input is empty.
      {
          .vocab = {"a", "abc", "abcdefghi", "de", "defgxy", "deh", "f", "ghz",
                    "<unk>"},
          .unk_token = "<unk>",
          .suffix_indicator = "",
          .max_bytes_per_token = 100,
          .input = "",
          .expected_tokens = {},
          .expected_token_ids = {},
          .expected_token_start_offsets = {},
          .expected_token_end_offsets = {},
      },

      // Test suite 13, multi-bytes chars in vocab and input.
      // The following codepoints and their utf-8 encodings are used here:
      //  * U+03B1 (Greek Small Letter Alpha): "\xCE\xB1"
      //  * U+03B2 (Greek Small Letter Beta): "\xCE\xB2"
      //  * U+2EDA (Cjk Radical C-Simplified Leaf): b'\xE2\xBB\x9A'
      //  * U+2EDB (Cjk Radical C-Simplified Wind): b'\xE2\xBB\x9B'
      // Test 71: multi-bytes chars in the vocab.
      {
          .vocab = {"<unk>", "abc", "a", "##bc", "a\xCE\xB1\xCE\xB2",
                    "\xCE\xB1", "##\xCE\xB1", "##\xCE\xB2", "\xE2\xBB\x9A"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "abc",
          .expected_tokens = {"abc"},
          .expected_token_ids = {1},
          .expected_token_start_offsets = {0},
          .expected_token_end_offsets = {3},
      },
      // Test 72: input contains 2-bytes chars.
      {
          .vocab = {"<unk>", "abc", "a", "##bc", "a\xCE\xB1\xCE\xB2",
                    "\xCE\xB1", "##\xCE\xB1", "##\xCE\xB2", "\xE2\xBB\x9A"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "a\xCE\xB1\xCE\xB2\xCE\xB1\xCE\xB2",
          .expected_tokens = {"a\xCE\xB1\xCE\xB2", "##\xCE\xB1", "##\xCE\xB2"},
          .expected_token_ids = {4, 6, 7},
          .expected_token_start_offsets = {0, 5, 7},
          .expected_token_end_offsets = {5, 7, 9},
      },
      // Test 73: input contains 3-bytes chars.
      {
          .vocab = {"<unk>", "abc", "a", "##bc", "a\xCE\xB1\xCE\xB2",
                    "\xCE\xB1", "##\xCE\xB1", "##\xCE\xB2", "\xE2\xBB\x9A"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "\xE2\xBB\x9A"
                   "bc\xCE\xB1",
          .expected_tokens = {"\xE2\xBB\x9A", "##bc", "##\xCE\xB1"},
          .expected_token_ids = {8, 3, 6},
          .expected_token_start_offsets = {0, 3, 5},
          .expected_token_end_offsets = {3, 5, 7},
      },
      // Test 74: input contains unseen multi-bytes chars.
      {
          .vocab = {"<unk>", "abc", "a", "##bc", "a\xCE\xB1\xCE\xB2",
                    "\xCE\xB1", "##\xCE\xB1", "##\xCE\xB2", "\xE2\xBB\x9A"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "\xE2\xBB\x9B",
          .expected_tokens = {"<unk>"},
          .expected_token_ids = {0},
          .expected_token_start_offsets = {0},
          .expected_token_end_offsets = {3},
      },
  };
  return v;
}

using TestTokenizeSingleWord = testing::TestWithParam<Spec>;

TEST_P(TestTokenizeSingleWord, Test) {
  const Spec& spec = GetParam();
  ASSERT_OK_AND_ASSIGN(
      std::string flatbuffer,
      BuildModelAndExportToFlatBuffer(spec.vocab, spec.max_bytes_per_token,
                                      spec.suffix_indicator, spec.unk_token,
                                      /*no_pretokenization=*/true));
  ASSERT_OK_AND_ASSIGN(auto tokenizer,
                       FastWordpieceTokenizer::Create(flatbuffer.data()));

  std::vector<std::string> output_tokens;
  std::vector<int> output_ids;
  std::vector<int> output_begin_offsets;
  std::vector<int> output_end_offsets;
  tokenizer.Tokenize(spec.input, &output_tokens, &output_ids,
                     &output_begin_offsets, &output_end_offsets);
  EXPECT_THAT(output_tokens, spec.expected_tokens);
  EXPECT_THAT(output_ids, spec.expected_token_ids);
  EXPECT_THAT(output_begin_offsets, spec.expected_token_start_offsets);
  EXPECT_THAT(output_end_offsets, spec.expected_token_end_offsets);
}

TEST_P(TestTokenizeSingleWord, TestNoOutputPieces) {
  const Spec& spec = GetParam();
  ASSERT_OK_AND_ASSIGN(
      std::string flatbuffer,
      BuildModelAndExportToFlatBuffer(spec.vocab, spec.max_bytes_per_token,
                                      spec.suffix_indicator, spec.unk_token,
                                      true /* no_pretokenization */));
  ASSERT_OK_AND_ASSIGN(auto tokenizer,
                       FastWordpieceTokenizer::Create(flatbuffer.data()));

  std::vector<int> output_ids;
  std::vector<int> output_begin_offsets;
  std::vector<int> output_end_offsets;
  tokenizer.Tokenize(spec.input, &output_ids, &output_begin_offsets,
                     &output_end_offsets);
  EXPECT_THAT(output_ids, spec.expected_token_ids);
  EXPECT_THAT(output_begin_offsets, spec.expected_token_start_offsets);
  EXPECT_THAT(output_end_offsets, spec.expected_token_end_offsets);
}

TEST_P(TestTokenizeSingleWord, TestNoOutputPiecesOnlyOutputIds) {
  const Spec& spec = GetParam();
  ASSERT_OK_AND_ASSIGN(
      std::string flatbuffer,
      BuildModelAndExportToFlatBuffer(spec.vocab, spec.max_bytes_per_token,
                                      spec.suffix_indicator, spec.unk_token,
                                      true /* no_pretokenization */));
  ASSERT_OK_AND_ASSIGN(auto tokenizer,
                       FastWordpieceTokenizer::Create(flatbuffer.data()));

  std::vector<int> output_ids;
  tokenizer.Tokenize(spec.input, &output_ids);
  EXPECT_THAT(output_ids, spec.expected_token_ids);
}

TEST_P(TestTokenizeSingleWord, TestNoOutputPiecesWithPositiveSentenceOffsets) {
  const Spec& spec = GetParam();
  const int offset_in_sentence = 123;
  ASSERT_OK_AND_ASSIGN(
      std::string flatbuffer,
      BuildModelAndExportToFlatBuffer(spec.vocab, spec.max_bytes_per_token,
                                      spec.suffix_indicator, spec.unk_token,
                                      true /* no_pretokenization */));
  ASSERT_OK_AND_ASSIGN(auto tokenizer,
                       FastWordpieceTokenizer::Create(flatbuffer.data()));

  std::vector<int> output_ids;
  std::vector<int> output_begin_offsets;
  std::vector<int> output_end_offsets;
  std::vector<int> expected_token_start_offsets(
      spec.expected_token_start_offsets);
  std::vector<int> expected_token_end_offsets(spec.expected_token_end_offsets);

  for (int& offset : expected_token_start_offsets) {
    offset += offset_in_sentence;
  }
  for (int& offset : expected_token_end_offsets) {
    offset += offset_in_sentence;
  }

  tokenizer.Tokenize(spec.input, &output_ids, &output_begin_offsets,
                     &output_end_offsets,
                     /*input_word_offset_in_text=*/offset_in_sentence);
  EXPECT_THAT(output_begin_offsets, expected_token_start_offsets);
  EXPECT_THAT(output_end_offsets, expected_token_end_offsets);
}

INSTANTIATE_TEST_SUITE_P(
    FastWordpieceTokenizerParameterizedTest, TestTokenizeSingleWord,
    testing::ValuesIn(GetTestSpecsForTokenizeSingleWord()));

// Test End-to-end FastWordPieceTokenization for tokenizing general texts.
const std::vector<Spec>& GetTestSpecsForTokenizeText() {
  static const std::vector<Spec>& v = *new std::vector<Spec>{
      // Test suite 1. End-to-end test including whitespace tokenization.
      // Test 0: Input is empty.
      {
          .vocab = {"a", "abc", "abcdefghi", "##de", "##defgxy", "##deh", "##f",
                    "##ghz", "<unk>"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "",
          .expected_tokens = {},
          .expected_token_ids = {},
          .expected_token_start_offsets = {},
          .expected_token_end_offsets = {},
      },
      // Test 1: Input has only spaces.
      {
          .vocab = {"a", "abc", "abcdefghi", "##de", "##defgxy", "##deh", "##f",
                    "##ghz", "<unk>"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = " \t ",
          .expected_tokens = {},
          .expected_token_ids = {},
          .expected_token_start_offsets = {},
          .expected_token_end_offsets = {},
      },
      // Test 2: Input is a single word. Result is OK.
      {
          .vocab = {"a", "abc", "abcdefghi", "##de", "##defgxy", "##deh", "##f",
                    "##ghz", "<unk>"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "abcdef",
          .expected_tokens = {"abc", "##de", "##f"},
          .expected_token_ids = {1, 3, 6},
          .expected_token_start_offsets = {0, 3, 5},
          .expected_token_end_offsets = {3, 5, 6},
      },
      // Test 3: Input is a single word. Result is <unk>.
      {
          .vocab = {"a", "abc", "abcdefghi", "##de", "##defgxy", "##deh", "##f",
                    "##ghz", "<unk>"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "abcd",
          .expected_tokens = {"<unk>"},
          .expected_token_ids = {8},
          .expected_token_start_offsets = {0},
          .expected_token_end_offsets = {4},
      },
      // Test 4: Input contains multiple words, with several whitespaces in the
      // middle. Result is OK.
      {
          .vocab = {"a", "abc", "abcdefghi", "##de", "##defgxy", "##deh", "##f",
                    "##ghz", "<unk>"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "abcdef \t\t \tabcf",
          .expected_tokens = {"abc", "##de", "##f", "abc", "##f"},
          .expected_token_ids = {1, 3, 6, 1, 6},
          .expected_token_start_offsets = {0, 3, 5, 11, 14},
          .expected_token_end_offsets = {3, 5, 6, 14, 15},
      },
      // Test 5: Input has multiple words, with leading and trailing spaces.
      {
          .vocab = {"a", "abc", "abcdefghi", "##de", "##defgxy", "##deh", "##f",
                    "##ghz", "<unk>"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "\tabcdef  abcf ",
          .expected_tokens = {"abc", "##de", "##f", "abc", "##f"},
          .expected_token_ids = {1, 3, 6, 1, 6},
          .expected_token_start_offsets = {1, 4, 6, 9, 12},
          .expected_token_end_offsets = {4, 6, 7, 12, 13},
      },
      // Test 6: Input contains suffix indicator as words. Suffix indicator is
      // in vocab.
      {
          .vocab = {"a", "abc", "abcdefghi", "##de", "##defgxy", "##deh", "##f",
                    "##ghz", "<unk>", "##"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "## abcde ##  ##a",
          .expected_tokens = {"<unk>", "<unk>", "abc", "##de", "<unk>", "<unk>",
                              "<unk>", "<unk>", "a"},
          .expected_token_ids = {8, 8, 1, 3, 8, 8, 8, 8, 0},
          .expected_token_start_offsets = {0, 1, 3, 6, 9, 10, 13, 14, 15},
          .expected_token_end_offsets = {1, 2, 6, 8, 10, 11, 14, 15, 16},
      },
      // Test 7: Input contains suffix indicator as words. Suffix indicator is
      // in vocab.
      {
          .vocab = {"a", "abc", "abcdefghi", "##de", "##defgxy", "##deh", "##f",
                    "##ghz", "<unk>", "##"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "## abcde ##  ##a ##f",
          .expected_tokens = {"<unk>", "<unk>", "abc", "##de", "<unk>", "<unk>",
                              "<unk>", "<unk>", "a", "<unk>", "<unk>", "<unk>"},
          .expected_token_ids = {8, 8, 1, 3, 8, 8, 8, 8, 0, 8, 8, 8},
          .expected_token_start_offsets = {0, 1, 3, 6, 9, 10, 13, 14, 15, 17,
                                           18, 19},
          .expected_token_end_offsets = {1, 2, 6, 8, 10, 11, 14, 15, 16, 18, 19,
                                         20},
      },
      // Test 8: Input contains suffix indicator as words. Suffix indicator is
      // not in vocab.
      {
          .vocab = {"a", "abc", "abcdefghi", "##de", "##defgxy", "##deh", "##f",
                    "##ghz", "<unk>"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "##",
          .expected_tokens = {"<unk>", "<unk>"},
          .expected_token_ids = {8, 8},
          .expected_token_start_offsets = {0, 1},
          .expected_token_end_offsets = {1, 2},
      },
      // Test 9: Input contains unseen character words.
      {
          .vocab = {"a", "abc", "abcdefghi", "##de", "##defgxy", "##deh", "##f",
                    "##ghz", "<unk>"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = " a \tabcdeX \rabcdefghz abcdeXfghz Xabc abcd",
          .expected_tokens = {"a", "<unk>", "abc", "##de", "##f", "##ghz",
                              "<unk>", "<unk>", "<unk>"},
          .expected_token_ids = {0, 8, 1, 3, 6, 7, 8, 8, 8},
          .expected_token_start_offsets = {1, 4, 12, 15, 17, 18, 22, 33, 38},
          .expected_token_end_offsets = {2, 10, 15, 17, 18, 21, 32, 37, 42},
      },
      // Test 10: Input contains untokenizable words. No spaces before or after.
      {
          .vocab = {"a", "abc", "abcdefghi", "##de", "##defgxy", "##deh", "##f",
                    "##ghz", "<unk>"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "abcdefgx",
          .expected_tokens = {"<unk>"},
          .expected_token_ids = {8},
          .expected_token_start_offsets = {0},
          .expected_token_end_offsets = {8},
      },
      // Test 11: Input contains untokenizable words. One space before.
      {
          .vocab = {"a", "abc", "abcdefghi", "##de", "##defgxy", "##deh", "##f",
                    "##ghz", "<unk>"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = " abcdefgx",
          .expected_tokens = {"<unk>"},
          .expected_token_ids = {8},
          .expected_token_start_offsets = {1},
          .expected_token_end_offsets = {9},
      },
      // Test 12: Input contains untokenizable words. One space after.
      {
          .vocab = {"a", "abc", "abcdefghi", "##de", "##defgxy", "##deh", "##f",
                    "##ghz", "<unk>"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "abcdefgx ",
          .expected_tokens = {"<unk>"},
          .expected_token_ids = {8},
          .expected_token_start_offsets = {0},
          .expected_token_end_offsets = {8},
      },
      // Test 13: Input has untokenizable words. One space before and after.
      {
          .vocab = {"a", "abc", "abcdefghi", "##de", "##defgxy", "##deh", "##f",
                    "##ghz", "<unk>"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = " abcdefgx ",
          .expected_tokens = {"<unk>"},
          .expected_token_ids = {8},
          .expected_token_start_offsets = {1},
          .expected_token_end_offsets = {9},
      },
      // Test 14: Input contains mix words with unseen characters.
      {
          .vocab = {"a", "abc", "abcdefghi", "##de", "##defgxy", "##deh", "##f",
                    "##ghz", "<unk>"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = " a \tabcdeX \rabcdefghz abcdeXfghz Xabc",
          .expected_tokens = {"a", "<unk>", "abc", "##de", "##f", "##ghz",
                              "<unk>", "<unk>"},
          .expected_token_ids = {0, 8, 1, 3, 6, 7, 8, 8},
          .expected_token_start_offsets = {1, 4, 12, 15, 17, 18, 22, 33},
          .expected_token_end_offsets = {2, 10, 15, 17, 18, 21, 32, 37},
      },
      // Test 15: Another basic test.
      {
          .vocab = {"<unk>", "<s>", "</s>", "want", "##want", "##ed", "wa",
                    "un", "runn", "##ing"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "unwanted running",
          .expected_tokens = {"un", "##want", "##ed", "runn", "##ing"},
          .expected_token_ids = {7, 4, 5, 8, 9},
          .expected_token_start_offsets = {0, 2, 6, 9, 13},
          .expected_token_end_offsets = {2, 6, 8, 13, 16},
      },
      // Test 16: Input has unseen characters.
      {
          .vocab = {"<unk>", "<s>", "</s>", "want", "##want", "##ed", "wa",
                    "un", "runn", "##ing"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "unwantedX running",
          .expected_tokens = {"<unk>", "runn", "##ing"},
          .expected_token_ids = {0, 8, 9},
          .expected_token_start_offsets = {0, 10, 14},
          .expected_token_end_offsets = {9, 14, 17},
      },
      // Test 17: Input contains mix words with untokenizable words.
      {
          .vocab = {"a", "abc", "abcdefghi", "##de", "##defgxy", "##deh", "##f",
                    "##ghz", "<unk>"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = " a \tabcdeX \rabcdefghz abcdeXfghz ab",
          .expected_tokens = {"a", "<unk>", "abc", "##de", "##f", "##ghz",
                              "<unk>", "<unk>"},
          .expected_token_ids = {0, 8, 1, 3, 6, 7, 8, 8},
          .expected_token_start_offsets = {1, 4, 12, 15, 17, 18, 22, 33},
          .expected_token_end_offsets = {2, 10, 15, 17, 18, 21, 32, 35},
      },
      // Test 18: Input and vocab contains Unicode tokens. The Trie matching
      // loop would stop at matching a partial word.
      {
          .vocab = {"\xE2\x82\xAC", "a", "abc", "<unk>"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = " \xE2\x82\xAD abc",
          .expected_tokens = {"<unk>", "abc"},
          .expected_token_ids = {3, 2},
          .expected_token_start_offsets = {1, 5},
          .expected_token_end_offsets = {4, 8},
      },
      // Test 19: Contains suffix indicator as a word.
      {
          .vocab = {"<pad>", "<unk>", "<s>", "want", "##want", "##ed", "wa",
                    "un", "runn", "##ing", ".", "##.", "...", "#", "###"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "##",
          .expected_tokens = {"#", "#"},
          .expected_token_ids = {13, 13},
          .expected_token_start_offsets = {0, 1},
          .expected_token_end_offsets = {1, 2},
      },
      // Test 20: unknown words.
      {
          .vocab = {"<pad>", "<unk>", "<s>", "want", "##want", "##ed", "wa",
                    "un", "runn", "##ing", ".", "##.", "..."},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = " X wantXwanted. \t ",
          .expected_tokens = {"<unk>", "<unk>", "."},
          .expected_token_ids = {1, 1, 10},
          .expected_token_start_offsets = {1, 3, 14},
          .expected_token_end_offsets = {2, 14, 15},
      },
      // Test 21: After the loop, the next character is whitespace.
      {
          .vocab = {"<pad>", "<unk>", "<s>", "want", "##want", "##ed", "wa",
                    "un", "runn", "##ing", ".", "##.", "..."},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "  wanted.  \t wa..",
          .expected_tokens = {"want", "##ed", ".", "wa", ".", "."},
          .expected_token_ids = {3, 5, 10, 6, 10, 10},
          .expected_token_start_offsets = {2, 6, 8, 13, 15, 16},
          .expected_token_end_offsets = {6, 8, 9, 15, 16, 17},
      },
      // Test 22: After the loop, the next character is not a whitespace.
      {
          .vocab = {"<pad>", "<unk>", "<s>", "want", "##want", "##ed", "wa",
                    "un", "runn", "##ing", ".", "##.", "..."},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "  wanted.x  \t wa..",
          .expected_tokens = {"want", "##ed", ".", "<unk>", "wa", ".", "."},
          .expected_token_ids = {3, 5, 10, 1, 6, 10, 10},
          .expected_token_start_offsets = {2, 6, 8, 9, 14, 16, 17},
          .expected_token_end_offsets = {6, 8, 9, 10, 16, 17, 18},
      },
      // Test 23: After the loop, the next character is not a whitespace. And a
      // trailing space.
      {
          .vocab = {"<pad>", "<unk>", "<s>", "want", "##want", "##ed", "wa",
                    "un", "runn", "##ing", ".", "##.", "..."},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "  wanted.x  \t wa.. \n",
          .expected_tokens = {"want", "##ed", ".", "<unk>", "wa", ".", "."},
          .expected_token_ids = {3, 5, 10, 1, 6, 10, 10},
          .expected_token_start_offsets = {2, 6, 8, 9, 14, 16, 17},
          .expected_token_end_offsets = {6, 8, 9, 10, 16, 17, 18},
      },
      // Test 24: After the loop, it's in the middle of a whitespace. The
      // previous is tokenizable.
      {
          .vocab = {"<unk>", "want", "##want", "##ed", "wa", ".", "##.", "...",
                    "##\xc2\xa1"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "  wanted\xc2\xa0\t wa",
          .expected_tokens = {"want", "##ed", "wa"},
          .expected_token_ids = {1, 3, 4},
          .expected_token_start_offsets = {2, 6, 12},
          .expected_token_end_offsets = {6, 8, 14},
      },
      // Test 25: After the loop, it's in the middle of a whitespace. The
      // previous is tokenizable (a punctuation).
      {
          .vocab = {"<unk>", "want", "##want", "##ed", "wa", ".", "##.", "...",
                    "\xc2\xa1", "##\xc2\xa1"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "  wanted.\xc2\xa0\t wa",
          .expected_tokens = {"want", "##ed", ".", "wa"},
          .expected_token_ids = {1, 3, 5, 4},
          .expected_token_start_offsets = {2, 6, 8, 13},
          .expected_token_end_offsets = {6, 8, 9, 15},
      },
      // Test 26: After the loop, it's in the middle of a whitespace. The
      // previous is untokenizable.
      {
          .vocab = {"<unk>", "want", "##want", "##ed", "wa", ".", "##.", "...",
                    "##e\xC2\xA1", "##\xC2\xA1"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "  wante\xc2\xa0\t wa",
          .expected_tokens = {"<unk>", "wa"},
          .expected_token_ids = {0, 4},
          .expected_token_start_offsets = {2, 11},
          .expected_token_end_offsets = {7, 13},
      },

      // Test suite 2. End-to-end test including whitespace tokenization and
      // split on punctuation.
      // Test 27. Basic case 1.
      {
          .vocab =
              {
                  "<unk>",  "don",   "##'",   "##t",  "tread", "##ness",
                  "hel",    "##lo",  "there", "my",   "na",    "##me",
                  "is",     "ter",   "##ry",  "what", "##cha", "##ma",
                  "##call", "##it?", "you",   "said",
              },
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "hello there my name is terry",
          .expected_tokens = {"hel", "##lo", "there", "my", "na", "##me", "is",
                              "ter", "##ry"},
          .expected_token_ids = {6, 7, 8, 9, 10, 11, 12, 13, 14},
          .expected_token_start_offsets = {0, 3, 6, 12, 15, 17, 20, 23, 26},
          .expected_token_end_offsets = {3, 5, 11, 14, 17, 19, 22, 26, 28},
      },
      // Test 28. Basic case 2.
      {
          .vocab =
              {
                  "<unk>",  "don",   "##'",   "##t",  "tread", "##ness",
                  "hel",    "##lo",  "there", "my",   "na",    "##me",
                  "is",     "ter",   "##ry",  "what", "##cha", "##ma",
                  "##call", "##it?", "you",   "said",
              },
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "whatchamacallit? you said",
          .expected_tokens = {"<unk>", "<unk>", "you", "said"},
          .expected_token_ids = {0, 0, 20, 21},
          .expected_token_start_offsets = {0, 15, 17, 21},
          .expected_token_end_offsets = {15, 16, 20, 25},
      },
      // Test 29. Basic case 3. Punctuation is an independant word in the vocab.
      {
          .vocab =
              {
                  "<unk>",  "don",   "##'",   "##t",  "tread", "##ness",
                  "hel",    "##lo",  "there", "my",   "na",    "##me",
                  "is",     "ter",   "##ry",  "what", "##cha", "##ma",
                  "##call", "##it?", "you",   "said", "##it",  "?",
              },
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "whatchamacallit? you said",
          .expected_tokens = {"what", "##cha", "##ma", "##call", "##it", "?",
                              "you", "said"},
          .expected_token_ids = {15, 16, 17, 18, 22, 23, 20, 21},
          .expected_token_start_offsets = {0, 4, 7, 9, 13, 15, 17, 21},
          .expected_token_end_offsets = {4, 7, 9, 13, 15, 16, 20, 25},
      },
      // Test 30. Basic case 4 with untokenizable words.
      {
          .vocab =
              {
                  "<unk>",  "don",   "'",     "t",    "tread", "##ness",
                  "hel",    "##lo",  "there", "my",   "na",    "##me",
                  "is",     "ter",   "##ry",  "what", "##cha", "##ma",
                  "##call", "##it?", "you",   "said",
              },
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "don't tread cantfindme treadcantfindme",
          .expected_tokens = {"don", "'", "t", "tread", "<unk>", "<unk>"},
          .expected_token_ids = {1, 2, 3, 4, 0, 0},
          .expected_token_start_offsets = {0, 3, 4, 6, 12, 23},
          .expected_token_end_offsets = {3, 4, 5, 11, 22, 38},
      },
      // Test 31: Basic case 5.
      {
          .vocab = {"<pad>", "<unk>", "<s>", "want", "##want", "##ed", "wa",
                    "un", "runn", "##ing", ".", "##."},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "unwanted.",
          .expected_tokens = {"un", "##want", "##ed", "."},
          .expected_token_ids = {7, 4, 5, 10},
          .expected_token_start_offsets = {0, 2, 6, 8},
          .expected_token_end_offsets = {2, 6, 8, 9},
      },
      // Test 32: Basic case 6.
      {
          .vocab = {"<pad>", "<unk>", "<s>", "want", "##want", "##ed", "wa",
                    "un", "runn", "##ing", ".", "##.", "..."},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "  want.wanted. \t ",
          .expected_tokens = {"want", ".", "want", "##ed", "."},
          .expected_token_ids = {3, 10, 3, 5, 10},
          .expected_token_start_offsets = {2, 6, 7, 11, 13},
          .expected_token_end_offsets = {6, 7, 11, 13, 14},
      },
      // Test 33: Basic with unseen characters (as a single word).
      {
          .vocab = {"<pad>", "<unk>", "<s>", "want", "##want", "##ed", "wa",
                    "un", "runn", "##ing", ".", "##.", "..."},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = " X want.wanted. \t ",
          .expected_tokens = {"<unk>", "want", ".", "want", "##ed", "."},
          .expected_token_ids = {1, 3, 10, 3, 5, 10},
          .expected_token_start_offsets = {1, 3, 7, 8, 12, 14},
          .expected_token_end_offsets = {2, 7, 8, 12, 14, 15},
      },
      // Test 34: Basic with unseen characters (in a word before a punctuation).
      {
          .vocab = {"<pad>", "<unk>", "<s>", "want", "##want", "##ed", "wa",
                    "un", "runn", "##ing", ".", "##.", "..."},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = " X wantX.wanted. \t ",
          .expected_tokens = {"<unk>", "<unk>", ".", "want", "##ed", "."},
          .expected_token_ids = {1, 1, 10, 3, 5, 10},
          .expected_token_start_offsets = {1, 3, 8, 9, 13, 15},
          .expected_token_end_offsets = {2, 8, 9, 13, 15, 16},
      },
      // Test 35: Basic with unseen characters (in the middle of a word).
      {
          .vocab = {"<pad>", "<unk>", "<s>", "want", "##want", "##ed", "wa",
                    "un", "runn", "##ing", ".", "##.", "..."},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = " X wantXwanted. \t ",
          .expected_tokens = {"<unk>", "<unk>", "."},
          .expected_token_ids = {1, 1, 10},
          .expected_token_start_offsets = {1, 3, 14},
          .expected_token_end_offsets = {2, 14, 15},
      },
      // Test 36: Basic with unseen characters and a leading period.
      {
          .vocab = {"<pad>", "<unk>", "<s>", "want", "##want", "##ed", "wa",
                    "un", "runn", "##ing", ".", "##.", "..."},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = " X .wantXwanted. \t ",
          .expected_tokens = {"<unk>", ".", "<unk>", "."},
          .expected_token_ids = {1, 10, 1, 10},
          .expected_token_start_offsets = {1, 3, 4, 15},
          .expected_token_end_offsets = {2, 4, 15, 16},
      },
      // Test 37: Contains ellipsis (as ".....").
      {
          .vocab = {"<pad>", "<unk>", "<s>", "want", "##want", "##ed", "wa",
                    "un", "runn", "##ing", ".", "##.", "..."},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "  wanted.  \t wa.....",
          .expected_tokens = {"want", "##ed", ".", "wa", ".", ".", ".", ".",
                              "."},
          .expected_token_ids = {3, 5, 10, 6, 10, 10, 10, 10, 10},
          .expected_token_start_offsets = {2, 6, 8, 13, 15, 16, 17, 18, 19},
          .expected_token_end_offsets = {6, 8, 9, 15, 16, 17, 18, 19, 20},
      },
      // Test 38: After the loop, the next character is an unknown punctuation;
      // the previous can be tokenized.
      {
          .vocab = {"<pad>", "<unk>", "<s>", "want", "##want", "##ed", "wa",
                    "un", "runn", "##ing", ".", "##.", "..."},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "  wanted,  \t wa",
          .expected_tokens = {"want", "##ed", "<unk>", "wa"},
          .expected_token_ids = {3, 5, 1, 6},
          .expected_token_start_offsets = {2, 6, 8, 13},
          .expected_token_end_offsets = {6, 8, 9, 15},
      },
      // Test 39: After the loop, the next character is an unknown punctuation;
      // the previous can be tokenized.
      {
          .vocab = {"<pad>", "<unk>", "<s>", "want", "##want", "##ed", "wa",
                    "un", "runn", "##ing", ".", "##.", "..."},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "  wanted.,  \t wa",
          .expected_tokens = {"want", "##ed", ".", "<unk>", "wa"},
          .expected_token_ids = {3, 5, 10, 1, 6},
          .expected_token_start_offsets = {2, 6, 8, 9, 14},
          .expected_token_end_offsets = {6, 8, 9, 10, 16},
      },
      // Test 40: After the loop, the next character is an unknown punctuation;
      // the previous is empty.
      {
          .vocab = {"<pad>", "<unk>", "<s>", "want", "##want", "##ed", "wa",
                    "un", "runn", "##ing", ".", "##.", "..."},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = " , wanted,  \t wa",
          .expected_tokens = {"<unk>", "want", "##ed", "<unk>", "wa"},
          .expected_token_ids = {1, 3, 5, 1, 6},
          .expected_token_start_offsets = {1, 3, 7, 9, 14},
          .expected_token_end_offsets = {2, 7, 9, 10, 16},
      },
      // Test 41: After the loop, the next character is an unknown punctuation;
      // the previous can not be tokenized.
      {
          .vocab = {"<pad>", "<unk>", "<s>", "want", "##want", "##ed", "wa",
                    "un", "runn", "##ing", ".", "##.", "..."},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "  wante,  \t wa",
          .expected_tokens = {"<unk>", "<unk>", "wa"},
          .expected_token_ids = {1, 1, 6},
          .expected_token_start_offsets = {2, 7, 12},
          .expected_token_end_offsets = {7, 8, 14},
      },
      // Test 42: After the loop, in the middle of an unknown punctuation.
      // Previous is tokenizable.
      {
          .vocab = {"<unk>", "want", "##want", "##ed", "wa", ".", "##.", "...",
                    /*U+05C3*/ "\xD7\x83"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "  wanted\xd7\x86xyz  \t wa",
          .expected_tokens = {"want", "##ed", "<unk>", "<unk>", "wa"},
          .expected_token_ids = {1, 3, 0, 0, 4},
          .expected_token_start_offsets = {2, 6, 8, 10, 17},
          .expected_token_end_offsets = {6, 8, 10, 13, 19},
      },
      // Test 43: After the loop, in the middle of an unknown punctuation.
      // Previous is tokenizable.
      {
          .vocab = {"<unk>", "want", "##want", "##ed", "wa", ".", "##.", "...",
                    /*U+05C3*/ "\xD7\x83"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "  wanted.\xd7\x86xyz  \t wa",
          .expected_tokens = {"want", "##ed", ".", "<unk>", "<unk>", "wa"},
          .expected_token_ids = {1, 3, 5, 0, 0, 4},
          .expected_token_start_offsets = {2, 6, 8, 9, 11, 18},
          .expected_token_end_offsets = {6, 8, 9, 11, 14, 20},
      },
      // Test 44: After the loop, in the middle of an unknown punctuation.
      // Previous is not tokenizable.
      {
          .vocab = {"<unk>", "want", "##want", "##ed", "wa", ".", "##.", "...",
                    /*U+05C3*/ "##e\xD7\x83",
                    /*U+05C3*/ "\xD7\x83"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "  wante\xd7\x86xyz  \t wa",
          .expected_tokens = {"<unk>", "<unk>", "<unk>", "wa"},
          .expected_token_ids = {0, 0, 0, 4},
          .expected_token_start_offsets = {2, 7, 9, 16},
          .expected_token_end_offsets = {7, 9, 12, 18},
      },
      // Test 45: Fails to match the first character in the beginning.
      {
          .vocab = {"<pad>", "<unk>", "<s>", "want", "##want", "##ed", "wa",
                    "un", "runn", "##ing", ".", "##.", "..."},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "xyz  \t wa",
          .expected_tokens = {"<unk>", "wa"},
          .expected_token_ids = {1, 6},
          .expected_token_start_offsets = {0, 7},
          .expected_token_end_offsets = {3, 9},
      },
      // Test 46: After the loop, the next character is not a whitespace nor
      // punctuation. Trie fails to recognize the first character.
      {
          .vocab = {"<pad>", "<unk>", "<s>", "want", "##want", "##ed", "wa",
                    "un", "runn", "##ing", ".", "##.", "..."},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "  wanted.xyz  \t wa",
          .expected_tokens = {"want", "##ed", ".", "<unk>", "wa"},
          .expected_token_ids = {3, 5, 10, 1, 6},
          .expected_token_start_offsets = {2, 6, 8, 9, 16},
          .expected_token_end_offsets = {6, 8, 9, 12, 18},
      },
      // Test 47: After the loop, the next character is not a whitespace nor
      // punctuation. Previous is not tokenizable.
      {
          .vocab = {"<pad>", "<unk>", "<s>", "want", "##want", "##ed", "wa",
                    "un", "runn", "##ing", ".", "##.", "..."},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "  wantedxyz  \t wa",
          .expected_tokens = {"<unk>", "wa"},
          .expected_token_ids = {1, 6},
          .expected_token_start_offsets = {2, 15},
          .expected_token_end_offsets = {11, 17},
      },
      // Test 48: After the loop, the next character is not a whitespace nor
      // punctuation. Previous is not tokenizable.
      {
          .vocab = {"<pad>", "<unk>", "<s>", "want", "##want", "##ed", "wa",
                    "un", "runn", "##ing", ".", "##.", "..."},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "  wantexyz  \t wa",
          .expected_tokens = {"<unk>", "wa"},
          .expected_token_ids = {1, 6},
          .expected_token_start_offsets = {2, 14},
          .expected_token_end_offsets = {10, 16},
      },
      // Test 49: Unknown punctuation followed by unseen character.
      {
          .vocab = {"<unk>", "want", "##want", "##ed", "wa", ".", "##.", "...",
                    /*U+05C3*/ "##e\xD7\x83",
                    /*U+05C3*/ "\xD7\x83"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "wanted\xd7\x86xyz",
          .expected_tokens = {"want", "##ed", "<unk>", "<unk>"},
          .expected_token_ids = {1, 3, 0, 0},
          .expected_token_start_offsets = {0, 4, 6, 8},
          .expected_token_end_offsets = {4, 6, 8, 11},
      },
      // Test 50: Ellipsis is mapped to "<unk>"s when "." is not in vocab.
      {
          .vocab = {"<unk>", "want", "##want", "##ed", "wa", "..."},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "wanted...",
          .expected_tokens = {"want", "##ed", "<unk>", "<unk>", "<unk>"},
          .expected_token_ids = {1, 3, 0, 0, 0},
          .expected_token_start_offsets = {0, 4, 6, 7, 8},
          .expected_token_end_offsets = {4, 6, 7, 8, 9},
      },

      // Test suite 3. End-to-end test including whitespace and punctuation
      // tokenization on max_bytes_per_token = 10.
      // Test 51: Word length = 9 (i.e., max_bytes_per_token-1).
      {
          .vocab = {"<unk>", "01234", "##5678", "##56789",
                    /*U+05C3*/ "##\xD7\x83"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 10,
          .input = "  012345678 ",
          .expected_tokens = {"01234", "##5678"},
          .expected_token_ids = {1, 2},
          .expected_token_start_offsets = {2, 7},
          .expected_token_end_offsets = {7, 11},
      },
      // Test 52: Word length = 10 (i.e., max_bytes_per_token).
      {
          .vocab = {"<unk>", "01234", "##5678", "##56789",
                    /*U+05C3*/ "##\xD7\x83"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 10,
          .input = "  0123456789 ",
          .expected_tokens = {"01234", "##56789"},
          .expected_token_ids = {1, 3},
          .expected_token_start_offsets = {2, 7},
          .expected_token_end_offsets = {7, 12},
      },
      // Test 53: Word length = 9, followed by a multi-bytes Unicode punctuation
      // char, which is a hebrew punctuation "sof pasquq".
      {
          .vocab = {"<unk>", "01234", "##5678", "##56789",
                    /*U+05C3*/ "##\xD7\x83"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 10,
          .input = "  012345678\xD7\x83 ",
          .expected_tokens = {"01234", "##5678", "<unk>"},
          .expected_token_ids = {1, 2, 0},
          .expected_token_start_offsets = {2, 7, 11},
          .expected_token_end_offsets = {7, 11, 13},
      },
      // Test 54: Word length = 11 (i.e., max_bytes_per_token+1). The 10th
      // char is on Unicode boundary.
      {
          .vocab = {"<unk>", "01234", "##5678", "##56789",
                    /*U+05C3*/ "##\xD7\x83", "##a"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 10,
          .input = "  0123456789a ",
          .expected_tokens = {"<unk>"},
          .expected_token_ids = {0},
          .expected_token_start_offsets = {2},
          .expected_token_end_offsets = {13},
      },
      // Test 55: Word length = 10 (i.e., max_bytes_per_token). The next char
      // (\xe2\x80\x80) is a whitespace.
      {
          .vocab = {"<unk>", "01234", "##5678", "##56789",
                    /*U+05C3*/ "##\xD7\x83", "##a"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 10,
          .input = "  0123456789\xe2\x80\x80 ",
          .expected_tokens = {"01234", "##56789"},
          .expected_token_ids = {1, 3},
          .expected_token_start_offsets = {2, 7},
          .expected_token_end_offsets = {7, 12},
      },
      // Test 56:  Word length = 9 (i.e., max_bytes_per_token-1). The next is
      // a multi-byte whitespace. The 10th char is in the middle of the
      // whitespace.
      {
          .vocab = {"<unk>", "01234", "##5678", "##56789",
                    /*U+05C3*/ "##\xD7\x83", "##a", "##\xe2\x80\x8B"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 10,
          .input = "  012345678\xe2\x80\x80 ",
          .expected_tokens = {"01234", "##5678"},
          .expected_token_ids = {1, 2},
          .expected_token_start_offsets = {2, 7},
          .expected_token_end_offsets = {7, 11},
      },
      // Test 57: Word length = 9 (i.e., max_bytes_per_token-1). The next is a
      // multi-byte whitespace. The 10th char is in the middle of the
      // whitespace. The word is not tokenizable.
      {
          .vocab = {"<unk>", "01234", "##56789", "##5678\xe2\x80\x8B",
                    /*U+05C3*/ "##\xD7\x83", "##a", "##\xe2\x80\x8B"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 10,
          .input = "  012345678\xe2\x80\x80 ",
          .expected_tokens = {"<unk>"},
          .expected_token_ids = {0},
          .expected_token_start_offsets = {2},
          .expected_token_end_offsets = {11},
      },
      // Test 58:  Word length = 9 (i.e., max_bytes_per_token-1) plus a
      // trailing punctuation.
      {
          .vocab = {"<unk>", "01234", "##5678", "##56789",
                    /*U+05C3*/ "##\xD7\x83", "##a", "."},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 10,
          .input = "  .012345678. ",
          .expected_tokens = {".", "01234", "##5678", "."},
          .expected_token_ids = {6, 1, 2, 6},
          .expected_token_start_offsets = {2, 3, 8, 12},
          .expected_token_end_offsets = {3, 8, 12, 13},
      },
      // Test 59:  Word length = 9 (i.e., max_bytes_per_token-1) plus a
      // trailing punctuation, followed by more words.
      {
          .vocab = {"<unk>", "01234", "##5678", "##56789",
                    /*U+05C3*/ "\xD7\x83", "##a", ".", "...", "a"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 10,
          .input = "  .012345678.a ",
          .expected_tokens = {".", "01234", "##5678", ".", "a"},
          .expected_token_ids = {6, 1, 2, 6, 8},
          .expected_token_start_offsets = {2, 3, 8, 12, 13},
          .expected_token_end_offsets = {3, 8, 12, 13, 14},
      },
      // Test 60:  Word length = 10 (i.e., max_bytes_per_token) plus a
      // trailing punctuation, and the word is tokenizable.
      {
          .vocab = {"<unk>", "01234", "##5678", "##56789",
                    /*U+05C3*/ "\xD7\x83", "##a", ".", "..."},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 10,
          .input = "  .0123456789. ",
          .expected_tokens = {".", "01234", "##56789", "."},
          .expected_token_ids = {6, 1, 3, 6},
          .expected_token_start_offsets = {2, 3, 8, 13},
          .expected_token_end_offsets = {3, 8, 13, 14},
      },
      // Test 61:  Word length = 10 (i.e., max_bytes_per_token) plus a
      // trailing unknown punctuation, and the word is tokenizable.
      {
          .vocab = {"<unk>", "01234", "##5678", "##56789", "##a", ".", "..."},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 10,
          .input = "  .0123456789\xD7\x83 ",
          .expected_tokens = {".", "01234", "##56789", "<unk>"},
          .expected_token_ids = {5, 1, 3, 0},
          .expected_token_start_offsets = {2, 3, 8, 13},
          .expected_token_end_offsets = {3, 8, 13, 15},
      },
      // Test 62:  Word length = 11 (i.e., max_bytes_per_token+1).
      {
          .vocab = {"<unk>", "01234", "##5678", "##56789",
                    /*U+05C3*/ "\xD7\x83", "##a", ".", "..."},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 10,
          .input = "  .0123456789Z ",
          .expected_tokens = {".", "<unk>"},
          .expected_token_ids = {6, 0},
          .expected_token_start_offsets = {2, 3},
          .expected_token_end_offsets = {3, 14},
      },
      // Test 63:  Word length = 11 (i.e., max_bytes_per_token+1).
      // The input would be tokenizable if `max_byte_per_token` is set to be
      // greater or equal to `word_length`.
      {
          .vocab = {"<unk>", "0123456789", "##0123456789", "##012345678abc",
                    /*U+05C3*/ "\xD7\x83", "##a", ".", "..."},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 10,
          .input = "  .012345678a. ",
          .expected_tokens = {".", "<unk>", "."},
          .expected_token_ids = {6, 0, 6},
          .expected_token_start_offsets = {2, 3, 13},
          .expected_token_end_offsets = {3, 13, 14},
      },
      // Test 64:  Input is "<unk>".
      {
          .vocab = {"<unk>", "0123456789", "##0123456789", "##012345678abc",
                    /*U+05C3*/ "\xD7\x83", "##a", ".", "...", ">"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "<unk>.",
          .expected_tokens = {"<unk>", "<unk>", ">", "."},
          .expected_token_ids = {0, 0, 8, 6},
          .expected_token_start_offsets = {0, 1, 4, 5},
          .expected_token_end_offsets = {1, 4, 5, 6},
      },

      // Test suite 4: Test different suffix indicators.
      // Test 65: Suffix indicator is "##". Input contains "##".
      {
          .vocab = {"<pad>", "<unk>", "<s>", "want", "##want", "##ed", "wa",
                    "un", "runn", "##ing", ".", "##.", "...", "#", "##", "###"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "## running",
          .expected_tokens = {"#", "#", "runn", "##ing"},
          .expected_token_ids = {13, 13, 8, 9},
          .expected_token_start_offsets = {0, 1, 3, 7},
          .expected_token_end_offsets = {1, 2, 7, 10},
      },
      // Test 66: Test suffix indicator "<suffix>".
      {
          .vocab = {"<unk>", "want", "<suffix>want", "<suffix>ed", "wa", "un",
                    "runn", "<suffix>ing", "#", "."},
          .unk_token = "<unk>",
          .suffix_indicator = "<suffix>",
          .max_bytes_per_token = 100,
          .input = "## running. <",
          .expected_tokens = {"#", "#", "runn", "<suffix>ing", ".", "<unk>"},
          .expected_token_ids = {8, 8, 6, 7, 9, 0},
          .expected_token_start_offsets = {0, 1, 3, 7, 10, 12},
          .expected_token_end_offsets = {1, 2, 7, 10, 11, 13},
      },
      // Test 67: Test suffix indicator "suffix>". Suffix indicator appears in
      // the input as a single word after a punctuation.
      {
          .vocab = {"<unk>", "want", "suffix>want", "suffix>ed", "wa", "un",
                    "runn", "suffix>ing", "#", "su", "suffix>ffix", "suffix"},
          .unk_token = "<unk>",
          .suffix_indicator = "suffix>",
          .max_bytes_per_token = 100,
          .input = "#suffix> running",
          .expected_tokens = {"#", "suffix", "<unk>", "runn", "suffix>ing"},
          .expected_token_ids = {8, 11, 0, 6, 7},
          .expected_token_start_offsets = {0, 1, 7, 9, 13},
          .expected_token_end_offsets = {1, 7, 8, 13, 16},
      },
      // Test 68: Test suffix indicator "suffix>". Suffix indicator appears in
      // the input as a single word after a punctuation.
      {
          .vocab = {"<unk>", "want", "suffix>want", "suffix>ed", "wa", "un",
                    "runn", "suffix>ing", "#", "su", "suffix>ffix"},
          .unk_token = "<unk>",
          .suffix_indicator = "suffix>",
          .max_bytes_per_token = 100,
          .input = "#suffix> running",
          .expected_tokens = {"#", "su", "suffix>ffix", "<unk>", "runn",
                              "suffix>ing"},
          .expected_token_ids = {8, 9, 10, 0, 6, 7},
          .expected_token_start_offsets = {0, 1, 3, 7, 9, 13},
          .expected_token_end_offsets = {1, 3, 7, 8, 13, 16},
      },
      // Test 69: Test suffix indicator "<suffix". Suffix indicator appears in
      // the input as a single word after a punctuation.
      {
          .vocab = {"<unk>", "runn", "<suffixing", "#", "su", "<suffixffix"},
          .unk_token = "<unk>",
          .suffix_indicator = "<suffix",
          .max_bytes_per_token = 100,
          .input = "#<suffix running",
          .expected_tokens = {"#", "<unk>", "su", "<suffixffix", "runn",
                              "<suffixing"},
          .expected_token_ids = {3, 0, 4, 5, 1, 2},
          .expected_token_start_offsets = {0, 1, 2, 4, 9, 13},
          .expected_token_end_offsets = {1, 2, 4, 8, 13, 16},
      },
      // Test 70: Test suffix indicator "<suffix". Input "<suffixing" appears in
      // the vocab as a leading prefix of a word.
      {
          .vocab = {"<unk>", "runn", "<suffixing", "<", "su", "<suffixffix"},
          .unk_token = "<unk>",
          .suffix_indicator = "<suffix",
          .max_bytes_per_token = 100,
          .input = "<suffixing running",
          .expected_tokens = {"<", "su", "<suffixffix", "<suffixing", "runn",
                              "<suffixing"},
          .expected_token_ids = {3, 4, 5, 2, 1, 2},
          .expected_token_start_offsets = {0, 1, 3, 7, 11, 15},
          .expected_token_end_offsets = {1, 3, 7, 10, 15, 18},
      },
      // Test 71: Test suffix indicator ">>>". Suffix indicator appears in the
      // input.
      {
          .vocab = {"<unk>", "want", ">>>want", ">>>ed", "wa", "un", "runn",
                    ">>>ing", "#", "su", ">>>ffix"},
          .unk_token = "<unk>",
          .suffix_indicator = ">>>",
          .max_bytes_per_token = 100,
          .input = "#suffix>>> running",
          .expected_tokens = {"#", "su", ">>>ffix", "<unk>", "<unk>", "<unk>",
                              "runn", ">>>ing"},
          .expected_token_ids = {8, 9, 10, 0, 0, 0, 6, 7},
          .expected_token_start_offsets = {0, 1, 3, 7, 8, 9, 11, 15},
          .expected_token_end_offsets = {1, 3, 7, 8, 9, 10, 15, 18},
      },
      // Test 72: Test suffix indicator "<<suffix". Suffix indicator appears in
      // the input and the vocab.
      {
          .vocab = {"<unk>", "runn", "<<suffixing", "<", "su", "<<suffixffix"},
          .unk_token = "<unk>",
          .suffix_indicator = "<<suffix",
          .max_bytes_per_token = 100,
          .input = "<<suffix running",
          .expected_tokens = {"<", "<", "su", "<<suffixffix", "runn",
                              "<<suffixing"},
          .expected_token_ids = {3, 3, 4, 5, 1, 2},
          .expected_token_start_offsets = {0, 1, 2, 4, 9, 13},
          .expected_token_end_offsets = {1, 2, 4, 8, 13, 16},
      },
      // Test 73: Test suffix indicator "XYZ". Input contains "XYZ".
      {
          .vocab = {"<unk>", "runn", "XYZing", "<", "X", "XYZYZ"},
          .unk_token = "<unk>",
          .suffix_indicator = "XYZ",
          .max_bytes_per_token = 100,
          .input = "XYZ running",
          .expected_tokens = {"X", "XYZYZ", "runn", "XYZing"},
          .expected_token_ids = {4, 5, 1, 2},
          .expected_token_start_offsets = {0, 1, 4, 8},
          .expected_token_end_offsets = {1, 3, 8, 11},
      },
      // Test 74: Test suffix indicator "XYZ", which appears in the
      // vocab and input sentence as a single word.
      {
          .vocab = {"<unk>", "runn", "XYZing", "<", "X", "XYZYZ", "XYZ"},
          .unk_token = "<unk>",
          .suffix_indicator = "XYZ",
          .max_bytes_per_token = 100,
          .input = "XYZ running",
          .expected_tokens = {"XYZ", "runn", "XYZing"},
          .expected_token_ids = {6, 1, 2},
          .expected_token_start_offsets = {0, 4, 8},
          .expected_token_end_offsets = {3, 8, 11},
      },
      // Test suite 5: Test multi-byte punctuation and Chinese characters.
      // Test 75: Contains a multi-bytes Unicode punctuation char "\xEF\xBC\x8C"
      // followed by a tokenizable word.
      {
          .vocab = {"<unk>", "want", "##ed", "ABC", "\xEF\xBC\x8C", "##ABC"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 10,
          .input = "wanted\xEF\xBC\x8C"
                   "ABC",
          .expected_tokens = {"want", "##ed", "\xEF\xBC\x8C", "ABC"},
          .expected_token_ids = {1, 2, 4, 3},
          .expected_token_start_offsets = {0, 4, 6, 9},
          .expected_token_end_offsets = {4, 6, 9, 12},
      },
      // Test 76: Contains a multi-bytes Unicode punctuation char "\xEF\xBC\x8C"
      // (absent in the vocab) followed by a tokenizable word.
      {
          .vocab = {"<unk>", "want", "##ed", "ABC", "\xEF\xBC\x8C", "##ABC"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 10,
          .input = "wanted\xD7\x83"
                   "ABC",
          .expected_tokens = {"want", "##ed", "<unk>", "ABC"},
          .expected_token_ids = {1, 2, 0, 3},
          .expected_token_start_offsets = {0, 4, 6, 8},
          .expected_token_end_offsets = {4, 6, 8, 11},
      },
      // Test 77: Contains a multi-bytes Unicode chinese character \xe4\xb8\x81,
      // which is considered as a single word in Bert, so it's treated in the
      // same way as punctuation characters by the tokenizer.
      {
          .vocab = {"<unk>", "want", "##ed", "ABC", "\xe4\xb8\x81", "##ABC"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 10,
          .input = "wanted\xe4\xb8\x81"
                   "ABC",
          .expected_tokens = {"want", "##ed", "\xe4\xb8\x81", "ABC"},
          .expected_token_ids = {1, 2, 4, 3},
          .expected_token_start_offsets = {0, 4, 6, 9},
          .expected_token_end_offsets = {4, 6, 9, 12},
      },
      // Test 78: Contains a multi-bytes Unicode chinese character \xe4\xb8\x81.
      {
          .vocab = {"<unk>", "want", "##ed", "ABC", "##ABC",
                    "wanted\xe4\xb8\x81"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 10,
          .input = "wanted\xe4\xb8\x81"
                   "ABC",
          .expected_tokens = {"want", "##ed", "<unk>", "ABC"},
          .expected_token_ids = {1, 2, 0, 3},
          .expected_token_start_offsets = {0, 4, 6, 9},
          .expected_token_end_offsets = {4, 6, 9, 12},
      },
      // Test 79: Contains a multi-bytes Unicode chinese character \xe4\xb8\x81,
      // which is included in the vocab as the suffix of a word.
      {
          .vocab = {"<unk>", "want", "##ed", "ABC", "##ABC",
                    "wanted\xe4\xb8\x81"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 10,
          .input = "wanted\xe4\xb8\x81"
                   "ABC",
          .expected_tokens = {"want", "##ed", "<unk>", "ABC"},
          .expected_token_ids = {1, 2, 0, 3},
          .expected_token_start_offsets = {0, 4, 6, 9},
          .expected_token_end_offsets = {4, 6, 9, 12},
      }};
  return v;
}

using TestTokenizeText = testing::TestWithParam<Spec>;

TEST_P(TestTokenizeText, Test) {
  const Spec& spec = GetParam();
  ASSERT_OK_AND_ASSIGN(
      std::string flatbuffer,
      BuildModelAndExportToFlatBuffer(spec.vocab, spec.max_bytes_per_token,
                                      spec.suffix_indicator, spec.unk_token));
  ASSERT_OK_AND_ASSIGN(auto tokenizer,
                       FastWordpieceTokenizer::Create(flatbuffer.data()));

  std::vector<std::string> output_tokens;
  std::vector<int> output_ids;
  std::vector<int> output_begin_offsets;
  std::vector<int> output_end_offsets;
  tokenizer.Tokenize(spec.input, &output_tokens, &output_ids,
                     &output_begin_offsets, &output_end_offsets);
  EXPECT_THAT(output_tokens, spec.expected_tokens);
  EXPECT_THAT(output_ids, spec.expected_token_ids);
  EXPECT_THAT(output_begin_offsets, spec.expected_token_start_offsets);
  EXPECT_THAT(output_end_offsets, spec.expected_token_end_offsets);
}

TEST_P(TestTokenizeText, TestNoOutputPieces) {
  const Spec& spec = GetParam();
  ASSERT_OK_AND_ASSIGN(
      std::string flatbuffer,
      BuildModelAndExportToFlatBuffer(spec.vocab, spec.max_bytes_per_token,
                                      spec.suffix_indicator, spec.unk_token));
  ASSERT_OK_AND_ASSIGN(auto tokenizer,
                       FastWordpieceTokenizer::Create(flatbuffer.data()));

  std::vector<int> output_ids;
  std::vector<int> output_begin_offsets;
  std::vector<int> output_end_offsets;
  tokenizer.Tokenize(spec.input, &output_ids, &output_begin_offsets,
                     &output_end_offsets);
  EXPECT_THAT(output_ids, spec.expected_token_ids);
  EXPECT_THAT(output_begin_offsets, spec.expected_token_start_offsets);
  EXPECT_THAT(output_end_offsets, spec.expected_token_end_offsets);
}

TEST_P(TestTokenizeText, TestNoOutputPiecesOnlyOutputIds) {
  const Spec& spec = GetParam();
  ASSERT_OK_AND_ASSIGN(
      std::string flatbuffer,
      BuildModelAndExportToFlatBuffer(spec.vocab, spec.max_bytes_per_token,
                                      spec.suffix_indicator, spec.unk_token));
  ASSERT_OK_AND_ASSIGN(auto tokenizer,
                       FastWordpieceTokenizer::Create(flatbuffer.data()));

  std::vector<int> output_ids;
  tokenizer.Tokenize(spec.input, &output_ids);
  EXPECT_THAT(output_ids, spec.expected_token_ids);
}

INSTANTIATE_TEST_SUITE_P(EndToEndFastWordpieceTokenizerParameterizedTest,
                         TestTokenizeText,
                         testing::ValuesIn(GetTestSpecsForTokenizeText()));

// Test the detokenization function of FastWordPieceTokenizer.
const std::vector<Spec>& GetTestSpecsForTokenizeDetokenize() {
  static const std::vector<Spec>& v = *new std::vector<Spec>{
      // Test 0: Input is a single word.
      {
          .vocab = {"a", "abc", "##de", "##defgxy", "##deh", "##f", "##ghz",
                    "<unk>"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "abcdefghz",
          .expected_token_ids = {1, 2, 5, 6},
          .expected_detokenized_text = "abcdefghz",
      },
      // Test 1: Input is a sentence.
      {
          .vocab = {"a", "abc", "##de", "##c", "##f", "<unk>"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "a abc abcde ab",
          .expected_token_ids = {0, 1, 1, 2, 5},
          .expected_detokenized_text = "a abc abcde <unk>",
      },
      // Test 2: Input has the leading suffix indicator.
      {
          .vocab = {"a", "abc", "##de", "##deh", "##f", "<unk>"},
          .unk_token = "<unk>",
          .suffix_indicator = "##",
          .max_bytes_per_token = 100,
          .input = "##deh abcde",
          .expected_token_ids = {3, 1, 2},
          .expected_detokenized_text = "##deh abcde",
      },
  };
  return v;
}
using TestTokenizeDetokenize = testing::TestWithParam<Spec>;

TEST_P(TestTokenizeDetokenize, Test) {
  const Spec& spec = GetParam();
  ASSERT_OK_AND_ASSIGN(
      std::string flatbuffer,
      BuildModelAndExportToFlatBuffer(spec.vocab, spec.max_bytes_per_token,
                                      spec.suffix_indicator, spec.unk_token,
                                      /*no_pretokenization=*/true,
                                      /*support_detokenization=*/true));
  ASSERT_OK_AND_ASSIGN(auto tokenizer,
                       FastWordpieceTokenizer::Create(flatbuffer.data()));

  // Test detokenization.
  ASSERT_OK_AND_ASSIGN(auto output_text,
                       tokenizer.Detokenize(spec.expected_token_ids));
  EXPECT_THAT(output_text, spec.expected_detokenized_text);
}

INSTANTIATE_TEST_SUITE_P(
    FastWordpieceTokenizerDetokenizeParameterizedTest, TestTokenizeDetokenize,
    testing::ValuesIn(GetTestSpecsForTokenizeDetokenize()));

}  // namespace
}  // namespace text
}  // namespace tensorflow
