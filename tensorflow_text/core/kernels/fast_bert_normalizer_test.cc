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

#include "tensorflow_text/core/kernels/fast_bert_normalizer.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow_text/core/kernels/fast_bert_normalizer_model_builder.h"

namespace tensorflow {
namespace text {
namespace {

template <typename T>
std::string ListToString(const std::vector<T>& list) {
  return absl::StrCat("[", absl::StrJoin(list, ", "), "]");
}

// Testing spec struct for parameterized tests.
struct Spec {
  friend std::ostream& operator<<(std::ostream& os, const Spec& s) {
    return os << "input: " << s.input << ", "
              << "lower_case_nfd_strip_accents:"
              << s.lower_case_nfd_strip_accents << ", "
              << "expected_output:" << s.expected_output << ", "
              << "expected_offset_mapping:"
              << ListToString(s.expected_offset_mapping) << std::endl;
  }

  std::string input;
  bool lower_case_nfd_strip_accents = false;
  std::string expected_output;
  std::vector<int> expected_offset_mapping;
};

// Parameterized tests specs for FastBertNormalizer.
const std::vector<Spec>& GetTestSpecs() {
  static const std::vector<Spec>& v = *new std::vector<Spec>{
      // Test Suite 1: No lower case.
      // Test 0: Empty input.
      {
          .input = "",
          .lower_case_nfd_strip_accents = false,
          .expected_output = "",
          .expected_offset_mapping = {0},
      },
      // Test 1: All ascii, digit, and normal letters.
      {
          .input = "Test #1.",
          .lower_case_nfd_strip_accents = false,
          .expected_output = "Test #1.",
          .expected_offset_mapping = {0, 1, 2, 3, 4, 5, 6, 7, 8},
      },
      // Test 2: Multi-byte letters.
      // "\xC3\x80" is U+00C0 "Latin Capital Letter A with Grave".
      // "\x41\xCC\x80" is the decomposition of U+00C0 "Latin Capital Letter A
      // with Grave".
      {
          .input = "Test\xC3\x80\x41\xCC\x80 #1.",
          .lower_case_nfd_strip_accents = false,
          .expected_output = "Test\xC3\x80\x41\xCC\x80 #1.",
          .expected_offset_mapping = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                      13},
      },
      // Test 3: Control chars normalized into whitespaces.
      {
          .input = "Te\x11st #1.",
          .lower_case_nfd_strip_accents = false,
          .expected_output = "Te st #1.",
          .expected_offset_mapping = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
      },
      // Test 4: Tabs and newlines normalized into whitespaces.
      {
          .input = "Test \t\n#1.",
          .lower_case_nfd_strip_accents = false,
          .expected_output = "Test   #1.",
          .expected_offset_mapping = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
      },
      // Test Suite 2: Lower case.
      // Test 5: Empty input.
      {
          .input = "",
          .lower_case_nfd_strip_accents = true,
          .expected_output = "",
          .expected_offset_mapping = {0},
      },
      // Test 6: All ascii, digit, and normal letters.
      {
          .input = "Test #1.",
          .lower_case_nfd_strip_accents = true,
          .expected_output = "test #1.",
          .expected_offset_mapping = {0, 1, 2, 3, 4, 5, 6, 7, 8},
      },
      // Test 7: Multi-byte letters.
      // "\xC3\x80" is U+00C0 "Latin Capital Letter A with Grave", which is
      // normalized to "a". "\x41\xCC\x80" is the decomposition of U+00C0 "Latin
      // Capital Letter A with Grave", which is normalized to "a".
      {
          .input = "Test\xC3\x80\x41\xCC\x80 #1.",
          .lower_case_nfd_strip_accents = true,
          .expected_output = "testaa #1.",
          .expected_offset_mapping = {0, 1, 2, 3, 4, 6, 9, 10, 11, 12, 13},
      },
      // Test 8: Control chars normalized into whitespaces.
      {
          .input = "Te\x11st #1.",
          .lower_case_nfd_strip_accents = true,
          .expected_output = "te st #1.",
          .expected_offset_mapping = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
      },
      // Test 9: Tabs and newlines normalized into whitespaces.
      {
          .input = "Test \t\n#1.",
          .lower_case_nfd_strip_accents = true,
          .expected_output = "test   #1.",
          .expected_offset_mapping = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
      },
      // Test 10: Multibytes string normalized into multibytes string.
      // "\xC2\xBC" (2 bytes) is normalized into "1\xE2\x81\x84""4" (5 bytes).
      {
          .input = "a\xC2\xBC",
          .lower_case_nfd_strip_accents = true,
          .expected_output = "a1\xE2\x81\x84"
                             "4",
          .expected_offset_mapping = {0, 1, 1, 1, 1, 1, 3},
      },
      // Test 11: Multibytes string normalized into multibytes string.
      // "\xC7\xB2" (2 bytes) is normalized into "dz" (2 bytes).
      {
          .input = "a\xC7\xB2",
          .lower_case_nfd_strip_accents = true,
          .expected_output = "adz",
          .expected_offset_mapping = {0, 1, 1, 3},
      },
      // Test 12: Multibytes string normalized into multibytes string.
      // "\xCE\xB9" (2 bytes) is normalized into "\xCE\xB7" (2 bytes).
      {
          .input = "a\xCE\x89",
          .lower_case_nfd_strip_accents = true,
          .expected_output = "a\xCE\xB7",
          .expected_offset_mapping = {0, 1, 1, 3},
      },
      // Test 13: Invalid UTF8 input. lower_case_nfd_strip_accents = false.
      {
          .input = "a\x80 \xFF \xF8 a\xE0\x61 \xF3\x9C\x9D",
          .lower_case_nfd_strip_accents = false,
          .expected_output = "a\x80 \xFF \xF8 a\xE0\x61 \xF3\x9C\x9D",
          .expected_offset_mapping = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                      13, 14},
      },
      // Test 14: Invalid UTF8 input. lower_case_nfd_strip_accents = true.
      {
          .input = "a\x80 \xFF \xF8 a\xE0\x61 \xF3\x9C\x9D",
          .lower_case_nfd_strip_accents = true,
          .expected_output = "a\x80 \xFF \xF8 a\xE0\x61 \xF3\x9C\x9D",
          .expected_offset_mapping = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                      13, 14},
      },
  };
  return v;
}

using TestNormalization = testing::TestWithParam<Spec>;

TEST_P(TestNormalization, TestGetOffsets) {
  const auto spec = GetParam();
  const auto fast_bert_normalizer =
      FastBertNormalizerFactory::GetInstance(spec.lower_case_nfd_strip_accents)
          .GetNormalizer();

  std::string output_normalized_text = "Something existing";
  std::vector<int> output_normalized_offset_mapping;
  bool is_normalized_identical;
  fast_bert_normalizer->NormalizeText</*kGetOffsets=*/true>(
      spec.input, &is_normalized_identical, &output_normalized_text,
      &output_normalized_offset_mapping);
  if (is_normalized_identical) {
    ASSERT_THAT(output_normalized_text, "");
    ASSERT_THAT(spec.input, spec.expected_output);
    ASSERT_THAT(output_normalized_offset_mapping, testing::ElementsAre());
  } else {
    ASSERT_THAT(output_normalized_text, spec.expected_output);
    ASSERT_THAT(output_normalized_offset_mapping, spec.expected_offset_mapping);
  }
}

TEST_P(TestNormalization, TestNoGetOffsets) {
  const auto spec = GetParam();
  const auto fast_bert_normalizer =
      FastBertNormalizerFactory::GetInstance(spec.lower_case_nfd_strip_accents)
          .GetNormalizer();

  std::string output_normalized_text;
  std::vector<int> output_normalized_offset_mapping;
  bool is_normalized_identical;
  fast_bert_normalizer->NormalizeText</*kGetOffsets=*/false>(
      spec.input, &is_normalized_identical, &output_normalized_text,
      /*output_normalized_offset_mapping=*/nullptr);
  if (is_normalized_identical) {
    ASSERT_THAT(spec.input, spec.expected_output);
    ASSERT_THAT(output_normalized_text, "");
  } else {
    ASSERT_THAT(output_normalized_text, spec.expected_output);
  }
}

INSTANTIATE_TEST_SUITE_P(FastBertNormalizerTest, TestNormalization,
                         testing::ValuesIn(GetTestSpecs()));
}  // namespace
}  // namespace text
}  // namespace tensorflow
