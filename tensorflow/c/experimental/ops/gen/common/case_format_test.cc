/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/c/experimental/ops/gen/common/case_format.h"

#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace generator {

namespace {

// For each test case, we manually construct the 4 variations in string case and
// test all 16 conversions: from and to each of the 4 string case variations.
struct Variations {
  string lower_camel;
  string lower_snake;
  string upper_camel;
  string upper_snake;
};

void TestSingleVariation(const string &str, Variations expected,
                         char delimiter = '_') {
  EXPECT_EQ(expected.lower_camel, toLowerCamel(str, delimiter));
  EXPECT_EQ(expected.lower_snake, toLowerSnake(str, delimiter));
  EXPECT_EQ(expected.upper_camel, toUpperCamel(str, delimiter));
  EXPECT_EQ(expected.upper_snake, toUpperSnake(str, delimiter));
}

void TestAllVariations(Variations variations, char delimiter = '_') {
  TestSingleVariation(variations.lower_camel, variations, delimiter);
  TestSingleVariation(variations.lower_snake, variations, delimiter);
  TestSingleVariation(variations.upper_camel, variations, delimiter);
  TestSingleVariation(variations.upper_snake, variations, delimiter);
}

TEST(CppOpGenCaseFormat, test_single_word) {
  TestAllVariations(Variations{
      "three",
      "three",
      "Three",
      "THREE",
  });
}

TEST(CppOpGenCaseFormat, test_complex_string) {
  TestAllVariations(Variations{
      "threeNTest33Words",
      "three_n_test33_words",
      "ThreeNTest33Words",
      "THREE_N_TEST33_WORDS",
  });
}

TEST(CppOpGenCaseFormat, test_hyphen_delimiter) {
  TestAllVariations(
      Variations{
          "threeNTest33Words",
          "three-n-test33-words",
          "ThreeNTest33Words",
          "THREE-N-TEST33-WORDS",
      },
      '-');
}

TEST(CppOpGenCaseFormat, test_trailing_underscore) {
  TestAllVariations(Variations{
      "threeNTest33Words_",
      "three_n_test33_words_",
      "ThreeNTest33Words_",
      "THREE_N_TEST33_WORDS_",
  });
}

TEST(CppOpGenCaseFormat, test_double_trailing_underscores) {
  TestAllVariations(Variations{
      "xxY__",
      "xx_y__",
      "XxY__",
      "XX_Y__",
  });
}

TEST(CppOpGenCaseFormat, test_leading_underscore) {
  TestAllVariations(Variations{
      "_threeNTest33Words",
      "_three_n_test33_words",
      "_ThreeNTest33Words",
      "_THREE_N_TEST33_WORDS",
  });
}

TEST(CppOpGenCaseFormat, test_double_leading_underscores) {
  TestAllVariations(Variations{
      "__threeNTest33Words",
      "__three_n_test33_words",
      "__ThreeNTest33Words",
      "__THREE_N_TEST33_WORDS",
  });
}

TEST(CppOpGenCaseFormat, test_leading_and_trailing_underscores) {
  TestAllVariations(Variations{
      "__threeNTest33Words____",
      "__three_n_test33_words____",
      "__ThreeNTest33Words____",
      "__THREE_N_TEST33_WORDS____",
  });
}

}  // namespace

}  // namespace generator
}  // namespace tensorflow
