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

#include "tensorflow/compiler/tf2xla/str_util.h"

#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace str_util {

class ReplaceAllPairsTest : public ::testing::Test {
 protected:
  void ExpectReplaceAllPairs(
      string text, const std::vector<std::pair<string, string>>& replace,
      StringPiece want) {
    ReplaceAllPairs(&text, replace);
    EXPECT_EQ(text, want);
  }
};

TEST_F(ReplaceAllPairsTest, Simple) {
  ExpectReplaceAllPairs("", {}, "");
  ExpectReplaceAllPairs("", {{"", ""}}, "");
  ExpectReplaceAllPairs("", {{"", "X"}}, "X");
  ExpectReplaceAllPairs("", {{"", "XYZ"}}, "XYZ");
  ExpectReplaceAllPairs("", {{"", "XYZ"}, {"", "_"}}, "_X_Y_Z_");
  ExpectReplaceAllPairs("", {{"", "XYZ"}, {"", "_"}, {"_Y_", "a"}}, "_XaZ_");
  ExpectReplaceAllPairs("banana", {}, "banana");
  ExpectReplaceAllPairs("banana", {{"", ""}}, "banana");
  ExpectReplaceAllPairs("banana", {{"", "_"}}, "_b_a_n_a_n_a_");
  ExpectReplaceAllPairs("banana", {{"", "__"}}, "__b__a__n__a__n__a__");
  ExpectReplaceAllPairs("banana", {{"a", "a"}}, "banana");
  ExpectReplaceAllPairs("banana", {{"a", ""}}, "bnn");
  ExpectReplaceAllPairs("banana", {{"a", "X"}}, "bXnXnX");
  ExpectReplaceAllPairs("banana", {{"a", "XX"}}, "bXXnXXnXX");
  ExpectReplaceAllPairs("banana", {{"a", "XX"}, {"XnX", "z"}}, "bXzzX");
  ExpectReplaceAllPairs("a{{foo}}b{{bar}}c{{foo}}",
                        {{"{{foo}}", "0"}, {"{{bar}}", "123456789"}},
                        "a0b123456789c0");
}

}  // namespace str_util
}  // namespace tensorflow
