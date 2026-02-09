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

#include "tensorflow_text/core/kernels/utf8_binarize.h"
#include <vector>

#include <gmock/gmock.h>
#include "absl/types/span.h"

namespace tensorflow {
namespace text {
namespace {

using ::testing::ElementsAre;

TEST(UnicodeTest, Utf8Binarize) {
  std::vector<float> out1(3 * 4);
  Utf8Binarize("hello", /*word_length=*/3, /*bits_per_char=*/4,
               /*replacement=*/3, /*result=*/absl::MakeSpan(out1));
                                               // L-endian 4 lowest bits of:
  EXPECT_THAT(out1, ElementsAre(0, 0, 0, 1,    // "h"
                                1, 0, 1, 0,    // "e"
                                0, 0, 1, 1));  // "l"

  std::vector<float> out2(4 * 5);
  Utf8Binarize("爱上一个不回", /*word_length=*/4, /*bits_per_char=*/5,
               /*replacement=*/7, /*result=*/absl::MakeSpan(out2));
                                                  // L-endian 5 lowest bits of:
  EXPECT_THAT(out2, ElementsAre(1, 0, 0, 0, 1,    // "爱"
                                0, 1, 0, 1, 0,    // "上"
                                0, 0, 0, 0, 0,    // "一"
                                0, 1, 0, 1, 0));  // "个"

  // Notable example:
  // - (Unicode) characters are padded, not truncated as above (zero-padding);
  // - the UTF-8 sequence is invalid, so we get a replacement bit pattern.
  std::vector<float> out3(3 * 6);
  Utf8Binarize("\xc3(", /*word_length=*/3, /*bits_per_char=*/6,
               /*replacement=*/35, /*result=*/absl::MakeSpan(out3));
                                                     // LE 6 lowest bits of:
  EXPECT_THAT(out3, ElementsAre(1, 1, 0, 0, 0, 1,    // Replacement.
                                0, 0, 0, 1, 0, 1,    // "(".
                                0, 0, 0, 0, 0, 0));  // Padding.
}

}  // namespace
}  // namespace text
}  // namespace tensorflow
