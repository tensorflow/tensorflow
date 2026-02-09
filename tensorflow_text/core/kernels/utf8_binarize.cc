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
#include <algorithm>
#include <cassert>

#include "absl/strings/string_view.h"
#include "icu4c/source/common/unicode/utf8.h"

namespace tensorflow {
namespace text {

void Utf8Binarize(
    absl::string_view input, int word_length, int bits_per_char,
    int replacement, /* out */ absl::Span<float> result) {
  assert(result.size() == word_length * bits_per_char);

  const int input_size = input.size();
  int string_pos = 0;
  int chars = 0;
  int result_pos = 0;
  while (string_pos < input_size && chars < word_length) {
    UChar32 chr;
    U8_NEXT(input, string_pos, input_size, chr);
    if (chr < 0) {
      // Decoding failure.
      chr = replacement;
    }
    int bits = bits_per_char;
    while (bits-- != 0) {
      result[result_pos++] = (chr & 1) == 1 ? 1.0f : 0.0f;
      chr >>= 1;
    }
    ++chars;
  }

  std::fill(result.begin() + result_pos, result.end(), 0.0f);
}

}  // namespace text
}  // namespace tensorflow
