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

#ifndef THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_UTF8_BINARIZE_H_
#define THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_UTF8_BINARIZE_H_

#include "absl/strings/string_view.h"
#include "absl/types/span.h"

namespace tensorflow {
namespace text {

// Stores low-endian floating-point bitwise representations of Unicode code
// points of `input` in `result` (`result.size()` is required to be exactly
// `word_length * bits_per_char` - output is padded / truncated accordingly).
// Replacements (for invalid UTF sequences) are represented by the
// `bits_per_char` lowest bits of `replacement`.
void Utf8Binarize(absl::string_view input, int word_length, int bits_per_char,
                  int replacement, /* out */ absl::Span<float> result);

}  // namespace text
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_UTF8_BINARIZE_H_
