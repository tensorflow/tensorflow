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

#ifndef TENSORFLOW_TEXT_CORE_KERNELS_SENTENCE_BREAKING_UTILS_H_
#define TENSORFLOW_TEXT_CORE_KERNELS_SENTENCE_BREAKING_UTILS_H_

#include <string>
#include "absl/strings/string_view.h"
#include "icu4c/source/common/unicode/ucnv.h"
#include "icu4c/source/common/unicode/ucnv_err.h"
#include "icu4c/source/common/unicode/utypes.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace text {

// A class of utils for identifying certain classes and properties of unicode
// characters.
class UnicodeUtil {
 public:
  // `converter` not owned.
  explicit UnicodeUtil(UConverter* converter) : converter_(converter) {}

  // Returns true iff a string is terminal punctuation.
  absl::Status IsTerminalPunc(const absl::string_view& input,
                              bool* result) const;

  // Returns true iff a string is close punctuation (close quote or close
  // paren).
  absl::Status IsClosePunc(const absl::string_view& input, bool* result) const;

  // Returns true iff a string is an open paren.
  absl::Status IsOpenParen(const absl::string_view& input, bool* result) const;

  // Returns true iff a string is a close paren.
  absl::Status IsCloseParen(const absl::string_view& input, bool* result) const;

  // Returns true iff a word is made of punctuation characters only.
  absl::Status IsPunctuationWord(const absl::string_view& input,
                                 bool* result) const;

  // Returns true iff a string is an ellipsis token ("...").
  absl::Status IsEllipsis(const absl::string_view& input, bool* result) const;

 private:
  absl::Status GetOneUChar(const absl::string_view&,
                           bool* has_more_than_one_char, UChar32* result) const;

  // not owned. mutable because UConverter contains some internal options and
  // buffer.
  mutable UConverter* converter_;
};

}  // namespace text
}  // namespace tensorflow

#endif  // TENSORFLOW_TEXT_CORE_KERNELS_SENTENCE_BREAKING_UTILS_H_
