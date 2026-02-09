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

#ifndef TENSORFLOW_TEXT_CORE_KERNELS_WORDPIECE_TOKENIZER_H_
#define TENSORFLOW_TEXT_CORE_KERNELS_WORDPIECE_TOKENIZER_H_

#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"

namespace tensorflow {
namespace text {

struct LookupStatus {
  LookupStatus() : error_msg(""), success(true) {}
  LookupStatus(std::string msg) : error_msg(std::move(msg)), success(false) {}
  std::string error_msg;
  bool success;

  static LookupStatus OK() { return LookupStatus(); }
};

class WordpieceVocab {
 public:
  virtual ~WordpieceVocab() {}
  virtual LookupStatus Contains(const absl::string_view key,
                                bool* value) const = 0;
};

LookupStatus WordpieceTokenize(
    const absl::string_view& token, const int max_bytes_per_token,
    const int max_chars_per_subtoken, const std::string& suffix_indicator,
    bool use_unknown_token, const std::string& unknown_token,
    bool split_unknown_characters, const WordpieceVocab* vocab_map,
    std::vector<std::string>* subwords, std::vector<int>* begin_offset,
    std::vector<int>* end_offset, int* num_word_pieces);

// As above but with `max_bytes_per_subtoken` unknown,
// and split_unknown_characters=false. (For backwards compatability.)
LookupStatus WordpieceTokenize(
    const absl::string_view& token, const int max_bytes_per_token,
    const std::string& suffix_indicator, bool use_unknown_token,
    const std::string& unknown_token, const WordpieceVocab* vocab_map,
    std::vector<std::string>* subwords, std::vector<int>* begin_offset,
    std::vector<int>* end_offset, int* num_word_pieces);

}  // namespace text
}  // namespace tensorflow

#endif  // TENSORFLOW_TEXT_CORE_KERNELS_WORDPIECE_TOKENIZER_H_
