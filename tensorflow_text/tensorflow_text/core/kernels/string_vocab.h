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

#ifndef THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_STRING_VOCAB_H_
#define THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_STRING_VOCAB_H_

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow_text/core/kernels/wordpiece_tokenizer.h"

namespace tensorflow {
namespace text {

// An implementation of WordpieceVocab, used (1) to store the input vocabulary
// and (2) to call the original implementation of WordPiece tokenization to
// pre-compute the result for the suffix indicator string.
class StringVocab : public WordpieceVocab {
 public:
  explicit StringVocab(const std::vector<std::string>& vocab);
  LookupStatus Contains(absl::string_view key, bool* value) const override;
  absl::optional<int> LookupId(absl::string_view key) const;
  // Returns the key of `vocab_id` or empty if `vocab_id` is not valid.
  absl::optional<absl::string_view> LookupWord(int vocab_id) const;
  int Size() const { return index_map_.size(); }

 private:
  std::vector<std::string> vocab_;
  absl::flat_hash_map<absl::string_view, int> index_map_;
};
}  // namespace text
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_STRING_VOCAB_H_
