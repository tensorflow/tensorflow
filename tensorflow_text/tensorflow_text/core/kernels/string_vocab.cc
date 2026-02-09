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

#include "tensorflow_text/core/kernels/string_vocab.h"

namespace tensorflow {
namespace text {

StringVocab::StringVocab(const std::vector<std::string>& vocab)
    : vocab_(vocab) {
  for (int i = 0; i < vocab.size(); ++i) {
    index_map_[vocab_[i]] = i;
  }
}

LookupStatus StringVocab::Contains(absl::string_view key, bool* value) const {
  *value = index_map_.contains(key);
  return LookupStatus();
}

absl::optional<int> StringVocab::LookupId(absl::string_view key) const {
  auto it = index_map_.find(key);
  if (it == index_map_.end()) {
    return absl::nullopt;
  } else {
    return it->second;
  }
}

// Returns the key of `vocab_id` or empty if `vocab_id` is not valid.
absl::optional<absl::string_view> StringVocab::LookupWord(int vocab_id) const {
  if (vocab_id >= vocab_.size() || vocab_id < 0) {
    return absl::nullopt;
  }
  return vocab_[vocab_id];
}
}  // namespace text
}  // namespace tensorflow
