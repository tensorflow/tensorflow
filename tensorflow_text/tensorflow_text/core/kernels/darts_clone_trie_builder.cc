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

#include "tensorflow_text/core/kernels/darts_clone_trie_builder.h"

#include <algorithm>
#include <memory>
#include <numeric>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "include/darts.h"

namespace tensorflow {
namespace text {
namespace trie_utils {

absl::StatusOr<std::vector<uint32_t>> BuildDartsCloneTrie(
    const std::vector<std::string>& keys) {
  std::vector<int> values(keys.size());
  std::iota(values.begin(), values.end(), 0);
  return BuildDartsCloneTrie(keys, values);
}

absl::StatusOr<std::vector<uint32_t>> BuildDartsCloneTrie(
    const std::vector<std::string>& keys, const std::vector<int>& values) {
  if (keys.size() != values.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The sizes of 'keys' and 'values' must be equal! Keys size: ",
        keys.size(), " . Values size: ", values.size()));
  }

  {
    // Make sure there are no duplicated elements or empty strings in 'keys'.
    absl::flat_hash_set<absl::string_view> unique_keys;
    for (const auto& key : keys) {
      if (key.empty()) {
        return absl::InvalidArgumentError(
            "The empty string \"\" is found in 'keys', which is not "
            "supported.");
      }
      if (!unique_keys.insert(key).second) {
        return absl::InvalidArgumentError(
            absl::StrCat("Duplicated key: ", key, "."));
      }
    }
  }

  // Make sure all values are non-negative.
  for (int i = 0; i < keys.size(); ++i) {
    if (values[i] < 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "All values must be non-negative! Found value: ", values[i],
          " for key: ", keys[i], ", at index: ", i));
    }
  }

  // Create a vector to hold the indexes.
  std::vector<int> vocab_index_sorted(keys.size());
  std::iota(vocab_index_sorted.begin(), vocab_index_sorted.end(), 0);

  // Sort the index by keys.
  std::sort(
      vocab_index_sorted.begin(), vocab_index_sorted.end(),
      [&keys](const int x, const int y) { return keys.at(x) < keys.at(y); });

  // Create vectors to build the trie.
  std::vector<const char*> trie_keys;
  std::vector<int> trie_values;
  trie_keys.reserve(keys.size());
  trie_values.reserve(keys.size());
  for (const auto index : vocab_index_sorted) {
    trie_keys.push_back(keys.at(index).c_str());
    trie_values.push_back(values[index]);
  }

  // Build the trie.
  auto trie = std::make_unique<Darts::DoubleArray>();
  trie->build(trie_keys.size(), const_cast<char**>(&trie_keys[0]), nullptr,
              const_cast<int*>(&trie_values[0]));

  // Return the data of darts_clone (an array of 32-bit unsigned int).
  const uint32_t* trie_array = static_cast<const uint32_t*>(trie->array());
  return std::vector<uint32_t>(trie_array, trie_array + trie->size());
}

}  // namespace trie_utils
}  // namespace text
}  // namespace tensorflow
