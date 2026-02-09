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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow_text/core/kernels/sentencepiece/double_array_trie_builder.h"

#include <algorithm>
#include <memory>

#include "include/darts.h"

namespace tensorflow {
namespace text {
namespace sentencepiece {

std::vector<uint32_t> BuildTrie(const std::vector<std::string>& data) {
  std::vector<int> ids;
  ids.reserve(data.size());
  for (int i = 0; i < data.size(); ++i) {
    ids.push_back(i);
  }
  return BuildTrie(data, ids);
}

std::vector<uint32_t> BuildTrie(const std::vector<std::string>& data,
                                const std::vector<int>& ids) {
  // We make strong assumptions about binary structure of trie.
  struct OneElement {
    OneElement(const std::string* key_, int index_)
        : key(key_), index(index_) {}
    const std::string* key;
    int index;
    bool operator<(const OneElement& el) const { return *key < *el.key; }
  };
  std::vector<OneElement> elements;
  elements.reserve(data.size());
  auto data_iterator = std::begin(data);
  auto ids_iterator = std::begin(ids);
  for (; data_iterator != std::end(data) && ids_iterator != std::end(ids);
       ++data_iterator, ++ids_iterator) {
    elements.emplace_back(&(*data_iterator), *ids_iterator);
  }
  // Sort by keys.
  std::sort(elements.begin(), elements.end());

  // Create vectors to build the trie.
  std::vector<const char*> strings;
  std::vector<int32_t> indexes;
  strings.reserve(data.size());
  indexes.reserve(data.size());
  for (const auto& el : elements) {
    strings.push_back(el.key->c_str());
    indexes.push_back(el.index);
  }
  auto trie = std::make_unique<Darts::DoubleArray>();
  trie->build(data.size(), const_cast<char**>(&strings[0]), nullptr,
              &indexes[0]);
  // We make strong assumptions about internal Darts trie structure:
  // - it is a vector of 32 bit signed integers
  // - the "array" is the only one structure that contains all information about
  // the trie.
  const uint32_t* trie_data = static_cast<const uint32_t*>(trie->array());
  return std::vector<uint32_t>(trie_data, trie_data + trie->size());
}

}  // namespace sentencepiece
}  // namespace text
}  // namespace tensorflow
