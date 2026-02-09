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

#ifndef TENSORFLOW_LITE_SUPPORT_CUSTOM_OPS_KERNEL_SENTENCEPIECE_DOUBLE_ARRAY_TRIE_H_
#define TENSORFLOW_LITE_SUPPORT_CUSTOM_OPS_KERNEL_SENTENCEPIECE_DOUBLE_ARRAY_TRIE_H_

#include <functional>
#include <vector>

#include "tensorflow_text/core/kernels/sentencepiece/config_generated.h"
#include "tensorflow_text/core/kernels/sentencepiece/utils.h"

namespace tensorflow {
namespace text {
namespace sentencepiece {

// A trie node specifies a node in the tree, either an intermediate node or
// a leaf node.
// A leaf node contains the id as an int of the string match. This id is encoded
// in the lower 31 bits, thus the number of distinct ids is 2^31.
// An intermediate node has an associated label and an offset to its children.
// The label is encoded in the least significant byte and must match the input
// character during matching.

// A memory mappable trie, compatible with Darts::DoubleArray.
class DoubleArrayTrie {
 public:
  struct Match {
    Match() {}
    Match(int id, int match_length) : id(id), match_length(match_length) {}
    int id = -1;
    int match_length = -1;
    bool empty() const { return match_length == -1; }
    bool operator==(const Match& m) const {
      return m.id == id && m.match_length == match_length;
    }
  };

  // nodes and nodes_length specify the array of the nodes of the trie.
  explicit DoubleArrayTrie(const flatbuffers::Vector<uint32_t>* nodes)
      : nodes_(nodes) {}

  // Finds matches that are prefixes of a string.
  template <typename callback>
  void IteratePrefixMatches(const utils::string_view& input,
                            callback update_fn) const;

  // Finds the longest prefix match of a string.
  Match LongestPrefixMatch(const utils::string_view& input) const {
    Match match;
    IteratePrefixMatches(input, [&match](const Match& m) { match = m; });
    return match;
  }

 private:
  // Returns whether a node as a leaf as a child.
  bool has_leaf(uint32_t i) const { return ((*nodes_)[i]) & 0x100; }

  // Returns a value associated with a node. Available when a node is a leaf.
  int value(uint32_t i) const {
    return static_cast<int>(((*nodes_)[i]) & 0x7fffffff);
  }

  // Returns a label associated with a node.
  // A leaf node will have the MSB set and thus return an invalid label.
  int32_t label(uint32_t i) const { return ((*nodes_)[i]) & 0x800000ff; }

  // Returns offset to children.
  int32_t offset(uint32_t i) const {
    const uint32_t node = (*nodes_)[i];
    return (node >> 10) << ((node & 0x200) >> 6);
  }

  const flatbuffers::Vector<uint32_t>* nodes_;
};

template <typename callback>
void DoubleArrayTrie::IteratePrefixMatches(const utils::string_view& input,
                                           callback update_fn) const {
  if (nodes_->size() == 0) {
    return;
  }
  uint32_t pos = offset(0);
  for (int i = 0; i < input.length(); ++i) {
    pos ^= static_cast<unsigned char>(input.at(i));
    if (pos < 0 || pos >= nodes_->size() || label(pos) != input.at(i)) {
      // No match, exit.
      return;
    }
    const bool node_has_leaf = has_leaf(pos);
    pos ^= offset(pos);
    if (pos < 0 || pos >= nodes_->size()) {
      // We can get here only if the structure is corrupted.
      return;
    }
    if (node_has_leaf) {
      update_fn(Match(value(pos), i + 1));
    }
  }
}

}  // namespace sentencepiece
}  // namespace text
}  // namespace tensorflow

#endif  // TENSORFLOW_LITE_SUPPORT_CUSTOM_OPS_KERNEL_SENTENCEPIECE_DOUBLE_ARRAY_TRIE_H_
