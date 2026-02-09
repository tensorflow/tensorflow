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

// Builder utils for Darts-clone tries.
//
// Darts-clone is a compact and efficient implementation of Darts (Double-ARray
// Trie System). For more info, see https://github.com/s-yata/darts-clone.
//
// This header file contains utils that build a darts-clone trie. To access such
// a darts-clone trie, use the utils from the companion header file
// darts_clone_trie_wrapper.h.
#ifndef THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_DARTS_CLONE_TRIE_BUILDER_H_
#define THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_DARTS_CLONE_TRIE_BUILDER_H_

#include <stdint.h>
#include <string>
#include <vector>

#include "absl/status/statusor.h"

namespace tensorflow {
namespace text {
namespace trie_utils {

// Builds the trie given keys and values, and returns the darts_clone trie
// array data. `keys` and `values` should have the same size; `values[i]` is the
// value for `keys[i]`. `keys` should not contain duplicated elements. In
// addition, the empty string "" should not be in `keys`, because darts_clone
// does not support that. Furthermore, all `values` should be non-negative.
absl::StatusOr<std::vector<uint32_t>> BuildDartsCloneTrie(
    const std::vector<std::string>& keys, const std::vector<int>& values);

// A variant where the values are indexes in the keys: i.e., the value for
// `keys[i]` is the index `i`.
absl::StatusOr<std::vector<uint32_t>> BuildDartsCloneTrie(
    const std::vector<std::string>& keys);

}  // namespace trie_utils
}  // namespace text
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_DARTS_CLONE_TRIE_BUILDER_H_
