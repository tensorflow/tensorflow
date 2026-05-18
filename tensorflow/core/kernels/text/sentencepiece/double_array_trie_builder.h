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

#ifndef TENSORFLOW_CORE_KERNELS_TEXT_SENTENCEPIECE_DOUBLE_ARRAY_TRIE_BUILDER_H_
#define TENSORFLOW_CORE_KERNELS_TEXT_SENTENCEPIECE_DOUBLE_ARRAY_TRIE_BUILDER_H_

#include <string>
#include <vector>

#include "tensorflow/core/kernels/text/sentencepiece/config_generated.h"
#include "tensorflow/core/kernels/text/sentencepiece/utils.h"

namespace tensorflow {
namespace text {
namespace sentencepiece {

std::vector<uint32_t> BuildTrie(const std::vector<std::string>& data,
                                const std::vector<int>& ids);

// A variant where ids are indexes in data.
std::vector<uint32_t> BuildTrie(const std::vector<std::string>& data);

}  // namespace sentencepiece
}  // namespace text
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_TEXT_SENTENCEPIECE_DOUBLE_ARRAY_TRIE_BUILDER_H_
