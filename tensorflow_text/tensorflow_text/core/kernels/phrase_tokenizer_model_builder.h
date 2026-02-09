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

#ifndef THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_PHRASE_TOKENIZER_MODEL_BUILDER_H_
#define THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_PHRASE_TOKENIZER_MODEL_BUILDER_H_

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"

namespace tensorflow {
namespace text {

// Builds a PhraseTokenizer model in flatbuffer format.
//
// Args:
//  * vocab: The phrase vocabulary.
//  * unk_token: The unknown token string.
//. * support_detokenization: Whether to enable the detokenization function.
//    Setting it to true expands the size of the flatbuffer.
//  * prob: Probability of emitting a phrase when there is a match.
// Returns:
//  The bytes of the flatbuffer that stores the model.
absl::StatusOr<std::string> BuildPhraseModelAndExportToFlatBuffer(
    const std::vector<std::string>& vocab, const std::string& unk_token,
    bool support_detokenization = false, int prob = 0,
    bool split_end_punctuation = false);
}  // namespace text
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_PHRASE_TOKENIZER_MODEL_BUILDER_H_
