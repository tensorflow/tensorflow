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

#ifndef THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_FAST_WORDPIECE_TOKENIZER_MODEL_BUILDER_H_
#define THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_FAST_WORDPIECE_TOKENIZER_MODEL_BUILDER_H_

#include <string>
#include <vector>

#include "absl/status/statusor.h"

namespace tensorflow {
namespace text {

// Builds a FastWordpieceTokenizer model in flatbuffer format.
//
// Args:
//  * vocab: The WordPiece vocabulary.
//  * max_bytes_per_token: The max size of the input token. If the input
//    length is longer than this, it will be mapped to unk_token.
//  * suffix_indicator: Characters prepended to a wordpiece to indicate that
//    it is a suffix to another subword, such as "##".
//  * unk_token: The unknown token string.
//  * no_pretokenization: Whether to pretokenize on punctuation & whitespace.
//    Set to `false` when the model is used for general text end-to-end
//    tokenization, which combines pre-tokenization (splitting text into words
//    on punctuation/whitespaces) and WordPiece (breaking words into subwords)
//    into one pass.
//. * support_detokenization: Whether to enable the detokenization function.
//    Setting it to true expands the size of the flatbuffer. As a reference,
//    When using 120k multilingual BERT WordPiece vocab, the flatbuffer's size
//    increases from ~5MB to ~6MB.
// Returns:
//  The bytes of the flatbuffer that stores the model.
absl::StatusOr<std::string> BuildModelAndExportToFlatBuffer(
    const std::vector<std::string>& vocab, int max_bytes_per_token,
    absl::string_view suffix_indicator, absl::string_view unk_token,
    bool no_pretokenization = false, bool support_detokenization = false);
}  // namespace text
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_FASt_WORDPIECE_TOKENIZER_MODEL_BUILDER_H_
