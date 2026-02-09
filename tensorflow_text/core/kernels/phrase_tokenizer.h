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

#ifndef THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_Phrase_TOKENIZER_H_
#define THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_Phrase_TOKENIZER_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/random/random.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "tensorflow_text/core/kernels/phrase_tokenizer_model_generated.h"
#include "tensorflow_text/core/kernels/sentencepiece/double_array_trie.h"
#include "tensorflow_text/core/kernels/string_vocab.h"
#include "tensorflow_text/core/kernels/whitespace_tokenizer.h"

namespace tensorflow {
namespace text {

class PhraseTokenizer {
 public:
  // Creates an instance.
  //
  // Args:
  //  * config_flatbuffer: the pointer to the PhraseTokenizerConfig
  //    flatbuffer, which is not owned by this instance and should be kept
  //    alive through the lifetime of the instance.
  static absl::StatusOr<PhraseTokenizer> Create(const void* config_flatbuffer);

  // Tokenizes a string (or series of character codepoints) by Phrase.
  //
  // Example:
  // input = "Show me the way."
  // output = ["Show me", "the", "way."]
  //
  // The input should be UTF-8 but the tokenization is performed on Unicode
  // codepoints.
  //
  // Args:
  //  * input: The UTF-8 string of an input.
  //  * tokens: The output tokens.
  void Tokenize(const absl::string_view input,
                std::vector<std::string>* result_tokens,
                std::vector<int>* result_token_ids);

  // Detokenizer the input into a single string.
  absl::StatusOr<std::string> Detokenize(
      const absl::Span<const int> input) const;

 private:
  // Detokenizer the input into vector of strings.
  absl::StatusOr<std::vector<std::string>> DetokenizeToTokens(
      const absl::Span<const int> input) const;

  // Find the phrase tokens based on the current phrase.
  void FindPhraseTokens(const std::string& cur_phrase,
                        std::vector<std::string>* phrase_tokens,
                        std::vector<int>* phrase_token_ids);

  // Lookup the phrase in the token string from current index.
  // Args:
  //  * token: The input token string to find the next phrase.
  //  * cur: The starting point to search for the phrase.
  //  * in_trie: Whether there is a phrase in DoubleArrayTrie.
  //  * emitted_phrase_id: The emitted phrase id.
  //  * emitted_phrase_length: The length of the emitted phrase.
  void PhraseLookup(const std::string& token, int cur, bool* in_trie,
                    int* emitted_phrase_id, int* emitted_phrase_length);

  std::unique_ptr<StringVocab> vocab_ = nullptr;
  const PhraseTokenizerConfig* phrase_config_;
  absl::string_view whitespace_config_str_;
  std::unique_ptr<sentencepiece::DoubleArrayTrie> trie_ = nullptr;
  float prob_;
  absl::BitGen gen_;
  std::unique_ptr<WhitespaceTokenizer> whitespace_tokenizer_ = nullptr;
  bool split_end_punctuation_ = false;
  const absl::flat_hash_set<std::string> special_tokens_ = {
      "'t", "'s", ".", ",", "!", "?", "'m", "'re", "'ll", "'d", "'ve"};
};

}  // namespace text
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_Phrase_TOKENIZER_H_
