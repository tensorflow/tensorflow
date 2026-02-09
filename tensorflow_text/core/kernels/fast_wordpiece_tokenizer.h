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

#ifndef THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_FAST_WORDPIECE_TOKENIZER_H_
#define THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_FAST_WORDPIECE_TOKENIZER_H_

#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "tensorflow_text/core/kernels/darts_clone_trie_wrapper.h"
#include "tensorflow_text/core/kernels/fast_wordpiece_tokenizer_model_generated.h"
#include "tensorflow_text/core/kernels/fast_wordpiece_tokenizer_utils.h"

namespace tensorflow {
namespace text {

// Applies WordPiece tokenization with an existing WordPiece vocabulary.
//
// Example:
// input = unaffable
// output = un ##aff ##able
//
// One important edge case is that if the input word contains a Unicode
// character that is not seen in the vocabulary, the entire word is mapped
// to the unknown token, which is "<unk>" by default. Otherwise, in the "worst"
// case, the word is split into characters.
//
// This is based on the WordPiece/Subword tokenizer from tensor2tensor.
// https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/data_generators/text_encoder.py
class FastWordpieceTokenizer {
 public:
  // Creates an instance.
  //
  // Args:
  //  * config_flatbuffer: the pointer to the FastWordpieceTokenizerConfig
  //    flatbuffer, which is not owned by this instance and should be kept alive
  //    through the lifetime of the instance.
  static absl::StatusOr<FastWordpieceTokenizer> Create(
      const void* config_flatbuffer);

  // Tokenizes `input` into its word pieces (i.e., subword tokens) and
  // appends the new tokens to the end of the outputs.
  // When `config_->end_to_end() is `false`, `input` should be a single
  // word (after pre-tokenization by whitespaces and/or punctuations).
  // Otherwise, `input` should be general text consisting of potentially many
  // words.
  //
  // The input should be UTF-8 but the tokenization is performed on Unicode
  // codepoints.
  //
  //
  // Args:
  //  * input: The UTF-8 string of an input.
  //  * output_pieces: The output tokens.
  //  * output_ids: The output token ids.
  //  * output_start_offsets: The start offsets of output tokens in the input
  //    text, in utf-8 bytes.
  //  * output_end_offsets: The end offsets of output tokens in the input
  //    text, in utf-8 bytes.
  //  * input_word_offset_in_text: The relative offset of the input word in
  //    the whole text. Only used when not using end-to-end tokenizer.
  //  * error: If not null, this will be set to true if the tokenizer failed to
  //    make progress in decoding the input.
  // Note: the start offsets are inclusive and the end offsets are exclusive.
  void Tokenize(absl::string_view input,
                std::vector<std::string>* output_pieces,
                std::vector<int>* output_ids,
                std::vector<int>* output_start_offsets,
                std::vector<int>* output_end_offsets,
                int input_word_offset_in_text = 0, bool* error = nullptr) const;

  // An override not returning `output_pieces`.
  void Tokenize(absl::string_view input, std::vector<int>* output_ids,
                std::vector<int>* output_start_offsets,
                std::vector<int>* output_end_offsets,
                int input_word_offset_in_text = 0) const;

  // An override only returning `output_ids`.
  void Tokenize(absl::string_view input, std::vector<int>* output_ids,
                int input_word_offset_in_text = 0) const;

  // Detokenizes wordpiece ids into a vector of tokens.
  absl::StatusOr<std::vector<std::string>> DetokenizeToTokens(
      const absl::Span<const int> input) const;

  // Detokenizes wordpiece ids to a text. If the input string to the tokenizer
  // is normalized and the tokenized wordpieces don't contain `<unk>`, the
  // detokenized result of the tokenized wordpieces is the same as the original
  // input text.
  absl::StatusOr<std::string> Detokenize(
      const absl::Span<const int> input) const;

 private:
  // The actual implementation of `Tokenize` when configured for single words.
  //
  // The template parameters `kGetPieces`, `kGetIds', and `kGetOffsets` control
  // which parts of the output we generate. At least one of `kGetPieces` and
  // `kGetIds` should be true.
  template <bool kGetPieces, bool kGetIds, bool kGetOffsets>
  void TokenizeSingleWordImpl(absl::string_view input_word,
                              int input_word_offset_in_text,
                              std::vector<std::string>* output_pieces,
                              std::vector<int>* output_ids,
                              std::vector<int>* output_start_offsets,
                              std::vector<int>* output_end_offsets) const;

  // The actual implementation of `Tokenize` when configured for general texts.
  //
  // The work of this method is equivalent to first splitting `input_text` into
  // words (by splitting on punctuation and whitespaces, and next running
  // `TokenizeSingleWordImpl` on each word.
  template <bool kGetPieces, bool kGetIds, bool kGetOffsets>
  void TokenizeTextImpl(absl::string_view input_text,
                        std::vector<std::string>* output_pieces,
                        std::vector<int>* output_ids,
                        std::vector<int>* output_start_offsets,
                        std::vector<int>* output_end_offsets,
                        bool* error) const;

  // Try following the failure link to make the transition when trie matching
  // fails.
  //
  // If f(node) (i.e., failure link) is not null, it does the following:
  //  (1) collects tokens F(node) (i.e., failure pops) and appends to the end of
  //      `output_ids`, `output_pieces`, and/or `output_start_offsets` and
  //      `output_end_offsets`,
  //  (2) moves `cur_offset_in_input_word` accordingly to pass the collected
  //      tokens when `kGetPieces=true` or `kGetOffsets=true`, in order to
  //      calculate the start/end offsets of tokens and to get the token
  //      strings. Otherwise, `cur_offset_in_input_word` is ignored.
  //  (3) transits `node` to f(node) following the failure link,
  //  (4) returns true.
  //
  // If f(node) is null, it does not change anything and returns false.
  //
  // Args:
  //  * cur_offset_in_input_word: The current offset in `input_word` that
  //    corresponds to the start offset of the tokens that are going to be
  //    collected in this function. This value is used if 'kGetPieces=true' or
  //    'kGetOffsets=true', and when so, this value will be updated accordingly
  //    after the new word piece tokens have been appended to the output.
  template <bool kGetPieces, bool kGetIds, bool kGetOffsets>
  bool TryFollowFailureLinkAndCollectTokens(
      absl::string_view input_word, int input_word_offset_in_text,
      int& cur_offset_in_input_word,
      trie_utils::DartsCloneTrieWrapper::TraversalCursor& node,
      std::vector<std::string>* output_pieces, std::vector<int>* output_ids,
      std::vector<int>* output_start_offsets,
      std::vector<int>* output_end_offsets) const;

  // Appends a word piece token (represented by `encoded_token_value`) to the
  // output.
  //
  // Args:
  //  * cur_offset_in_input_word: The current offset in `input_word` that
  //    corresponds to the start offset of the wordpiece token. This value
  //    is used if `kGetPieces=true` or `kGetOffsets=true`, and when so, this
  //    value will be updated accordingly after the wordpiece token has been
  //    appended to the output.
  //  * encoded_token_value: the encoded value of the word piece token to be
  //    appended. See EncodeToken() in fast_wordpiece_tokenizer_utils.h.
  template <bool kGetPieces, bool kGetIds, bool kGetOffsets>
  void AppendTokenToOutput(absl::string_view input_word,
                           int input_word_offset_in_text,
                           int& cur_offset_in_input_word,
                           int encoded_token_value,
                           std::vector<std::string>* output_pieces,
                           std::vector<int>* output_ids,
                           std::vector<int>* output_start_offsets,
                           std::vector<int>* output_end_offsets) const;

  // This method is called when the trie matching loop encounters a word
  // boundary (e.g., the end-of-input). This method segments the remaining
  // string on the trie path into pieces and appends them to the outputs. If
  // that is not possible with the current vocabulary, this method resets the
  // outputs and appends unk_token.
  //
  // Example 1: suppose the vocabulary is {ab, abcd}. If the input word is "ab",
  // after matching "ab", we processed all input characters and now meets the
  // end-of-input. Note that the string "ab" is stored on the trie path that we
  // just traversed along. This function recognizes it as the token "ab" and
  // puts the token into the output as expected.
  //
  // Example 2: for the same vocabulary {ab, abcd}, suppose the input word is
  // "abc". After the trie matching loop, we matched "abc" and encountered the
  // end-of-input. Now the string "abc" is stored on the trie path, which we
  // haven't segmented into tokens yet. So this function closes it by trying to
  // segment "abc" into tokens. It fails since the remaining string "abc" cannot
  // be tokenized into tokens given the vocabulary. In this case, it resets the
  // outputs and appends unk_token at the end as expected.
  template <bool kGetPieces, bool kGetIds, bool kGetOffsets>
  void HandleTheRemainingStringOnTriePath(
      absl::string_view input_word, int input_word_offset_in_text,
      trie_utils::DartsCloneTrieWrapper::TraversalCursor& cur_node,
      int& original_num_tokens, int& cur_offset_in_input_word,
      std::vector<std::string>* output_pieces, std::vector<int>* output_ids,
      std::vector<int>* output_start_offsets,
      std::vector<int>* output_end_offsets) const;

  // Resets the output and appends unk_token.
  //
  // We call this method when we find that the input word cannot be tokenized.
  // We clear all new tokens recognized so far and replace them with a single
  // unk_token.
  //
  // Args:
  //  * input_word_offset_in_text: The offset of the current word in the
  //    input text.
  //  * input_size: The length of the current input word, in utf-8 bytes.
  //  * original_num_tokens: The original number of tokens in the output before
  //    we started the tokenization of the current input word. It is updated
  //    after this method.
  template <bool kGetPieces, bool kGetIds, bool kGetOffsets>
  void ResetOutputAppendUnknownToken(
      int input_word_offset_in_text, int input_size, int& original_num_tokens,
      std::vector<std::string>* output_pieces, std::vector<int>* output_ids,
      std::vector<int>* output_start_offsets,
      std::vector<int>* output_end_offsets) const;

  // Try handling the special case when the input word is the suffix indicator
  // itself. If so, appends the precomputed result to output_pieces and
  // output_ids, and returns true. Otherwise, it does nothing and returns false.
  template <bool kGetPieces, bool kGetIds, bool kGetOffsets>
  bool TryHandleTheInputWordBeingSuffixIndicatorItself(
      absl::string_view input_word, int input_word_offset_in_text,
      const trie_utils::DartsCloneTrieWrapper::TraversalCursor& cur_node,
      int& cur_offset_in_input_word, int original_num_tokens,
      std::vector<std::string>* output_pieces, std::vector<int>* output_ids,
      std::vector<int>* output_start_offsets,
      std::vector<int>* output_end_offsets) const;

  // Returns the position (in bytes) immediately after the end of the word.
  int SkipTheRemainingOfWordAndTrailingWhiteSpaces(absl::string_view input,
                                                   int& cur_pos) const;

  // Points to the FastWordpieceTokenizer config flatbuffer (not owned).
  const FastWordpieceTokenizerConfig* config_ = nullptr;

  // A wrapper to access the trie encoded inside the flatbuffer that `config_`
  // points to.
  std::unique_ptr<trie_utils::DartsCloneTrieWrapper> trie_ = nullptr;
};

}  // namespace text
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_FAST_WORDPIECE_TOKENIZER_H_
