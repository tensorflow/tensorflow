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

// To optimize speed/memory usage, we assume:
//  * The WordPiece vocabulary has at most 2^22 = 4M tokens.
//  * No token from the vocabulary has more than 256 bytes.
//
// The assumptions are adjustable by setting the constants defined in this file.
//
// Note: by recompiling the underlying trie library and the helper functions in
// this file to use 64-bit (or even larger) integers, we can support even a
// larger vocab size and longer vocab tokens. Still, we believe the current
// implementation covers all real cases.
#ifndef THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_FAST_WORDPIECE_TOKENIZER_UTILS_H_
#define THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_FAST_WORDPIECE_TOKENIZER_UTILS_H_

#include <stdint.h>

#include <limits>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "icu4c/source/common/unicode/uchar.h"
#include "icu4c/source/common/unicode/umachine.h"

namespace tensorflow {
namespace text {
namespace fast_wordpiece_tokenizer_utils {

// This header assumes that <int> is 32-bit integer types.
static_assert(sizeof(int) == 4, "FastWordpieceTokenizer requires 4-byte int.");

////////////////////////////////////////////////////////////////////////////////
// Constants for token encoding.
//
// The constants below define a 32-bit compact token representation that encodes
// (1) the token id, (2) the token length (minus 1, and without the suffix
// indicator, in utf-8 bytes), and (3) is_suffix_token (i.e., the token starts
// with the suffix indicator (say) "##").
//
// The encoded value is stored on the darts_clone trie as well as in the
// `failure_pops_pool` (see FastWordpieceTokenizerConfig in
// fast_wordpiece_tokenizer_model.fbs). As required by darts_clone_trie, the
// type of the encoded value should be 32-bit signed int, and the top bit is
// reserved to be always 0.
//
// Examples (given the existing constants; bits are numbered 0 to 31 from
// right/lower to left/upper; the top bit is reserved by darts_clone trie and is
// always 0):
//  * Token "a", token id 0 -> The encoded value is 0x0:
//    * bit 31: 0.
//    * bit 30: 0, since token "a" is not a suffix token.
//    * bits 29-8: 0, since the token id is 0.
//    * bits 7-0: 0, since the encoded token length is 0 (see below comments).
//  * Token "b", token id 1 -> The encoded value is 0x100:
//    * bit 31: 0.
//    * bit 30: 0, since token "b" is not a suffix token.
//    * bits 29-8: 1, since the token id is 1.
//    * bits 7-0: 0, since the encoded token length is 0 (see below comments).
//  * Token "##b", token id 2 -> The encoded value is 0x40000200:
//    * bit 31: 0.
//    * bit 30: 1, since token "##b" is a suffix token.
//    * bits 29-8: 2, since the token id is 2.
//    * bits 7-0: 0, since the encoded token length is 0 (see below comments).
//  * Token "bc", token id 3 -> The encoded value is 0x301:
//    * bit 31: 0.
//    * bit 30: 0, since token "bc" is not a suffix token.
//    * bits 29-8: 3, since the token id is 3.
//    * bits 7-0: 1, since the encoded token length is 1 (see below comments).
//  * Token "##bcd", token id 5 -> The encoded value is 0x40000502:
//    * bit 31: 0.
//    * bit 30: 1, since token "##bcd" is a suffix token.
//    * bits 29-8: 5, since the token id is 5.
//    * bits 7-0: 2, since the encoded token length is 2 (see below comments).
//
// One special case is that when the suffix indicator is the empty string "". In
// this case, `is_suffix_token` is false for all tokens.
//
// Another special case is that when the suffix indicator string happens to be a
// token in the vocabulary. When encoding such a token like "##", by design,
// `is_suffix_token` is false, and the encoded token length is the full length
// of the suffix indicator string.
//
////////////////////////////////////////////////////////////////////////////////

// The (right-to-left 0-based) bit to encode whether the token is a suffix
// token.
static constexpr uint32_t kBitToIndicateSuffixToken = 30;

// The number of low bits to encode the vocab token length into a compact
// representation. Technically, we encode the length of the token without the
// suffix indicator (if any) minus 1. Examples:
//  * Token "a" -> we encode 1-1 = 0.
//  * Token "abc" -> we encode 3-1 = 0.
//  * Token "##abc" -> we encode 2, as before (we ignore the suffix indicator).
static constexpr uint32_t kBitsToEncodeVocabTokenLength = 8;

// The bit mask to get the vocab token length from the compact representation.
static constexpr uint32_t kMaskToEncodeVocabTokenLength =
    (1 << kBitsToEncodeVocabTokenLength) - 1;

// Max vocab token length supported (given `kBitsToEncodeVocabTokenLength`).
static constexpr uint32_t kMaxVocabTokenLengthInUTF8Bytes =
    (1 << kBitsToEncodeVocabTokenLength);

// The maximum vocab size supported by our 32-bit encoding. Using right-to-left
// 0-based numbering, Bit 31 is reserved by darts_clone trie. Bit 30 indicates
// whether the token is a suffix token. The low `kBitsToEncodeVocabTokenLength`
// bits encode the token length. Given `kBitsToEncodeVocabTokenLength=8`, this
// leaves 32-1-1-8=22 bits for token ids, i.e., a max vocab size of 2^22 = 4M.
static constexpr uint32_t kMaxSupportedVocabSize =
    (1 << (32 - 1 - 1 - kBitsToEncodeVocabTokenLength));

// The bit mask to get the vocab token id from the compact representation.
static constexpr uint32_t kMaskToEncodeVocabTokenId =
    ((1 << kBitToIndicateSuffixToken) - 1) ^ kMaskToEncodeVocabTokenLength;

////////////////////////////////////////////////////////////////////////////////
// Helpers for encoding / decoding tokens.
////////////////////////////////////////////////////////////////////////////////

// Encodes a token into the encoded value. `token_length` is without the suffix
// indicator. The result is always a non-negative integer. Only used in building
// the model (in flatbuffer), not in doing WordPiece tokenization.
inline absl::StatusOr<int> EncodeToken(int token_id, int token_length,
                                       bool is_suffix_token) {
  const int encoded_value = (is_suffix_token << kBitToIndicateSuffixToken) |
                            (token_id << kBitsToEncodeVocabTokenLength) |
                            (token_length - 1);
  if (encoded_value < 0) {
    return absl::FailedPreconditionError(absl::StrCat(
        "EncodeToken() must return a non-negative value! Found encoded value: ",
        encoded_value, " for input token id: ", token_id, ", token_length: ",
        token_length, ", is_suffix_token: ", is_suffix_token));
  }
  return encoded_value;
}

// Gets whether it is a suffix token from the encoded value.
inline bool IsSuffixToken(int token_encoded_value) {
  return static_cast<bool>(token_encoded_value >> kBitToIndicateSuffixToken);
}

// Gets the token id from the encoded value.
inline int GetTokenId(int token_encoded_value) {
  return (token_encoded_value & kMaskToEncodeVocabTokenId) >>
         kBitsToEncodeVocabTokenLength;
}

// Gets the token length (without the suffix indicator) from the encoded value.
inline int GetTokenLength(int token_encoded_value) {
  return (token_encoded_value & kMaskToEncodeVocabTokenLength) + 1;
}

////////////////////////////////////////////////////////////////////////////////
// Constants for encoding failure pop lists.
//
// We put all failure pop lists into a common pool. The constants below define
// the compact representation that encodes (1) the offset, and (2) the length
// (minus 1) for a failure pop list in the common pool.
//
// Examples (given the existing constants; bits are numbered 0 to 31 from
// right/lower to left/upper):
//  * failure pop list A, whose offset is 0 and length is 1 -> The encoded value
//  is 0x0:
//    * bits 31-8: 0, since the offset is 0.
//    * bits 7-0: 0, since the encoded length is 0 (=1-1).
//  * failure pop list B, whose offset is 0 and length is 3 -> The encoded value
//  is 0x2:
//    * bits 31-8: 0, since the offset is 0.
//    * bits 7-0: 2, since the encoded length is 2 (=3-1).
//  * failure pop list C, whose offset is 11 and the length is 10 -> The encoded
//  value is 0xB09:
//    * bits 31-8: 0xB, since the offset is 11.
//    * bits 7-0: 9, since the encoded length is 9 (=10-1).
////////////////////////////////////////////////////////////////////////////////

// The number of low bits used to encode the length of failure pops minus 1 in
// the compact representation. This value should be less than or equal to
// `kBitsToEncodeVocabTokenLength`, since the size of failure pops is bounded by
// the maximum token length in the vocabulary.
static constexpr uint32_t kBitsToEncodeFailurePopsListSize =
    kBitsToEncodeVocabTokenLength;

// The bit mask to get the length of the failure pop list (without any suffix
// indicator, and minus 1) from the compact representation.
static constexpr uint32_t kMaskToEncodeFailurePopsListSize =
    (1 << kBitsToEncodeFailurePopsListSize) - 1;

// Max length of the failure pop list supported (given
// `kBitsToEncodeFailurePopsListSize`).
static constexpr uint32_t kMaxFailurePopsListSize =
    (1 << kBitsToEncodeFailurePopsListSize);

// The maximum valid offset in the failure pool, excluding the largest one
// (i.e., 0xFF...F), which is reserved to denote a null failure pop list (see
// `kNullFailurePopsList`).
static constexpr uint32_t kMaxSupportedFailurePoolOffset =
    (1 << (32 - kBitsToEncodeFailurePopsListSize)) - 1 - 1;

// Represents the null failure pops list, because 0xFF...F is not a valid of
// offset (see `kMaxSupportedFailurePoolOffset`).
static constexpr uint32_t kNullFailurePopsList =
    std::numeric_limits<uint32_t>::max();

////////////////////////////////////////////////////////////////////////////////
// Helpers for encoding / decoding failure pop lists
////////////////////////////////////////////////////////////////////////////////

// Encodes the offset (in the failure pop pool) and the length of a failure pop
// list into an integer for a compact representation.
inline uint32_t EncodeFailurePopList(int offset, int length) {
  return (offset << kBitsToEncodeFailurePopsListSize) | (length - 1);
}

// Decodes the offset (in the failure pop pool) and the length of a failure pop
// list from the compact representation (an integer).
inline void GetFailurePopsOffsetAndLength(uint32_t offset_and_length,
                                          int& out_offset, int& out_length) {
  out_offset = offset_and_length >> kBitsToEncodeFailurePopsListSize;
  out_length = (offset_and_length & kMaskToEncodeFailurePopsListSize) + 1;
}

////////////////////////////////////////////////////////////////////////////////
// Constants related to the Trie structure.
////////////////////////////////////////////////////////////////////////////////

// Represents the null node id. Different from any normal node.
static constexpr uint32_t kNullNode = std::numeric_limits<uint32_t>::max();

// The maximum trie size supported. Because std::numeric_limits<uint32_t>::max()
// (i.e., 0xFFFFFFFF) is reserved to represent the null node, the total trie
// size needs to be smaller or equal to 0xFFFFFFFF.
static constexpr uint32_t kMaxSupportedTrieSize =
    std::numeric_limits<uint32_t>::max();

////////////////////////////////////////////////////////////////////////////////
// Helpers for analyzing Unicode characters.
////////////////////////////////////////////////////////////////////////////////
inline bool IsPunctuationOrChineseChar(UChar32 char_value) {
  uint32_t cp = static_cast<uint32_t>(char_value);
  // Chinese characters that are treated as punctuation in Bert.
  if ((cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) ||
      (cp >= 0x20000 && cp <= 0x2A6DF) || (cp >= 0x2A700 && cp <= 0x2B73F) ||
      (cp >= 0x2B740 && cp <= 0x2B81F) || (cp >= 0x2B820 && cp <= 0x2CEAF) ||
      (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0x2F800 && cp <= 0x2FA1F)) {
    return true;
  }
  // Some special chars e.g. ">", "$" that are not covered by the u_ispunct are
  // considered as punctuation chars.
  if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) ||
      (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126)) {
    return true;
  }
  return u_ispunct(char_value);
}
}  // namespace fast_wordpiece_tokenizer_utils
}  // namespace text
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_FAST_WORDPIECE_TOKENIZER_UTILS_H_
