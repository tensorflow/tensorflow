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

#ifndef THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_WHITESPACE_TOKENIZER_CONFIG_BUILDER_H_
#define THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_WHITESPACE_TOKENIZER_CONFIG_BUILDER_H_

#include <string>


namespace tensorflow {
namespace text {

// Builds a WhitespaceTokenizer config object. This contains the Unicode
// codepoints which are considered whitespaces.
//
// The config object is a series of bytes, where each bit represents a Unicode
// character and is 1 if it is a whitespace character, and 0 otherwise.
//
// Returns:
//   The bytes of the config as a string.
std::string BuildWhitespaceTokenizerConfig();

// Builds a string full of all the whitespace characters. It is mainly used
// for testing and validation.
//
// Returns:
//   A string of Unicode whitespace characters.
std::string BuildWhitespaceString();

}  // namespace text
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_WHITESPACE_TOKENIZER_CONFIG_BUILDER_H_
