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

#include "tensorflow_text/core/kernels/sentencepiece/optimized_encoder.h"

#include <algorithm>
#include <tuple>

#include "tensorflow_text/core/kernels/sentencepiece/double_array_trie.h"
#include "tensorflow_text/core/kernels/sentencepiece/encoder_config_generated.h"

namespace tensorflow {
namespace text {
namespace sentencepiece {
namespace {

const char kSpaceSymbol[] = "\xe2\x96\x81";

template <typename processing_callback>
std::tuple<std::string, std::vector<int>> process_string(
    const std::string& input, const std::vector<int>& offsets,
    const processing_callback& pc) {
  std::string result_string;
  result_string.reserve(input.size());
  std::vector<int> result_offsets;
  result_offsets.reserve(offsets.size());
  for (int i = 0, j = 0; i < input.size();) {
    auto result = pc(input.data() + i, input.size() - i);
    auto consumed = std::get<0>(result);
    auto new_string = std::get<1>(result);
    if (consumed == 0) {
      // Skip the current byte and move forward.
      result_string.push_back(input[i]);
      result_offsets.push_back(offsets[j]);
      i++;
      j++;
      continue;
    }
    result_string.append(new_string.data(), new_string.length());
    for (int i = 0; i < new_string.length(); ++i) {
      result_offsets.push_back(offsets[j]);
    }
    j += consumed;
    i += consumed;
  }
  return std::make_tuple(result_string, result_offsets);
}

inline char is_whitespace(char c) {
  return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

std::tuple<int, utils::string_view> remove_extra_whitespaces(const char* data,
                                                             int len) {
  if (len == 0 || !is_whitespace(*data)) {
    return std::make_tuple(0, utils::string_view(nullptr, 0));
  }
  int num_consumed = 1;
  for (; num_consumed < len && is_whitespace(data[num_consumed]);
       ++num_consumed) {
  }
  return num_consumed > 1
             ? std::make_tuple(num_consumed, utils::string_view(" ", 1))
             : std::make_tuple(0, utils::string_view(nullptr, 0));
}

std::tuple<int, utils::string_view> find_replacement(
    const char* data, int len, const DoubleArrayTrie& dat,
    const flatbuffers::Vector<int8_t>& replacements) {
  const auto max_match = dat.LongestPrefixMatch(utils::string_view(data, len));
  if (!max_match.empty()) {
    // Because flatbuffer byte is signed char which is not the same as char,
    // there is the reinterpret_cast here.
    const char* replaced_string_ptr =
        reinterpret_cast<const char*>(replacements.data() + max_match.id);
    return std::make_tuple(max_match.match_length,
                           utils::string_view(replaced_string_ptr));
  }
  return std::make_tuple(0, utils::string_view(nullptr, 0));
}
}  // namespace

std::tuple<std::string, std::vector<int>> NormalizeString(
    const std::string& in_string, const EncoderConfig& config) {
  std::vector<int> output_offsets;
  std::string result = in_string;
  output_offsets.reserve(in_string.length());
  for (int i = 0; i < in_string.length(); ++i) {
    output_offsets.push_back(i);
  }
  if (in_string.empty()) {
    return std::make_tuple(result, output_offsets);
  }
  if (config.add_dummy_prefix()) {
    result.insert(result.begin(), ' ');
    output_offsets.insert(output_offsets.begin(), 0);
  }
  // Greedely replace normalized_prefixes with normalized_replacements
  if (config.normalized_prefixes() != nullptr &&
      config.normalized_replacements() != nullptr) {
    const DoubleArrayTrie normalized_prefixes_matcher(
        config.normalized_prefixes()->nodes());
    const auto norm_replace = [&config, &normalized_prefixes_matcher](
                                  const char* data, int len) {
      return find_replacement(data, len, normalized_prefixes_matcher,
                              *config.normalized_replacements());
    };
    std::tie(result, output_offsets) =
        process_string(result, output_offsets, norm_replace);
  }
  if (config.remove_extra_whitespaces()) {
    std::tie(result, output_offsets) =
        process_string(result, output_offsets, remove_extra_whitespaces);
    if (!result.empty() && is_whitespace(result.back())) {
      result.pop_back();
      output_offsets.pop_back();
    }
  }
  if (config.escape_whitespaces()) {
    const auto replace_whitespaces = [](const char* data, int len) {
      if (len > 0 && is_whitespace(*data)) {
        return std::make_tuple(1, utils::string_view(kSpaceSymbol));
      }
      return std::make_tuple(0, utils::string_view(nullptr, 0));
    };
    std::tie(result, output_offsets) =
        process_string(result, output_offsets, replace_whitespaces);
  }

  return std::make_tuple(result, output_offsets);
}

EncoderResult EncodeNormalizedString(const std::string& str,
                                     const std::vector<int>& offsets,
                                     const EncoderConfig& config, bool add_bos,
                                     bool add_eos, bool reverse) {
  const DoubleArrayTrie piece_matcher(config.pieces()->nodes());
  const flatbuffers::Vector<float>* piece_scores = config.pieces_scores();
  const int unknown_code = config.unknown_code();
  const float unknown_penalty = config.unknown_penalty();
  struct LatticeElement {
    float score = 0;
    int code = -1;
    int prev_position = -1;
    LatticeElement(float score_, int code_, int prev_position_)
        : score(score_), code(code_), prev_position(prev_position_) {}
    LatticeElement() {}
  };
  const int length = str.length();
  std::vector<LatticeElement> lattice(length + 1);
  for (int i = 0; i < length; ++i) {
    if (i > 0 && lattice[i].prev_position < 0) {
      // This state is unreachable.
      continue;
    }
    if (unknown_code >= 0) {
      // Put unknown code.
      const float penalized_score = lattice[i].score + unknown_penalty;
      const int pos = i + 1;
      LatticeElement& current_element = lattice[pos];
      if (current_element.prev_position < 0 ||
          current_element.score < penalized_score) {
        current_element = LatticeElement(
            penalized_score, unknown_code,
            // If the current state is already reached by unknown code, merge
            // states.
            lattice[i].code == unknown_code ? lattice[i].prev_position : i);
      }
    }
    auto lattice_update = [&lattice, i,
                           piece_scores](const DoubleArrayTrie::Match& m) {
      LatticeElement& target_element = lattice[i + m.match_length];
      const float score = lattice[i].score + (*piece_scores)[m.id];
      if (target_element.prev_position < 0 || target_element.score < score) {
        target_element = LatticeElement(score, m.id, i);
      }
    };
    piece_matcher.IteratePrefixMatches(
        utils::string_view(str.data() + i, length - i), lattice_update);
  }

  EncoderResult result;
  if (add_eos) {
    result.codes.push_back(config.end_code());
    result.offsets.push_back(length);
  }
  if (lattice[length].prev_position >= 0) {
    for (int pos = length; pos > 0;) {
      auto code = lattice[pos].code;
      if (code != config.unknown_code()) {
        code += config.encoding_offset();
      }
      result.codes.push_back(code);
      pos = lattice[pos].prev_position;
      result.offsets.push_back(offsets[pos]);
    }
  }
  if (add_bos) {
    result.codes.push_back(config.start_code());
    result.offsets.push_back(0);
  }
  if (!reverse) {
    std::reverse(result.codes.begin(), result.codes.end());
    std::reverse(result.offsets.begin(), result.offsets.end());
  }
  return result;
}

EncoderResult EncodeString(const std::string& string, const void* config_buffer,
                           bool add_bos, bool add_eos, bool reverse) {
  // Get the config from the buffer.
  const EncoderConfig* config = GetEncoderConfig(config_buffer);
  if (config->version() != EncoderVersion::EncoderVersion_SENTENCE_PIECE) {
    EncoderResult result;
    result.type = EncoderResultType::WRONG_CONFIG;
    return result;
  }
  std::string normalized_string;
  std::vector<int> offsets;
  std::tie(normalized_string, offsets) = NormalizeString(string, *config);
  return EncodeNormalizedString(normalized_string, offsets, *config, add_bos,
                                add_eos, reverse);
}

}  // namespace sentencepiece
}  // namespace text
}  // namespace tensorflow
