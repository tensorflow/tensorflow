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

#include "tensorflow_text/core/kernels/sentencepiece/optimized_decoder.h"

#include <string>
#include <tuple>

#include "tensorflow_text/core/kernels/sentencepiece/decoder_config_generated.h"
#include "tensorflow_text/core/kernels/sentencepiece/double_array_trie.h"

namespace tensorflow {
namespace text {
namespace sentencepiece {

DecoderResult DecodeString(const std::vector<int>& encoded,
                           const void* config_buffer) {
  DecoderResult result;

  // Get the config from the buffer.
  const DecoderConfig* config = GetDecoderConfig(config_buffer);
  if (config->version() != EncoderVersion::EncoderVersion_SENTENCE_PIECE) {
    result.type = DecoderResultType::WRONG_CONFIG;
    return result;
  }
  bool remove_dummy_prefix = config->remove_dummy_prefix();
  const auto config_pieces = config->decode_pieces();
  for (const auto code : encoded) {
    const int real_code = code - config->encoding_offset();
    if (real_code >= config_pieces->size()) {
      result.type = DecoderResultType::INVALID_INPUT;
      return result;
    }
    const auto& piece_text = config_pieces->GetAsString(real_code);
    const char* piece_str = piece_text->c_str();
    if (remove_dummy_prefix && *piece_str == ' ') {
      ++piece_str;
    }
    result.decoded.append(piece_str);
    remove_dummy_prefix = false;
  }
  // TODO(mgubin): Denormalize the string, haven't seen any Sentencepiece model
  // with a denormalizer.
  return result;
}

}  // namespace sentencepiece
}  // namespace text
}  // namespace tensorflow
