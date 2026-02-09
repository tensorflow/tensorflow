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

#ifndef TENSORFLOW_LITE_SUPPORT_CUSTOM_OPS_KERNEL_SENTENCEPIECE_OPTIMIZED_ENCODER_H_
#define TENSORFLOW_LITE_SUPPORT_CUSTOM_OPS_KERNEL_SENTENCEPIECE_OPTIMIZED_ENCODER_H_

// Sentencepiece encoder optimized with memmapped model.

#include <string>
#include <tuple>
#include <vector>

#include "tensorflow_text/core/kernels/sentencepiece/encoder_config_generated.h"

namespace tensorflow {
namespace text {
namespace sentencepiece {

enum class EncoderResultType { SUCCESS = 0, WRONG_CONFIG = 1 };

struct EncoderResult {
  EncoderResultType type = EncoderResultType::SUCCESS;
  std::vector<int> codes;
  std::vector<int> offsets;
};
std::tuple<std::string, std::vector<int>> NormalizeString(
    const std::string& in_string, const EncoderConfig& config);

// Encodes one string and returns ids and offsets. Takes the configuration as a
// type-erased buffer.
EncoderResult EncodeString(const std::string& string, const void* config_buffer,
                           bool add_bos, bool add_eos, bool reverse);

}  // namespace sentencepiece
}  // namespace text
}  // namespace tensorflow

#endif  // TENSORFLOW_LITE_SUPPORT_CUSTOM_OPS_KERNEL_SENTENCEPIECE_OPTIMIZED_ENCODER_H_
