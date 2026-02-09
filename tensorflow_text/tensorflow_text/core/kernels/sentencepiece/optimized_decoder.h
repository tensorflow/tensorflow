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

#ifndef TENSORFLOW_LITE_SUPPORT_CUSTOM_OPS_KERNEL_SENTENCEPIECE_OPTIMIZED_DECODER_H_
#define TENSORFLOW_LITE_SUPPORT_CUSTOM_OPS_KERNEL_SENTENCEPIECE_OPTIMIZED_DECODER_H_

// Sentencepiece decoder optimized with memmapped model.

#include <string>
#include <vector>

namespace tensorflow {
namespace text {
namespace sentencepiece {

enum class DecoderResultType {
  SUCCESS = 0,
  WRONG_CONFIG = 1,
  INVALID_INPUT = 2
};

struct DecoderResult {
  DecoderResultType type = DecoderResultType::SUCCESS;
  std::string decoded;
};

// Decodes one string from a vector of id. Takes the configuration as a
// type-erased  buffer.
DecoderResult DecodeString(const std::vector<int>& encoded,
                           const void* config_buffer);

}  // namespace sentencepiece
}  // namespace text
}  // namespace tensorflow

#endif  // TENSORFLOW_LITE_SUPPORT_CUSTOM_OPS_KERNEL_SENTENCEPIECE_OPTIMIZED_DECODER_H_
