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

#include "tensorflow_text/core/kernels/sentencepiece/py_tflite_registerer.h"

namespace tflite {
namespace ops {
namespace custom {
TfLiteRegistration* Register_FAST_SENTENCEPIECE_TOKENIZER();
TfLiteRegistration* Register_FAST_SENTENCEPIECE_DETOKENIZER();

namespace text {

extern "C" void AddFastSentencepieceTokenize(
    tflite::MutableOpResolver* resolver) {
  resolver->AddCustom(
      "TFText>FastSentencepieceTokenize",
      ::tflite::ops::custom::Register_FAST_SENTENCEPIECE_TOKENIZER());
}

extern "C" void AddFastSentencepieceDetokenize(
    tflite::MutableOpResolver* resolver) {
  resolver->AddCustom(
      "TFText>FastSentencepieceDetokenize",
      ::tflite::ops::custom::Register_FAST_SENTENCEPIECE_DETOKENIZER());
}

}  // namespace text
}  // namespace custom
}  // namespace ops
}  // namespace tflite
