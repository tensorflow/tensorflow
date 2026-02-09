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

#include "tensorflow_text/core/kernels/fast_wordpiece_tokenizer_tflite.h"

#include "tensorflow/lite/kernels/shim/tflite_op_shim.h"
#include "tensorflow_text/core/kernels/fast_wordpiece_tokenizer_kernel_template.h"

namespace tflite {
namespace ops {
namespace custom {
namespace text {

using TokenizeOpKernel = tflite::shim::TfLiteOpKernel<
    tensorflow::text::FastWordpieceTokenizeWithOffsetsOp>;

using DetokenizeOpKernel =
    tflite::shim::TfLiteOpKernel<tensorflow::text::FastWordpieceDetokenizeOp>;

extern "C" void AddFastWordpieceTokenize(tflite::MutableOpResolver* resolver) {
  TokenizeOpKernel::Add(resolver);
}

extern "C" void AddFastWordpieceDetokenize(
    tflite::MutableOpResolver* resolver) {
  DetokenizeOpKernel::Add(resolver);
}

}  // namespace text
}  // namespace custom
}  // namespace ops
}  // namespace tflite
