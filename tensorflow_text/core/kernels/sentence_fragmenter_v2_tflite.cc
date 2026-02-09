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

#include "tensorflow_text/core/kernels/sentence_fragmenter_v2_tflite.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/shim/tflite_op_shim.h"
#include "tensorflow_text/core/kernels/sentence_fragmenter_v2_kernel_template.h"

namespace tflite {
namespace ops {
namespace custom {
namespace text {

extern "C" void AddSentenceFragmenterV2(tflite::MutableOpResolver* resolver) {
  tflite::shim::TfLiteOpKernel<
      tensorflow::text::SentenceFragmenterV2Op>::Add(resolver);
}

}  // namespace text
}  // namespace custom
}  // namespace ops
}  // namespace tflite
