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

#ifndef TENSORFLOW_CORE_KERNELS_TEXT_WHITESPACE_TOKENIZER_KERNEL_H_
#define TENSORFLOW_CORE_KERNELS_TEXT_WHITESPACE_TOKENIZER_KERNEL_H_

#include "tensorflow/core/kernels/text/whitespace_tokenizer_kernel_template.h"
#include "tensorflow/lite/kernels/shim/tf_op_shim.h"

namespace tensorflow {
namespace text {

class WhitespaceTokenizeWithOffsetsV2OpKernel
    : public tflite::shim::TfOpKernel<WhitespaceTokenizeWithOffsetsV2Op> {
 public:
  using TfOpKernel::TfOpKernel;
};

}  // namespace text
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_TEXT_WHITESPACE_TOKENIZER_KERNEL_H_
