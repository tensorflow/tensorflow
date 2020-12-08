/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_TENSOR_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_TENSOR_H_

// Most functionality has been moved into a version of this file that doesn't
// rely on std::string, so that it can be used in TFL Micro.
#include "tensorflow/lite/kernels/internal/portable_tensor.h"
#include "tensorflow/lite/string_util.h"

namespace tflite {

template <>
class SequentialTensorWriter<string> {
 public:
  SequentialTensorWriter(const TfLiteTensor* input, TfLiteTensor* output)
      : input_(input), output_(output) {}
  ~SequentialTensorWriter() { buffer_.WriteToTensor(output_, nullptr); }

  void Write(int position) { this->WriteN(position, 1); }
  void WriteN(int position, int len) {
    for (int i = 0; i < len; i++) {
      buffer_.AddString(GetString(input_, position + i));
    }
  }

 private:
  const TfLiteTensor* input_;
  TfLiteTensor* output_;
  DynamicBuffer buffer_;
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_TENSOR_H_
