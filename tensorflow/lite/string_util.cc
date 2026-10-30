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

#include "tensorflow/lite/string_util.h"

#include <stddef.h>

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "tensorflow/compiler/mlir/lite/utils/string_utils.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"

namespace tflite {


TfLiteStatus DynamicBuffer::AddString(const StringRef& string) {
  return AddString(string.str, string.len);
}

TfLiteStatus DynamicBuffer::AddString(const char* str, size_t len) {
  if (SimpleDynamicBuffer::AddString(str, len)) {
    return kTfLiteOk;
  }
  return kTfLiteError;
}

void DynamicBuffer::AddJoinedString(const std::vector<StringRef>& strings,
                                    char separator) {
  StringRef ref;
  ref.str = &separator;
  ref.len = 1;
  AddJoinedString(strings, ref);
}

void DynamicBuffer::AddJoinedString(const std::vector<StringRef>& strings,
                                    StringRef separator) {
  if (strings.empty()) {
    AddString("", 0);
    return;
  }
  const size_t num_separators = strings.size() - 1;
  if (separator.len > 0 && num_separators > max_length_ / separator.len) {
    return;
  }
  size_t total_len = num_separators * separator.len;
  for (StringRef ref : strings) {
    if (ref.len > max_length_ - total_len) {
      return;
    }
    total_len += ref.len;
  }
  if (total_len > max_length_ || data_.size() > max_length_ - total_len) {
    return;
  }
  if (total_len == 0) {
    offset_.push_back(offset_.back());
    return;
  }
  data_.resize(data_.size() + total_len);

  char* dst = data_.data() + offset_.back();
  for (size_t i = 0; i < strings.size(); ++i) {
    // Fill separator if not first string.
    if (i != 0 && separator.len > 0) {
      memcpy(dst, separator.str, separator.len);
      dst += separator.len;
    }

    // Fill content of the string.
    if (strings[i].len > 0) {
      memcpy(dst, strings[i].str, strings[i].len);
      dst += strings[i].len;
    }
  }
  offset_.push_back(offset_.back() + total_len);
}

#ifndef TF_LITE_STATIC_MEMORY
void DynamicBuffer::WriteToTensorAsVector(TfLiteTensor* tensor) {
  auto dims = TfLiteIntArrayCreate(1);
  dims->data[0] = offset_.size() - 1;  // Store number of strings.
  WriteToTensor(tensor, dims);
}

void DynamicBuffer::WriteToTensor(TfLiteTensor* tensor,
                                  TfLiteIntArray* new_shape) {
  char* tensor_buffer;
  int bytes = WriteToBuffer(&tensor_buffer);

  if (new_shape == nullptr) {
    new_shape = TfLiteIntArrayCopy(tensor->dims);
  }

  // Set tensor content pointer to tensor_buffer, and release original data.
  TfLiteTensorReset(tensor->type, tensor->name, new_shape, tensor->params,
                    tensor_buffer, bytes, kTfLiteDynamic, tensor->allocation,
                    tensor->is_variable, tensor);
}
#endif  // TF_LITE_STATIC_MEMORY

}  // namespace tflite
