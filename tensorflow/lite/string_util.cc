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
#include <stdint.h>

#include <cstdlib>
#include <cstring>
#include <vector>

#include "tensorflow/lite/c/common.h"

namespace tflite {

void DynamicBuffer::AddString(const char* str, size_t len) {
  data_.resize(data_.size() + len);
  memcpy(data_.data() + offset_.back(), str, len);
  offset_.push_back(offset_.back() + len);
}

void DynamicBuffer::AddString(const StringRef& string) {
  AddString(string.str, string.len);
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
  // Resize the data buffer.
  int total_len = (strings.size() - 1) * separator.len;
  for (StringRef ref : strings) {
    total_len += ref.len;
  }
  data_.resize(data_.size() + total_len);

  char* dst = data_.data() + offset_.back();
  for (int i = 0; i < strings.size(); ++i) {
    // Fill separator if not first string.
    if (i != 0) {
      memcpy(dst, separator.str, separator.len);
      dst += separator.len;
    }

    // Fill content of the string.
    memcpy(dst, strings[i].str, strings[i].len);
    dst += strings[i].len;
  }
  offset_.push_back(offset_.back() + total_len);
}

int DynamicBuffer::WriteToBuffer(char** buffer) {
  // Allocate sufficient memory to tensor buffer.
  int32_t num_strings = offset_.size() - 1;
  // Total bytes include:
  //   * size of content (data_.size)
  //   * offset of each tensor (sizeof(int32_t) * num_strings)
  //   * length of whole buffer (int32_t)
  //   * num of strings (int32_t).
  int32_t bytes = data_.size()                            // size of content
                  + sizeof(int32_t) * (num_strings + 2);  // size of header

  // Caller will take ownership of buffer.
  *buffer = reinterpret_cast<char*>(malloc(bytes));

  // Set num of string
  memcpy(*buffer, &num_strings, sizeof(int32_t));

  // Set offset of strings.
  int32_t start = sizeof(int32_t) * (num_strings + 2);
  for (size_t i = 0; i < offset_.size(); i++) {
    int32_t offset = start + offset_[i];
    memcpy(*buffer + sizeof(int32_t) * (i + 1), &offset, sizeof(int32_t));
  }

  // Copy data of strings.
  memcpy(*buffer + start, data_.data(), data_.size());
  return bytes;
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

int GetStringCount(const void* raw_buffer) {
  // The first integers in the raw buffer is the number of strings.
  return *static_cast<const int32_t*>(raw_buffer);
}

int GetStringCount(const TfLiteTensor* tensor) {
  // The first integers in the raw buffer is the number of strings.
  return GetStringCount(tensor->data.raw);
}

StringRef GetString(const void* raw_buffer, int string_index) {
  const int32_t* offset =
      static_cast<const int32_t*>(raw_buffer) + (string_index + 1);
  return StringRef{
      static_cast<const char*>(raw_buffer) + (*offset),
      (*(offset + 1)) - (*offset),
  };
}

StringRef GetString(const TfLiteTensor* tensor, int string_index) {
  return GetString(tensor->data.raw, string_index);
}

}  // namespace tflite
