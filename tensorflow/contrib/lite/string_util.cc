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

#include "tensorflow/contrib/lite/string_util.h"

#include <string.h>
#include <vector>
#include "tensorflow/contrib/lite/context.h"
#include "tensorflow/contrib/lite/interpreter.h"

namespace tflite {
namespace {

// Convenient method to get pointer to int32_t.
int32_t* GetIntPtr(char* ptr) { return reinterpret_cast<int32_t*>(ptr); }
}  // namespace

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
  // Resize the data buffer.
  int total_len = strings.size() - 1;
  for (StringRef ref : strings) {
    total_len += ref.len;
  }
  data_.resize(data_.size() + total_len);

  int current_idx = 0;
  for (StringRef ref : strings) {
    char* dst = data_.data() + offset_.back() + current_idx;

    // Fill separator if not first string.
    if (current_idx != 0) {
      *dst = separator;
      ++dst;
      ++current_idx;
    }

    // Fill content of the string.
    memcpy(dst, ref.str, ref.len);
    current_idx += ref.len;
  }
  offset_.push_back(offset_.back() + total_len);
}

void DynamicBuffer::WriteToTensor(TfLiteTensor* tensor) {
  // Allocate sufficient memory to tensor buffer.
  int32_t num_strings = offset_.size() - 1;
  // Total bytes include:
  //   * size of content (data_.size)
  //   * offset of each tensor (sizeof(int32_t) * num_strings)
  //   * length of whole buffer (int32_t)
  //   * num of strings (int32_t).
  int32_t bytes = data_.size()                            // size of content
                  + sizeof(int32_t) * (num_strings + 2);  // size of header

  // Output tensor will take over the ownership of tensor_buffer, and free it
  // during Interpreter destruction.
  char* tensor_buffer = static_cast<char*>(malloc(bytes));

  // Set num of string
  memcpy(tensor_buffer, &num_strings, sizeof(int32_t));

  // Set offset of strings.
  int32_t start = sizeof(int32_t) * (num_strings + 2);
  for (int i = 0; i < offset_.size(); i++) {
    int32_t offset = start + offset_[i];
    memcpy(tensor_buffer + sizeof(int32_t) * (i + 1), &offset, sizeof(int32_t));
  }

  // Copy data of strings.
  memcpy(tensor_buffer + start, data_.data(), data_.size());

  // Set tensor content pointer to tensor_buffer, and release original data.
  auto dims = TfLiteIntArrayCreate(1);
  dims->data[0] = num_strings;
  TfLiteTensorReset(tensor->type, tensor->name, dims, tensor->params,
                    tensor_buffer, bytes, kTfLiteDynamic, tensor->allocation,
                    tensor);
}

int GetStringCount(const TfLiteTensor* tensor) {
  // The first integers in the raw buffer is the number of strings.
  return *GetIntPtr(tensor->data.raw);
}

StringRef GetString(const TfLiteTensor* tensor, int string_index) {
  int32_t* offset =
      GetIntPtr(tensor->data.raw + sizeof(int32_t) * (string_index + 1));
  return {
      tensor->data.raw + (*offset),
      (*(offset + 1)) - (*offset),
  };
}

}  // namespace tflite
