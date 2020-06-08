/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_RESOURCE_LOOKUP_UTIL_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_RESOURCE_LOOKUP_UTIL_H_

#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/string_util.h"

namespace tflite {
namespace resource {
namespace internal {

/// Helper class for accessing TFLite tensor data.
template <typename T>
class TensorReader {
 public:
  explicit TensorReader(const TfLiteTensor* input) {
    input_data_ = GetTensorData<T>(input);
  }

  // Returns the corresponding scalar data at the given index position.
  // In here, it does not check the validity of the index should be guaranteed
  // in order not to harm the performance. Caller should take care of it.
  T GetData(int index) { return input_data_[index]; }

 private:
  const T* input_data_;
};

/// Helper class for accessing TFLite tensor data. This specialized class is for
/// std::string type.
template <>
class TensorReader<std::string> {
 public:
  explicit TensorReader(const TfLiteTensor* input) : input_(input) {}

  // Returns the corresponding string data at the given index position.
  // In here, it does not check the validity of the index should be guaranteed
  // in order not to harm the performance. Caller should take care of it.
  std::string GetData(int index) {
    auto string_ref = GetString(input_, index);
    return std::string(string_ref.str, string_ref.len);
  }

 private:
  const TfLiteTensor* input_;
};

/// WARNING: Experimental interface, subject to change.
/// Helper class for writing TFLite tensor data.
template <typename ValueType>
class TensorWriter {
 public:
  explicit TensorWriter(TfLiteTensor* values) {
    output_data_ = GetTensorData<ValueType>(values);
  }

  // Sets the given value to the given index position of the tensor storage.
  // In here, it does not check the validity of the index should be guaranteed
  // in order not to harm the performance. Caller should take care of it.
  void SetData(int index, ValueType& value) { output_data_[index] = value; }

  // Commit updates. In this case, it does nothing since the SetData method
  // writes data directly.
  void Commit() {
    // Noop.
  }

 private:
  ValueType* output_data_;
};

/// WARNING: Experimental interface, subject to change.
/// Helper class for writing TFLite tensor data. This specialized class is for
/// std::string type.
template <>
class TensorWriter<std::string> {
 public:
  explicit TensorWriter(TfLiteTensor* values) : values_(values) {}

  // Queues the given string value to the buffer regardless of the provided
  // index.
  // In here, it does not check the validity of the index should be guaranteed
  // in order not to harm the performance. Caller should take care of it.
  void SetData(int index, const std::string& value) {
    buf_.AddString(value.data(), value.length());
  }

  // Commit updates. The stored data in DynamicBuffer will be written into the
  // tensor storage.
  void Commit() { buf_.WriteToTensor(values_, nullptr); }

 private:
  TfLiteTensor* values_;
  DynamicBuffer buf_;
};

}  // namespace internal
}  // namespace resource
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RESOURCE_LOOKUP_UTIL_H_
