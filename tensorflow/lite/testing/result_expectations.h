/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_TESTING_RESULT_EXPECTATIONS_H_
#define TENSORFLOW_LITE_TESTING_RESULT_EXPECTATIONS_H_

#include <complex>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "absl/strings/escaping.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/testing/split.h"

namespace tflite {
namespace testing {

// Class for comparing the values of expectations against the values computed by
// the model.
class DataExpectation {
 public:
  //  Constructs a DataExpectation with the given relative threshold, absolute
  //  threshold, and quantization error multiplier.
  //
  //  The relative threshold is the maximum allowed difference between the
  //  expected value and the actual value, expressed as a percentage of the
  //  expected value. The absolute threshold is the maximum allowed difference
  //  between the expected value and the actual value, in absolute terms. The
  //  quantization error multiplier is the factor by which the expected value
  //  should be quantized.
  DataExpectation(double relative_threshold, double absolute_threshold,
                  int quantization_error_multiplier);

  //  Sets the data for the tensor. The data is expected to be in CSV format,
  //  with each value separated by a comma. The function will split the CSV
  //  values into a vector of values and then set the data for the tensor to the
  //  vector.
  template <typename T>
  void SetData(const std::string& csv_values) {
    const auto values = testing::Split<T>(csv_values, ",");
    num_elements_ = values.size();
    data_ = make_type_erased_array<T>(num_elements_);
    SetTensorData(values, data_.get());
  }

  //  Checks the data against the expectation.
  //
  //  Returns true if the data matches the expectation, false otherwise.
  bool Check(bool verbose, const TfLiteTensor& tensor);

 private:
  bool CompareTwoValuesHelper(float v1, float v2);

  bool CompareTwoValuesHelper(double v1, double v2);

  bool CompareTwoValues(std::complex<float> v1, std::complex<float> v2) {
    return CompareTwoValues(v1.real(), v2.real()) ||
           CompareTwoValues(v1.imag(), v2.imag());
  }

  bool CompareTwoValues(std::complex<double> v1, std::complex<double> v2) {
    return CompareTwoValues(v1.real(), v2.real()) ||
           CompareTwoValues(v1.imag(), v2.imag());
  }

  bool CompareTwoValues(float v1, float v2) {
    return CompareTwoValuesHelper(v1, v2);
  }

  bool CompareTwoValues(double v1, double v2) {
    return CompareTwoValuesHelper(v1, v2);
  }

  // Creates a type-erased array.
  template <typename T>
  std::unique_ptr<void, void (*)(void*)> make_type_erased_array(size_t size) {
    return std::unique_ptr<void, void (*)(void*)>(
        static_cast<void*>(new T[size]),
        [](void* data) { delete[] static_cast<T*>(data); });
  }

  template <typename T>
  void SetTensorData(const std::vector<T>& values, void* data) {
    T* input_ptr = static_cast<T*>(data);
    std::copy(values.begin(), values.end(), input_ptr);
  }

  template <typename T, typename TS>
  bool TypedCheck(bool verbose, const TfLiteTensor& tensor);

  bool TypedCheckString(bool verbose, const TfLiteTensor& tensor);
  bool QuantizedCheck(bool verbose, const TfLiteTensor& tensor);

  std::unique_ptr<void, void (*)(void*)> data_;
  size_t num_elements_;
  double relative_threshold_;
  double absolute_threshold_;
  int quantization_error_multiplier_;
};

// SetData specializations.
template <>
inline void DataExpectation::SetData<std::string>(
    const std::string& csv_values) {
  std::string s = absl::HexStringToBytes(csv_values);
  data_ = make_type_erased_array<char>(s.size());
  memcpy(data_.get(), s.data(), s.size());
}

// Class for comparing the expected shape against the shape of data computed by
// the model.
class ShapeExpectation {
 public:
  //  Constructs a ShapeExpectation with the given shape.
  //
  //  The shape is a vector of integers, where each integer represents the
  //  size of a dimension.
  explicit ShapeExpectation(const std::string& csv_values);

  //  Checks the shape of the data against the expectation.
  //
  //  Returns true if the shape of the data matches the expectation, false
  //  otherwise.
  bool CheckShape(bool verbose, const TfLiteTensor& tensor);

 private:
  std::vector<int32_t> shape_;
};

}  // namespace testing
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TESTING_RESULT_EXPECTATIONS_H_
