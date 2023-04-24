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
#include "tensorflow/lite/testing/result_expectations.h"

#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/testing/split.h"

namespace tflite {
namespace testing {
namespace {
// Returns the value in the given position in a tensor.
template <typename T>
T Value(void* data, int index) {
  return static_cast<T*>(data)[index];
}

bool InterpretAsQuantized(const TfLiteTensor& tensor) {
  if (tensor.quantization.type == kTfLiteNoQuantization) return false;

  // Quantized single-op models with uint8 input/output type are only used for
  // EdgeTPU tests.
  // EdgeTPU tests need to read the quantized values as-is to check for
  // bit-exactness. As a result we don't interpret the tensor as quantized.
  // TODO(b/176121243): Add an option to interpret uint8 buffers as
  // non-quantized type and set if from the child class.
  if (tensor.type == kTfLiteUInt8) return false;

  if (tensor.quantization.params != nullptr) {
    auto* quantization =
        reinterpret_cast<TfLiteAffineQuantization*>(tensor.quantization.params);
    if (quantization->scale != nullptr && quantization->scale->size == 1 &&
        quantization->zero_point != nullptr &&
        quantization->zero_point->size == 1) {
      return true;
    }
  }
  return false;
}
}  // namespace

DataExpectation::DataExpectation(double relative_threshold,
                                 double absolute_threshold,
                                 int quantization_error_multiplier)
    : data_(nullptr, nullptr),
      num_elements_(0),
      relative_threshold_(relative_threshold),
      absolute_threshold_(absolute_threshold),
      quantization_error_multiplier_(quantization_error_multiplier) {}

bool DataExpectation::Check(bool verbose, const TfLiteTensor& tensor) {
  if (InterpretAsQuantized(tensor)) {
    return QuantizedCheck(verbose, tensor);
  }

  switch (tensor.type) {
    case kTfLiteFloat32:
      return TypedCheck<float, float>(verbose, tensor);
    case kTfLiteInt32:
      return TypedCheck<int32_t, float>(verbose, tensor);
    case kTfLiteUInt32:
      return TypedCheck<uint32_t, float>(verbose, tensor);
    case kTfLiteInt64:
      return TypedCheck<int64_t, float>(verbose, tensor);
    case kTfLiteUInt64:
      return TypedCheck<uint64_t, float>(verbose, tensor);
    case kTfLiteUInt8:
      return TypedCheck<uint8_t, float>(verbose, tensor);
    case kTfLiteInt8:
      return TypedCheck<int8_t, float>(verbose, tensor);
    case kTfLiteUInt16:
      return TypedCheck<uint16_t, float>(verbose, tensor);
    case kTfLiteInt16:
      return TypedCheck<int16_t, float>(verbose, tensor);
    case kTfLiteBool:
      return TypedCheck<bool, float>(verbose, tensor);
    case kTfLiteString:
      return TypedCheckString(verbose, tensor);
    case kTfLiteComplex64:
      return TypedCheck<std::complex<float>, std::complex<float>>(verbose,
                                                                  tensor);
    case kTfLiteComplex128:
      return TypedCheck<std::complex<double>, std::complex<double>>(verbose,
                                                                    tensor);
    case kTfLiteFloat64:
      return TypedCheck<double, double>(verbose, tensor);
    case kTfLiteFloat16:
      return TypedCheck<Eigen::half, float>(verbose, tensor);
    default:
      fprintf(stderr, "Unsupported type %d in Check\n", tensor.type);
      return false;
  }
}

bool DataExpectation::CompareTwoValuesHelper(float v1, float v2) {
  if (std::isnan(v1) || std::isnan(v2)) {
    return !(std::isnan(v1) && std::isnan(v2));
  }

  float diff = std::abs(v1 - v2);
  bool error_is_large = false;
  // For very small numbers, try absolute error, otherwise go with
  // relative.
  if (std::abs(v2) < relative_threshold_) {
    error_is_large = (diff > absolute_threshold_);
  } else {
    error_is_large = (diff > relative_threshold_ * std::abs(v2));
  }
  return error_is_large;
}

bool DataExpectation::CompareTwoValuesHelper(double v1, double v2) {
  if (std::isnan(v1) || std::isnan(v2)) {
    return !(std::isnan(v1) && std::isnan(v2));
  }

  double diff = std::abs(v1 - v2);
  bool error_is_large = false;
  // For very small numbers, try absolute error, otherwise go with
  // relative.
  if (std::abs(v2) < relative_threshold_) {
    error_is_large = (diff > absolute_threshold_);
  } else {
    error_is_large = (diff > relative_threshold_ * std::abs(v2));
  }
  return error_is_large;
}

template <typename T, typename TS>
bool DataExpectation::TypedCheck(bool verbose, const TfLiteTensor& tensor) {
  size_t tensor_size = tensor.bytes / sizeof(T);

  if (tensor_size != num_elements_) {
    std::cerr << "Expected a tensor with " << num_elements_ << " elements, got "
              << tensor_size << std::endl;
    std::cerr << "while checking tensor " << tensor.name << std::endl;
    return false;
  }

  bool good_output = true;
  for (int i = 0; i < tensor_size; ++i) {
    TS computed = Value<T>(tensor.data.raw, i);
    TS reference = Value<T>(data_.get(), i);
    if (CompareTwoValues(computed, reference)) {
      good_output = false;
      if (verbose) {
        std::cerr << "  Tensor[" << tensor.name << "] index " << i << ": got "
                  << computed << ", but expected " << reference << std::endl;
      }
    }
  }
  return good_output;
}

bool DataExpectation::TypedCheckString(bool verbose,
                                       const TfLiteTensor& tensor) {
  if (tensor.data.raw == nullptr) {
    if (verbose) {
      std::cerr << "  got empty string" << std::endl;
    }
    return false;
  }
  int expected_num_strings = GetStringCount(data_.get());
  int returned_num_strings = GetStringCount(&tensor);
  if (expected_num_strings != returned_num_strings) {
    if (verbose) {
      std::cerr << "  string count differ: got " << returned_num_strings
                << ", but expected " << expected_num_strings << std::endl;
    }
    return false;
  }
  for (int i = 0; i < returned_num_strings; ++i) {
    auto expected_ref = GetString(data_.get(), i);
    auto returned_ref = GetString(&tensor, i);
    if (expected_ref.len != returned_ref.len) {
      if (verbose) {
        std::cerr << "  index " << i << ": got string of size "
                  << returned_ref.len << ", but expected size "
                  << expected_ref.len << std::endl;
      }
      return false;
    }
    if (strncmp(expected_ref.str, returned_ref.str, returned_ref.len) != 0) {
      if (verbose) {
        std::cerr << "  index " << i << ": strings are different" << std::endl;
      }
      return false;
    }
  }

  return true;
}

bool DataExpectation::QuantizedCheck(bool verbose, const TfLiteTensor& tensor) {
  auto* quantization =
      reinterpret_cast<TfLiteAffineQuantization*>(tensor.quantization.params);
  const float scale = quantization->scale->data[0];
  const int32_t zero_point = quantization->zero_point->data[0];

  bool good_result = true;
  int int_size = tensor.type == kTfLiteInt8 ? 1 : 2;
  for (int i = 0; i < tensor.bytes / int_size; i++) {
    int32_t computed =
        tensor.type == kTfLiteInt8 ? tensor.data.int8[i] : tensor.data.i16[i];
    const float dequantized =
        static_cast<float>(scale * (computed - zero_point));
    int error_multiplier = quantization_error_multiplier_;
    // If we are doing int16 symmetric quantization of activations, we need to
    // bump up the potential error. Since the weights are quantized to 8 bits
    // and the activations are 16bits, the output is could be getting
    // effectively 8bit error instead of 16bit error. So we need to multiply the
    // error mulitplier by 255 (the difference in number of values between a
    // 16bit and 8bit number)
    if (tensor.type == kTfLiteInt16) error_multiplier *= 255;
    const float reference = Value<float>(data_.get(), i);
    if (std::abs(dequantized - reference) > error_multiplier * scale) {
      if (verbose) {
        std::cerr << "  index " << i << ": got " << dequantized
                  << ", but expected " << reference << std::endl;
      }
      good_result = false;
    }
  }
  return good_result;
}

ShapeExpectation::ShapeExpectation(const std::string& csv_values)
    : shape_(testing::Split<int32_t>(csv_values, ",")) {}

bool ShapeExpectation::CheckShape(bool verbose, const TfLiteTensor& tensor) {
  bool valid = true;
  if (tensor.dims->size == shape_.size()) {
    for (int i = 0; i < shape_.size(); ++i) {
      if (shape_[i] != tensor.dims->data[i]) {
        valid = false;
      }
    }
  } else {
    valid = false;
  }
  if (!valid && verbose) {
    std::cerr << "Incorrect output shape while checking tensor " << tensor.name
              << std::endl;
    std::cerr << "TFLite output shape: ";
    for (int i = 0; i < tensor.dims->size; ++i) {
      std::cerr << tensor.dims->data[i] << ", ";
    }
    std::cerr << std::endl;
    std::cerr << "Expected output shape: ";
    for (int i = 0; i < shape_.size(); ++i) {
      std::cerr << shape_[i] << ", ";
    }
    std::cerr << std::endl;
  }
  return valid;
}

}  // namespace testing
}  // namespace tflite
