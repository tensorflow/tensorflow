/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_METRICS_UTIL_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_METRICS_UTIL_H_

#include <functional>

#include "mlir/IR/Location.h"  // from @llvm-project
#include "tensorflow/lite/python/metrics/converter_error_data.pb.h"

namespace mlir {
namespace TFL {

// The hash function for mlir::Location.
struct LocationHash {
  std::size_t operator()(const Location& v) const noexcept {
    return hash_value(v);
  }
};

// The hash function for ConverterErrorData.
struct ConverterErrorDataHash {
  std::size_t operator()(
      const tflite::metrics::ConverterErrorData& v) const noexcept {
    std::size_t hash_result = std::hash<std::string>{}(v.error_message());
    if (v.has_subcomponent()) {
      hash_result ^= std::hash<std::string>{}(v.subcomponent()) << 1;
    }
    if (v.has_error_code()) {
      hash_result ^= std::hash<int>{}(v.error_code()) << 2;
    }
    if (v.has_operator_() && v.operator_().has_name()) {
      hash_result ^= std::hash<std::string>{}(v.operator_().name()) << 3;
    }
    return hash_result;
  }
};

// The comparison function for ConverterErrorData.
struct ConverterErrorDataComparison {
  std::size_t operator()(
      const tflite::metrics::ConverterErrorData& a,
      const tflite::metrics::ConverterErrorData& b) const noexcept {
    return ConverterErrorDataHash()(a) == ConverterErrorDataHash()(b);
  }
};

// Helper function to create a new ConverterErrorData.
tflite::metrics::ConverterErrorData NewConverterErrorData(
    const std ::string& pass_name, const std::string& error_message,
    tflite::metrics::ConverterErrorData::ErrorCode error_code,
    const std::string& op_name, const Location& location);

}  // namespace TFL
}  // namespace mlir
#endif  // TENSORFLOW_COMPILER_MLIR_LITE_METRICS_UTIL_H_
