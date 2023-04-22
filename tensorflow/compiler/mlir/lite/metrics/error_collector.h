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
#ifndef TENSORFLOW_COMPILER_MLIR_LITE_METRICS_ERROR_COLLECTOR_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_METRICS_ERROR_COLLECTOR_H_

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "tensorflow/compiler/mlir/lite/metrics/types_util.h"
#include "tensorflow/lite/python/metrics_wrapper/converter_error_data.pb.h"

namespace mlir {
namespace TFL {

// A singleton to store errors collected by the instrumentation.
class ErrorCollector {
  using ConverterErrorData = tflite::metrics::ConverterErrorData;
  using ConverterErrorDataSet =
      std::unordered_set<ConverterErrorData, ConverterErrorDataHash,
                         ConverterErrorDataComparision>;

 public:
  const ConverterErrorDataSet &CollectedErrors() { return collected_errors_; }

  void ReportError(const ConverterErrorData &error) {
    collected_errors_.insert(error);
  }

  // Clear the set of collected errors.
  void Clear() { collected_errors_.clear(); }

  // Returns the global instance of ErrorCollector.
  static ErrorCollector* GetErrorCollector();

 private:
  ErrorCollector() {}

  ConverterErrorDataSet collected_errors_;

  static ErrorCollector* error_collector_instance_;
};

}  // namespace TFL
}  // namespace mlir
#endif  // TENSORFLOW_COMPILER_MLIR_LITE_METRICS_ERROR_COLLECTOR_H_
