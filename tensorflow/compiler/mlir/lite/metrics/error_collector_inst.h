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
#ifndef TENSORFLOW_COMPILER_MLIR_LITE_METRICS_ERROR_COLLECTOR_INST_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_METRICS_ERROR_COLLECTOR_INST_H_

#include <string>
#include <utility>

#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Pass/PassInstrumentation.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/metrics/error_collector.h"
#include "tensorflow/compiler/mlir/lite/metrics/types_util.h"
#include "tensorflow/lite/python/metrics/converter_error_data.pb.h"

namespace mlir {
namespace TFL {

// Collects errors when running the pass manager.
class ErrorCollectorInstrumentation : public PassInstrumentation {
  using ConverterErrorData = tflite::metrics::ConverterErrorData;
  using ErrorCode = ConverterErrorData::ErrorCode;

 public:
  explicit ErrorCollectorInstrumentation(MLIRContext *context);

 private:
  // Instrumentation hooks. These hooks don't need to be thread-safe. The pass
  // manager runs each pass for the entire module, then it walks through
  // each op in the module and runs the pass on them, may be in async mode.
  void runBeforePass(Pass *pass, Operation *module) override;
  void runAfterPass(Pass *pass, Operation *module) override;
  void runAfterPassFailed(Pass *pass, Operation *module) override;

  // The handler to capture error messages.
  std::unique_ptr<ScopedDiagnosticHandler> handler_;
  // A map from location to op name.
  std::unordered_map<Location, std::string, LocationHash> loc_to_name_;
  // Stores the error message for errors without op name and error code.
  std::string common_error_message_;
  // Name of the running pass.
  std::string pass_name_;
  // Pointer to the global ErrorCollector instance.
  ErrorCollector *error_collector_;
};

// Prefix when adding error code as a note in Diagnostic.
constexpr char kErrorCodePrefix[] = "Error code: ";

// Adds error code to a newly created InFlightDiagnostic.
inline InFlightDiagnostic AttachErrorCode(InFlightDiagnostic &&diag,
                                          int error_code) {
  using tflite::metrics::ConverterErrorData;
  diag.attachNote() << kErrorCodePrefix
                    << ConverterErrorData::ErrorCode_Name(error_code);
  return std::move(diag);
}

}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_METRICS_ERROR_COLLECTOR_INST_H_
