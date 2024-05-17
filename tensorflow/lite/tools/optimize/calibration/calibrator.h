/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_TOOLS_OPTIMIZE_CALIBRATION_CALIBRATOR_H_
#define TENSORFLOW_LITE_TOOLS_OPTIMIZE_CALIBRATION_CALIBRATOR_H_

#include <memory>

#include "tensorflow/lite/allocation.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/core/model.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/tools/optimize/calibration/calibration_reader.h"

namespace tflite {
namespace optimize {
namespace calibration {

// Warning: This is not a public API and subject to change.

// Builds a interpreter that logs the calibration data in memory.
// The calibration data can be recovered using |calibration_reader|.
//
// Sample usage:
// std::unique_ptr<Interpreter> interpreter;
// std::unique_ptr<CalibrationReader> calibration_reader;
// BuiltinOpResolver resolver = ...
// FlatBufferModel model = ..
//
// BuildLoggingInterpreter(model, resolver, &interpreter,
//  &calibration_reader);
//
//
// * Allocate tensors...
// * Call interpreter->invoke on calibration dataset.
//
// Calibration data can be read either directly by calling
// std::unordered_map<int,  CalibrationStats>> tensor_index_to_stats;
// calibration_reader->GetTensorStatsAsMap(&tensor_index_to_stats);
//
// or adding calibration data to model itself.
// ModelT * original_floating_point_model = ...
// calibration_reader->AddCalibrationToModel(original_floating_point_model,
// false);
//
TfLiteStatus BuildLoggingInterpreter(
    const FlatBufferModel& model, const OpResolver& op_resolver,
    std::unique_ptr<Interpreter>* interpreter,
    std::unique_ptr<CalibrationReader>* calibration_reader);

// Same as above, except gets separate tflite::Model and ErrorReporter pointers.
TfLiteStatus BuildLoggingInterpreter(
    const tflite::Model* model, ErrorReporter* error_reporter,
    const OpResolver& op_resolver, std::unique_ptr<Interpreter>* interpreter,
    std::unique_ptr<CalibrationReader>* calibration_reader,
    const Allocation* allocation = nullptr);

}  // namespace calibration
}  // namespace optimize
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_OPTIMIZE_CALIBRATION_CALIBRATOR_H_
