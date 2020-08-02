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
#ifndef TENSORFLOW_LITE_TOOLS_OPTIMIZE_CALIBRATION_BUILTIN_LOGGING_OPS_LSTM_H_
#define TENSORFLOW_LITE_TOOLS_OPTIMIZE_CALIBRATION_BUILTIN_LOGGING_OPS_LSTM_H_

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/tools/optimize/calibration/calibration_logger.h"

namespace tflite {
namespace optimize {
namespace calibration {
namespace builtin {

TfLiteStatus lstm_logging_kernel(TfLiteContext* context, TfLiteNode* node,
                                 Logger* logger, ErrorReporter* error_reporter);

}  // namespace builtin
}  // namespace calibration
}  // namespace optimize
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_OPTIMIZE_CALIBRATION_BUILTIN_LOGGING_OPS_LSTM_H_
