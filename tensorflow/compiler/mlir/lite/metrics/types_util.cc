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
#include "tensorflow/compiler/mlir/lite/metrics/types_util.h"

namespace mlir {
namespace TFL {

tflite::metrics::ConverterErrorData NewConverterErrorData(
    const std ::string& pass_name, const std::string& error_message,
    tflite::metrics::ConverterErrorData::ErrorCode error_code,
    const std::string& op_name) {
  using tflite::metrics::ConverterErrorData;
  ConverterErrorData error;
  if (!pass_name.empty()) {
    error.set_subcomponent(pass_name);
  }

  if (!error_message.empty()) {
    error.set_error_message(error_message);
  }

  if (!op_name.empty()) {
    error.mutable_operator_()->set_name(op_name);
  }

  error.set_error_code(error_code);
  return error;
}

}  // namespace TFL
}  // namespace mlir
