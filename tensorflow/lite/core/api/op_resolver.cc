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

#include "tensorflow/lite/core/api/op_resolver.h"

namespace tflite {

TfLiteStatus GetRegistrationFromOpCode(
    const OperatorCode* opcode, const OpResolver& op_resolver,
    ErrorReporter* error_reporter, const TfLiteRegistration** registration) {
  TfLiteStatus status = kTfLiteOk;
  *registration = nullptr;
  auto builtin_code = opcode->builtin_code();
  int version = opcode->version();

  if (builtin_code > BuiltinOperator_MAX ||
      builtin_code < BuiltinOperator_MIN) {
    error_reporter->Report(
        "Op builtin_code out of range: %d. Are you using old TFLite binary "
        "with newer model?",
        builtin_code);
    status = kTfLiteError;
  } else if (builtin_code != BuiltinOperator_CUSTOM) {
    *registration = op_resolver.FindOp(builtin_code, version);
    if (*registration == nullptr) {
      error_reporter->Report(
          "Didn't find op for builtin opcode '%s' version '%d'\n",
          EnumNameBuiltinOperator(builtin_code), version);
      status = kTfLiteError;
    }
  } else if (!opcode->custom_code()) {
    error_reporter->Report(
        "Operator with CUSTOM builtin_code has no custom_code.\n");
    status = kTfLiteError;
  } else {
    const char* name = opcode->custom_code()->c_str();
    *registration = op_resolver.FindOp(name, version);
    if (*registration == nullptr) {
      error_reporter->Report(
          "Didn't find custom op for name '%s' with version %d\n", name,
          version);
      status = kTfLiteError;
    }
  }
  return status;
}

}  // namespace tflite
