/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/contrib/lite/tools/mutable_op_resolver.h"

namespace tflite {

TfLiteRegistration* MutableOpResolver::FindOp(
    tflite::BuiltinOperator op) const {
  auto it = builtins_.find(op);
  return it != builtins_.end() ? it->second : nullptr;
}

TfLiteRegistration* MutableOpResolver::FindOp(const char* op) const {
  auto it = custom_ops_.find(op);
  return it != custom_ops_.end() ? it->second : nullptr;
}

void MutableOpResolver::AddBuiltin(tflite::BuiltinOperator op,
                                   TfLiteRegistration* registration) {
  registration->builtin_code = op;
  builtins_.insert(std::make_pair(op, registration));
}

void MutableOpResolver::AddCustom(const char* name,
                                  TfLiteRegistration* registration) {
  registration->builtin_code = BuiltinOperator_CUSTOM;
  custom_ops_.insert(std::make_pair(std::string(name), registration));
}

}  // namespace tflite
