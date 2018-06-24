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

#include "tensorflow/contrib/lite/op_resolver.h"
#include "tensorflow/contrib/lite/context.h"

namespace tflite {

const TfLiteRegistration* MutableOpResolver::FindOp(tflite::BuiltinOperator op,
                                                    int version) const {
  auto it = builtins_.find(std::make_pair(op, version));
  return it != builtins_.end() ? &it->second : nullptr;
}

const TfLiteRegistration* MutableOpResolver::FindOp(const char* op,
                                                    int version) const {
  auto it = custom_ops_.find(std::make_pair(op, version));
  return it != custom_ops_.end() ? &it->second : nullptr;
}

void MutableOpResolver::AddBuiltin(tflite::BuiltinOperator op,
                                   TfLiteRegistration* registration,
                                   int min_version, int max_version) {
  for (int version = min_version; version <= max_version; ++version) {
    TfLiteRegistration new_registration = *registration;
    new_registration.builtin_code = op;
    new_registration.version = version;
    auto op_key = std::make_pair(op, version);
    builtins_[op_key] = new_registration;
  }
}

void MutableOpResolver::AddCustom(const char* name,
                                  TfLiteRegistration* registration,
                                  int min_version, int max_version) {
  for (int version = min_version; version <= max_version; ++version) {
    TfLiteRegistration new_registration = *registration;
    new_registration.builtin_code = BuiltinOperator_CUSTOM;
    new_registration.version = version;
    auto op_key = std::make_pair(name, version);
    custom_ops_[op_key] = new_registration;
  }
}

}  // namespace tflite
