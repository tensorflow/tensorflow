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

#include "tensorflow/lite/experimental/micro/micro_mutable_op_resolver.h"

namespace tflite {

const TfLiteRegistration* MicroMutableOpResolver::FindOp(
    tflite::BuiltinOperator op, int version) const {
  for (int i = 0; i < registrations_len_; ++i) {
    const TfLiteRegistration& registration = registrations_[i];
    if ((registration.builtin_code == op) &&
        (registration.version == version)) {
      return &registration;
    }
  }
  return nullptr;
}

const TfLiteRegistration* MicroMutableOpResolver::FindOp(const char* op,
                                                         int version) const {
  for (int i = 0; i < registrations_len_; ++i) {
    const TfLiteRegistration& registration = registrations_[i];
    if ((registration.builtin_code == BuiltinOperator_CUSTOM) &&
        (strcmp(registration.custom_name, op) == 0) &&
        (registration.version == version)) {
      return &registration;
    }
  }
  return nullptr;
}

void MicroMutableOpResolver::AddBuiltin(tflite::BuiltinOperator op,
                                        TfLiteRegistration* registration,
                                        int min_version, int max_version) {
  for (int version = min_version; version <= max_version; ++version) {
    if (registrations_len_ >= TFLITE_REGISTRATIONS_MAX) {
      // TODO(petewarden) - Add error reporting hooks so we can report this!
      return;
    }
    TfLiteRegistration* new_registration = &registrations_[registrations_len_];
    registrations_len_ += 1;

    *new_registration = *registration;
    new_registration->builtin_code = op;
    new_registration->version = version;
  }
}

void MicroMutableOpResolver::AddCustom(const char* name,
                                       TfLiteRegistration* registration,
                                       int min_version, int max_version) {
  for (int version = min_version; version <= max_version; ++version) {
    if (registrations_len_ >= TFLITE_REGISTRATIONS_MAX) {
      // TODO(petewarden) - Add error reporting hooks so we can report this!
      return;
    }
    TfLiteRegistration* new_registration = &registrations_[registrations_len_];
    registrations_len_ += 1;

    *new_registration = *registration;
    new_registration->builtin_code = BuiltinOperator_CUSTOM;
    new_registration->custom_name = name;
    new_registration->version = version;
  }
}

}  // namespace tflite
