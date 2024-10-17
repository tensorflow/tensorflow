/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/c/c_api_internal.h"

#include "tensorflow/lite/core/c/common.h"

namespace tflite {
namespace internal {

void InitTfLiteRegistration(TfLiteRegistration* registration,
                            TfLiteOperator* registration_external) {
  registration->builtin_code = registration_external->builtin_code;
  registration->custom_name = registration_external->custom_name;
  registration->version = registration_external->version;
  registration->registration_external = registration_external;
}

TfLiteRegistration* OperatorToRegistration(
    const TfLiteOperator* registration_external) {
  // All TfLiteOperator objects are dynamically allocated via
  // TfLiteOperatorCreate(), so they are guaranteed
  // to be mutable, hence the const_cast below should be safe.
  auto registration_external_non_const =
      const_cast<TfLiteOperator*>(registration_external);
  TfLiteRegistration* new_registration = new TfLiteRegistration{};
  InitTfLiteRegistration(new_registration, registration_external_non_const);
  return new_registration;
}

}  // namespace internal
}  // namespace tflite
