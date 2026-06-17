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
#include "tensorflow/lite/mutable_op_resolver_utils.h"

#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {

void AddOp(MutableOpResolver* mutable_op_resolver, const TfLiteOperator* op,
           int min_version, int max_version) {
  TfLiteRegistration registration{};
  registration.builtin_code = TfLiteOperatorGetBuiltInCode(op);
  registration.custom_name = TfLiteOperatorGetCustomName(op);
  registration.version = TfLiteOperatorGetVersion(op);
  // This const cast is safe because TfLiteOperator is an opaque
  // type and TfLiteOperator objects are always allocated with
  // TfLiteOperatorCreate() which allocates non-const objects.
  registration.registration_external = const_cast<TfLiteOperator*>(op);
  if (registration.custom_name != nullptr) {
    mutable_op_resolver->AddCustom(registration.custom_name, &registration,
                                   min_version, max_version);
  } else {
    mutable_op_resolver->AddBuiltin(BuiltinOperator(registration.builtin_code),
                                    &registration, min_version, max_version);
  }
}

void AddOp(MutableOpResolver* mutable_op_resolver, const TfLiteOperator* op) {
  int version = TfLiteOperatorGetVersion(op);
  AddOp(mutable_op_resolver, op, version, version);
}

}  // namespace tflite
