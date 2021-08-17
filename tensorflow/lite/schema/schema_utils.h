/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_SCHEMA_SCHEMA_UTILS_H_
#define TENSORFLOW_LITE_SCHEMA_SCHEMA_UTILS_H_

#include "flatbuffers/flatbuffers.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {

// The following methods are introduced to resolve op builtin code shortage
// problem. The new builtin operator will be assigned to the extended builtin
// code field in the flatbuffer schema. Those methods helps to hide builtin code
// details.
BuiltinOperator GetBuiltinCode(const OperatorCode *op_code);

BuiltinOperator GetBuiltinCode(const OperatorCodeT *op_code);

}  // namespace tflite

#endif  // TENSORFLOW_LITE_SCHEMA_SCHEMA_UTILS_H_
