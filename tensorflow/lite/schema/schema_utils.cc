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
#include "tensorflow/lite/schema/schema_utils.h"

#include "tensorflow/lite/kernels/internal/compatibility.h"

namespace tflite {

// The following GetBuiltinCode methods are the utility methods for reading
// builtin operatore code. Later, theses method will be used for upcoming
// builtin code compatibility changes.

BuiltinOperator GetBuiltinCode(const OperatorCode *op_code) {
  // Caller should guarantee that the given argument value is not a nullptr.
  TFLITE_DCHECK(op_code != nullptr);

  return op_code->builtin_code();
}

BuiltinOperator GetBuiltinCode(const OperatorCodeT *op_code) {
  // Caller should guarantee that the given argument value is not a nullptr.
  TFLITE_DCHECK(op_code != nullptr);

  return op_code->builtin_code;
}

}  // namespace tflite
