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
#include "tensorflow/compiler/mlir/lite/schema/schema_utils.h"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>

#include "tensorflow/compiler/mlir/lite/kernels/internal/compatibility_macros.h"
#include "tensorflow/compiler/mlir/lite/schema/schema_generated.h"

namespace tflite {

// The following GetBuiltinCode methods are the utility methods for reading
// builtin operator code, ensuring compatibility issues between v3 and v3a
// schema. Always the maximum value of the two fields always will be the correct
// value as follows:
//
// - Supporting schema version v3 models
//
// The `builtin_code` field is not available in the v3 models. Flatbuffer
// library will feed zero value, which is the default value in the v3a schema.
// The actual builtin operator code value will exist in the
// `deprecated_builtin_code` field. At the same time, it implies that
// `deprecated_builtin_code` >= `builtin_code` and the maximum value of the two
// fields will be same with `deprecated_builtin_code'.
//
// - Supporting builtin operator codes beyonds 127
//
// New builtin operators, whose operator code is larger than 127, can not be
// assigned to the `deprecated_builtin_code` field. In such cases, the
// value of the `builtin_code` field should be used for the builtin operator
// code. In the case, the maximum value of the two fields will be the value of
// the `builtin_code` as the right value.

BuiltinOperator GetBuiltinCode(const OperatorCode* op_code) {
  // Caller should guarantee that the given argument value is not a nullptr.
  TFLITE_DCHECK(op_code != nullptr);

  return std::max(
      op_code->builtin_code(),
      static_cast<BuiltinOperator>(op_code->deprecated_builtin_code()));
}

BuiltinOperator GetBuiltinCode(const OperatorCodeT* op_code) {
  // Caller should guarantee that the given argument value is not a nullptr.
  TFLITE_DCHECK(op_code != nullptr);

  return std::max(op_code->builtin_code, static_cast<BuiltinOperator>(
                                             op_code->deprecated_builtin_code));
}

size_t TensorTypeGetSize(::tflite::TensorType data_type) {
  switch (data_type) {
    case ::tflite::TensorType_FLOAT32:
      static_assert(sizeof(float) == 4, "");
      return 4;
    case ::tflite::TensorType_FLOAT16:
      static_assert(sizeof(int16_t) == 2, "");
      return 2;
    case ::tflite::TensorType_INT32:
      static_assert(sizeof(int32_t) == 4, "");
      return 4;
    case ::tflite::TensorType_UINT8:
      static_assert(sizeof(uint8_t) == 1, "");
      return 1;
    case ::tflite::TensorType_INT64:
      static_assert(sizeof(int64_t) == 8, "");
      return 8;
    case ::tflite::TensorType_BOOL:
      return sizeof(bool);
    case ::tflite::TensorType_INT16:
      static_assert(sizeof(int16_t) == 2, "");
      return 2;
    case ::tflite::TensorType_COMPLEX64:
      static_assert(sizeof(std::complex<float>) == 8, "");
      return 8;
    case ::tflite::TensorType_INT8:
      static_assert(sizeof(int8_t) == 1, "");
      return 1;
    case ::tflite::TensorType_FLOAT64:
      static_assert(sizeof(double) == 8, "");
      return 8;
    case ::tflite::TensorType_COMPLEX128:
      static_assert(sizeof(std::complex<double>) == 16, "");
      return 16;
    case ::tflite::TensorType_UINT64:
      static_assert(sizeof(uint64_t) == 8, "");
      return 8;
    case ::tflite::TensorType_UINT32:
      static_assert(sizeof(uint32_t) == 4, "");
      return 4;
    case ::tflite::TensorType_UINT16:
      static_assert(sizeof(uint16_t) == 2, "");
      return 2;
    default:
      return 0;
  }
}
}  // namespace tflite
