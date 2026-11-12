/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/utils/tf_convert_type.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/mlir/lite/schema/schema_generated.h"
#include "tensorflow/core/framework/types.pb.h"

namespace tflite {

tensorflow::DataType TflTypeToTfType(tflite::TensorType type) {
  switch (type) {
    case tflite::TensorType_BOOL:
      return tensorflow::DT_BOOL;
    case tflite::TensorType_COMPLEX64:
      return tensorflow::DT_COMPLEX64;
    case tflite::TensorType_COMPLEX128:
      return tensorflow::DT_COMPLEX128;
    case tflite::TensorType_FLOAT16:
      return tensorflow::DT_HALF;
    case tflite::TensorType_BFLOAT16:
      return tensorflow::DT_BFLOAT16;
    case tflite::TensorType_FLOAT32:
      return tensorflow::DT_FLOAT;
    case tflite::TensorType_FLOAT64:
      return tensorflow::DT_DOUBLE;
    case tflite::TensorType_FLOAT8_E4M3FN:
      return tensorflow::DT_FLOAT8_E4M3FN;
    case tflite::TensorType_FLOAT8_E5M2:
      return tensorflow::DT_FLOAT8_E5M2;
    // TODO(b/246806634): Tensorflow DT_INT2/4 type doesn't exist yet
    case tflite::TensorType_INT2:
      return tensorflow::DT_INT8;
    case tflite::TensorType_INT4:
      return tensorflow::DT_INT8;
    case tflite::TensorType_UINT4:
      return tensorflow::DT_UINT4;
    case tflite::TensorType_INT8:
      return tensorflow::DT_INT8;
    case tflite::TensorType_INT16:
      return tensorflow::DT_INT16;
    case tflite::TensorType_INT32:
      return tensorflow::DT_INT32;
    case tflite::TensorType_UINT32:
      return tensorflow::DT_UINT32;
    case tflite::TensorType_INT64:
      return tensorflow::DT_INT64;
    case tflite::TensorType_STRING:
      return tensorflow::DT_STRING;
    case tflite::TensorType_UINT8:
      return tensorflow::DT_UINT8;
    case tflite::TensorType_UINT16:
      return tensorflow::DT_UINT16;
    case tflite::TensorType_UINT64:
      return tensorflow::DT_UINT64;
    case tflite::TensorType_RESOURCE:
      return tensorflow::DT_RESOURCE;
    case tflite::TensorType_VARIANT:
      return tensorflow::DT_VARIANT;
  }
}

absl::StatusOr<tflite::TensorType> TfTypeToTflType(tensorflow::DataType type) {
  switch (type) {
    case tensorflow::DT_BOOL:
      return tflite::TensorType_BOOL;
    case tensorflow::DT_COMPLEX64:
      return tflite::TensorType_COMPLEX64;
    case tensorflow::DT_COMPLEX128:
      return tflite::TensorType_COMPLEX128;
    case tensorflow::DT_HALF:
      return tflite::TensorType_FLOAT16;
    case tensorflow::DT_BFLOAT16:
      return tflite::TensorType_BFLOAT16;
    case tensorflow::DT_FLOAT:
      return tflite::TensorType_FLOAT32;
    case tensorflow::DT_DOUBLE:
      return tflite::TensorType_FLOAT64;
    case tensorflow::DT_FLOAT8_E4M3FN:
      return tflite::TensorType_FLOAT8_E4M3FN;
    case tensorflow::DT_FLOAT8_E5M2:
      return tflite::TensorType_FLOAT8_E5M2;
    case tensorflow::DT_UINT4:
      return tflite::TensorType_UINT4;
    case tensorflow::DT_INT8:
      return tflite::TensorType_INT8;
    case tensorflow::DT_INT16:
      return tflite::TensorType_INT16;
    case tensorflow::DT_INT32:
      return tflite::TensorType_INT32;
    case tensorflow::DT_UINT32:
      return tflite::TensorType_UINT32;
    case tensorflow::DT_INT64:
      return tflite::TensorType_INT64;
    case tensorflow::DT_UINT64:
      return tflite::TensorType_UINT64;
    case tensorflow::DT_STRING:
      return tflite::TensorType_STRING;
    case tensorflow::DT_UINT8:
      return tflite::TensorType_UINT8;
    case tensorflow::DT_RESOURCE:
      return tflite::TensorType_RESOURCE;
    case tensorflow::DT_VARIANT:
      return tflite::TensorType_VARIANT;
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("unsupported tensor data type", type));
  }
}

}  // namespace tflite
