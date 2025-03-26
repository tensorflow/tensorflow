// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_COMMON_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_COMMON_H_

#include "third_party/qairt/latest/include/QNN/QnnCommon.h"
#include "third_party/qairt/latest/include/QNN/QnnInterface.h"
#include "third_party/qairt/latest/include/QNN/QnnTypes.h"
#include "third_party/qairt/latest/include/QNN/System/QnnSystemInterface.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#define LITERT_RETURN_STATUS_IF_QNN_NOT_OK(expr) \
  if (QNN_SUCCESS != (expr)) {                   \
    return kLiteRtStatusErrorNotFound;           \
  }

// Pointers to functions of a dynamically loaded QNN library.
typedef QNN_INTERFACE_VER_TYPE QnnApi;

// Pointers to functions of a dynamically loaded QNN system library.
typedef QNN_SYSTEM_INTERFACE_VER_TYPE QnnSystemApi;

// QNN backend library should be on DT_RUNPATH (-rpath).
static const char kLibQnnHtpSo[] = "libQnnHtp.so";

// QNN backend library should be on DT_RUNPATH (-rpath).
static const char kLibQnnSystemSo[] = "libQnnSystem.so";

// Map LiteRT element type to Qnn counterpart.
inline LiteRtStatus LegalizeElementType(litert::ElementType litert_type,
                                        Qnn_DataType_t* qnn_type) {
  switch (litert_type) {
    case litert::ElementType::Bool:
      *qnn_type = QNN_DATATYPE_BOOL_8;
      break;
    case litert::ElementType::Int4:
      *qnn_type = QNN_DATATYPE_SFIXED_POINT_4;
      break;
    case litert::ElementType::Int8:
      *qnn_type = QNN_DATATYPE_INT_8;
      break;
    case litert::ElementType::Int16:
      *qnn_type = QNN_DATATYPE_INT_16;
      break;
    case litert::ElementType::Int32:
      *qnn_type = QNN_DATATYPE_INT_32;
      break;
    case litert::ElementType::Int64:
      *qnn_type = QNN_DATATYPE_INT_64;
      break;
    case litert::ElementType::UInt8:
      *qnn_type = QNN_DATATYPE_UINT_8;
      break;
    case litert::ElementType::UInt16:
      *qnn_type = QNN_DATATYPE_UINT_16;
      break;
    case litert::ElementType::UInt32:
      *qnn_type = QNN_DATATYPE_UINT_32;
      break;
    case litert::ElementType::UInt64:
      *qnn_type = QNN_DATATYPE_UINT_64;
      break;
    case litert::ElementType::Float16:
      *qnn_type = QNN_DATATYPE_FLOAT_16;
      break;
    case litert::ElementType::Float32:
      *qnn_type = QNN_DATATYPE_FLOAT_32;
      break;
    case litert::ElementType::Float64:
      *qnn_type = QNN_DATATYPE_FLOAT_64;
      break;
    default:
      return kLiteRtStatusErrorUnsupported;
  }
  return kLiteRtStatusOk;
}

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_COMMON_H_
