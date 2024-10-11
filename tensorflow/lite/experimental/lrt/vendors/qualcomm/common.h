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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LRT_VENDORS_QUALCOMM_COMMON_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LRT_VENDORS_QUALCOMM_COMMON_H_

#include "third_party/qairt/include/QNN/QnnInterface.h"
#include "third_party/qairt/include/QNN/QnnTypes.h"
#include "third_party/qairt/include/QNN/System/QnnSystemInterface.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_common.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_model.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Pointers to functions of a dynamically loaded QNN library.
typedef QNN_INTERFACE_VER_TYPE QnnApi;

// Pointers to functions of a dynamically loaded QNN system library.
typedef QNN_SYSTEM_INTERFACE_VER_TYPE QnnSystemApi;

// QNN backend library should be on DT_RUNPATH (-rpath).
static const char kLibQnnHtpSo[] = "libQnnHtp.so";

// QNN backend library should be on DT_RUNPATH (-rpath).
static const char kLibQnnSystemSo[] = "libQnnSystem.so";

// Map LiteRT element type to Qnn counterpart.
inline LrtStatus LegalizeElementType(LrtElementType lrt_type,
                                     Qnn_DataType_t* qnn_type) {
  switch (lrt_type) {
    case kLrtElementTypeBool:
      *qnn_type = QNN_DATATYPE_BOOL_8;
      break;
    case kLrtElementTypeInt4:
      *qnn_type = QNN_DATATYPE_SFIXED_POINT_4;
      break;
    case kLrtElementTypeInt8:
      *qnn_type = QNN_DATATYPE_INT_8;
      break;
    case kLrtElementTypeInt16:
      *qnn_type = QNN_DATATYPE_INT_16;
      break;
    case kLrtElementTypeInt32:
      *qnn_type = QNN_DATATYPE_INT_32;
      break;
    case kLrtElementTypeInt64:
      *qnn_type = QNN_DATATYPE_INT_64;
      break;
    case kLrtElementTypeUInt8:
      *qnn_type = QNN_DATATYPE_UINT_8;
      break;
    case kLrtElementTypeUInt16:
      *qnn_type = QNN_DATATYPE_UINT_16;
      break;
    case kLrtElementTypeUInt32:
      *qnn_type = QNN_DATATYPE_UINT_32;
      break;
    case kLrtElementTypeUInt64:
      *qnn_type = QNN_DATATYPE_UINT_64;
      break;
    case kLrtElementTypeFloat16:
      *qnn_type = QNN_DATATYPE_FLOAT_16;
      break;
    case kLrtElementTypeFloat32:
      *qnn_type = QNN_DATATYPE_FLOAT_32;
      break;
    case kLrtElementTypeFloat64:
      *qnn_type = QNN_DATATYPE_FLOAT_64;
      break;
    default:
      return kLrtStatusErrorUnsupported;
  }
  return kLrtStatusOk;
}

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LRT_VENDORS_QUALCOMM_COMMON_H_
