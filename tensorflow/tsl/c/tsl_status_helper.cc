/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/tsl/c/tsl_status_helper.h"

#include "tensorflow/tsl/c/tsl_status_internal.h"
#include "tensorflow/tsl/platform/errors.h"

namespace tsl {

void Set_TSL_Status_from_Status(TSL_Status* tsl_status, const Status& status) {
  tensorflow::error::Code code = status.code();
  const char* message(status.error_message().c_str());

  switch (code) {
    case tensorflow::error::OK:
      assert(TSL_GetCode(tsl_status) == TSL_OK);
      break;
    case tensorflow::error::CANCELLED:
      TSL_SetStatus(tsl_status, TSL_CANCELLED, message);
      break;
    case tensorflow::error::UNKNOWN:
      TSL_SetStatus(tsl_status, TSL_UNKNOWN, message);
      break;
    case tensorflow::error::INVALID_ARGUMENT:
      TSL_SetStatus(tsl_status, TSL_INVALID_ARGUMENT, message);
      break;
    case tensorflow::error::DEADLINE_EXCEEDED:
      TSL_SetStatus(tsl_status, TSL_DEADLINE_EXCEEDED, message);
      break;
    case tensorflow::error::NOT_FOUND:
      TSL_SetStatus(tsl_status, TSL_NOT_FOUND, message);
      break;
    case tensorflow::error::ALREADY_EXISTS:
      TSL_SetStatus(tsl_status, TSL_ALREADY_EXISTS, message);
      break;
    case tensorflow::error::PERMISSION_DENIED:
      TSL_SetStatus(tsl_status, TSL_PERMISSION_DENIED, message);
      break;
    case tensorflow::error::UNAUTHENTICATED:
      TSL_SetStatus(tsl_status, TSL_UNAUTHENTICATED, message);
      break;
    case tensorflow::error::RESOURCE_EXHAUSTED:
      TSL_SetStatus(tsl_status, TSL_RESOURCE_EXHAUSTED, message);
      break;
    case tensorflow::error::FAILED_PRECONDITION:
      TSL_SetStatus(tsl_status, TSL_FAILED_PRECONDITION, message);
      break;
    case tensorflow::error::ABORTED:
      TSL_SetStatus(tsl_status, TSL_ABORTED, message);
      break;
    case tensorflow::error::OUT_OF_RANGE:
      TSL_SetStatus(tsl_status, TSL_OUT_OF_RANGE, message);
      break;
    case tensorflow::error::UNIMPLEMENTED:
      TSL_SetStatus(tsl_status, TSL_UNIMPLEMENTED, message);
      break;
    case tensorflow::error::INTERNAL:
      TSL_SetStatus(tsl_status, TSL_INTERNAL, message);
      break;
    case tensorflow::error::UNAVAILABLE:
      TSL_SetStatus(tsl_status, TSL_UNAVAILABLE, message);
      break;
    case tensorflow::error::DATA_LOSS:
      TSL_SetStatus(tsl_status, TSL_DATA_LOSS, message);
      break;
    default:
      assert(0);
      break;
  }

  errors::CopyPayloads(status, tsl_status->status);
}

Status StatusFromTSL_Status(const TSL_Status* tsl_status) {
  return tsl_status->status;
}

}  // namespace tsl
