/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_C_TF_STATUS_H_
#define TENSORFLOW_C_TF_STATUS_H_

#include "tensorflow/c/c_api_macros.h"
#include "tsl/c/tsl_status.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TSL_Status TF_Status;

// --------------------------------------------------------------------------
// TF_Code holds an error code.  The enum values here are identical to
// corresponding values in error_codes.proto.
typedef TSL_Code TF_Code;
#define TF_OK TSL_OK
#define TF_CANCELLED TSL_CANCELLED
#define TF_UNKNOWN TSL_UNKNOWN
#define TF_INVALID_ARGUMENT TSL_INVALID_ARGUMENT
#define TF_DEADLINE_EXCEEDED TSL_DEADLINE_EXCEEDED
#define TF_NOT_FOUND TSL_NOT_FOUND
#define TF_ALREADY_EXISTS TSL_ALREADY_EXISTS
#define TF_PERMISSION_DENIED TSL_PERMISSION_DENIED
#define TF_UNAUTHENTICATED TSL_UNAUTHENTICATED
#define TF_RESOURCE_EXHAUSTED TSL_RESOURCE_EXHAUSTED
#define TF_FAILED_PRECONDITION TSL_FAILED_PRECONDITION
#define TF_ABORTED TSL_ABORTED
#define TF_OUT_OF_RANGE TSL_OUT_OF_RANGE
#define TF_UNIMPLEMENTED TSL_UNIMPLEMENTED
#define TF_INTERNAL TSL_INTERNAL
#define TF_UNAVAILABLE TSL_UNAVAILABLE
#define TF_DATA_LOSS TSL_DATA_LOSS

// --------------------------------------------------------------------------

// Return a new status object.
TF_CAPI_EXPORT extern TF_Status* TF_NewStatus(void);

// Delete a previously created status object.
TF_CAPI_EXPORT extern void TF_DeleteStatus(TF_Status*);

// Record <code, msg> in *s.  Any previous information is lost.
// A common use is to clear a status: TF_SetStatus(s, TF_OK, "");
TF_CAPI_EXPORT extern void TF_SetStatus(TF_Status* s, TF_Code code,
                                        const char* msg);

// Record <key, value> as a payload in *s. The previous payload having the
// same key (if any) is overwritten. Payload will not be added if the Status
// is OK.
TF_CAPI_EXPORT void TF_SetPayload(TF_Status* s, const char* key,
                                  const char* value);

// Iterates over the stored payloads and calls the `visitor(key, value)`
// callable for each one. `key` and `value` is only usable during the callback.
// `capture` will be passed to the callback without modification.
#define TF_PayloadVisitor TSL_PayloadVisitor
TF_CAPI_EXPORT extern void TF_ForEachPayload(const TF_Status* s,
                                             TF_PayloadVisitor visitor,
                                             void* capture);

// Convert from an I/O error code (e.g., errno) to a TF_Status value.
// Any previous information is lost. Prefer to use this instead of TF_SetStatus
// when the error comes from I/O operations.
TF_CAPI_EXPORT extern void TF_SetStatusFromIOError(TF_Status* s, int error_code,
                                                   const char* context);

// Return the code record in *s.
TF_CAPI_EXPORT extern TF_Code TF_GetCode(const TF_Status* s);

// Return a pointer to the (null-terminated) error message in *s.  The
// return value points to memory that is only usable until the next
// mutation to *s.  Always returns an empty string if TF_GetCode(s) is
// TF_OK.
TF_CAPI_EXPORT extern const char* TF_Message(const TF_Status* s);

#ifdef __cplusplus
} /* end extern "C" */
#endif

#endif  // TENSORFLOW_C_TF_STATUS_H_
