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

#ifdef SWIG
#define TF_CAPI_EXPORT
#else
#if defined(_WIN32)
#ifdef TF_COMPILE_LIBRARY
#define TF_CAPI_EXPORT __declspec(dllexport)
#else
#define TF_CAPI_EXPORT __declspec(dllimport)
#endif  // TF_COMPILE_LIBRARY
#else
#define TF_CAPI_EXPORT __attribute__((visibility("default")))
#endif  // _WIN32
#endif  // SWIG

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TF_Status TF_Status;

// --------------------------------------------------------------------------
// TF_Code holds an error code.  The enum values here are identical to
// corresponding values in error_codes.proto.
typedef enum TF_Code {
  TF_OK = 0,
  TF_CANCELLED = 1,
  TF_UNKNOWN = 2,
  TF_INVALID_ARGUMENT = 3,
  TF_DEADLINE_EXCEEDED = 4,
  TF_NOT_FOUND = 5,
  TF_ALREADY_EXISTS = 6,
  TF_PERMISSION_DENIED = 7,
  TF_UNAUTHENTICATED = 16,
  TF_RESOURCE_EXHAUSTED = 8,
  TF_FAILED_PRECONDITION = 9,
  TF_ABORTED = 10,
  TF_OUT_OF_RANGE = 11,
  TF_UNIMPLEMENTED = 12,
  TF_INTERNAL = 13,
  TF_UNAVAILABLE = 14,
  TF_DATA_LOSS = 15,
} TF_Code;

// --------------------------------------------------------------------------

// Return a new status object.
TF_CAPI_EXPORT extern TF_Status* TF_NewStatus(void);

// Delete a previously created status object.
TF_CAPI_EXPORT extern void TF_DeleteStatus(TF_Status*);

// Record <code, msg> in *s.  Any previous information is lost.
// A common use is to clear a status: TF_SetStatus(s, TF_OK, "");
TF_CAPI_EXPORT extern void TF_SetStatus(TF_Status* s, TF_Code code,
                                        const char* msg);

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
