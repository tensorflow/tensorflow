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

#ifndef XLA_TSL_C_TSL_STATUS_H_
#define XLA_TSL_C_TSL_STATUS_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TSL_Status TSL_Status;

// --------------------------------------------------------------------------
// TSL_Code holds an error code.  The enum values here are identical to
// corresponding values in error_codes.proto.
typedef enum TSL_Code {
  TSL_OK = 0,
  TSL_CANCELLED = 1,
  TSL_UNKNOWN = 2,
  TSL_INVALID_ARGUMENT = 3,
  TSL_DEADLINE_EXCEEDED = 4,
  TSL_NOT_FOUND = 5,
  TSL_ALREADY_EXISTS = 6,
  TSL_PERMISSION_DENIED = 7,
  TSL_UNAUTHENTICATED = 16,
  TSL_RESOURCE_EXHAUSTED = 8,
  TSL_FAILED_PRECONDITION = 9,
  TSL_ABORTED = 10,
  TSL_OUT_OF_RANGE = 11,
  TSL_UNIMPLEMENTED = 12,
  TSL_INTERNAL = 13,
  TSL_UNAVAILABLE = 14,
  TSL_DATA_LOSS = 15,
} TSL_Code;

// --------------------------------------------------------------------------

// Return a new status object.
extern TSL_Status* TSL_NewStatus(void);

// Delete a previously created status object.
extern void TSL_DeleteStatus(TSL_Status*);

// Record <code, msg> in *s.  Any previous information is lost.
// A common use is to clear a status: TSL_SetStatus(s, TSL_OK, "");
extern void TSL_SetStatus(TSL_Status* s, TSL_Code code, const char* msg);

// Record <key, value> as a payload in *s. The previous payload having the
// same key (if any) is overwritten. Payload will not be added if the Status
// is OK.
extern void TSL_SetPayload(TSL_Status* s, const char* key, const char* value);

// Iterates over the stored payloads and calls the `visitor(key, value)`
// callable for each one. `key` and `value` is only usable during the callback.
// `capture` will be passed to the callback without modification.
typedef void (*TSL_PayloadVisitor)(const char* key, const char* value,
                                   void* capture);
extern void TSL_ForEachPayload(const TSL_Status* s, TSL_PayloadVisitor visitor,
                               void* capture);

// Convert from an I/O error code (e.g., errno) to a TSL_Status value.
// Any previous information is lost. Prefer to use this instead of TSL_SetStatus
// when the error comes from I/O operations.
extern void TSL_SetStatusFromIOError(TSL_Status* s, int error_code,
                                     const char* context);

// Return the code record in *s.
extern TSL_Code TSL_GetCode(const TSL_Status* s);

// Return a pointer to the (null-terminated) error message in *s.  The
// return value points to memory that is only usable until the next
// mutation to *s.  Always returns an empty string if TSL_GetCode(s) is
// TSL_OK.
extern const char* TSL_Message(const TSL_Status* s);

#ifdef __cplusplus
} /* end extern "C" */
#endif

#endif  // XLA_TSL_C_TSL_STATUS_H_
