/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_SERVICE_CUSTOM_CALL_STATUS_H_
#define XLA_SERVICE_CUSTOM_CALL_STATUS_H_

#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

// ABI-stable public interfaces for XlaCustomCallStatus.

// Represents the result of a CustomCall: success or failure, with an associated
// error message in the failure case.
typedef struct XlaCustomCallStatus_ XlaCustomCallStatus;

// Set the XlaCustomCallStatus to a success state. This is the default state.
void XlaCustomCallStatusSetSuccess(XlaCustomCallStatus* status);

// Set the XlaCustomCallStatus to a failure state with the given error message.
// Does not take ownership of the supplied message string; instead copies the
// first 'message_len' bytes, or up to the null terminator, whichever comes
// first.
void XlaCustomCallStatusSetFailure(XlaCustomCallStatus* status,
                                   const char* message, size_t message_len);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // XLA_SERVICE_CUSTOM_CALL_STATUS_H_
