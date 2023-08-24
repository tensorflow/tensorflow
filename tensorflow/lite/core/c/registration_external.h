/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
/// \warning Users of TensorFlow Lite should not include this file directly,
/// but should instead include "third_party/tensorflow/lite/c/c_api.h".
/// Only the TensorFlow Lite implementation itself should include this
/// file directly.
#ifndef TENSORFLOW_LITE_CORE_C_REGISTRATION_EXTERNAL_H_
#define TENSORFLOW_LITE_CORE_C_REGISTRATION_EXTERNAL_H_

#include <stdlib.h>

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/core/async/c/types.h"
#include "tensorflow/lite/core/c/c_api_types.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// TfLiteRegistrationExternal is an external version of TfLiteRegistration to
// use custom op registration API.
//
// \warning This is an experimental type and subject to change.
typedef struct TfLiteRegistrationExternal TfLiteRegistrationExternal;

// Returns a new TfLiteRegistrationExternal instance.
//
// \note The caller retains ownership and should ensure that
// the lifetime of the `TfLiteRegistrationExternal` must be at least as long as
// the lifetime of the `TfLiteInterpreter`.
//
// \warning This is an experimental API and subject to change.
TFL_CAPI_EXPORT extern TfLiteRegistrationExternal*
TfLiteRegistrationExternalCreate(TfLiteBuiltinOperator builtin_code,
                                 const char* custom_name, int version);

// Destroys the TfLiteRegistrationExternal instance.
//
// \warning This is an experimental API and subject to change.
TFL_CAPI_EXPORT extern void TfLiteRegistrationExternalDelete(
    TfLiteRegistrationExternal* registration);

// Return the builtin op code of the provided external 'registration'.
//
// \warning This is an experimental API and subject to change.
TFL_CAPI_EXPORT extern TfLiteBuiltinOperator
TfLiteRegistrationExternalGetBuiltInCode(
    const TfLiteRegistrationExternal* registration);

/// Returns the custom name of the provided 'registration'. The returned pointer
/// will be non-null iff the op is a custom op.
///
/// \warning This is an experimental API and subject to change.
TFL_CAPI_EXPORT extern const char* TfLiteRegistrationExternalGetCustomName(
    const TfLiteRegistrationExternal* registration);

/// Return the OP version of the provided external 'registration'.  Return -1
/// in case of error, or if the provided address is null.
///
/// \warning This is an experimental API and subject to change.
TFL_CAPI_EXPORT extern int TfLiteRegistrationExternalGetVersion(
    const TfLiteRegistrationExternal* registration);

// Sets the initialization callback for the registration.
//
// The callback is called to initialize the op from serialized data.
// Please refer `init` of `TfLiteRegistration` for the detail.
//
// \warning This is an experimental API and subject to change.
TFL_CAPI_EXPORT extern void TfLiteRegistrationExternalSetInit(
    TfLiteRegistrationExternal* registration,
    void* (*init)(TfLiteOpaqueContext* context, const char* buffer,
                  size_t length));

// Sets the deallocation callback for the registration.
//
// This callback is called to deallocate the data returned by the init callback.
// The value passed in the `data` parameter is the value that was returned by
// the `init` callback.
// Please refer `free` of `TfLiteRegistration` for the detail.
//
// \warning This is an experimental API and subject to change.
TFL_CAPI_EXPORT extern void TfLiteRegistrationExternalSetFree(
    TfLiteRegistrationExternal* registration,
    void (*free)(TfLiteOpaqueContext* context, void* data));

// Sets the preparation callback for the registration.
//
// The callback is called when the inputs of operator have been resized.
// Please refer `prepare` of `TfLiteRegistration` for the detail.
//
// \warning This is an experimental API and subject to change.
TFL_CAPI_EXPORT extern void TfLiteRegistrationExternalSetPrepare(
    TfLiteRegistrationExternal* registration,
    TfLiteStatus (*prepare)(TfLiteOpaqueContext* context,
                            TfLiteOpaqueNode* node));

// Sets the invocation callback for the registration.
//
// The callback is called when the operator is executed.
// Please refer `invoke` of `TfLiteRegistration` for the detail.
//
// \warning This is an experimental API and subject to change.
TFL_CAPI_EXPORT extern void TfLiteRegistrationExternalSetInvoke(
    TfLiteRegistrationExternal* registration,
    TfLiteStatus (*invoke)(TfLiteOpaqueContext* context,
                           TfLiteOpaqueNode* node));

/// Sets the async kernel accessor callback for the registration.
///
/// The callback is called to retrieve the async kernel if the delegate supports
/// it. If the delegate does not support async execution, either this function
/// should not be called, or `async_kernel` needs to be nullptr.
/// `node` is the delegate TfLiteNode created by `ModifyGraphWithDelegate`.
/// Please refer `async_kernel` of `TfLiteRegistration` for the detail.
/// \warning This is an experimental API and subject to change.
TFL_CAPI_EXPORT extern void TfLiteRegistrationExternalSetAsyncKernel(
    TfLiteRegistrationExternal* registration,
    struct TfLiteAsyncKernel* (*async_kernel)(TfLiteOpaqueContext* context,
                                              TfLiteOpaqueNode* node));

/// Sets the inplace_operator field of the external registration.
///
/// This is a bitmask. Please refer to `inplace_operator` field of
/// `TfLiteRegistration` for details.
///
/// \warning This is an experimental API and subject to change.
TFL_CAPI_EXPORT extern void TfLiteRegistrationExternalSetInplaceOperator(
    TfLiteRegistrationExternal* registration, uint64_t inplace_operator);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_CORE_C_REGISTRATION_EXTERNAL_H_
