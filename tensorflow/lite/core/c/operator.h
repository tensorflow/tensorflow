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
///
/// The types and functions declared in operator.h are
/// part of the TensorFlow Lite Extension APIs.
/// We reserve the right to make changes to this API in future releases,
/// potentially including non-backwards-compatible changes, on a different
/// schedule than for the other TensorFlow Lite APIs. See
/// https://www.tensorflow.org/guide/versions#separate_version_number_for_tensorflow_lite_extension_apis.
#ifndef TENSORFLOW_LITE_CORE_C_OPERATOR_H_
#define TENSORFLOW_LITE_CORE_C_OPERATOR_H_

#include <stdint.h>
#include <stdlib.h>

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/core/async/c/types.h"
#include "tensorflow/lite/core/c/c_api_types.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

/// TfLiteOperator is an opaque version of TfLiteRegistration,
/// and is used for registering custom ops.  It represents a definition of a
/// custom op or a builtin op.
///
/// \warning This is an experimental type and subject to change.
typedef struct TfLiteOperator TfLiteOperator;

/// Returns a new TfLiteOperator instance.
///
/// The returned TfLiteOperator instance represents a definition
/// of an operator with the identity (builtin_code/custom_name and
/// version) specified by the parameters, but with all callbacks initially
/// unset.
///
/// Evaluation of any operation using this operator will be done using
/// the "prepare" and "invoke" callbacks, which can be set using
/// `TfLiteOperatorSetPrepare` and
/// `TfLiteOperatorSetInvoke`, or for async execution
/// the "prepare", "eval", and "wait" callbacks of the `TfLiteAsyncKernel`,
/// which can be set using `TfLiteOperatorSetAsyncKernel`.
/// If the relevant callbacks are not set, then such evaluation will result
/// in an error status.  So normally any use of this function should be followed
/// by appropriate calls to set those callbacks.
///
/// \note The caller retains ownership and should ensure that
/// the lifetime of the `TfLiteOperator` must be at least as long as
/// the lifetime of any `TfLiteInterpreter` or `tflite::Interpreter` that it is
/// used in.
///
/// \param builtin_code Enumeration code specifying which builtin operator this
///                     defines, or `TfLiteBuiltinCustom` to define a custom op.
/// \param custom_name  Name of the custom op, or `nullptr` for a builtin op.
///                     If `custom_name` is non-null, then `builtin_code` should
///                     be `TfLiteBuiltinCustom`.
/// \param version      Version of the op.  See
///                     https://www.tensorflow.org/lite/guide/ops_version
///
/// \return \a newly created TfLiteOperator on success, \a nullptr on failure
///
/// Deprecated: Use `TfLiteOperatorCreateWithData`
TFL_CAPI_EXPORT extern TfLiteOperator* TfLiteOperatorCreate(
    TfLiteBuiltinOperator builtin_code, const char* custom_name, int version);

/// Returns a new TfLiteOperator instance.
///
/// The returned TfLiteOperator instance represents a definition
/// of an operator with the identity (builtin_code/custom_name and
/// version) specified by the parameters, but with all callbacks initially
/// unset.
///
/// Evaluation of any operation using this operator will be done using
/// the "prepare" and "invoke" callbacks, which can be set using
/// `TfLiteOperatorSetPrepare` and
/// `TfLiteOperatorSetInvoke`, or for async execution
/// the "prepare", "eval", and "wait" callbacks of the `TfLiteAsyncKernel`,
/// which can be set using `TfLiteOperatorSetAsyncKernel`.
/// If the relevant callbacks are not set, then such evaluation will result
/// in an error status.  So normally any use of this function should be followed
/// by appropriate calls to set those callbacks.
///
/// \note The caller retains ownership and should ensure that
/// the lifetime of the `TfLiteOperator` must be at least as long as
/// the lifetime of any `TfLiteInterpreter` or `tflite::Interpreter` that it is
/// used in.
///
/// \param builtin_code Enumeration code specifying which builtin operator this
///                     defines, or `TfLiteBuiltinCustom` to define a custom op.
/// \param custom_name  Name of the custom op, or `nullptr` for a builtin op.
///                     If `custom_name` is non-null, then `builtin_code` should
///                     be `TfLiteBuiltinCustom`.
/// \param version      Version of the op.  See
///                     https://www.tensorflow.org/lite/guide/ops_version
/// \param user_data    Opaque pointer passed to the operator's callbacks set
///                     with functions such as `TfLiteOperatorSetXXXWithData`.
///                     The user is expected to manage the memory pointed by
///                     this field and the lifetime of that memory should extend
///                     at least from the call to `TfLiteOperatorCreateWithData`
///                     to the invocation of the callback set with
///                     `TfLiteOperatorSetFreeWithData`.
///
/// \return a newly created TfLiteOperator on success, or a nullptr on failure
TFL_CAPI_EXPORT extern TfLiteOperator* TfLiteOperatorCreateWithData(
    TfLiteBuiltinOperator builtin_code, const char* custom_name, int version,
    void* user_data);

/// Destroys the TfLiteOperator instance.
///
TFL_CAPI_EXPORT extern void TfLiteOperatorDelete(TfLiteOperator* registration);

/// Return the builtin op code of the provided external 'registration'.
///
TFL_CAPI_EXPORT extern TfLiteBuiltinOperator TfLiteOperatorGetBuiltInCode(
    const TfLiteOperator* registration);

/// Returns the custom name of the provided 'registration'. The returned pointer
/// will be non-null iff the op is a custom op.
///
TFL_CAPI_EXPORT extern const char* TfLiteOperatorGetCustomName(
    const TfLiteOperator* registration);

/// Return the OP version of the provided external 'registration'.  Return -1
/// in case of error, or if the provided address is null.
///
TFL_CAPI_EXPORT extern int TfLiteOperatorGetVersion(
    const TfLiteOperator* registration);

/// Return the user data field of the provided external 'registration', or
/// nullptr if none was set.
///
TFL_CAPI_EXPORT extern void* TfLiteOperatorGetUserData(
    const TfLiteOperator* registration);

/// Sets the initialization callback for the registration.
///
/// The callback is called to initialize the op from serialized data.
/// Please refer `init` of `TfLiteRegistration` for the detail.
///
/// Deprecated: Use `TfLiteOperatorSetInitWithData`
TFL_CAPI_EXPORT extern void TfLiteOperatorSetInit(
    TfLiteOperator* registration,
    void* (*init)(TfLiteOpaqueContext* context, const char* buffer,
                  size_t length));

/// Sets the initialization callback for the registration. The function returns
/// an error upon failure.
///
/// The callback is called to initialize the op from serialized data. The value
/// passed in the `user_data` parameter is the value that was passed to
/// `TfLiteOperatorCreateWithData`.  Please refer `init` of `TfLiteRegistration`
/// for the detail.
///
TFL_CAPI_EXPORT extern TfLiteStatus TfLiteOperatorSetInitWithData(
    TfLiteOperator* registration,
    void* (*init)(void* user_data, TfLiteOpaqueContext* context,
                  const char* buffer, size_t length));

/// Sets the deallocation callback for the registration.
///
/// This callback is called to deallocate the data returned by the init
/// callback. The value passed in the `data` parameter is the value that was
/// returned by the `init` callback. Please refer `free` of `TfLiteRegistration`
/// for the detail.
///
/// Deprecated: Use `TfLiteOperatorSetFreeWithData`
TFL_CAPI_EXPORT extern void TfLiteOperatorSetFree(
    TfLiteOperator* registration,
    void (*free)(TfLiteOpaqueContext* context, void* data));

/// Sets the deallocation callback for the registration, similarly to
/// `TfLiteOperatorSetFree`. The function returns an error upon failure.
///
/// This callback is called to deallocate the data returned by the init
/// callback. The value passed in the `data` parameter is the value that was
/// returned by the `init` callback. The value passed in the `user_data`
/// parameter is the value that was passed to `TfLiteOperatorCreateWithData`.
/// Please refer `free` of `TfLiteRegistration` for the detail.
///
TFL_CAPI_EXPORT extern TfLiteStatus TfLiteOperatorSetFreeWithData(
    TfLiteOperator* registration,
    void (*free)(void* user_data, TfLiteOpaqueContext* context, void* data));

/// Sets the preparation callback for the registration.
///
/// The callback is called when the inputs of operator have been resized.
/// Please refer `prepare` of `TfLiteRegistration` for the detail.
///
/// Deprecated: Use `TfLiteOperatorSetPrepareWithData`
TFL_CAPI_EXPORT extern void TfLiteOperatorSetPrepare(
    TfLiteOperator* registration,
    TfLiteStatus (*prepare)(TfLiteOpaqueContext* context,
                            TfLiteOpaqueNode* node));

/// Sets the preparation callback for the registration. The function returns an
/// error upon failure.
///
/// The callback is called when the inputs of operator have been resized.  The
/// value passed in the `user_data` parameter is the value that was passed to
/// `TfLiteOperatorCreateWithData`.  Please refer `prepare` of
/// `TfLiteRegistration` for the detail.
///
TFL_CAPI_EXPORT extern TfLiteStatus TfLiteOperatorSetPrepareWithData(
    TfLiteOperator* registration,
    TfLiteStatus (*prepare)(void* user_data, TfLiteOpaqueContext* context,
                            TfLiteOpaqueNode* node));

/// Sets the invocation callback for the registration.
///
/// The callback is called when the operator is executed.
/// Please refer `invoke` of `TfLiteRegistration` for the detail.
///
/// Deprecated: Use `TfLiteOperatorSetInvokeWithData`
TFL_CAPI_EXPORT extern void TfLiteOperatorSetInvoke(
    TfLiteOperator* registration,
    TfLiteStatus (*invoke)(TfLiteOpaqueContext* context,
                           TfLiteOpaqueNode* node));

/// Sets the invocation callback for the registration. The function returns an
/// error upon failure.
///
/// The callback is called when the operator is executed.  The value passed in
/// the `user_data` parameter is the value that was passed to
/// `TfLiteOperatorCreate`.  Please refer `invoke` of `TfLiteRegistration` for
/// the detail.
///
TFL_CAPI_EXPORT extern TfLiteStatus TfLiteOperatorSetInvokeWithData(
    TfLiteOperator* registration,
    TfLiteStatus (*invoke)(void* user_data, TfLiteOpaqueContext* context,
                           TfLiteOpaqueNode* node));

/// Sets the async kernel accessor callback for the registration.
///
/// The callback is called to retrieve the async kernel if the delegate supports
/// it. If the delegate does not support async execution, either this function
/// should not be called, or `async_kernel` needs to be nullptr.
/// `node` is the delegate TfLiteNode created by `ModifyGraphWithDelegate`.
/// Please refer `async_kernel` of `TfLiteRegistration` for the detail.
///
/// \warning This is an experimental API and subject to change.
/// Deprecated: Use `TfLiteOperatorSetAsyncKernelWithData`
TFL_CAPI_EXPORT extern void TfLiteOperatorSetAsyncKernel(
    TfLiteOperator* registration,
    struct TfLiteAsyncKernel* (*async_kernel)(TfLiteOpaqueContext* context,
                                              TfLiteOpaqueNode* node));

/// Sets the async kernel accessor callback for the registration. The function
/// returns an error upon failure.
///
/// The callback is called to retrieve the async kernel if the delegate supports
/// it. If the delegate does not support async execution, either this function
/// should not be called, or `async_kernel` needs to be nullptr.  `node` is the
/// delegate TfLiteNode created by `ModifyGraphWithDelegate`.  The value passed
/// in the `user_data` parameter is the value that was passed to
/// `TfLiteOperatorCreate`.  Please refer `async_kernel` of `TfLiteRegistration`
/// for the detail.
///
/// \warning This is an experimental API and subject to change.
TFL_CAPI_EXPORT extern TfLiteStatus TfLiteOperatorSetAsyncKernelWithData(
    TfLiteOperator* registration,
    struct TfLiteAsyncKernel* (*async_kernel)(void* user_data,
                                              TfLiteOpaqueContext* context,
                                              TfLiteOpaqueNode* node));

/// Sets the inplace_operator field of the external registration.
///
/// This is a bitmask. Please refer to `inplace_operator` field of
/// `TfLiteRegistration` for details.
///
TFL_CAPI_EXPORT extern void TfLiteOperatorSetInplaceOperator(
    TfLiteOperator* registration, uint64_t inplace_operator);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_CORE_C_OPERATOR_H_
