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

#ifndef TENSORFLOW_C_KERNELS_EXPERIMENTAL_H_
#define TENSORFLOW_C_KERNELS_EXPERIMENTAL_H_

#include "tensorflow/c/kernels.h"

// --------------------------------------------------------------------------
// Experimental kernel C API for TensorFlow.
//
// The API here is subject to changes in the future.
// --------------------------------------------------------------------------

// Macro to control visibility of exported symbols in the shared library (.so,
// .dylib, .dll).
// This duplicates the TF_EXPORT macro definition in
// tensorflow/core/platform/macros.h in order to keep this .h file independent
// of any other includes.
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

typedef struct TF_VariableInputLockHolder TF_VariableInputLockHolder;

// Expose higher level Assignment operation for Pluggable vendors to implement
// in the plugin for Training. The API takes in the context with indices for
// the input and value tensors. It also accepts the copy callback provided by
// pluggable vendor to do the copying of the tensors. The caller takes ownership
// of the `source` and `dest` tensors and is responsible for freeing them with
// TF_DeleteTensor. This function will return an error when the following
// conditions are met:
//   1. `validate_shape` is set to `true`
//   2. The variable is initialized
//   3. The shape of the value tensor doesn't match the shape of the variable
//      tensor.
TF_CAPI_EXPORT extern void TF_AssignVariable(
    TF_OpKernelContext* ctx, int input_index, int value_index,
    bool validate_shape,
    void (*copyFunc)(TF_OpKernelContext* ctx, TF_Tensor* source,
                     TF_Tensor* dest),
    TF_Status* status);

// Expose higher level Assignment operation for Pluggable vendors to implement
// in the plugin for Training on ref variables. The API takes in the context
// with indices for the input and value tensors. It also accepts the copy
// callback provided by pluggable vendor to do the copying of the tensors. The
// caller takes ownership of the `source` and `dest` tensors and is responsible
// for freeing them with TF_DeleteTensor.
TF_CAPI_EXPORT extern void TF_AssignRefVariable(
    TF_OpKernelContext* ctx, int input_ref_index, int output_ref_index,
    int value_index, bool use_locking, bool validate_shape,
    void (*copyFunc)(TF_OpKernelContext* ctx, TF_Tensor* source,
                     TF_Tensor* dest),
    TF_Status* status);

// Expose higher level AssignUpdate operation for Pluggable vendors to implement
// in the plugin for Training. The API takes in the context with indices for the
// input and value tensors. It also accepts the copy callback provided by
// pluggable vendor to do the copying of the tensors and the update callback to
// apply the arithmetic operation. The caller takes ownership of the `source`,
// `dest`, `tensor` and `value` tensors and is responsible for freeing them with
// TF_DeleteTensor.
TF_CAPI_EXPORT extern void TF_AssignUpdateVariable(
    TF_OpKernelContext* ctx, int input_index, int value_index, int Op,
    int isVariantType,
    void (*copyFunc)(TF_OpKernelContext* ctx, TF_Tensor* source,
                     TF_Tensor* dest),
    void (*updateFunc)(TF_OpKernelContext* ctx, TF_Tensor* tensor,
                       TF_Tensor* value, int Op),
    TF_Status* status);

// This is a helper function which acquires mutexes in-order to provide
// thread-safe way of performing weights update during the optimizer op. It
// returns an opaque LockHolder handle back to plugin. This handle is passed to
// the Release API for releasing the locks when the weight update is done. The
// caller takes ownership of the `source` and `dest` tensors and is responsible
// for freeing them with TF_DeleteTensor.
TF_CAPI_EXPORT extern void TF_MaybeLockVariableInputMutexesInOrder(
    TF_OpKernelContext* ctx, bool do_lock, bool sparse, const int* const inputs,
    size_t len,
    void (*copyFunc)(TF_OpKernelContext* ctx, TF_Tensor* source,
                     TF_Tensor* dest),
    TF_VariableInputLockHolder** lockHolder, TF_Status* status);

// This interface returns `out` tensor which is updated corresponding to the
// variable passed with input index. The caller takes ownership of the `source`
// and `dest` tensors and is responsible for freeing them with TF_DeleteTensor.
TF_CAPI_EXPORT extern void TF_GetInputTensorFromVariable(
    TF_OpKernelContext* ctx, int input, bool lock_held, bool isVariantType,
    bool sparse,
    void (*copyFunc)(TF_OpKernelContext* ctx, TF_Tensor* source,
                     TF_Tensor* dest),
    TF_Tensor** out, TF_Status* status);

// This interface forwards the reference from input to the output tensors
// corresponding to the indices provided with `input_index` and `output_index`
TF_CAPI_EXPORT extern void TF_OpKernelContext_ForwardRefInputToRefOutput(
    TF_OpKernelContext* ctx, int32_t input_index, int32_t output_index);

// The API releases the opaque lock handle returned with
// `TF_MaybeLockVariableInputMutexesInOrder` API
TF_CAPI_EXPORT extern void TF_ReleaseVariableInputLockHolder(
    TF_VariableInputLockHolder* lockHolder);

// Allows plugin to get TF_Tensor when passed its input_name
TF_CAPI_EXPORT extern void TF_GetInputByName(TF_OpKernelContext* ctx,
                                             const char* inputName,
                                             TF_Tensor** tensor,
                                             TF_Status* status);

// Interprets the named kernel construction attribute as a shape attribute and
// fills in `vals` with the size of each dimension. `vals` must point to an
// array of length at least `max_values` (ideally set to total_size from
// TF_OpKernelConstruction_GetAttrSize(ctx, attr_name, &list_size,
// &total_size)).
TF_CAPI_EXPORT extern void TF_OpKernelConstruction_GetAttrTensorShape(
    TF_OpKernelConstruction* ctx, const char* attr_name, int64_t* dims,
    size_t num_dims, TF_Status* status);

TF_CAPI_EXPORT extern bool TF_IsRefInput(TF_OpKernelContext* ctx, int i,
                                         TF_Status* status);

// Expose higher level AddN operation for Pluggable vendors to implement
// in the plugin for Variant data types. The API takes in the context and a
// callback provided by pluggable vendor to do a Binary Add operation on the
// tensors unwrapped from the Variant tensors. The caller takes ownership of the
// `a`, `b` and `out` tensors and is responsible for freeing them with
// TF_DeleteTensor.
TF_CAPI_EXPORT extern void TF_AddNVariant(
    TF_OpKernelContext* ctx,
    void (*binary_add_func)(TF_OpKernelContext* ctx, TF_Tensor* a, TF_Tensor* b,
                            TF_Tensor* out),
    TF_Status* status);

// Expose higher level ZerosLike operation for Pluggable vendors to implement
// in the plugin for Variant data types. The API takes in the context and a
// callback provided by pluggable vendor to do a ZerosLike operation on the
// tensors unwrapped from the Variant tensors. The caller takes ownership of the
// `input` and `out` tensors and is responsible for freeing them with
// TF_DeleteTensor.
TF_CAPI_EXPORT extern void TF_ZerosLikeVariant(
    TF_OpKernelContext* ctx,
    void (*zeros_like_func)(TF_OpKernelContext* ctx, TF_Tensor* input,
                            TF_Tensor* out),
    TF_Status* status);

#ifdef __cplusplus
} /* end extern "C" */
#endif

#endif  // TENSORFLOW_C_KERNELS_EXPERIMENTAL_H_
