/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_C_C_API_OPAQUE_H_
#define TENSORFLOW_LITE_C_C_API_OPAQUE_H_

#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/c_api_types.h"  // IWYU pragma: export
#include "tensorflow/lite/c/common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// --------------------------------------------------------------------------
/// C API for TensorFlow Lite Opaque Types.
///
/// These APIs are accessors for TFLite Opaque Types.  These APIs are primarily
/// intended to be used by delegates and custom OP implementations.
///
/// WARNING: This is an experimental API and subject to change.

// --------------------------------------------------------------------------
// Accessors for TfLiteOpaqueTensor.

// Returns the type of a tensor element.
TFL_CAPI_EXPORT extern TfLiteType TfLiteOpaqueTensorType(
    const TfLiteOpaqueTensor* opaque_tensor);

// Returns the number of dimensions that the tensor has.
TFL_CAPI_EXPORT extern int32_t TfLiteOpaqueTensorNumDims(
    const TfLiteOpaqueTensor* opaque_tensor);

// Returns the length of the tensor in the "dim_index" dimension.
TFL_CAPI_EXPORT extern int32_t TfLiteOpaqueTensorDim(
    const TfLiteOpaqueTensor* opaque_tensor, int32_t dim_index);

// Returns the size of the underlying data in bytes.
TFL_CAPI_EXPORT extern size_t TfLiteOpaqueTensorByteSize(
    const TfLiteOpaqueTensor* opaque_tensor);

// Returns a pointer to the underlying data buffer.
TFL_CAPI_EXPORT extern void* TfLiteOpaqueTensorData(
    const TfLiteOpaqueTensor* opaque_tensor);

// Returns the (null-terminated) name of the tensor.
TFL_CAPI_EXPORT extern const char* TfLiteOpaqueTensorName(
    const TfLiteOpaqueTensor* opaque_tensor);

// Copies from the provided input buffer into the tensor's buffer.
TFL_CAPI_EXPORT extern TfLiteStatus TfLiteOpaqueTensorCopyFromBuffer(
    TfLiteOpaqueTensor* opaque_tensor, const void* input_data,
    size_t input_data_size);

// Copies to the provided output buffer from the tensor's buffer.
TFL_CAPI_EXPORT extern TfLiteStatus TfLiteOpaqueTensorCopyToBuffer(
    const TfLiteOpaqueTensor* opaque_tensor, void* output_data,
    size_t output_data_size);

// --------------------------------------------------------------------------
// Accessors for TfLiteOpaqueNode.

// Returns the input tensor of the given node.
TFL_CAPI_EXPORT extern const TfLiteOpaqueTensor* TfLiteOpaqueNodeGetInput(
    TfLiteOpaqueContext* opaque_context, const TfLiteOpaqueNode* opaque_node,
    int index);

// Returns the output tensor of the given node.
TFL_CAPI_EXPORT extern TfLiteOpaqueTensor* TfLiteOpaqueNodeGetOutput(
    TfLiteOpaqueContext* opaque_context, const TfLiteOpaqueNode* opaque_node,
    int index);

// Gets the number of input tensors of the provided 'opaque_node'.
TFL_CAPI_EXPORT int TfLiteOpaqueNodeNumberOfInputs(
    const TfLiteOpaqueNode* opaque_node);

// Gets the number of output tensors of the provided 'opaque_node'.
TFL_CAPI_EXPORT int TfLiteOpaqueNodeNumberOfOutputs(
    const TfLiteOpaqueNode* opaque_node);

// Returns opaque data provided by the node implementer. The value returned
// from this function is the value that was returned from the `init` callback
// that was passed to `TfLiteRegistrationExternalSetInit`.
TFL_CAPI_EXPORT extern void* TfLiteOpaqueNodeGetUserData(
    const TfLiteOpaqueNode* opaque_node);

// --------------------------------------------------------------------------
// Accessors for TfLiteOpaqueContext.

typedef struct TfLiteIntArray TfLiteIntArray;

// Loads the provided `execution_plan` associated with the provided
// `opaque_context`.  Returns `kTfLiteOk` if the `execution_plan` was
// successfully loaded.  A return value different from `kTfLiteOk` indicates a
// failure and the `execution_plan` will be left in an unspecified state.
TFL_CAPI_EXPORT extern TfLiteStatus TfLiteOpaqueContextGetExecutionPlan(
    TfLiteOpaqueContext* opaque_context, TfLiteIntArray** execution_plan);

// Given the specified 'opaque_context' and 'node_index', load the caller's
// opaque '*node' and '*registration_external' pointer.  Return 'kTfLiteOk' if
// both the '*node' as well as the '*registration_external' have been loaded
// correctly.  Any other return code indicates a failure and both '*node' as
// well as '*registration_external' will be in an unspecified state.
//
// A caller can obtain a node's index by calling
// 'TfLiteOpaqueContextGetExecutionPlan', which provides an array of node
// indices, sorted in execution order.  A node index might also come from the
// data structures passed to the delegate kernel's callback parameters, like the
// delegate parameters data structure passed to the 'init' callback that
// contains an array of node indices that are meant to be handled by the
// delegate kernel.
//
// This function is expected to be called from within a delegate callback, like
// 'Prepare', or a delegate kernel callback (i.e., a callback registered with
// a 'TfLiteRegistrationExternal' object).
//
// The loaded '*node' and '*registration_external' pointers will generally
// remain valid for the lifetime of the associated 'opaque_context', but can be
// invalidated through API calls where delegates get un-applied, like API calls
// that modify the model graph via a delegate, or if input tensors get re-sized.
//
// TODO(b/237983452): Further clarify the lifetime guarantees of pointers that
// are returned to the users and which actions invalidate them.
TFL_CAPI_EXPORT TfLiteStatus TfLiteOpaqueContextGetNodeAndRegistration(
    struct TfLiteOpaqueContext* opaque_context, int node_index,
    TfLiteOpaqueNode** node,
    TfLiteRegistrationExternal** registration_external);

// WARNING: This is an experimental API and subject to change.
// Entry point for C API ReplaceNodeSubsetsWithDelegateKernels
//
// Replaces the specified `nodes_to_replace` that are associated with the
// provided `opaque_context` with delegate kernels.  The provided
// `registration_external` represents the delegate kernel and will be used for
// each node subset that will be delegate to the provided `opaque_delegate`.
//
// The TF Lite runtime will take ownership of the `registration_external` and
// will delete it when the associated `opaque_context` gets destroyed.
//
// The ownership of the `nodes_to_replace` and the `opaque_delegate` remains
// with the caller.
TfLiteStatus TfLiteOpaqueContextReplaceNodeSubsetsWithDelegateKernels(
    struct TfLiteOpaqueContext* opaque_context,
    TfLiteRegistrationExternal* registration_external,
    const TfLiteIntArray* nodes_to_replace,
    struct TfLiteOpaqueDelegateStruct* opaque_delegate);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_C_C_API_OPAQUE_H_
