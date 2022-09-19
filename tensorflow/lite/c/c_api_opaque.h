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

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// --------------------------------------------------------------------------
/// C API for TensorFlow Lite Opaque Types.
///
/// These APIs are accessors for TFLite Opaque Types.
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

// Returns opaque data provided by the node implementer. The value returned
// from this function is the value that was returned from the `init` callback
// that was passed to `TfLiteRegistrationExternalSetInit`.
TFL_CAPI_EXPORT extern void* TfLiteOpaqueNodeGetUserData(
    const TfLiteOpaqueNode* opaque_node);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_C_C_API_OPAQUE_H_
