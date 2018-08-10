/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CONTRIB_LITE_EXPERIMENTAL_C_C_API_H_
#define TENSORFLOW_CONTRIB_LITE_EXPERIMENTAL_C_C_API_H_

#include <stdint.h>

// Eventually the various C APIs defined in context.h will be migrated into
// the appropriate /c/c_api*.h header. For now, we pull in existing definitions
// for convenience.
#include "tensorflow/contrib/lite/context.h"

// --------------------------------------------------------------------------
// Experimental C API for TensorFlowLite.
//
// The API leans towards simplicity and uniformity instead of convenience, as
// most usage will be by language-specific wrappers.
//
// Conventions:
// * We use the prefix TFL_ for everything in the API.

#ifdef SWIG
#define TFL_CAPI_EXPORT
#else
#if defined(_WIN32)
#ifdef TF_COMPILE_LIBRARY
#define TFL_CAPI_EXPORT __declspec(dllexport)
#else
#define TFL_CAPI_EXPORT __declspec(dllimport)
#endif  // TF_COMPILE_LIBRARY
#else
#define TFL_CAPI_EXPORT __attribute__((visibility("default")))
#endif  // _WIN32
#endif  // SWIG

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef TfLiteTensor TFL_Tensor;
typedef TfLiteStatus TFL_Status;
typedef TfLiteType TFL_Type;

// --------------------------------------------------------------------------
// TFL_Interpreter provides inference from a provided model.
typedef struct _TFL_Interpreter TFL_Interpreter;

// Returns an interpreter for the provided model, or null on failure.
//
// NOTE: The client *must* explicitly allocate tensors before attempting to
// access input tensor data or invoke the interpreter.
TFL_CAPI_EXPORT extern TFL_Interpreter* TFL_NewInterpreter(
    const void* model_data, int32_t model_size);

// Destroys the interpreter.
TFL_CAPI_EXPORT extern void TFL_DeleteInterpreter(TFL_Interpreter* interpreter);

// Returns the number of input tensors associated with the model.
TFL_CAPI_EXPORT extern int TFL_InterpreterGetInputTensorCount(
    const TFL_Interpreter* interpreter);

// Returns the tensor associated with the input index.
// REQUIRES: 0 <= input_index < TFL_InterpreterGetInputTensorCount(tensor)
TFL_CAPI_EXPORT extern TFL_Tensor* TFL_InterpreterGetInputTensor(
    const TFL_Interpreter* interpreter, int32_t input_index);

// Attempts to resize the specified input tensor.
// NOTE: After a resize, the client *must* explicitly allocate tensors before
// attempting to access the resized tensor data or invoke the interpreter.
// REQUIRES: 0 <= input_index < TFL_InterpreterGetInputTensorCount(tensor)
TFL_CAPI_EXPORT extern TFL_Status TFL_InterpreterResizeInputTensor(
    TFL_Interpreter* interpreter, int32_t input_index, const int* input_dims,
    int32_t input_dims_size);

// Updates allocations for all tensors, resizing dependent tensors using the
// specified input tensor dimensionality.
//
// This is a relatively expensive operation, and need only be called after
// creating the graph and/or resizing any inputs.
TFL_CAPI_EXPORT extern TFL_Status TFL_InterpreterAllocateTensors(
    TFL_Interpreter* interpreter);

// Runs inference for the loaded graph.
//
// NOTE: It is possible that the interpreter is not in a ready state to
// evaluate (e.g., if a ResizeInputTensor() has been performed without a call to
// AllocateTensors()).
TFL_CAPI_EXPORT extern TFL_Status TFL_InterpreterInvoke(
    TFL_Interpreter* interpreter);

// Returns the number of output tensors associated with the model.
TFL_CAPI_EXPORT extern int32_t TFL_InterpreterGetOutputTensorCount(
    const TFL_Interpreter* interpreter);

// Returns the tensor associated with the output index.
// REQUIRES: 0 <= input_index < TFL_InterpreterGetOutputTensorCount(tensor)
TFL_CAPI_EXPORT extern const TFL_Tensor* TFL_InterpreterGetOutputTensor(
    const TFL_Interpreter* interpreter, int32_t output_index);

// --------------------------------------------------------------------------
// TFL_Tensor wraps data associated with a graph tensor.
//
// Note that, while the TFL_Tensor struct is not currently opaque, and its
// fields can be accessed directly, these methods are still convenient for
// language bindings. In the future the tensor struct will likely be made opaque
// in the public API.

// Returns the type of a tensor element.
TFL_CAPI_EXPORT extern TFL_Type TFL_TensorType(const TFL_Tensor* tensor);

// Returns the number of dimensions that the tensor has.
TFL_CAPI_EXPORT extern int32_t TFL_TensorNumDims(const TFL_Tensor* tensor);

// Returns the length of the tensor in the "dim_index" dimension.
// REQUIRES: 0 <= dim_index < TFLiteTensorNumDims(tensor)
TFL_CAPI_EXPORT extern int32_t TFL_TensorDim(const TFL_Tensor* tensor,
                                             int32_t dim_index);

// Returns the size of the underlying data in bytes.
TFL_CAPI_EXPORT extern size_t TFL_TensorByteSize(const TFL_Tensor* tensor);

// Copies from the provided input buffer into the tensor's buffer.
// REQUIRES: input_data_size == TFL_TensorByteSize(tensor)
TFL_CAPI_EXPORT extern TFL_Status TFL_TensorCopyFromBuffer(
    TFL_Tensor* tensor, const void* input_data, int32_t input_data_size);

// Copies to the provided output buffer from the tensor's buffer.
// REQUIRES: output_data_size == TFL_TensorByteSize(tensor)
TFL_CAPI_EXPORT extern TFL_Status TFL_TensorCopyToBuffer(
    const TFL_Tensor* output_tensor, void* output_data,
    int32_t output_data_size);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // TENSORFLOW_CONTRIB_LITE_EXPERIMENTAL_C_C_API_H_
