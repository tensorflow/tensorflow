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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_C_C_API_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_C_C_API_H_

#include <stdarg.h>
#include <stdint.h>

// Eventually the various C APIs defined in context.h will be migrated into
// the appropriate /c/c_api*.h header. For now, we pull in existing definitions
// for convenience.
#include "c_api_types.h"

// --------------------------------------------------------------------------
// Experimental C API for TensorFlowLite.
//
// The API leans towards simplicity and uniformity instead of convenience, as
// most usage will be by language-specific wrappers.
//
// Conventions:
// * We use the prefix TfLite for everything in the API.
// * size_t is used to represent byte sizes of objects that are
//   materialized in the address space of the calling process.
// * int is used as an index into arrays.

#ifdef SWIG
#define TFL_CAPI_EXPORT
#else
#if defined(_WIN32)
#ifdef TFL_COMPILE_LIBRARY
#define TFL_CAPI_EXPORT __declspec(dllexport)
#else
#define TFL_CAPI_EXPORT __declspec(dllimport)
#endif  // TFL_COMPILE_LIBRARY
#else
#define TFL_CAPI_EXPORT __attribute__((visibility("default")))
#endif  // _WIN32
#endif  // SWIG

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// --------------------------------------------------------------------------
// TfLiteVersion returns a string describing version information of the
// TensorFlow Lite library. TensorFlow Lite uses semantic versioning.
TFL_CAPI_EXPORT extern const char* TfLiteVersion(void);

// --------------------------------------------------------------------------
// TfLiteModel wraps a loaded TensorFlow Lite model.
typedef struct TfLiteModel TfLiteModel;

// Returns a model from the provided buffer, or null on failure.
TFL_CAPI_EXPORT extern TfLiteModel* TfLiteModelCreate(const void* model_data,
                                                      size_t model_size);

// Returns a model from the provided file, or null on failure.
TFL_CAPI_EXPORT extern TfLiteModel* TfLiteModelCreateFromFile(
    const char* model_path);

// Destroys the model instance.
TFL_CAPI_EXPORT extern void TfLiteModelDelete(TfLiteModel* model);

// --------------------------------------------------------------------------
// TfLiteInterpreterOptions allows customized interpreter configuration.
typedef struct TfLiteInterpreterOptions TfLiteInterpreterOptions;

// Returns a new interpreter options instances.
TFL_CAPI_EXPORT extern TfLiteInterpreterOptions*
TfLiteInterpreterOptionsCreate();

// Destroys the interpreter options instance.
TFL_CAPI_EXPORT extern void TfLiteInterpreterOptionsDelete(
    TfLiteInterpreterOptions* options);

// Sets the number of CPU threads to use for the interpreter.
TFL_CAPI_EXPORT extern void TfLiteInterpreterOptionsSetNumThreads(
    TfLiteInterpreterOptions* options, int32_t num_threads);

// Adds a delegate to be applied during `TfLiteInterpreter` creation.
//
// If delegate application fails, interpreter creation will also fail with an
// associated error logged.
//
// NOTE: The caller retains ownership of the delegate and should ensure that it
// remains valid for the duration of any created interpreter's lifetime.
TFL_CAPI_EXPORT extern void TfLiteInterpreterOptionsAddDelegate(
    TfLiteInterpreterOptions* options, TfLiteDelegate* delegate);

// Sets a custom error reporter for interpreter execution.
//
// * `reporter` takes the provided `user_data` object, as well as a C-style
//   format string and arg list (see also vprintf).
// * `user_data` is optional. If provided, it is owned by the client and must
//   remain valid for the duration of the interpreter lifetime.
TFL_CAPI_EXPORT extern void TfLiteInterpreterOptionsSetErrorReporter(
    TfLiteInterpreterOptions* options,
    void (*reporter)(void* user_data, const char* format, va_list args),
    void* user_data);

// --------------------------------------------------------------------------
// TfLiteInterpreter provides inference from a provided model.
typedef struct TfLiteInterpreter TfLiteInterpreter;

// Returns a new interpreter using the provided model and options, or null on
// failure.
//
// * `model` must be a valid model instance. The caller retains ownership of the
//   object, and can destroy it immediately after creating the interpreter; the
//   interpreter will maintain its own reference to the underlying model data.
// * `optional_options` may be null. The caller retains ownership of the object,
//   and can safely destroy it immediately after creating the interpreter.
//
// NOTE: The client *must* explicitly allocate tensors before attempting to
// access input tensor data or invoke the interpreter.
TFL_CAPI_EXPORT extern TfLiteInterpreter* TfLiteInterpreterCreate(
    const TfLiteModel* model, const TfLiteInterpreterOptions* optional_options);

// Destroys the interpreter.
TFL_CAPI_EXPORT extern void TfLiteInterpreterDelete(
    TfLiteInterpreter* interpreter);

// Returns the number of input tensors associated with the model.
TFL_CAPI_EXPORT extern int TfLiteInterpreterGetInputTensorCount(
    const TfLiteInterpreter* interpreter);

// Returns the tensor associated with the input index.
// REQUIRES: 0 <= input_index < TfLiteInterpreterGetInputTensorCount(tensor)
TFL_CAPI_EXPORT extern TfLiteTensor* TfLiteInterpreterGetInputTensor(
    const TfLiteInterpreter* interpreter, int32_t input_index);

// Resizes the specified input tensor.
//
// NOTE: After a resize, the client *must* explicitly allocate tensors before
// attempting to access the resized tensor data or invoke the interpreter.
// REQUIRES: 0 <= input_index < TfLiteInterpreterGetInputTensorCount(tensor)
TFL_CAPI_EXPORT extern TfLiteStatus TfLiteInterpreterResizeInputTensor(
    TfLiteInterpreter* interpreter, int32_t input_index, const int* input_dims,
    int32_t input_dims_size);

// Updates allocations for all tensors, resizing dependent tensors using the
// specified input tensor dimensionality.
//
// This is a relatively expensive operation, and need only be called after
// creating the graph and/or resizing any inputs.
TFL_CAPI_EXPORT extern TfLiteStatus TfLiteInterpreterAllocateTensors(
    TfLiteInterpreter* interpreter);

// Runs inference for the loaded graph.
//
// NOTE: It is possible that the interpreter is not in a ready state to
// evaluate (e.g., if a ResizeInputTensor() has been performed without a call to
// AllocateTensors()).
TFL_CAPI_EXPORT extern TfLiteStatus TfLiteInterpreterInvoke(
    TfLiteInterpreter* interpreter);

// Returns the number of output tensors associated with the model.
TFL_CAPI_EXPORT extern int32_t TfLiteInterpreterGetOutputTensorCount(
    const TfLiteInterpreter* interpreter);

// Returns the tensor associated with the output index.
// REQUIRES: 0 <= input_index < TfLiteInterpreterGetOutputTensorCount(tensor)
//
// NOTE: The shape and underlying data buffer for output tensors may be not
// be available until after the output tensor has been both sized and allocated.
// In general, best practice is to interact with the output tensor *after*
// calling TfLiteInterpreterInvoke().
TFL_CAPI_EXPORT extern const TfLiteTensor* TfLiteInterpreterGetOutputTensor(
    const TfLiteInterpreter* interpreter, int32_t output_index);

// --------------------------------------------------------------------------
// TfLiteTensor wraps data associated with a graph tensor.
//
// Note that, while the TfLiteTensor struct is not currently opaque, and its
// fields can be accessed directly, these methods are still convenient for
// language bindings. In the future the tensor struct will likely be made opaque
// in the public API.

// Returns the type of a tensor element.
TFL_CAPI_EXPORT extern TfLiteType TfLiteTensorType(const TfLiteTensor* tensor);

// Returns the number of dimensions that the tensor has.
TFL_CAPI_EXPORT extern int32_t TfLiteTensorNumDims(const TfLiteTensor* tensor);

// Returns the length of the tensor in the "dim_index" dimension.
// REQUIRES: 0 <= dim_index < TFLiteTensorNumDims(tensor)
TFL_CAPI_EXPORT extern int32_t TfLiteTensorDim(const TfLiteTensor* tensor,
                                               int32_t dim_index);

// Returns the size of the underlying data in bytes.
TFL_CAPI_EXPORT extern size_t TfLiteTensorByteSize(const TfLiteTensor* tensor);

// Returns a pointer to the underlying data buffer.
//
// NOTE: The result may be null if tensors have not yet been allocated, e.g.,
// if the Tensor has just been created or resized and `TfLiteAllocateTensors()`
// has yet to be called, or if the output tensor is dynamically sized and the
// interpreter hasn't been invoked.
TFL_CAPI_EXPORT extern void* TfLiteTensorData(const TfLiteTensor* tensor);

// Returns the (null-terminated) name of the tensor.
TFL_CAPI_EXPORT extern const char* TfLiteTensorName(const TfLiteTensor* tensor);

// Returns the parameters for asymmetric quantization. The quantization
// parameters are only valid when the tensor type is `kTfLiteUInt8` and the
// `scale != 0`. Quantized values can be converted back to float using:
//    real_value = scale * (quantized_value - zero_point);
TFL_CAPI_EXPORT extern TfLiteQuantizationParams TfLiteTensorQuantizationParams(
    const TfLiteTensor* tensor);

// Copies from the provided input buffer into the tensor's buffer.
// REQUIRES: input_data_size == TfLiteTensorByteSize(tensor)
TFL_CAPI_EXPORT extern TfLiteStatus TfLiteTensorCopyFromBuffer(
    TfLiteTensor* tensor, const void* input_data, size_t input_data_size);

// Copies to the provided output buffer from the tensor's buffer.
// REQUIRES: output_data_size == TfLiteTensorByteSize(tensor)
TFL_CAPI_EXPORT extern TfLiteStatus TfLiteTensorCopyToBuffer(
    const TfLiteTensor* output_tensor, void* output_data,
    size_t output_data_size);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_C_C_API_H_
