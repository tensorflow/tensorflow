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
#ifndef TENSORFLOW_LITE_C_C_API_H_
#define TENSORFLOW_LITE_C_C_API_H_

#include <stdarg.h>
#include <stdint.h>
#include <stdlib.h>

#include "tensorflow/lite/c/c_api_types.h"  // IWYU pragma: export

// --------------------------------------------------------------------------
/// C API for TensorFlow Lite.
///
/// The API leans towards simplicity and uniformity instead of convenience, as
/// most usage will be by language-specific wrappers. It provides largely the
/// same set of functionality as that of the C++ TensorFlow Lite `Interpreter`
/// API, but is useful for shared libraries where having a stable ABI boundary
/// is important.
///
/// Conventions:
/// * We use the prefix TfLite for everything in the API.
/// * size_t is used to represent byte sizes of objects that are
///   materialized in the address space of the calling process.
/// * int is used as an index into arrays.
///
/// Usage:
/// <pre><code>
/// // Create the model and interpreter options.
/// TfLiteModel* model = TfLiteModelCreateFromFile("/path/to/model.tflite");
/// TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
/// TfLiteInterpreterOptionsSetNumThreads(options, 2);
///
/// // Create the interpreter.
/// TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);
///
/// // Allocate tensors and populate the input tensor data.
/// TfLiteInterpreterAllocateTensors(interpreter);
/// TfLiteTensor* input_tensor =
///     TfLiteInterpreterGetInputTensor(interpreter, 0);
/// TfLiteTensorCopyFromBuffer(input_tensor, input.data(),
///                            input.size() * sizeof(float));
///
/// // Execute inference.
/// TfLiteInterpreterInvoke(interpreter);
///
/// // Extract the output tensor data.
/// const TfLiteTensor* output_tensor =
//      TfLiteInterpreterGetOutputTensor(interpreter, 0);
/// TfLiteTensorCopyToBuffer(output_tensor, output.data(),
///                          output.size() * sizeof(float));
///
/// // Dispose of the model and interpreter objects.
/// TfLiteInterpreterDelete(interpreter);
/// TfLiteInterpreterOptionsDelete(options);
/// TfLiteModelDelete(model);
///
/// </code></pre>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// --------------------------------------------------------------------------
// Opaque types used by the C API.

// TfLiteModel wraps a loaded TensorFlow Lite model.
typedef struct TfLiteModel TfLiteModel;

// TfLiteInterpreterOptions allows customized interpreter configuration.
typedef struct TfLiteInterpreterOptions TfLiteInterpreterOptions;

// Allows delegation of nodes to alternative backends.
typedef struct TfLiteDelegate TfLiteDelegate;

// TfLiteInterpreter provides inference from a provided model.
typedef struct TfLiteInterpreter TfLiteInterpreter;

// A tensor in the interpreter system which is a wrapper around a buffer of
// data including a dimensionality (or NULL if not currently defined).
typedef struct TfLiteTensor TfLiteTensor;

// TfLiteOpaqueContext is an opaque version of TfLiteContext;
// WARNING: This is an experimental type and subject to change.
typedef struct TfLiteOpaqueContext TfLiteOpaqueContext;

// TfLiteOpaqueNode is an opaque version of TfLiteNode;
// WARNING: This is an experimental type and subject to change.
typedef struct TfLiteOpaqueNode TfLiteOpaqueNode;

// TfLiteRegistrationExternal is an external version of TfLiteRegistration to
// use custom op registration API.
// WARNING: This is an experimental type and subject to change.
typedef struct TfLiteRegistrationExternal TfLiteRegistrationExternal;

// --------------------------------------------------------------------------
// TfLiteVersion returns a string describing version information of the
// TensorFlow Lite library. TensorFlow Lite uses semantic versioning.
TFL_CAPI_EXPORT extern const char* TfLiteVersion(void);

// Returns a model from the provided buffer, or null on failure.
//
// NOTE: The caller retains ownership of the `model_data` and should ensure that
// the lifetime of the `model_data` must be at least as long as the lifetime
// of the `TfLiteModel`.
TFL_CAPI_EXPORT extern TfLiteModel* TfLiteModelCreate(const void* model_data,
                                                      size_t model_size);

// Returns a model from the provided file, or null on failure.
TFL_CAPI_EXPORT extern TfLiteModel* TfLiteModelCreateFromFile(
    const char* model_path);

// Destroys the model instance.
TFL_CAPI_EXPORT extern void TfLiteModelDelete(TfLiteModel* model);

// Returns a new TfLiteRegistrationExternal instance.
//
// NOTE: The caller retains ownership and should ensure that
// the lifetime of the `TfLiteRegistrationExternal` must be at least as long as
// the lifetime of the `TfLiteInterpreter`.
// WARNING: This is an experimental API and subject to change.
TFL_CAPI_EXPORT extern TfLiteRegistrationExternal*
TfLiteRegistrationExternalCreate(const char* custom_name, const int version);

// Destroys the TfLiteRegistrationExternal instance.
// WARNING: This is an experimental API and subject to change.
TFL_CAPI_EXPORT extern void TfLiteRegistrationExternalDelete(
    TfLiteRegistrationExternal* registration);

// Sets the preparation callback for the registration.
//
// The callback is called when the inputs of operator have been resized.
// Please refer `prepare` of `TfLiteRegistration` for the detail.
// WARNING: This is an experimental API and subject to change.
TFL_CAPI_EXPORT extern void TfLiteRegistrationExternalSetPrepare(
    TfLiteRegistrationExternal* registration,
    TfLiteStatus (*prepare)(TfLiteOpaqueContext* context,
                            TfLiteOpaqueNode* node));

// Sets the invocation callback for the registration.
//
// The callback is called when the operator is executed.
// Please refer `invoke` of `TfLiteRegistration` for the detail.
// WARNING: This is an experimental API and subject to change.
TFL_CAPI_EXPORT extern void TfLiteRegistrationExternalSetInvoke(
    TfLiteRegistrationExternal* registration,
    TfLiteStatus (*invoke)(TfLiteOpaqueContext* context,
                           TfLiteOpaqueNode* node));

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
// * `user_data` is optional. If non-null, it is owned by the client and must
//   remain valid for the duration of the interpreter lifetime.
TFL_CAPI_EXPORT extern void TfLiteInterpreterOptionsSetErrorReporter(
    TfLiteInterpreterOptions* options,
    void (*reporter)(void* user_data, const char* format, va_list args),
    void* user_data);

// Adds an op registration to be applied during `TfLiteInterpreter` creation.
//
// The `TfLiteRegistrationExternal` object is needed to implement custom op of
// TFLite Interpreter via C API. Calling this function ensures that any
// `TfLiteInterpreter` created with the specified `options` can execute models
// that use the custom operator specified in `registration`.
// Please refer https://www.tensorflow.org/lite/guide/ops_custom for custom op
// support.
// NOTE: The caller retains ownership of the TfLiteRegistrationExternal object
// and should ensure that it remains valid for the duration of any created
// interpreter's lifetime.
// WARNING: This is an experimental API and subject to change.
TFL_CAPI_EXPORT extern void TfLiteInterpreterOptionsAddRegistrationExternal(
    TfLiteInterpreterOptions* options,
    TfLiteRegistrationExternal* registration);

// Returns a new interpreter using the provided model and options, or null on
// failure.
//
// * `model` must be a valid model instance. The caller retains ownership of the
//   object, and the model must outlive the interpreter.
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
TFL_CAPI_EXPORT extern int32_t TfLiteInterpreterGetInputTensorCount(
    const TfLiteInterpreter* interpreter);

// Returns the tensor associated with the input index.
// REQUIRES: 0 <= input_index < TfLiteInterpreterGetInputTensorCount(tensor)
TFL_CAPI_EXPORT extern TfLiteTensor* TfLiteInterpreterGetInputTensor(
    const TfLiteInterpreter* interpreter, int32_t input_index);

// Resizes the specified input tensor.
//
// NOTE: After a resize, the client *must* explicitly allocate tensors before
// attempting to access the resized tensor data or invoke the interpreter.
//
// REQUIRES: 0 <= input_index < TfLiteInterpreterGetInputTensorCount(tensor)
//
// This function makes a copy of the input dimensions, so the client can safely
// deallocate `input_dims` immediately after this function returns.
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
// Before calling this function, the caller should first invoke
// TfLiteInterpreterAllocateTensors() and should also set the values for the
// input tensors.  After successfully calling this function, the values for the
// output tensors will be set.
//
// NOTE: It is possible that the interpreter is not in a ready state to
// evaluate (e.g., if AllocateTensors() hasn't been called, or if a
// ResizeInputTensor() has been performed without a subsequent call to
// AllocateTensors()).
//
//   If the (experimental!) delegate fallback option was enabled in the
//   interpreter options, then the interpreter will automatically fall back to
//   not using any delegates if execution with delegates fails. For details, see
//   TfLiteInterpreterOptionsSetEnableDelegateFallback in c_api_experimental.h.
//
// Returns one of the following status codes:
//  - kTfLiteOk: Success. Output is valid.
//  - kTfLiteDelegateError: Execution with delegates failed, due to a problem
//    with the delegate(s). If fallback was not enabled, output is invalid.
//    If fallback was enabled, this return value indicates that fallback
//    succeeded, the output is valid, and all delegates previously applied to
//    the interpreter have been undone.
//  - kTfLiteApplicationError: Same as for kTfLiteDelegateError, except that
//    the problem was not with the delegate itself, but rather was
//    due to an incompatibility between the delegate(s) and the
//    interpreter or model.
//  - kTfLiteError: Unexpected/runtime failure. Output is invalid.

TFL_CAPI_EXPORT extern TfLiteStatus TfLiteInterpreterInvoke(
    TfLiteInterpreter* interpreter);

// Returns the number of output tensors associated with the model.
TFL_CAPI_EXPORT extern int32_t TfLiteInterpreterGetOutputTensorCount(
    const TfLiteInterpreter* interpreter);

// Returns the tensor associated with the output index.
// REQUIRES: 0 <= output_index < TfLiteInterpreterGetOutputTensorCount(tensor)
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

#endif  // TENSORFLOW_LITE_C_C_API_H_
