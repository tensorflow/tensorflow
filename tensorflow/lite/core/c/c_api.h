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
/// \warning Users of TensorFlow Lite should not include this file directly,
/// but should instead include "third_party/tensorflow/lite/c/c_api.h".
/// Only the TensorFlow Lite implementation itself should include this
/// file directly.
#ifndef TENSORFLOW_LITE_CORE_C_C_API_H_
#define TENSORFLOW_LITE_CORE_C_C_API_H_

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/core/c/c_api_types.h"  // IWYU pragma: export

// --------------------------------------------------------------------------
/// \file
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
///      TfLiteInterpreterGetOutputTensor(interpreter, 0);
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

// This header should be valid in both C (e.g. C99) and C++,
// so 'void' in parameters is not redundant.
// NOLINTBEGIN(modernize-redundant-void-arg)

// --------------------------------------------------------------------------
// Opaque types used by the C API.  (See also c_api_types.h.)

/// TfLiteModel wraps a loaded TensorFlow Lite model.
typedef struct TfLiteModel TfLiteModel;

/// TfLiteInterpreterOptions allows customized interpreter configuration.
typedef struct TfLiteInterpreterOptions TfLiteInterpreterOptions;

/// TfLiteInterpreter provides inference from a provided model.
typedef struct TfLiteInterpreter TfLiteInterpreter;

/// A tensor in the interpreter system which is a wrapper around a buffer of
/// data including a dimensionality (or NULL if not currently defined).
typedef struct TfLiteTensor TfLiteTensor;

/// TfLiteRegistrationExternal is an external version of TfLiteRegistration to
/// use custom op registration API.
/// \warning This is an experimental type and subject to change.
typedef struct TfLiteRegistrationExternal TfLiteRegistrationExternal;

// --------------------------------------------------------------------------
/// The TensorFlow Lite Runtime version.
///
/// Returns a pointer to a statically allocated string that is the version
/// number of the (potentially dynamically loaded) TF Lite Runtime library.
/// TensorFlow Lite uses semantic versioning, and the return value should be
/// in semver 2 format <http://semver.org>, starting with MAJOR.MINOR.PATCH,
/// e.g. "2.12.0" or "2.13.0-rc2".
TFL_CAPI_EXPORT extern const char* TfLiteVersion(void);

/// The supported TensorFlow Lite model file Schema version.
///
/// Returns the (major) version number of the Schema used for model
/// files that is supported by the (potentially dynamically loaded)
/// TensorFlow Lite Runtime.
///
/// Model files using schema versions different to this may not be supported by
/// the current version of the TF Lite Runtime.
TFL_CAPI_EXPORT int TfLiteSchemaVersion(void);

/// Returns a model from the provided buffer, or null on failure.
///
/// \note The caller retains ownership of the `model_data` buffer and should
/// ensure that the lifetime of the `model_data` buffer must be at least as long
/// as the lifetime of the `TfLiteModel` and of any `TfLiteInterpreter` objects
/// created from that `TfLiteModel`, and furthermore the contents of the
/// `model_data` buffer must not be modified during that time."
TFL_CAPI_EXPORT extern TfLiteModel* TfLiteModelCreate(const void* model_data,
                                                      size_t model_size);

/// Same as `TfLiteModelCreate` with customizble error reporter.
/// * `reporter` takes the provided `user_data` object, as well as a C-style
///   format string and arg list (see also vprintf).
/// * `user_data` is optional. If non-null, it is owned by the client and must
///   remain valid for the duration of the interpreter lifetime.
TFL_CAPI_EXPORT extern TfLiteModel* TfLiteModelCreateWithErrorReporter(
    const void* model_data, size_t model_size,
    void (*reporter)(void* user_data, const char* format, va_list args),
    void* user_data);

/// Returns a model from the provided file, or null on failure.
///
/// \note The file's contents must not be modified during the lifetime of the
/// `TfLiteModel` or of any `TfLiteInterpreter` objects created from that
/// `TfLiteModel`.
TFL_CAPI_EXPORT extern TfLiteModel* TfLiteModelCreateFromFile(
    const char* model_path);

/// Same as `TfLiteModelCreateFromFile` with customizble error reporter.
/// * `reporter` takes the provided `user_data` object, as well as a C-style
///   format string and arg list (see also vprintf).
/// * `user_data` is optional. If non-null, it is owned by the client and must
///   remain valid for the duration of the interpreter lifetime.
TFL_CAPI_EXPORT extern TfLiteModel* TfLiteModelCreateFromFileWithErrorReporter(
    const char* model_path,
    void (*reporter)(void* user_data, const char* format, va_list args),
    void* user_data);

/// Destroys the model instance.
TFL_CAPI_EXPORT extern void TfLiteModelDelete(TfLiteModel* model);

/// Returns a new interpreter options instances.
TFL_CAPI_EXPORT extern TfLiteInterpreterOptions*
TfLiteInterpreterOptionsCreate();

/// Creates and returns a shallow copy of an options object.
///
/// The caller is responsible for calling `TfLiteInterpreterOptionsDelete` to
/// deallocate the object pointed to by the returned pointer.
TFL_CAPI_EXPORT extern TfLiteInterpreterOptions* TfLiteInterpreterOptionsCopy(
    const TfLiteInterpreterOptions* from);

/// Destroys the interpreter options instance.
TFL_CAPI_EXPORT extern void TfLiteInterpreterOptionsDelete(
    TfLiteInterpreterOptions* options);

/// Sets the number of CPU threads to use for the interpreter.
TFL_CAPI_EXPORT extern void TfLiteInterpreterOptionsSetNumThreads(
    TfLiteInterpreterOptions* options, int32_t num_threads);

/// Adds a delegate to be applied during `TfLiteInterpreter` creation.
///
/// If delegate application fails, interpreter creation will also fail with an
/// associated error logged.
///
/// \note The caller retains ownership of the delegate and should ensure that it
/// remains valid for the duration of any created interpreter's lifetime.
///
/// If you are NOT using "TensorFlow Lite in Play Services", and NOT building
/// with `TFLITE_WITH_STABLE_ABI` or `TFLITE_USE_OPAQUE_DELEGATE` macros
/// enabled, it is possible to pass a `TfLiteDelegate*` rather than a
/// `TfLiteOpaqueDelegate*` to this function, since in those cases,
/// `TfLiteOpaqueDelegate` is just a typedef alias for `TfLiteDelegate`.
/// This is for compatibility with existing source code
/// and existing delegates.  For new delegates, it is recommended to
/// use `TfLiteOpaqueDelegate` rather than `TfLiteDelegate`.  (See
/// `TfLiteOpaqueDelegate` in tensorflow/lite/core/c/c_api_types.h.)
TFL_CAPI_EXPORT extern void TfLiteInterpreterOptionsAddDelegate(
    TfLiteInterpreterOptions* options, TfLiteOpaqueDelegate* delegate);

/// Sets a custom error reporter for interpreter execution.
///
/// * `reporter` takes the provided `user_data` object, as well as a C-style
///   format string and arg list (see also vprintf).
/// * `user_data` is optional. If non-null, it is owned by the client and must
///   remain valid for the duration of the interpreter lifetime.
TFL_CAPI_EXPORT extern void TfLiteInterpreterOptionsSetErrorReporter(
    TfLiteInterpreterOptions* options,
    void (*reporter)(void* user_data, const char* format, va_list args),
    void* user_data);

/// Adds an op registration to be applied during `TfLiteInterpreter` creation.
///
/// The `TfLiteRegistrationExternal` object is needed to implement custom op of
/// TFLite Interpreter via C API. Calling this function ensures that any
/// `TfLiteInterpreter` created with the specified `options` can execute models
/// that use the custom operator specified in `registration`.
/// Please refer https://www.tensorflow.org/lite/guide/ops_custom for custom op
/// support.
/// \note The caller retains ownership of the TfLiteRegistrationExternal object
/// and should ensure that it remains valid for the duration of any created
/// interpreter's lifetime.
/// \warning This is an experimental API and subject to change.
TFL_CAPI_EXPORT extern void TfLiteInterpreterOptionsAddRegistrationExternal(
    TfLiteInterpreterOptions* options,
    TfLiteRegistrationExternal* registration);

/// Enables users to cancel in-flight invocations with
/// `TfLiteInterpreterCancel`.
///
/// By default it is disabled and calling to `TfLiteInterpreterCancel` will
/// return kTfLiteError. See `TfLiteInterpreterCancel`.
///
/// \warning This is an experimental API and subject to change.
TFL_CAPI_EXPORT extern TfLiteStatus TfLiteInterpreterOptionsEnableCancellation(
    TfLiteInterpreterOptions* options, bool enable);

/// Returns a new interpreter using the provided model and options, or null on
/// failure.
///
/// * `model` must be a valid model instance. The caller retains ownership of
///   the object, and may destroy it (via TfLiteModelDelete) immediately after
///   creating the interpreter.  However, if the TfLiteModel was allocated with
///   TfLiteModelCreate, then the `model_data` buffer that was passed to
///   TfLiteModelCreate must outlive the lifetime of the TfLiteInterpreter
///   object that this function returns, and must not be modified during that
///   time; and if the TfLiteModel was allocated with TfLiteModelCreateFromFile,
///   then the contents of the model file must not be modified during the
///   lifetime of the TfLiteInterpreter object that this function returns.
/// * `optional_options` may be null. The caller retains ownership of the
///   object, and can safely destroy it (via TfLiteInterpreterOptionsDelete)
///   immediately after creating the interpreter.
///
/// \note The client *must* explicitly allocate tensors before attempting to
/// access input tensor data or invoke the interpreter.
TFL_CAPI_EXPORT extern TfLiteInterpreter* TfLiteInterpreterCreate(
    const TfLiteModel* model, const TfLiteInterpreterOptions* optional_options);

/// Destroys the interpreter.
TFL_CAPI_EXPORT extern void TfLiteInterpreterDelete(
    TfLiteInterpreter* interpreter);

/// Returns the number of input tensors associated with the model.
TFL_CAPI_EXPORT extern int32_t TfLiteInterpreterGetInputTensorCount(
    const TfLiteInterpreter* interpreter);

/// Returns a pointer to an array of input tensor indices.  The length of the
/// array can be obtained via a call to `TfLiteInterpreterGetInputTensorCount`.
///
/// Typically the input tensors associated with an `interpreter` would be set
/// during the initialization of the `interpreter`, through a mechanism like the
/// `InterpreterBuilder`, and remain unchanged throughout the lifetime of the
/// interpreter.  However, there are some circumstances in which the pointer may
/// not remain valid throughout the lifetime of the interpreter, because calls
/// to `SetInputs` on the interpreter invalidate the returned pointer.
///
/// The ownership of the array remains with the TFLite runtime.
TFL_CAPI_EXPORT const int* TfLiteInterpreterInputTensorIndices(
    const TfLiteInterpreter* interpreter);

/// Returns the tensor associated with the input index.
/// REQUIRES: 0 <= input_index < TfLiteInterpreterGetInputTensorCount(tensor)
TFL_CAPI_EXPORT extern TfLiteTensor* TfLiteInterpreterGetInputTensor(
    const TfLiteInterpreter* interpreter, int32_t input_index);

/// Resizes the specified input tensor.
///
/// \note After a resize, the client *must* explicitly allocate tensors before
/// attempting to access the resized tensor data or invoke the interpreter.
///
/// REQUIRES: 0 <= input_index < TfLiteInterpreterGetInputTensorCount(tensor)
///
/// This function makes a copy of the input dimensions, so the client can safely
/// deallocate `input_dims` immediately after this function returns.
TFL_CAPI_EXPORT extern TfLiteStatus TfLiteInterpreterResizeInputTensor(
    TfLiteInterpreter* interpreter, int32_t input_index, const int* input_dims,
    int32_t input_dims_size);

/// Updates allocations for all tensors, resizing dependent tensors using the
/// specified input tensor dimensionality.
///
/// This is a relatively expensive operation, and need only be called after
/// creating the graph and/or resizing any inputs.
TFL_CAPI_EXPORT extern TfLiteStatus TfLiteInterpreterAllocateTensors(
    TfLiteInterpreter* interpreter);

/// Runs inference for the loaded graph.
///
/// Before calling this function, the caller should first invoke
/// TfLiteInterpreterAllocateTensors() and should also set the values for the
/// input tensors.  After successfully calling this function, the values for the
/// output tensors will be set.
///
/// \note It is possible that the interpreter is not in a ready state to
/// evaluate (e.g., if AllocateTensors() hasn't been called, or if a
/// ResizeInputTensor() has been performed without a subsequent call to
/// AllocateTensors()).
///
///   If the (experimental!) delegate fallback option was enabled in the
///   interpreter options, then the interpreter will automatically fall back to
///   not using any delegates if execution with delegates fails. For details,
///   see TfLiteInterpreterOptionsSetEnableDelegateFallback in
///   c_api_experimental.h.
///
/// Returns one of the following status codes:
///  - kTfLiteOk: Success. Output is valid.
///  - kTfLiteDelegateError: Execution with delegates failed, due to a problem
///    with the delegate(s). If fallback was not enabled, output is invalid.
///    If fallback was enabled, this return value indicates that fallback
///    succeeded, the output is valid, and all delegates previously applied to
///    the interpreter have been undone.
///  - kTfLiteApplicationError: Same as for kTfLiteDelegateError, except that
///    the problem was not with the delegate itself, but rather was
///    due to an incompatibility between the delegate(s) and the
///    interpreter or model.
///  - kTfLiteError: Unexpected/runtime failure. Output is invalid.
TFL_CAPI_EXPORT extern TfLiteStatus TfLiteInterpreterInvoke(
    TfLiteInterpreter* interpreter);

/// Returns the number of output tensors associated with the model.
TFL_CAPI_EXPORT extern int32_t TfLiteInterpreterGetOutputTensorCount(
    const TfLiteInterpreter* interpreter);

/// Returns a pointer to an array of output tensor indices.  The length of the
/// array can be obtained via a call to `TfLiteInterpreterGetOutputTensorCount`.
///
/// Typically the output tensors associated with an `interpreter` would be set
/// during the initialization of the `interpreter`, through a mechanism like the
/// `InterpreterBuilder`, and remain unchanged throughout the lifetime of the
/// interpreter.  However, there are some circumstances in which the pointer may
/// not remain valid throughout the lifetime of the interpreter, because calls
/// to `SetOutputs` on the interpreter invalidate the returned pointer.
///
/// The ownership of the array remains with the TFLite runtime.
TFL_CAPI_EXPORT const int* TfLiteInterpreterOutputTensorIndices(
    const TfLiteInterpreter* interpreter);

/// Returns the tensor associated with the output index.
/// REQUIRES: 0 <= output_index < TfLiteInterpreterGetOutputTensorCount(tensor)
///
/// \note The shape and underlying data buffer for output tensors may be not
/// be available until after the output tensor has been both sized and
/// allocated.
/// In general, best practice is to interact with the output tensor *after*
/// calling TfLiteInterpreterInvoke().
TFL_CAPI_EXPORT extern const TfLiteTensor* TfLiteInterpreterGetOutputTensor(
    const TfLiteInterpreter* interpreter, int32_t output_index);

/// Returns modifiable access to the tensor that corresponds to the
/// specified `index` and is associated with the provided `interpreter`.
///
/// This requires the `index` to be between 0 and N - 1, where N is the
/// number of tensors in the model.
///
/// Typically the tensors associated with the `interpreter` would be set during
/// the `interpreter` initialization, through a mechanism like the
/// `InterpreterBuilder`, and remain unchanged throughout the lifetime of the
/// interpreter.  However, there are some circumstances in which the pointer may
/// not remain valid throughout the lifetime of the interpreter, because calls
/// to `AddTensors` on the interpreter invalidate the returned pointer.
///
/// Note the difference between this function and
/// `TfLiteInterpreterGetInputTensor` (or `TfLiteInterpreterGetOutputTensor` for
/// that matter): `TfLiteInterpreterGetTensor` takes an index into the array of
/// all tensors associated with the `interpreter`'s model, whereas
/// `TfLiteInterpreterGetInputTensor` takes an index into the array of input
/// tensors.
///
/// The ownership of the tensor remains with the TFLite runtime, meaning the
/// caller should not deallocate the pointer.
TFL_CAPI_EXPORT
TfLiteTensor* TfLiteInterpreterGetTensor(const TfLiteInterpreter* interpreter,
                                         int index);

/// Tries to cancel any in-flight invocation.
///
/// \note This only cancels `TfLiteInterpreterInvoke` calls that happen before
/// calling this and it does not cancel subsequent invocations.
/// \note Calling this function will also cancel any in-flight invocations of
/// SignatureRunners constructed from this interpreter.
/// Non-blocking and thread safe.
///
/// Returns kTfLiteError if cancellation is not enabled via
/// `TfLiteInterpreterOptionsEnableCancellation`.
///
/// \warning This is an experimental API and subject to change.
TFL_CAPI_EXPORT extern TfLiteStatus TfLiteInterpreterCancel(
    const TfLiteInterpreter* interpreter);

// --------------------------------------------------------------------------
// TfLiteTensor wraps data associated with a graph tensor.
//
// Note that, while the TfLiteTensor struct is not currently opaque, and its
// fields can be accessed directly, these methods are still convenient for
// language bindings. In the future the tensor struct will likely be made opaque
// in the public API.

/// Returns the type of a tensor element.
TFL_CAPI_EXPORT extern TfLiteType TfLiteTensorType(const TfLiteTensor* tensor);

/// Returns the number of dimensions that the tensor has.  Returns -1 in case
/// the 'opaque_tensor' does not have its dimensions property set.
TFL_CAPI_EXPORT extern int32_t TfLiteTensorNumDims(const TfLiteTensor* tensor);

/// Returns the length of the tensor in the "dim_index" dimension.
/// REQUIRES: 0 <= dim_index < TFLiteTensorNumDims(tensor)
TFL_CAPI_EXPORT extern int32_t TfLiteTensorDim(const TfLiteTensor* tensor,
                                               int32_t dim_index);

/// Returns the size of the underlying data in bytes.
TFL_CAPI_EXPORT extern size_t TfLiteTensorByteSize(const TfLiteTensor* tensor);

/// Returns a pointer to the underlying data buffer.
///
/// \note The result may be null if tensors have not yet been allocated, e.g.,
/// if the Tensor has just been created or resized and `TfLiteAllocateTensors()`
/// has yet to be called, or if the output tensor is dynamically sized and the
/// interpreter hasn't been invoked.
TFL_CAPI_EXPORT extern void* TfLiteTensorData(const TfLiteTensor* tensor);

/// Returns the (null-terminated) name of the tensor.
TFL_CAPI_EXPORT extern const char* TfLiteTensorName(const TfLiteTensor* tensor);

/// Returns the parameters for asymmetric quantization. The quantization
/// parameters are only valid when the tensor type is `kTfLiteUInt8` and the
/// `scale != 0`. Quantized values can be converted back to float using:
///    real_value = scale * (quantized_value - zero_point);
TFL_CAPI_EXPORT extern TfLiteQuantizationParams TfLiteTensorQuantizationParams(
    const TfLiteTensor* tensor);

/// Copies from the provided input buffer into the tensor's buffer.
/// REQUIRES: input_data_size == TfLiteTensorByteSize(tensor)
TFL_CAPI_EXPORT extern TfLiteStatus TfLiteTensorCopyFromBuffer(
    TfLiteTensor* tensor, const void* input_data, size_t input_data_size);

/// Copies to the provided output buffer from the tensor's buffer.
/// REQUIRES: output_data_size == TfLiteTensorByteSize(tensor)
TFL_CAPI_EXPORT extern TfLiteStatus TfLiteTensorCopyToBuffer(
    const TfLiteTensor* output_tensor, void* output_data,
    size_t output_data_size);

/// Returns a new TfLiteRegistrationExternal instance.
///
/// \note The caller retains ownership and should ensure that
/// the lifetime of the `TfLiteRegistrationExternal` must be at least as long as
/// the lifetime of the `TfLiteInterpreter`.
/// \warning This is an experimental API and subject to change.
TFL_CAPI_EXPORT extern TfLiteRegistrationExternal*
TfLiteRegistrationExternalCreate(TfLiteBuiltinOperator builtin_code,
                                 const char* custom_name, int version);

/// Return the builtin op code of the provided external 'registration'.
///
/// \warning This is an experimental API and subject to change.
TFL_CAPI_EXPORT extern TfLiteBuiltinOperator
TfLiteRegistrationExternalGetBuiltInCode(
    const TfLiteRegistrationExternal* registration);

/// Returns the custom name of the provided 'registration'. The returned pointer
/// will be non-null iff the op is a custom op.
///
/// \warning This is an experimental API and subject to change.
TFL_CAPI_EXPORT extern const char* TfLiteRegistrationExternalGetCustomName(
    const TfLiteRegistrationExternal* registration);

/// Destroys the TfLiteRegistrationExternal instance.
/// \warning This is an experimental API and subject to change.
TFL_CAPI_EXPORT extern void TfLiteRegistrationExternalDelete(
    TfLiteRegistrationExternal* registration);

/// Sets the initialization callback for the registration.
///
/// The callback is called to initialize the op from serialized data.
/// Please refer `init` of `TfLiteRegistration` for the detail.
/// \warning This is an experimental API and subject to change.
TFL_CAPI_EXPORT extern void TfLiteRegistrationExternalSetInit(
    TfLiteRegistrationExternal* registration,
    void* (*init)(TfLiteOpaqueContext* context, const char* buffer,
                  size_t length));

/// Sets the deallocation callback for the registration.
///
/// This callback is called to deallocate the data returned by the init
/// callback. The value passed in the `data` parameter is the value that was
/// returned by the `init` callback.
/// Please refer `free` of `TfLiteRegistration` for the detail.
/// \warning This is an experimental API and subject to change.
TFL_CAPI_EXPORT extern void TfLiteRegistrationExternalSetFree(
    TfLiteRegistrationExternal* registration,
    void (*free)(TfLiteOpaqueContext* context, void* data));

/// Sets the preparation callback for the registration.
///
/// The callback is called when the inputs of operator have been resized.
/// Please refer `prepare` of `TfLiteRegistration` for the detail.
/// \warning This is an experimental API and subject to change.
TFL_CAPI_EXPORT extern void TfLiteRegistrationExternalSetPrepare(
    TfLiteRegistrationExternal* registration,
    TfLiteStatus (*prepare)(TfLiteOpaqueContext* context,
                            TfLiteOpaqueNode* node));

/// Sets the invocation callback for the registration.
///
/// The callback is called when the operator is executed.
/// Please refer `invoke` of `TfLiteRegistration` for the detail.
/// \warning This is an experimental API and subject to change.
TFL_CAPI_EXPORT extern void TfLiteRegistrationExternalSetInvoke(
    TfLiteRegistrationExternal* registration,
    TfLiteStatus (*invoke)(TfLiteOpaqueContext* context,
                           TfLiteOpaqueNode* node));

// NOLINTEND(modernize-redundant-void-arg)

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_CORE_C_C_API_H_
