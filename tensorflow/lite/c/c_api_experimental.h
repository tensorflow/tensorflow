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
#ifndef TENSORFLOW_LITE_C_C_API_EXPERIMENTAL_H_
#define TENSORFLOW_LITE_C_C_API_EXPERIMENTAL_H_

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// --------------------------------------------------------------------------
// Opaque types used by the C API.

/// TfLiteSignatureRunner is used to run inference on a signature.
///
/// Note: A signature refers to a computation supported by a model, identified
/// by a distinct name, a list of named inputs and a list of named outputs. Each
/// named input/output is associated with a specific input/output tensor. A
/// model can have multiple signatures.
// To learn more about signatures in TFLite, refer to:
// https://www.tensorflow.org/lite/guide/signatures
///
/// Using the TfLiteSignatureRunner, for a particular signature, you can set its
/// inputs, invoke (i.e. execute) the computation, and retrieve its outputs.
typedef struct TfLiteSignatureRunner TfLiteSignatureRunner;

// --------------------------------------------------------------------------
/// Resets all variable tensors to zero.
///
/// WARNING: This is an experimental API and subject to change.
TFL_CAPI_EXPORT extern TfLiteStatus TfLiteInterpreterResetVariableTensors(
    TfLiteInterpreter* interpreter);

/// Adds an op registration for a builtin operator.
///
/// Op registrations are used to map ops referenced in the flatbuffer model
/// to executable function pointers (`TfLiteRegistration`s).
///
/// NOTE: The interpreter will make a shallow copy of `registration` internally,
/// so the caller should ensure that its contents (function pointers, etc...)
/// remain valid for the duration of the interpreter's lifetime. A common
/// practice is making the provided `TfLiteRegistration` instance static.
///
/// Code that uses this function should NOT call
/// `TfLiteInterpreterOptionsSetOpResolver` on the same options object.
///
/// WARNING: This is an experimental API and subject to change.
TFL_CAPI_EXPORT void TfLiteInterpreterOptionsAddBuiltinOp(
    TfLiteInterpreterOptions* options, TfLiteBuiltinOperator op,
    const TfLiteRegistration* registration, int32_t min_version,
    int32_t max_version);

/// Adds an op registration for a custom operator.
///
/// Op registrations are used to map ops referenced in the flatbuffer model
/// to executable function pointers (`TfLiteRegistration`s).
///
/// NOTE: The interpreter will make a shallow copy of `registration` internally,
/// so the caller should ensure that its contents (function pointers, etc...)
/// remain valid for the duration of any created interpreter's lifetime. A
/// common practice is making the provided `TfLiteRegistration` instance static.
///
/// The lifetime of the string pointed to by `name` must be at least as long
/// as the lifetime of the `TfLiteInterpreterOptions`.
///
/// Code that uses this function should NOT call
/// `TfLiteInterpreterOptionsSetOpResolver` on the same options object.
///
/// WARNING: This is an experimental API and subject to change.
TFL_CAPI_EXPORT void TfLiteInterpreterOptionsAddCustomOp(
    TfLiteInterpreterOptions* options, const char* name,
    const TfLiteRegistration* registration, int32_t min_version,
    int32_t max_version);

/// Registers callbacks for resolving builtin or custom operators.
///
/// The `TfLiteInterpreterOptionsSetOpResolver` function provides an alternative
/// method for registering builtin ops and/or custom ops, by providing operator
/// resolver callbacks.  Unlike using `TfLiteInterpreterOptionsAddBuiltinOp`
/// and/or `TfLiteInterpreterOptionsAddAddCustomOp`, these let you register all
/// the operators in a single call.
///
/// Code that uses this function should NOT call
/// `TfLiteInterpreterOptionsAddBuiltin` or
/// `TfLiteInterpreterOptionsAddCustomOp` on the same options object.
///
/// If `op_resolver_user_data` is non-null, its lifetime must be at least as
/// long as the lifetime of the `TfLiteInterpreterOptions`.
///
/// WARNING: This is an experimental API and subject to change.
void TfLiteInterpreterOptionsSetOpResolver(
    TfLiteInterpreterOptions* options,
    const TfLiteRegistration* (*find_builtin_op)(void* user_data,
                                                 TfLiteBuiltinOperator op,
                                                 int version),
    const TfLiteRegistration* (*find_custom_op)(void* user_data,
                                                const char* custom_op,
                                                int version),
    void* op_resolver_user_data);

/// Returns a new interpreter using the provided model and options, or null on
/// failure, where the model uses only the operators explicitly added to the
/// options.  This is the same as `TFLiteInterpreterCreate` from `c_api.h`,
/// except that the only operators that are supported are the ones registered
/// in `options` via calls to `TfLiteInterpreterOptionsSetOpResolver`,
/// `TfLiteInterpreterOptionsAddBuiltinOp`, and/or
/// `TfLiteInterpreterOptionsAddCustomOp`.
///
/// * `model` must be a valid model instance. The caller retains ownership of
///   the object, and can destroy it immediately after creating the interpreter;
///   the interpreter will maintain its own reference to the underlying model
///   data.
/// * `options` should not be null. The caller retains ownership of the object,
///   and can safely destroy it immediately after creating the interpreter.
///
/// NOTE: The client *must* explicitly allocate tensors before attempting to
/// access input tensor data or invoke the interpreter.
///
/// WARNING: This is an experimental API and subject to change.
TFL_CAPI_EXPORT extern TfLiteInterpreter*
TfLiteInterpreterCreateWithSelectedOps(const TfLiteModel* model,
                                       const TfLiteInterpreterOptions* options);

/// Enable or disable the NN API delegate for the interpreter (true to enable).
///
/// WARNING: This is an experimental API and subject to change.
TFL_CAPI_EXPORT extern void TfLiteInterpreterOptionsSetUseNNAPI(
    TfLiteInterpreterOptions* options, bool enable);

/// Enable or disable CPU fallback for the interpreter (true to enable).
/// If enabled, TfLiteInterpreterInvoke will do automatic fallback from
/// executing with delegate(s) to regular execution without delegates
/// (i.e. on CPU).
///
/// Allowing the fallback is suitable only if both of the following hold:
/// - The caller is known not to cache pointers to tensor data across
///   TfLiteInterpreterInvoke calls.
/// - The model is not stateful (no variables, no LSTMs) or the state isn't
///   needed between batches.
///
/// When delegate fallback is enabled, TfLiteInterpreterInvoke will
/// behave as follows:
///   If one or more delegates were set in the interpreter options
///   (see TfLiteInterpreterOptionsAddDelegate),
///   AND inference fails,
///   then the interpreter will fall back to not using any delegates.
///   In that case, the previously applied delegate(s) will be automatically
///   undone, and an attempt will be made to return the interpreter to an
///   invokable state, which may invalidate previous tensor addresses,
///   and the inference will be attempted again, using input tensors with
///   the same value as previously set.
///
/// WARNING: This is an experimental API and subject to change.
TFL_CAPI_EXPORT extern void TfLiteInterpreterOptionsSetEnableDelegateFallback(
    TfLiteInterpreterOptions* options, bool enable);

// Set if buffer handle output is allowed.
//
/// When using hardware delegation, Interpreter will make the data of output
/// tensors available in `tensor->data` by default. If the application can
/// consume the buffer handle directly (e.g. reading output from OpenGL
/// texture), it can set this flag to false, so Interpreter won't copy the
/// data from buffer handle to CPU memory. WARNING: This is an experimental
/// API and subject to change.
TFL_CAPI_EXPORT extern void TfLiteSetAllowBufferHandleOutput(
    const TfLiteInterpreter* interpreter, bool allow_buffer_handle_output);

/// Allow a delegate to look at the graph and modify the graph to handle
/// parts of the graph themselves. After this is called, the graph may
/// contain new nodes that replace 1 more nodes.
/// 'delegate' must outlive the interpreter.
/// Use `TfLiteInterpreterOptionsAddDelegate` instead of this unless
/// absolutely required.
/// Returns one of the following three status codes:
/// 1. kTfLiteOk: Success.
/// 2. kTfLiteDelegateError: Delegation failed due to an error in the
/// delegate. The Interpreter has been restored to its pre-delegation state.
/// NOTE: This undoes all delegates previously applied to the Interpreter.
/// 3. kTfLiteError: Unexpected/runtime failure.
/// WARNING: This is an experimental API and subject to change.
TFL_CAPI_EXPORT extern TfLiteStatus TfLiteInterpreterModifyGraphWithDelegate(
    const TfLiteInterpreter* interpreter, TfLiteDelegate* delegate);

/// Returns the tensor index corresponding to the input tensor
///
/// WARNING: This is an experimental API and subject to change.
TFL_CAPI_EXPORT extern int32_t TfLiteInterpreterGetInputTensorIndex(
    const TfLiteInterpreter* interpreter, int32_t input_index);

/// Returns the tensor index corresponding to the output tensor
///
/// WARNING: This is an experimental API and subject to change.
TFL_CAPI_EXPORT extern int32_t TfLiteInterpreterGetOutputTensorIndex(
    const TfLiteInterpreter* interpreter, int32_t output_index);

/// --------------------------------------------------------------------------
/// SignatureRunner APIs
///
/// You can run inference by either:
///
/// (i) (recommended) using the Interpreter to initialize SignatureRunner(s) and
///     then only using SignatureRunner APIs.
///
/// (ii) only using Interpreter APIs.
///
/// NOTE:
/// * Only use one of the above options to run inference, i.e, avoid mixing both
///   SignatureRunner APIs and Interpreter APIs to run inference as they share
///   the same underlying data (e.g. updating an input tensor “A” retrieved
///   using the Interpreter APIs will update the state of the input tensor “B”
///   retrieved using SignatureRunner APIs, if they point to the same underlying
///   tensor in the model; as it is not possible for a user to debug this by
///   analyzing the code, it can lead to undesirable behavior).
/// * The TfLiteSignatureRunner type is conditionally thread-safe, provided that
///   no two threads attempt to simultaneously access two TfLiteSignatureRunner
///   instances that point to the same underlying signature, or access a
///   TfLiteSignatureRunner and its underlying TfLiteInterpreter, unless all
///   such simultaneous accesses are reads (rather than writes).
/// * The lifetime of a TfLiteSignatureRunner object ends when
///   TfLiteSignatureRunnerDelete() is called on it (or when the lifetime of the
///   underlying TfLiteInterpreter ends -- but you should call
///   TfLiteSignatureRunnerDelete() before that happens in order to avoid
///   resource leaks).
/// * You can only apply delegates to the interpreter (via
///   TfLiteInterpreterOptions) and not to a signature.

/// Returns the number of signatures defined in the model.
///
/// WARNING: This is an experimental API and subject to change.
TFL_CAPI_EXPORT extern int32_t TfLiteInterpreterGetSignatureCount(
    const TfLiteInterpreter* interpreter);

/// Returns the name of the Nth signature in the model, where N is specified as
/// `signature_index`.
///
/// NOTE: The lifetime of the returned name is the same as (and depends on) the
/// lifetime of `interpreter`.
///
/// WARNING: This is an experimental API and subject to change.
TFL_CAPI_EXPORT extern const char* TfLiteInterpreterGetSignatureName(
    const TfLiteInterpreter* interpreter, int32_t signature_index);

/// Returns a new signature runner using the provided interpreter and signature
/// name, or nullptr on failure.
///
/// NOTE: `signature_name` is a null-terminated C string that must match the
/// name of a signature in the interpreter's model.
///
/// NOTE: The returned signature runner should be destroyed, by calling
/// TfLiteSignatureRunnerDelete(), before the interpreter is destroyed.
///
/// WARNING: This is an experimental API and subject to change.
TFL_CAPI_EXPORT extern TfLiteSignatureRunner*
TfLiteInterpreterGetSignatureRunner(const TfLiteInterpreter* interpreter,
                                    const char* signature_name);

/// Returns the number of inputs associated with a signature.
///
/// WARNING: This is an experimental API and subject to change.
TFL_CAPI_EXPORT extern size_t TfLiteSignatureRunnerGetInputCount(
    const TfLiteSignatureRunner* signature_runner);

/// Returns the (null-terminated) name of the Nth input in a signature, where N
/// is specified as `input_index`.
///
/// NOTE: The lifetime of the returned name is the same as (and depends on) the
/// lifetime of `signature_runner`.
///
/// WARNING: This is an experimental API and subject to change.
TFL_CAPI_EXPORT extern const char* TfLiteSignatureRunnerGetInputName(
    const TfLiteSignatureRunner* signature_runner, const int32_t input_index);

/// Resizes the input tensor identified as `input_name` to be the dimensions
/// specified by `input_dims` and `input_dims_size`. Only unknown dimensions can
/// be resized with this function. Unknown dimensions are indicated as `-1` in
/// the `dims_signature` attribute of a TfLiteTensor.
///
/// Returns status of failure or success. Note that this doesn't actually resize
/// any existing buffers. A call to TfLiteSignatureRunnerAllocateTensors() is
/// required to change the tensor input buffer.
///
/// NOTE: This function is similar to TfLiteInterpreterResizeInputTensorStrict()
/// and not TfLiteInterpreterResizeInputTensor().
///
/// NOTE: `input_name` must match the name of an input in the signature.
///
/// NOTE: This function makes a copy of the input dimensions, so the caller can
/// safely deallocate `input_dims` immediately after this function returns.
///
/// WARNING: This is an experimental API and subject to change.
TFL_CAPI_EXPORT extern TfLiteStatus TfLiteSignatureRunnerResizeInputTensor(
    TfLiteSignatureRunner* signature_runner, const char* input_name,
    const int* input_dims, int32_t input_dims_size);

/// Updates allocations for tensors associated with a signature and resizes
/// dependent tensors using the specified input tensor dimensionality.
/// This is a relatively expensive operation and hence should only be called
/// after initializing the signature runner object and/or resizing any inputs.
///
/// WARNING: This is an experimental API and subject to change.
TFL_CAPI_EXPORT extern TfLiteStatus TfLiteSignatureRunnerAllocateTensors(
    TfLiteSignatureRunner* signature_runner);

/// Returns the input tensor identified by `input_name` in the given signature.
/// Returns nullptr if the given name is not valid.
///
/// NOTE: The lifetime of the returned tensor is the same as (and depends on)
/// the lifetime of `signature_runner`.
///
/// WARNING: This is an experimental API and subject to change.
TFL_CAPI_EXPORT extern TfLiteTensor* TfLiteSignatureRunnerGetInputTensor(
    TfLiteSignatureRunner* signature_runner, const char* input_name);

/// Runs inference on a given signature.
///
/// Before calling this function, the caller should first invoke
/// TfLiteSignatureRunnerAllocateTensors() and should also set the values for
/// the input tensors. After successfully calling this function, the values for
/// the output tensors will be set.
///
/// WARNING: This is an experimental API and subject to change.
TFL_CAPI_EXPORT extern TfLiteStatus TfLiteSignatureRunnerInvoke(
    TfLiteSignatureRunner* signature_runner);

/// Returns the number of output tensors associated with the signature.
///
/// WARNING: This is an experimental API and subject to change.
TFL_CAPI_EXPORT extern size_t TfLiteSignatureRunnerGetOutputCount(
    const TfLiteSignatureRunner* signature_runner);

/// Returns the (null-terminated) name of the Nth output in a signature, where
/// N is specified as `output_index`.
///
/// NOTE: The lifetime of the returned name is the same as (and depends on) the
/// lifetime of `signature_runner`.
///
/// WARNING: This is an experimental API and subject to change.
TFL_CAPI_EXPORT extern const char* TfLiteSignatureRunnerGetOutputName(
    const TfLiteSignatureRunner* signature_runner, int32_t output_index);

/// Returns the output tensor identified by `output_name` in the given
/// signature. Returns nullptr if the given name is not valid.
///
/// NOTE: The lifetime of the returned tensor is the same as (and depends on)
/// the lifetime of `signature_runner`.
///
/// WARNING: This is an experimental API and subject to change.
TFL_CAPI_EXPORT extern const TfLiteTensor* TfLiteSignatureRunnerGetOutputTensor(
    const TfLiteSignatureRunner* signature_runner, const char* output_name);

/// Destroys the signature runner.
///
/// WARNING: This is an experimental API and subject to change.
TFL_CAPI_EXPORT extern void TfLiteSignatureRunnerDelete(
    TfLiteSignatureRunner* signature_runner);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_C_C_API_EXPERIMENTAL_H_
