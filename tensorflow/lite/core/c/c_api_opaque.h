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
#ifndef TENSORFLOW_LITE_CORE_C_C_API_OPAQUE_H_
#define TENSORFLOW_LITE_CORE_C_C_API_OPAQUE_H_

#include "tensorflow/lite/core/c/c_api.h"
#include "tensorflow/lite/core/c/c_api_types.h"  // IWYU pragma: export
#include "tensorflow/lite/core/c/common.h"

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

// Returns the number of dimensions that the tensor has.  Returns -1 in case
// the 'opaque_tensor' does not have its dimensions property set.
TFL_CAPI_EXPORT extern int32_t TfLiteOpaqueTensorNumDims(
    const TfLiteOpaqueTensor* opaque_tensor);

// Returns the length of the tensor in the "dim_index" dimension.
TFL_CAPI_EXPORT extern int32_t TfLiteOpaqueTensorDim(
    const TfLiteOpaqueTensor* opaque_tensor, int32_t dim_index);

// Loads into the provided 'num_dims' the number of dimensions that the tensor's
// signature has. Returns 'kTfLiteOk' if 'num_dims' was successfully loaded. Any
// other return code indicates an error and 'num_dims' won't be loaded.
//
// A tensor's dimension signature encodes shapes with unknown dimensions with
// -1.  E.g. for a tensor with three dimensions, whose first dimension has an
// unknown size, and the second and third dimension have a size of 2, the
// dimension signature is [-1,2,2], and 'TfLiteOpaqueTensorGetNumDimsSignature'
// loads 3 into 'num_dims'. If the tensor does not have its dimension signature
// field set then 'num_dims' is set to -1.
TFL_CAPI_EXPORT extern TfLiteStatus TfLiteOpaqueTensorGetNumDimsSignature(
    const TfLiteOpaqueTensor* opaque_tensor, int32_t* num_dims);

// Loads into the provided 'dim_length' the length of the tensor in the
// 'dim_index' signature dimension or -1 if that dimension has unknown length.
// Returns 'kTfLiteOk' if 'dim_length' was successfully loaded. Any
// other return code indicates an error and 'dim_length' won't be loaded.
TFL_CAPI_EXPORT extern TfLiteStatus TfLiteOpaqueTensorGetDimSignature(
    const TfLiteOpaqueTensor* opaque_tensor, int32_t dim_index,
    int32_t* dim_length);

// Returns 'non-zero' if the provided 'opaque_tensor' is a variable, and returns
// zero otherwise.
TFL_CAPI_EXPORT extern int TfLiteOpaqueTensorIsVariable(
    const TfLiteOpaqueTensor* opaque_tensor);

// Returns the size of the underlying data in bytes.
TFL_CAPI_EXPORT extern size_t TfLiteOpaqueTensorByteSize(
    const TfLiteOpaqueTensor* opaque_tensor);

// Returns a pointer to the underlying data buffer.
TFL_CAPI_EXPORT extern void* TfLiteOpaqueTensorData(
    const TfLiteOpaqueTensor* opaque_tensor);

// Returns the 'opaque_tensor's allocation type.
TFL_CAPI_EXPORT extern TfLiteAllocationType TfLiteOpaqueTensorGetAllocationType(
    const TfLiteOpaqueTensor* opaque_tensor);

// Returns the (null-terminated) name of the tensor.
TFL_CAPI_EXPORT extern const char* TfLiteOpaqueTensorName(
    const TfLiteOpaqueTensor* opaque_tensor);

// Returns the 'opaque_tensor's quantization information.
TFL_CAPI_EXPORT extern TfLiteQuantization TfLiteOpaqueTensorGetQuantization(
    const TfLiteOpaqueTensor* opaque_tensor);

// Returns the 'opaque_tensor's quantization parameters.
TFL_CAPI_EXPORT extern TfLiteQuantizationParams
TfLiteOpaqueTensorGetQuantizationParams(
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
    const TfLiteOpaqueContext* opaque_context,
    const TfLiteOpaqueNode* opaque_node, int index);

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

// Returns the builtin data associated with the provided 'opaque_node'.
//
// The builtin init data associated with a node would typically be set during
// the creation of the associated interpreter, through a mechanism like the
// interpreter builder that loads a TFLite model and initialises the
// interpreter's nodes accordingly.  Under these conditions the returned address
// remains valid throughout the lifetime of the 'opaque_node'.
TFL_CAPI_EXPORT extern void* TfLiteOpaqueNodeGetBuiltinData(
    const TfLiteOpaqueNode* opaque_node);

// Loads into the provided '*init_data' pointer the address of the custom init
// data associated with the provided 'opaque_node'.  The length of data is
// loaded into the provided 'size' pointer.  Returns 'kTfLiteOk' in case
// of success.  Any other return value indicates a failure and will leave
// 'init_data' and 'size' in an unspecified state.
//
// The custom init data associated with a node would typically be set during the
// creation of the associated interpreter, through a mechanism like the
// interpreter builder that loads a TFLite model and initialises the
// interpreter's nodes accordingly.  Under these conditions the returned address
// remains valid throughout the lifetime of the 'opaque_node'.
TFL_CAPI_EXPORT extern TfLiteStatus TfLiteOpaqueNodeGetCustomInitialData(
    const TfLiteOpaqueNode* opaque_node, const void** init_data, int* size);

// Loads into the provided '*inputs' pointer the starting address of an array
// of indices representing the tensors that are inputs of the provided
// 'opaque_node'. The length of the array is loaded into the provided
// 'num_inputs' pointer. Returns 'kTfLiteOk' in case of success.  Any other
// return value indicates a failure and will leave 'inputs' and
// 'num_inputs' in an unspecified state.
//
// The input tensors associated with a node would typically be set during the
// creation of the associated interpreter, through a mechanism like the
// interpreter builder that loads a TFLite model and initialises the
// interpreter's nodes accordingly.  Under these conditions the loaded address
// remains valid throughout the lifetime of the 'opaque_node'.
TFL_CAPI_EXPORT TfLiteStatus TfLiteOpaqueNodeInputs(
    const TfLiteOpaqueNode* opaque_node, const int** inputs, int* num_inputs);

// Loads into the provided '*outputs' pointer the starting address of an array
// of indices representing the tensors that are outputs of the provided
// 'opaque_node'. The length of the array is loaded into the provided
// 'num_outputs' pointer. Returns 'kTfLiteOk' in case of success.  Any other
// return value indicates a failure and will leave 'outputs' and
// 'num_outputs' in an unspecified state.
//
// The output tensors associated with a node would typically be set during the
// creation of the associated interpreter, through a mechanism like the
// interpreter builder that loads a TFLite model and initialises the
// interpreter's nodes accordingly.  Under these conditions the loaded address
// remains valid throughout the lifetime of the 'opaque_node'.
TFL_CAPI_EXPORT TfLiteStatus TfLiteOpaqueNodeOutputs(
    const TfLiteOpaqueNode* opaque_node, const int** outputs, int* num_outputs);

// Loads into the provided '*temporaries' pointer the starting address of an
// array of indices representing the temporary tensors associated with the
// provided 'opaque_node'. The length of the array is loaded into the provided
// 'num_temporaries' pointer. Returns 'kTfLiteOk' in case of success.  Any other
// return value indicates a failure and will leave 'temporaries' and
// 'num_temporaries' in an unspecified state.
//
// The temporary tensors associated with a node would typically be set during
// the creation of the associated interpreter, through a mechanism like the
// interpreter builder that loads a TFLite model and initialises the
// interpreter's nodes accordingly.  Under these conditions the loaded address
// remains valid throughout the lifetime of the 'opaque_node'.
TFL_CAPI_EXPORT
TfLiteStatus TfLiteOpaqueNodeTemporaries(const TfLiteOpaqueNode* opaque_node,
                                         const int** temporaries,
                                         int* num_temporaries);

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
TFL_CAPI_EXPORT TfLiteStatus
TfLiteOpaqueContextReplaceNodeSubsetsWithDelegateKernels(
    struct TfLiteOpaqueContext* opaque_context,
    TfLiteRegistrationExternal* registration_external,
    const TfLiteIntArray* nodes_to_replace,
    TfLiteOpaqueDelegate* opaque_delegate);

// Returns modifiable access to the opaque tensor that corresponds to the
// specified `index` and is associated with the provided `opaque_context`.
//
// This requires the `index` to be between 0 and N - 1, where N is the
// number of tensors in the model.
//
// Typically the tensors associated with the `context` would be set
// during the initialization of the `interpreter` that the `context` belongs to,
// through a mechanism like the `InterpreterBuilder`, and remain unchanged
// throughout the lifetime of the interpreter.  However, there are some
// circumstances in which the pointer may not remain valid throughout the
// lifetime of the interpreter, because calls to `AddTensors` on the interpreter
// invalidate the returned pointer.
//
// The ownership of the tensor remains with the TFLite runtime, meaning the
// caller should not deallocate the pointer.
TFL_CAPI_EXPORT
TfLiteOpaqueTensor* TfLiteOpaqueContextGetOpaqueTensor(
    const TfLiteOpaqueContext* opaque_context, int index);

// Loads into the provided '*inputs' pointer the starting address of an array
// of indices representing the tensors that are inputs to the subgraph that is
// associated with the provided 'opaque_context'.  The length of the array is
// loaded into the provided 'num_inputs' pointer.  Returns 'kTfLiteOk' in case
// of success.  Any other return value indicates a failure and will leave
// 'inputs' and 'num_inputs' in an unspecified state.  Calls to 'SetInputs' on
// the associated subgraph invalidate the loaded pointers.
TFL_CAPI_EXPORT
TfLiteStatus TfLiteOpaqueContextGetInputs(
    const struct TfLiteOpaqueContext* opaque_context, const int** inputs,
    int* num_inputs);

// Loads into the provided '*outputs' pointer the starting address of an array
// of indices representing the tensors that are outputs to the subgraph that is
// associated with the provided 'opaque_context'.  The length of the array is
// loaded into the provided 'num_outputs' pointer.  Returns 'kTfLiteOk' in case
// of success.  Any other return value indicates a failure and will leave
// 'outputs' and 'num_outputs' in an unspecified state.  Calls to 'SetOutputs'
// on the associated subgraph invalidate the loaded pointers.
TFL_CAPI_EXPORT
TfLiteStatus TfLiteOpaqueContextGetOutputs(
    const struct TfLiteOpaqueContext* opaque_context, const int** outputs,
    int* num_outputs);

// Loads into the provided '*variables' pointer the starting address of an array
// of indices representing the tensors that are variables to the subgraph that
// is associated with the provided 'opaque_context'.  The length of the array is
// loaded into the provided 'num_variables' pointer.  Returns 'kTfLiteOk' in
// case of success.  Any other return value indicates a failure and will leave
// 'variables' and 'num_variables' in an unspecified state.  Calls to
// 'SetVariables' on the associated subgraph invalidate the loaded pointers.
TFL_CAPI_EXPORT
TfLiteStatus TfLiteOpaqueContextGetVariables(
    const struct TfLiteOpaqueContext* opaque_context, const int** variables,
    int* num_variables);

// Returns the number of nodes associated with the provided 'opaque_context'.
TFL_CAPI_EXPORT
size_t TfLiteOpaqueContextGetNumNodes(
    const struct TfLiteOpaqueContext* opaque_context);

// Returns the number of tensors associated with the provided 'opaque_context'.
TFL_CAPI_EXPORT
size_t TfLiteOpaqueContextGetNumTensors(
    const struct TfLiteOpaqueContext* opaque_context);

// Returns the name of the subgraph that is associated with the provided
// 'opaque_context'.  Typically the returned pointer will remain valid
// throughout the lifetime of the subgraph, but may be invalidated by a call to
// 'Subgraph::SetName'.
TFL_CAPI_EXPORT
const char* TfLiteOpaqueContextGetName(
    const struct TfLiteOpaqueContext* opaque_context);

// Resizes the provided 'tensor' that is associated with the provided
// 'context' so that the 'tensor's shape matches the dimensionality specified
// via the provided 'new_size' array.  Returns 'kTfLiteOk' in
// case of success.  Any other return value indicates a failure and will leave
// the 'tensor' in an unspecified state.  The TF Lite runtime takes ownership
// of the 'new_size' array, even in case of failure.
TFL_CAPI_EXPORT
TfLiteStatus TfLiteOpaqueContextResizeTensor(TfLiteOpaqueContext* context,
                                             TfLiteOpaqueTensor* tensor,
                                             TfLiteIntArray* new_size);

// Reports an error message formed by using the provided 'format' string in
// combination with the data provided via the unnamed arguments following the
// the 'format' parameter ('...').  The intended usage and behavior is the same
// as with 'printf' with regards to how the data and the formatting string
// interact.  E.g.
// 'TfLiteOpaqueContextReportError(opaque_context, "a=%d b=%d", a, b);'
//
// The provided 'opaque_context' will be used for reporting the resulting error
// message.
//
// Note that TF Lite clients can use macros like 'TF_LITE_OPAQUE_ENSURE' to
// check for certain conditions to be true, and print an error message if the
// condition does not hold.  Direct usage of this function from application code
// should therefore be rare.
TFL_CAPI_EXPORT
void TfLiteOpaqueContextReportError(struct TfLiteOpaqueContext* opaque_context,
                                    const char* format, ...);

// Same as 'TfLiteOpaqueContextReportError', but with the variable arguments
// passed via a 'va_list' instead of directly.
//
// Callers that receive an ellipsis and want to forward it to
// to the opaque context error reporting API can add the ellipsis content to a
// 'va_list' and then call 'TfLiteOpaqueContextReportErrorVa'. E.g.:
//
// void MyErrorReporter(struct TfLiteOpaqueContext* opaque_context,
//                                     const char* format, ...) {
//   va_list vlist;
//   va_start(vlist, format);
//   TfLiteOpaqueContextReportErrorVa(opaque_context, format, vlist);
//   va_end(vlist);
// }
TFL_CAPI_EXPORT
void TfLiteOpaqueContextReportErrorVa(
    struct TfLiteOpaqueContext* opaque_context, const char* format,
    va_list vlist);

// Since we must not depend on any libraries, define a minimal subset of
// error macros while avoiding names that have pre-conceived meanings like
// assert and check.

// Try to make all reporting calls through TF_LITE_OPAQUE_KERNEL_LOG rather than
// calling the TfLiteOpaqueContextReportError function directly, so that message
// strings can be stripped out if the binary size needs to be severely
// optimized.
#ifndef TF_LITE_STRIP_ERROR_STRINGS

#if !defined(TF_LITE_OPAQUE_KERNEL_LOG)
#define TF_LITE_OPAQUE_KERNEL_LOG(opaque_context, ...)             \
  do {                                                             \
    TfLiteOpaqueContextReportError((opaque_context), __VA_ARGS__); \
  } while (false)
#endif

#if !defined(TF_LITE_OPAQUE_MAYBE_KERNEL_LOG)
#define TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(opaque_context, ...)         \
  do {                                                               \
    if ((opaque_context) != nullptr) {                               \
      TfLiteOpaqueContextReportError((opaque_context), __VA_ARGS__); \
    }                                                                \
  } while (false)
#endif

#else  // TF_LITE_STRIP_ERROR_STRINGS
#define ARGS_UNUSED(...) (void)sizeof(#__VA_ARGS__)

#if !defined(TF_LITE_OPAQUE_MAYBE_KERNEL_LOG)
#define TF_LITE_OPAQUE_KERNEL_LOG(opaque_context, ...) ARGS_UNUSED(__VA_ARGS__)
#endif

#if !defined(TF_LITE_OPAQUE_MAYBE_KERNEL_LOG)
#define TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(opaque_context, ...) \
  ARGS_UNUSED(__VA_ARGS__)
#endif

#endif  // TF_LITE_STRIP_ERROR_STRINGS

// Check whether value is true, and if not return kTfLiteError from
// the current function (and report the error string msg).
#if !defined(TF_LITE_OPAQUE_ENSURE_MSG)
#define TF_LITE_OPAQUE_ENSURE_MSG(opaque_context, value, msg)        \
  do {                                                               \
    if (!(value)) {                                                  \
      TF_LITE_OPAQUE_KERNEL_LOG((opaque_context), __FILE__ " " msg); \
      return kTfLiteError;                                           \
    }                                                                \
  } while (0)
#endif

// Check whether the value `a` is true, and if not return kTfLiteError from
// the current function, while also reporting the location of the error.
#if !defined(TF_LITE_OPAQUE_ENSURE)
#define TF_LITE_OPAQUE_ENSURE(opaque_context, a)                           \
  do {                                                                     \
    if (!(a)) {                                                            \
      TF_LITE_OPAQUE_KERNEL_LOG(opaque_context, "%s:%d: %s was not true.", \
                                __FILE__, __LINE__, #a);                   \
      return kTfLiteError;                                                 \
    }                                                                      \
  } while (0)
#endif

// Check whether the value `a == b` is true, and if not return kTfLiteError from
// the current function, while also reporting the location of the error.
// `a` and `b` may be evaluated more than once, so no side effects or
// extremely expensive computations should be done.
// NOTE: Use TF_LITE_ENSURE_TYPES_EQ if comparing TfLiteTypes.
#if !defined(TF_LITE_OPAQUE_ENSURE_EQ)
#define TF_LITE_OPAQUE_ENSURE_EQ(opaque_context, a, b)                         \
  do {                                                                         \
    if ((a) != (b)) {                                                          \
      TF_LITE_OPAQUE_KERNEL_LOG((opaque_context), "%s:%d %s != %s (%d != %d)", \
                                __FILE__, __LINE__, #a, #b, (a), (b));         \
      return kTfLiteError;                                                     \
    }                                                                          \
  } while (0)
#endif

#if !defined(TF_LITE_OPAQUE_ENSURE_TYPES_EQ)
#define TF_LITE_OPAQUE_ENSURE_TYPES_EQ(opaque_context, a, b)                   \
  do {                                                                         \
    if ((a) != (b)) {                                                          \
      TF_LITE_OPAQUE_KERNEL_LOG((opaque_context), "%s:%d %s != %s (%s != %s)", \
                                __FILE__, __LINE__, #a, #b,                    \
                                TfLiteTypeGetName(a), TfLiteTypeGetName(b));   \
      return kTfLiteError;                                                     \
    }                                                                          \
  } while (0)
#endif

#if !defined(TF_LITE_OPAQUE_ENSURE_NEAR)
#define TF_LITE_OPAQUE_ENSURE_NEAR(opaque_context, a, b, epsilon)            \
  do {                                                                       \
    auto delta = ((a) > (b)) ? ((a) - (b)) : ((b) - (a));                    \
    if (delta > epsilon) {                                                   \
      TF_LITE_OPAQUE_KERNEL_LOG(                                             \
          (opaque_context), "%s:%d %s not near %s (%f != %f)", __FILE__,     \
          __LINE__, #a, #b, static_cast<double>(a), static_cast<double>(b)); \
      return kTfLiteError;                                                   \
    }                                                                        \
  } while (0)
#endif

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_CORE_C_C_API_OPAQUE_H_
