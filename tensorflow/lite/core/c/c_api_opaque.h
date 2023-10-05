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

#include <stddef.h>

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
/// This API is part of the TensorFlow Lite Extension APIs.
/// We reserve the right to make changes to this API in future releases,
/// potentially including non-backwards-compatible changes, on a different
/// schedule than for the other TensorFlow Lite APIs. See
/// https://www.tensorflow.org/guide/versions#separate_version_number_for_tensorflow_lite_extension_apis.

/** \addtogroup c_api_opaque tensorflow/lite/c/c_api_opaque.h
 *  @{
 */

// --------------------------------------------------------------------------
// Accessors for TfLiteOpaqueTensor.

/// Returns the type of a tensor element.
TFL_CAPI_EXPORT extern TfLiteType TfLiteOpaqueTensorType(
    const TfLiteOpaqueTensor* opaque_tensor);

/// Returns the number of dimensions that the tensor has.  Returns -1 in case
/// the 'opaque_tensor' does not have its dimensions property set.
TFL_CAPI_EXPORT extern int32_t TfLiteOpaqueTensorNumDims(
    const TfLiteOpaqueTensor* opaque_tensor);

/// Returns the length of the tensor in the "dim_index" dimension.
TFL_CAPI_EXPORT extern int32_t TfLiteOpaqueTensorDim(
    const TfLiteOpaqueTensor* opaque_tensor, int32_t dim_index);

/// Loads into the provided 'num_dims' the number of dimensions that the
/// tensor's signature has. Returns 'kTfLiteOk' if 'num_dims' was successfully
/// loaded. Any other return code indicates an error and 'num_dims' won't be
/// loaded.
///
/// A tensor's dimension signature encodes shapes with unknown dimensions with
/// -1.  E.g. for a tensor with three dimensions, whose first dimension has an
/// unknown size, and the second and third dimension have a size of 2, the
/// dimension signature is [-1,2,2], and 'TfLiteOpaqueTensorGetNumDimsSignature'
/// loads 3 into 'num_dims'. If the tensor does not have its dimension signature
/// field set then 'num_dims' is set to -1.
TFL_CAPI_EXPORT extern TfLiteStatus TfLiteOpaqueTensorGetNumDimsSignature(
    const TfLiteOpaqueTensor* opaque_tensor, int32_t* num_dims);

/// Loads into the provided 'dim_length' the length of the tensor in the
/// 'dim_index' signature dimension or -1 if that dimension has unknown length.
/// Returns 'kTfLiteOk' if 'dim_length' was successfully loaded. Any
/// other return code indicates an error and 'dim_length' won't be loaded.
TFL_CAPI_EXPORT extern TfLiteStatus TfLiteOpaqueTensorGetDimSignature(
    const TfLiteOpaqueTensor* opaque_tensor, int32_t dim_index,
    int32_t* dim_length);

/// Returns 'non-zero' if the provided 'opaque_tensor' is a variable, and
/// returns zero otherwise.
TFL_CAPI_EXPORT extern int TfLiteOpaqueTensorIsVariable(
    const TfLiteOpaqueTensor* opaque_tensor);

/// Returns the size of the underlying data in bytes.
TFL_CAPI_EXPORT extern size_t TfLiteOpaqueTensorByteSize(
    const TfLiteOpaqueTensor* opaque_tensor);

/// Returns a pointer to the underlying data buffer.
TFL_CAPI_EXPORT extern void* TfLiteOpaqueTensorData(
    const TfLiteOpaqueTensor* opaque_tensor);

/// Returns the 'opaque_tensor's allocation type.
TFL_CAPI_EXPORT extern TfLiteAllocationType TfLiteOpaqueTensorGetAllocationType(
    const TfLiteOpaqueTensor* opaque_tensor);

/// Returns a tensor data allocation strategy.
TFL_CAPI_EXPORT extern TfLiteAllocationStrategy
TfLiteOpaqueTensorGetAllocationStrategy(const TfLiteOpaqueTensor* t);

/// Returns how stable a tensor data buffer address is across runs.
TFL_CAPI_EXPORT extern TfLiteRunStability
TfLiteOpaqueTensorGetBufferAddressStability(const TfLiteOpaqueTensor* t);

/// Returns how stable a tensor data values are across runs.
TFL_CAPI_EXPORT extern TfLiteRunStability TfLiteOpaqueTensorGetDataStability(
    const TfLiteOpaqueTensor* t);

/// Returns the operation step when the data of a tensor is populated.
TFL_CAPI_EXPORT extern TfLiteRunStep TfLiteOpaqueTensorGetDataKnownStep(
    const TfLiteOpaqueTensor* t);

/// Returns the operation step when the shape of a tensor is computed.
TFL_CAPI_EXPORT extern TfLiteRunStep TfLiteOpaqueTensorGetShapeKnownStep(
    const TfLiteOpaqueTensor* t);

/// Returns the (null-terminated) name of the tensor.
TFL_CAPI_EXPORT extern const char* TfLiteOpaqueTensorName(
    const TfLiteOpaqueTensor* opaque_tensor);

/// Returns the 'opaque_tensor's quantization information.
TFL_CAPI_EXPORT extern TfLiteQuantization TfLiteOpaqueTensorGetQuantization(
    const TfLiteOpaqueTensor* opaque_tensor);

/// Returns the 'opaque_tensor's quantization parameters.
TFL_CAPI_EXPORT extern TfLiteQuantizationParams
TfLiteOpaqueTensorGetQuantizationParams(
    const TfLiteOpaqueTensor* opaque_tensor);

/// Copies from the provided input buffer into the tensor's buffer.
TFL_CAPI_EXPORT extern TfLiteStatus TfLiteOpaqueTensorCopyFromBuffer(
    TfLiteOpaqueTensor* opaque_tensor, const void* input_data,
    size_t input_data_size);

/// Copies to the provided output buffer from the tensor's buffer.
TFL_CAPI_EXPORT extern TfLiteStatus TfLiteOpaqueTensorCopyToBuffer(
    const TfLiteOpaqueTensor* opaque_tensor, void* output_data,
    size_t output_data_size);

// Returns the number of strings stored in the provided 'tensor'.  Returns -1 in
// case of failure.
int TfLiteOpaqueTensorGetStringCount(const TfLiteOpaqueTensor* tensor);

// Stores the address of the n-th (denoted by the provided 'index') string
// contained in the provided 'tensor' in the provided '*str' pointer.  Stores
// the length of the string in the provided '*len' argument.
//
// Returns 'kTfLiteOk' if '*str' and '*len' have been set successfully.  Any
// other return value indicates a failure, which leaves '*str' and '*len' in an
// unspecified state.
//
// The range of valid indices is defined by the half open interval [0, N),
// where N == TfLiteOpaqueTensorGetStringCount(tensor).
//
// Note that 'str' is not guaranteed to be null-terminated. Also note that this
// function will not create a copy of the underlying string data.  The data is
// owned by the 'tensor'.
TfLiteStatus TfLiteOpaqueTensorGetString(const TfLiteOpaqueTensor* tensor,
                                         int index, const char** str, int* len);

// Writes the array of strings specified by 'str_array' into
// the specified 'tensor'.  The strings provided via the 'str_array' are being
// copied into the 'tensor'. Returns 'kTfLiteOk' in case of success.  Any other
// return value indicates a failure.
//
// The provided 'str_array_len' must denote the length of 'str_array'
// and 'str_n_len[i]' must denote the length of the i-th string.
//
// The provided strings don't need to be null terminated and may contain
// embedded null characters.  The amount of bytes copied into the 'tensor' is
// entirely determined by 'str_n_len[i]' and it is the caller's responsibility
// to set this value correctly to avoid undefined behavior.
//
// Also note that calling 'TfLiteOpaqueTensorWriteStrings' deallocates any
// previously stored data in the 'tensor'.
TfLiteStatus TfLiteOpaqueTensorWriteStrings(TfLiteOpaqueTensor* tensor,
                                            const char* const* str_array,
                                            int str_array_len,
                                            const int* str_n_len);

// Writes the string pointed to by the provided 'str' pointer of length 'len'
// into the provided 'tensor'.  The string provided via 'str' is
// copied into the 'tensor'.  Returns 'kTfLiteOk' in case of success.  Any
// other return value indicates a failure.
//
// Note that calling 'TfLiteOpaqueTensorWriteString' deallocates any
// previously stored data in the 'tensor'.  E.g. suppose 't' denotes a
// 'TfLiteOpaqueTensor*', then calling 'TfLiteOpaqueTensorWriteString(t, "AB",
// 2)' followed by a call to 'TfLiteOpaqueTensorWriteString(t, "CD", 2)' will
// lead to 't' containing 'CD', not 'ABCD'.
//
// 'TfLiteOpaqueTensorWriteString' is a convenience function for the use case
// of writing a single string to a tensor and its effects are identical to
// calling 'TfLiteOpaqueTensorWriteStrings' with an array of a single string.
TfLiteStatus TfLiteOpaqueTensorWriteString(TfLiteOpaqueTensor* tensor,
                                           const char* str, int len);

// An opaque type to create a tensor.
typedef struct TfLiteOpaqueTensorBuilder TfLiteOpaqueTensorBuilder;

// Creates an opaque tensor builder object.
TfLiteOpaqueTensorBuilder* TfLiteOpaqueTensorBuilderCreate();

// Deletes an opaque tensor builder object.
void TfLiteOpaqueTensorBuilderDelete(TfLiteOpaqueTensorBuilder* builder);

// Sets the 'TfLiteType' of the provided 'builder' to the provided 'type'.
// Returns the address of the provided 'builder', so that builder calls can be
// chained together.
TfLiteOpaqueTensorBuilder* TfLiteOpaqueTensorBuilderSetType(
    TfLiteOpaqueTensorBuilder* builder, TfLiteType type);

// Sets the raw data of the provided 'builder' to the provided 'data'. Returns
// the address of the provided 'builder', so that builder calls can be chained
// together.
TfLiteOpaqueTensorBuilder* TfLiteOpaqueTensorBuilderSetData(
    TfLiteOpaqueTensorBuilder* builder, void* data);

// Sets the allocation type of the provided 'builder' to the provided
// 'allocation_type'.  The 'allocation_type' must be one of the following:
// 'kTfLiteDynamic', 'kTfLiteArenaRw' or 'kTfLiteArenaRwPersistent'.  If the
// provided 'allocation_type' is not one of those values then
// 'TfLiteOpaqueContextAddTensor' will return an error. Returns the address of
// the provided 'builder', so that builder calls can be chained together.
TfLiteOpaqueTensorBuilder* TfLiteOpaqueTensorBuilderSetAllocationType(
    TfLiteOpaqueTensorBuilder* builder, TfLiteAllocationType allocation_type);

// Sets the quantization params of the provided 'builder' to the provided
// 'params'. Returns the address of the provided 'builder', so that builder
// calls can be chained together.
TfLiteOpaqueTensorBuilder* TfLiteOpaqueTensorBuilderSetQuantizationParams(
    TfLiteOpaqueTensorBuilder* builder, TfLiteQuantizationParams params);

// Sets the quantization of the provided 'builder' to the provided
// 'quantization'. Returns the address of the provided 'builder', so that
// builder calls can be chained together.
TfLiteOpaqueTensorBuilder* TfLiteOpaqueTensorBuilderSetQuantization(
    TfLiteOpaqueTensorBuilder* builder, TfLiteQuantization quantization);

// --------------------------------------------------------------------------
// Accessors for TfLiteOpaqueNode.

/// Returns the input tensor of the given node.
TFL_CAPI_EXPORT extern const TfLiteOpaqueTensor* TfLiteOpaqueNodeGetInput(
    const TfLiteOpaqueContext* opaque_context,
    const TfLiteOpaqueNode* opaque_node, int index);

/// Returns the output tensor of the given node.
TFL_CAPI_EXPORT extern TfLiteOpaqueTensor* TfLiteOpaqueNodeGetOutput(
    TfLiteOpaqueContext* opaque_context, const TfLiteOpaqueNode* opaque_node,
    int index);

/// Gets the number of input tensors of the provided 'opaque_node'.
TFL_CAPI_EXPORT int TfLiteOpaqueNodeNumberOfInputs(
    const TfLiteOpaqueNode* opaque_node);

/// Gets the number of output tensors of the provided 'opaque_node'.
TFL_CAPI_EXPORT int TfLiteOpaqueNodeNumberOfOutputs(
    const TfLiteOpaqueNode* opaque_node);

/// Returns opaque data provided by the node implementer. The value returned
/// from this function is the value that was returned from the `init` callback
/// that was passed to `TfLiteRegistrationExternalSetInit`.
TFL_CAPI_EXPORT extern void* TfLiteOpaqueNodeGetUserData(
    const TfLiteOpaqueNode* opaque_node);

/// Returns the builtin data associated with the provided 'opaque_node'.
///
/// The builtin init data associated with a node would typically be set during
/// the creation of the associated interpreter, through a mechanism like the
/// interpreter builder that loads a TFLite model and initialises the
/// interpreter's nodes accordingly.  Under these conditions the returned
/// address remains valid throughout the lifetime of the 'opaque_node'.
TFL_CAPI_EXPORT extern void* TfLiteOpaqueNodeGetBuiltinData(
    const TfLiteOpaqueNode* opaque_node);

/// Loads into the provided '*init_data' pointer the address of the custom init
/// data associated with the provided 'opaque_node'.  The length of data is
/// loaded into the provided 'size' pointer.  Returns 'kTfLiteOk' in case
/// of success.  Any other return value indicates a failure and will leave
/// 'init_data' and 'size' in an unspecified state.
///
/// The custom init data associated with a node would typically be set during
/// the creation of the associated interpreter, through a mechanism like the
/// interpreter builder that loads a TFLite model and initialises the
/// interpreter's nodes accordingly.  Under these conditions the returned
/// address remains valid throughout the lifetime of the 'opaque_node'.
TFL_CAPI_EXPORT extern TfLiteStatus TfLiteOpaqueNodeGetCustomInitialData(
    const TfLiteOpaqueNode* opaque_node, const void** init_data, int* size);

/// Loads into the provided '*inputs' pointer the starting address of an array
/// of indices representing the tensors that are inputs of the provided
/// 'opaque_node'. The length of the array is loaded into the provided
/// 'num_inputs' pointer. Returns 'kTfLiteOk' in case of success.  Any other
/// return value indicates a failure and will leave 'inputs' and
/// 'num_inputs' in an unspecified state.
///
/// The input tensors associated with a node would typically be set during the
/// creation of the associated interpreter, through a mechanism like the
/// interpreter builder that loads a TFLite model and initialises the
/// interpreter's nodes accordingly.  Under these conditions the loaded address
/// remains valid throughout the lifetime of the 'opaque_node'.
TFL_CAPI_EXPORT TfLiteStatus TfLiteOpaqueNodeInputs(
    const TfLiteOpaqueNode* opaque_node, const int** inputs, int* num_inputs);

/// Loads into the provided '*outputs' pointer the starting address of an array
/// of indices representing the tensors that are outputs of the provided
/// 'opaque_node'. The length of the array is loaded into the provided
/// 'num_outputs' pointer. Returns 'kTfLiteOk' in case of success.  Any other
/// return value indicates a failure and will leave 'outputs' and
/// 'num_outputs' in an unspecified state.
///
/// The output tensors associated with a node would typically be set during the
/// creation of the associated interpreter, through a mechanism like the
/// interpreter builder that loads a TFLite model and initialises the
/// interpreter's nodes accordingly.  Under these conditions the loaded address
/// remains valid throughout the lifetime of the 'opaque_node'.
TFL_CAPI_EXPORT TfLiteStatus TfLiteOpaqueNodeOutputs(
    const TfLiteOpaqueNode* opaque_node, const int** outputs, int* num_outputs);

/// Loads into the provided '*temporaries' pointer the starting address of an
/// array of indices representing the temporary tensors associated with the
/// provided 'opaque_node'. The length of the array is loaded into the provided
/// 'num_temporaries' pointer. Returns 'kTfLiteOk' in case of success.  Any
/// other return value indicates a failure and will leave 'temporaries' and
/// 'num_temporaries' in an unspecified state.
///
/// The temporary tensors associated with a node would typically be set during
/// the creation of the associated interpreter, through a mechanism like the
/// interpreter builder that loads a TFLite model and initialises the
/// interpreter's nodes accordingly.  Under these conditions the loaded address
/// remains valid throughout the lifetime of the 'opaque_node'.
TFL_CAPI_EXPORT
TfLiteStatus TfLiteOpaqueNodeTemporaries(const TfLiteOpaqueNode* opaque_node,
                                         const int** temporaries,
                                         int* num_temporaries);

// Given an 'index_of_input', which must be in the range of [0, N), where N is
// the number of input tensors of the provided 'opaque_node', returns the
// (global) index of the tensor that holds the input.  Returns -1 if
// 'index_of_input' is not within the [0, N) range.
TFL_CAPI_EXPORT
int TfLiteOpaqueNodeGetInputTensorIndex(const TfLiteOpaqueNode* opaque_node,
                                        int index_of_input);

// Given an 'index_of_output', which must be in the range of [0, N), where N is
// the number of output tensors of the provided 'opaque_node', returns the
// (global) index of the tensor that holds the output.  Returns -1 if
// 'index_of_output' is not within the [0, N) range.
TFL_CAPI_EXPORT
int TfLiteOpaqueNodeGetOutputTensorIndex(const TfLiteOpaqueNode* opaque_node,
                                         int index_of_output);

// --------------------------------------------------------------------------
// Accessors for TfLiteOpaqueContext.

typedef struct TfLiteIntArray TfLiteIntArray;

/// Loads the provided `execution_plan` associated with the provided
/// `opaque_context`.  Returns `kTfLiteOk` if the `execution_plan` was
/// successfully loaded.  A return value different from `kTfLiteOk` indicates a
/// failure and the `execution_plan` will be left in an unspecified state.
TFL_CAPI_EXPORT extern TfLiteStatus TfLiteOpaqueContextGetExecutionPlan(
    TfLiteOpaqueContext* opaque_context, TfLiteIntArray** execution_plan);

/// Given the specified 'opaque_context' and 'node_index', load the caller's
/// opaque '*node' and '*registration_external' pointer.  Return 'kTfLiteOk' if
/// both the '*node' as well as the '*registration_external' have been loaded
/// correctly.  Any other return code indicates a failure and both '*node' as
/// well as '*registration_external' will be in an unspecified state.
///
/// A caller can obtain a node's index by calling
/// 'TfLiteOpaqueContextGetExecutionPlan', which provides an array of node
/// indices, sorted in execution order.  A node index might also come from the
/// data structures passed to the delegate kernel's callback parameters, like
/// the delegate parameters data structure passed to the 'init' callback that
/// contains an array of node indices that are meant to be handled by the
/// delegate kernel.
///
/// This function is expected to be called from within a delegate callback, like
/// 'Prepare', or a delegate kernel callback (i.e., a callback registered with
/// a 'TfLiteRegistrationExternal' object).
///
/// The loaded '*node' and '*registration_external' pointers will generally
/// remain valid for the lifetime of the associated 'opaque_context', but can be
/// invalidated through API calls where delegates get un-applied, like API calls
/// that modify the model graph via a delegate, or if input tensors get
/// re-sized.
///
// TODO(b/237983452): Further clarify the lifetime guarantees of pointers that
// are returned to the users and which actions invalidate them.
TFL_CAPI_EXPORT TfLiteStatus TfLiteOpaqueContextGetNodeAndRegistration(
    struct TfLiteOpaqueContext* opaque_context, int node_index,
    TfLiteOpaqueNode** node,
    TfLiteRegistrationExternal** registration_external);

/// Entry point for C API ReplaceNodeSubsetsWithDelegateKernels
///
/// Replaces the specified `nodes_to_replace` that are associated with the
/// provided `opaque_context` with delegate kernels.  The provided
/// `registration_external` represents the delegate kernel and will be used for
/// each node subset that will be delegate to the provided `opaque_delegate`.
///
/// The TF Lite runtime will take ownership of the `registration_external` and
/// will delete it when the associated `opaque_context` gets destroyed.
///
/// The ownership of the `nodes_to_replace` and the `opaque_delegate` remains
/// with the caller.
TFL_CAPI_EXPORT TfLiteStatus
TfLiteOpaqueContextReplaceNodeSubsetsWithDelegateKernels(
    struct TfLiteOpaqueContext* opaque_context,
    TfLiteRegistrationExternal* registration_external,
    const TfLiteIntArray* nodes_to_replace,
    TfLiteOpaqueDelegate* opaque_delegate);

/// Returns modifiable access to the opaque tensor that corresponds to the
/// specified `index` and is associated with the provided `opaque_context`.
///
/// This requires the `index` to be between 0 and N - 1, where N is the
/// number of tensors in the model.
///
/// Typically the tensors associated with the `context` would be set
/// during the initialization of the `interpreter` that the `context` belongs
/// to, through a mechanism like the `InterpreterBuilder`, and remain unchanged
/// throughout the lifetime of the interpreter.  However, there are some
/// circumstances in which the pointer may not remain valid throughout the
/// lifetime of the interpreter, because calls to `AddTensors` on the
/// interpreter invalidate the returned pointer.
///
/// The ownership of the tensor remains with the TFLite runtime, meaning the
/// caller should not deallocate the pointer.
TFL_CAPI_EXPORT
TfLiteOpaqueTensor* TfLiteOpaqueContextGetOpaqueTensor(
    const TfLiteOpaqueContext* opaque_context, int index);

/// Loads into the provided '*inputs' pointer the starting address of an array
/// of indices representing the tensors that are inputs to the subgraph that is
/// associated with the provided 'opaque_context'.  The length of the array is
/// loaded into the provided 'num_inputs' pointer.  Returns 'kTfLiteOk' in case
/// of success.  Any other return value indicates a failure and will leave
/// 'inputs' and 'num_inputs' in an unspecified state.  Calls to 'SetInputs' on
/// the associated subgraph invalidate the loaded pointers.
TFL_CAPI_EXPORT
TfLiteStatus TfLiteOpaqueContextGetInputs(
    const struct TfLiteOpaqueContext* opaque_context, const int** inputs,
    int* num_inputs);

/// Loads into the provided '*outputs' pointer the starting address of an array
/// of indices representing the tensors that are outputs to the subgraph that is
/// associated with the provided 'opaque_context'.  The length of the array is
/// loaded into the provided 'num_outputs' pointer.  Returns 'kTfLiteOk' in case
/// of success.  Any other return value indicates a failure and will leave
/// 'outputs' and 'num_outputs' in an unspecified state.  Calls to 'SetOutputs'
/// on the associated subgraph invalidate the loaded pointers.
TFL_CAPI_EXPORT
TfLiteStatus TfLiteOpaqueContextGetOutputs(
    const struct TfLiteOpaqueContext* opaque_context, const int** outputs,
    int* num_outputs);

/// Loads into the provided '*variables' pointer the starting address of an
/// array of indices representing the tensors that are variables to the subgraph
/// that is associated with the provided 'opaque_context'.  The length of the
/// array is loaded into the provided 'num_variables' pointer.  Returns
/// 'kTfLiteOk' in case of success.  Any other return value indicates a failure
/// and will leave 'variables' and 'num_variables' in an unspecified state.
/// Calls to 'SetVariables' on the associated subgraph invalidate the loaded
/// pointers.
TFL_CAPI_EXPORT
TfLiteStatus TfLiteOpaqueContextGetVariables(
    const struct TfLiteOpaqueContext* opaque_context, const int** variables,
    int* num_variables);

/// Returns the number of nodes associated with the provided 'opaque_context'.
TFL_CAPI_EXPORT
size_t TfLiteOpaqueContextGetNumNodes(
    const struct TfLiteOpaqueContext* opaque_context);

/// Returns the number of tensors associated with the provided 'opaque_context'.
TFL_CAPI_EXPORT
size_t TfLiteOpaqueContextGetNumTensors(
    const struct TfLiteOpaqueContext* opaque_context);

/// Returns the name of the subgraph that is associated with the provided
/// 'opaque_context'.  Typically the returned pointer will remain valid
/// throughout the lifetime of the subgraph, but may be invalidated by a call to
/// 'Subgraph::SetName'.
TFL_CAPI_EXPORT
const char* TfLiteOpaqueContextGetName(
    const struct TfLiteOpaqueContext* opaque_context);

/// Resizes the provided 'tensor' that is associated with the provided
/// 'context' so that the 'tensor's shape matches the dimensionality specified
/// via the provided 'new_size' array.  Returns 'kTfLiteOk' in
/// case of success.  Any other return value indicates a failure and will leave
/// the 'tensor' in an unspecified state.  The TF Lite runtime takes ownership
/// of the 'new_size' array, even in case of failure.
TFL_CAPI_EXPORT
TfLiteStatus TfLiteOpaqueContextResizeTensor(TfLiteOpaqueContext* context,
                                             TfLiteOpaqueTensor* tensor,
                                             TfLiteIntArray* new_size);

/// Entry point for C API AcquireSubgraphContext.
///
/// Retrieves the corresponding TfLiteOpaqueContext of a subgraph given a
/// subgraph index and switches to the delegate context for this subgraph. If an
/// invalid subgraph index is given, then returns kTfLiteError.
/// NOTE: This function is expected to be paired with
/// TfLiteOpaqueContextReleaseSubgraphContext() once the delegate preparation is
/// done and/or the delegate context functions are no longer needed.
TFL_CAPI_EXPORT
TfLiteStatus TfLiteOpaqueContextAcquireSubgraphContext(
    struct TfLiteOpaqueContext* opaque_context, int subgraph_index,
    TfLiteOpaqueContext** acquired_opaque_context);

/// Entry point for C API ReleaseSubgraphContext.
///
/// Releases the corresponding TfLiteOpaqueContext by switching back to the
/// TFLite kernel context for this specified subgraph.
/// NOTE: This function is expected to be used after
/// TfLiteOpaqueContextAcquireSubgraphContext() once the delegate preparation is
/// done and/or the delegate context functions are no longer needed.
TFL_CAPI_EXPORT
TfLiteStatus TfLiteOpaqueContextReleaseSubgraphContext(
    struct TfLiteOpaqueContext* opaque_context, int subgraph_index);

/// Entry point for C API MarkSubgraphAsDelegationSkippable
///
/// Marks the subgraph with the given index as "delegation-skippable". Returns
/// kTfLiteOk if the given subgraph index is valid and is successfully marked
/// as delegation-skippable, and an error status if the subgraph index is
/// invalid.
/// If a subgraph is delegation-skippable, then the subgraph will be handled by
/// a specific TfLiteOpaqueDelegate that is already supposed to be
/// aware of this condition, and therefore, TfLiteInterpreter can skip invoking
/// `ModifyGraphWithDelegate` on this subgraph.
/// NOTE: This function is expected to be called only when the subgraph that
/// `subgraph_index` is pointing to should be skipped by
/// interpreter::ModifyGraphWithDelegate (e.g. the subgraph is part of the list
/// of callee subgraphs of the same control flow node, and all of those callees
/// are supported by the same delegate at once).
///
/// For  example, this function can be used when the delegate is handling
/// control flow ops such as while ops. For instance, a while op has a condition
/// subgraph indexed at `i` and a body subgraph indexed at `j`. The op can be
/// delegated when the following conditions hold:
///   1. The delegate supports while op
///   2. Both condition subgraph `i` and body subgraph `j` can be fully
///   delegated to the delegate.
/// Then if the delegate decides to support the while node along with both body
/// and condition subgraphs, it should mark subgraphs `i` and `j` skippable so
/// that those two subgraphs won't be delegated to another delegate.
/// WARNING: It is the delegate's responsibility to define when to skip
/// Subgraph::ModifyGraphWithDelegate, to check for any edge cases (i.e.
/// multiple references to the subgraph that `subgraph_index` is pointing to),
/// and to mark a subgraph as skippable by using this function.
TFL_CAPI_EXPORT
TfLiteStatus TfLiteOpaqueContextMarkSubgraphAsDelegationSkippable(
    TfLiteOpaqueContext* opaque_context, int subgraph_index);

// Loads metadata of a TF Lite node's custom initialization data.  Specifically:
// * Loads into the supplied 'fd' the file descriptor of the file that stores
//   the 'node's custom  initialization data.  This output parameter will be
//   loaded if the TF Lite runtime has access to the file descriptor, though
//   this is not always the case, e.g. if a client provides a tflite::Model
//   directly to the TF Lite runtime.  If 'fd' can be loaded then 'kTfLiteOk'
//   will be returned, otherwise 'kTfLiteError' is returned.
// * Loads into the supplied 'custom_initial_data_offset_in_file' pointer the
//   offset of the 'node's custom init data in the file associated with 'fd'.
//   This output parameter will be set to -1 if the 'node' does not have custom
//   init data set.
// * Loads into the supplied 'custom_initial_data_size' the size of the
//   custom initialization data.  This output parameter will be set to -1 if the
//   'node' does not have custom init data set.
//
// Returns 'kTfLiteOk' when 'fd' has been loaded successfully and 'kTfLiteError'
// otherwise.  Note that this means that 'kTfLiteOk' can be returned, even if
// the 'node' does not have custom init data set.
TFL_CAPI_EXPORT
TfLiteStatus TfLiteOpaqueContextGetNodeInitDataMmapInfo(
    const TfLiteOpaqueContext* context, const TfLiteOpaqueNode* node, int* fd,
    int64_t* custom_initial_data_offset_in_file,
    int64_t* custom_initial_data_size);

// Adds an additional tensor and configures its properties based on the provided
// 'builder', preserving pre-existing Tensor entries.  If non-null, the value
// pointed to by 'new_tensor_index' will be set to the index of the
// new tensor.  Returns 'kTfLiteOk' when the tensor has been added
// successfully.  Returns 'kTfLiteError' in case of failure.
TFL_CAPI_EXPORT
TfLiteStatus TfLiteOpaqueContextAddTensor(TfLiteOpaqueContext* context,
                                          TfLiteOpaqueTensorBuilder* builder,
                                          int* new_tensor_index);

// Populates the size in bytes of a provide 'type' into 'bytes'.  Returns
// 'kTfLiteOk' for valid types, and 'kTfLiteError' otherwise.
TFL_CAPI_EXPORT
TfLiteStatus TfLiteOpaqueContextGetSizeOfType(TfLiteOpaqueContext* context,
                                              TfLiteType type, size_t* bytes);

/// Reports an error message formed by using the provided 'format' string in
/// combination with the data provided via the unnamed arguments following
/// the 'format' parameter ('...').  The intended usage and behavior is the same
/// as with 'printf' with regards to how the data and the formatting string
/// interact.  E.g.
/// 'TfLiteOpaqueContextReportError(opaque_context, "a=%d b=%d", a, b);'
///
/// The provided 'opaque_context' will be used for reporting the resulting error
/// message.
///
/// Note that TF Lite clients can use macros like 'TF_LITE_OPAQUE_ENSURE' to
/// check for certain conditions to be true, and print an error message if the
/// condition does not hold.  Direct usage of this function from application
/// code should therefore be rare.
TFL_CAPI_EXPORT
void TfLiteOpaqueContextReportError(struct TfLiteOpaqueContext* opaque_context,
                                    const char* format, ...);

/// Same as 'TfLiteOpaqueContextReportError', but with the variable arguments
/// passed via a 'va_list' instead of directly.
///
/// Callers that receive an ellipsis and want to forward it to
/// to the opaque context error reporting API can add the ellipsis content to a
/// 'va_list' and then call 'TfLiteOpaqueContextReportErrorVa'. E.g.:
///
/// void MyErrorReporter(struct TfLiteOpaqueContext* opaque_context,
///                                     const char* format, ...) {
///   va_list vlist;
///   va_start(vlist, format);
///   TfLiteOpaqueContextReportErrorVa(opaque_context, format, vlist);
///   va_end(vlist);
/// }
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
#define TF_LITE_OPAQUE_ENSURE_EQ(opaque_context, a, b)                  \
  do {                                                                  \
    if ((a) != (b)) {                                                   \
      TF_LITE_OPAQUE_KERNEL_LOG((opaque_context),                       \
                                "%s:%d: %s != %s (%d != %d)", __FILE__, \
                                __LINE__, #a, #b, (a), (b));            \
      return kTfLiteError;                                              \
    }                                                                   \
  } while (0)
#endif

#if !defined(TF_LITE_OPAQUE_ENSURE_TYPES_EQ)
#define TF_LITE_OPAQUE_ENSURE_TYPES_EQ(opaque_context, a, b)                  \
  do {                                                                        \
    if ((a) != (b)) {                                                         \
      TF_LITE_OPAQUE_KERNEL_LOG(                                              \
          (opaque_context), "%s:%d: %s != %s (%s != %s)", __FILE__, __LINE__, \
          #a, #b, TfLiteTypeGetName(a), TfLiteTypeGetName(b));                \
      return kTfLiteError;                                                    \
    }                                                                         \
  } while (0)
#endif

#if !defined(TF_LITE_OPAQUE_ENSURE_NEAR)
#define TF_LITE_OPAQUE_ENSURE_NEAR(opaque_context, a, b, epsilon)             \
  do {                                                                        \
    double delta = ((a) > (b)) ? ((a) - (b)) : ((b) - (a));                   \
    if (delta > epsilon) {                                                    \
      TF_LITE_OPAQUE_KERNEL_LOG((opaque_context),                             \
                                "%s:%d: %s not near %s (%f != %f)", __FILE__, \
                                __LINE__, #a, #b, (double)(a), (double)(b));  \
      return kTfLiteError;                                                    \
    }                                                                         \
  } while (0)
#endif

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

/** @} */

#endif  // TENSORFLOW_LITE_CORE_C_C_API_OPAQUE_H_
