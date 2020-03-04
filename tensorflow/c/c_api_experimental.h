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

#ifndef TENSORFLOW_C_C_API_EXPERIMENTAL_H_
#define TENSORFLOW_C_C_API_EXPERIMENTAL_H_

#include <stddef.h>
#include <stdint.h>

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/eager/c_api.h"

// --------------------------------------------------------------------------
// Experimental C API for TensorFlow.
//
// The API here is subject to changes in the future.
// --------------------------------------------------------------------------

// Macro to control visibility of exported symbols in the shared library (.so,
// .dylib, .dll).
// This duplicates the TF_EXPORT macro definition in
// tensorflow/core/platform/macros.h in order to keep this .h file independent
// of any other includes.$a
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

// When `enable` is true, set
// tensorflow.ConfigProto.OptimizerOptions.global_jit_level to ON_1, and also
// set XLA flag values to prepare for XLA compilation. Otherwise set
// global_jit_level to OFF.
//
// This and the next API are syntax sugar over TF_SetConfig(), and is used by
// clients that cannot read/write the tensorflow.ConfigProto proto.
// TODO: Migrate to TF_CreateConfig() below.
TF_CAPI_EXPORT extern void TF_EnableXLACompilation(TF_SessionOptions* options,
                                                   unsigned char enable);

// Set XLA's internal BuildXlaOpsPassFlags.tf_xla_enable_lazy_compilation to the
// value of 'enabled'. Also returns the original value of that flag.
//
// Use in tests to allow XLA to fallback to TF classic. This has global effect.
TF_CAPI_EXPORT unsigned char TF_SetXlaEnableLazyCompilation(
    unsigned char enable);
TF_CAPI_EXPORT unsigned char TF_SetTfXlaCpuGlobalJit(unsigned char enable);

// Sets XLA's auto jit mode according to the specified string, which is parsed
// as if passed in XLA_FLAGS. This has global effect.
TF_CAPI_EXPORT void TF_SetXlaAutoJitMode(const char* mode);

// Sets XLA's minimum cluster size. This has global effect.
TF_CAPI_EXPORT void TF_SetXlaMinClusterSize(int size);

// Gets/Sets TF/XLA flag for whether(true) or not(false) to disable constant
// folding. This is for testing to ensure that XLA is being tested rather than
// Tensorflow's CPU implementation through constant folding.
TF_CAPI_EXPORT unsigned char TF_GetXlaConstantFoldingDisabled();
TF_CAPI_EXPORT void TF_SetXlaConstantFoldingDisabled(
    unsigned char should_enable);

// Create a serialized tensorflow.ConfigProto proto, where:
//
// a) ConfigProto.optimizer_options.global_jit_level is set to to ON_1 if
// `enable_xla_compilation` is non-zero, and OFF otherwise.
// b) ConfigProto.gpu_options.allow_growth is set to `gpu_memory_allow_growth`.
// c) ConfigProto.device_count is set to `num_cpu_devices`.
TF_CAPI_EXPORT extern TF_Buffer* TF_CreateConfig(
    unsigned char enable_xla_compilation, unsigned char gpu_memory_allow_growth,
    unsigned int num_cpu_devices);

// Create a serialized tensorflow.RunOptions proto, where RunOptions.trace_level
// is set to FULL_TRACE if `enable_full_trace` is non-zero, and NO_TRACE
// otherwise.
TF_CAPI_EXPORT extern TF_Buffer* TF_CreateRunOptions(
    unsigned char enable_full_trace);

// Returns the graph content in a human-readable format, with length set in
// `len`. The format is subject to change in the future.
// The returned string is heap-allocated, and caller should call free() on it.
TF_CAPI_EXPORT extern const char* TF_GraphDebugString(TF_Graph* graph,
                                                      size_t* len);

// Returns the function content in a human-readable format, with length set in
// `len`. The format is subject to change in the future.
// The returned string is heap-allocated, and caller should call free() on it.
//
// Do not return const char*, because some foreign language binding
// (e.g. swift) cannot then call free() on the returned pointer.
TF_CAPI_EXPORT extern char* TF_FunctionDebugString(TF_Function* func,
                                                   size_t* len);

// On success, dequeues a tensor from a TF-managed FifoQueue given by
// `tensor_id`, associated with `session`. There must be a graph node named
// "fifo_queue_dequeue_<tensor_id>", to be executed by this API call.

// Caller must call TF_DeleteTensor() over the returned tensor. If the queue is
// empty, this call is blocked.
//
// Tensors are enqueued via the corresponding TF enqueue op.
// TODO(hongm): Add support for `timeout_ms`.
TF_CAPI_EXPORT extern TF_Tensor* TF_DequeueNamedTensor(TF_Session* session,
                                                       int tensor_id,
                                                       TF_Status* status);

// On success, enqueues `tensor` into a TF-managed FifoQueue given by
// `tensor_id`, associated with `session`. There must be a graph node named
// "fifo_queue_enqueue_<tensor_id>", to be executed by this API call. It reads
// from a placeholder node "arg_tensor_enqueue_<tensor_id>".
//
// `tensor` is still owned by the caller. This call will be blocked if the queue
// has reached its capacity, and will be unblocked when the queued tensors again
// drop below the capacity due to dequeuing.
//
// Tensors are dequeued via the corresponding TF dequeue op.
// TODO(hongm): Add support for `timeout_ms`.
TF_CAPI_EXPORT extern void TF_EnqueueNamedTensor(TF_Session* session,
                                                 int tensor_id,
                                                 TF_Tensor* tensor,
                                                 TF_Status* status);
// Create a serialized tensorflow.ServerDef proto.
TF_Buffer* TFE_GetServerDef(const char* text_proto, TF_Status* status);

// TODO: remove this API in favor of the next one.
TF_CAPI_EXPORT extern TFE_Context* TFE_NewContextFromSession(
    const TFE_ContextOptions* opts, TF_Session* sess, TF_Status* status);

// Creates from `session` a new eager context to run a graph function or
// sends/recvs, so that these concurrent TFE executions can share (via
// `session` and its associated device mgr) the same set of fifo queue resource
// ops, used for host<->TF tensor transfers. This way the sends/recvs calls and
// graph function execution can access the same fifo queue resource handles
// (associated with devices managed by the device manager, which can be obtained
// from `session`).
//
// TODO: Remove this function once we migrate away from using session.
TF_CAPI_EXPORT extern TFE_Context* TFE_CreateContextFromSession(
    TF_Session* session, TF_Status* status);

// TODO: Retire this API in favor of the next one.
TF_CAPI_EXPORT extern TFE_TensorHandle* TFE_DequeueNamedTensor(
    TF_Session* session, int tensor_id, TF_DataType inputType,
    TF_Status* status);

TF_CAPI_EXPORT extern TFE_TensorHandle* TFE_DequeueNamedTensorFromCtx(
    TFE_Context* ctx, int tensor_id, TF_DataType inputType, TF_Status* status);

TF_CAPI_EXPORT extern void TFE_EnqueueNamedTensor(TF_Session* session,
                                                  int tensor_id,
                                                  TFE_TensorHandle* tensor,
                                                  TF_Status* status);

TF_CAPI_EXPORT extern void TFE_EnqueueNamedTensorFromCtx(
    TFE_Context* ctx, int tensor_id, TFE_TensorHandle* tensor,
    TF_Status* status);

// TODO: consider folding the 2 APIs below into the ones above.
TF_CAPI_EXPORT extern void TFE_EnqueueVariantTensor(TF_Session* session,
                                                    int tensor_id,
                                                    TFE_TensorHandle* tensor,
                                                    TF_Status* status);

TF_CAPI_EXPORT extern TFE_TensorHandle* TFE_DequeueVariantTensor(
    TF_Session* session, int tensor_id, TF_Status* status);

TF_CAPI_EXPORT extern void TF_MakeInternalErrorStatus(TF_Status* status,
                                                      const char* errMsg);

// TF_NewCheckpointReader() return the CheckpointReader that can be use to
// investigate or load the variable from the checkpoint file
typedef struct TF_CheckpointReader TF_CheckpointReader;
TF_CAPI_EXPORT extern TF_CheckpointReader* TF_NewCheckpointReader(
    const char* filename, TF_Status* status);
TF_CAPI_EXPORT extern void TF_DeleteCheckpointReader(
    TF_CheckpointReader* reader);
TF_CAPI_EXPORT extern int TF_CheckpointReaderHasTensor(
    TF_CheckpointReader* reader, const char* name);
// Get the variable name at the given index
TF_CAPI_EXPORT extern const char* TF_CheckpointReaderGetVariable(
    TF_CheckpointReader* reader, int index);
// Get the number of variable in the checkpoint
TF_CAPI_EXPORT extern int TF_CheckpointReaderSize(TF_CheckpointReader* reader);
// Get the DataType of a variable
TF_CAPI_EXPORT extern TF_DataType TF_CheckpointReaderGetVariableDataType(
    TF_CheckpointReader* reader, const char* name);
// Read the shape of a variable and write to `dims`
TF_CAPI_EXPORT extern void TF_CheckpointReaderGetVariableShape(
    TF_CheckpointReader* reader, const char* name, int64_t* dims, int num_dims,
    TF_Status* status);
// Get the number of dimension of a variable
TF_CAPI_EXPORT extern int TF_CheckpointReaderGetVariableNumDims(
    TF_CheckpointReader* reader, const char* name);
// Load the weight of a variable
TF_CAPI_EXPORT extern TF_Tensor* TF_CheckpointReaderGetTensor(
    TF_CheckpointReader* reader, const char* name, TF_Status* status);

// TF_NewAttrBuilder() returns an object that you can set attributes on as
// though it were an op. This allows querying properties of that op for
// type-checking purposes like if the op will run on a particular device type.
typedef struct TF_AttrBuilder TF_AttrBuilder;
TF_CAPI_EXPORT extern TF_AttrBuilder* TF_NewAttrBuilder(const char* op_name);
TF_CAPI_EXPORT extern void TF_DeleteAttrBuilder(TF_AttrBuilder* builder);
TF_CAPI_EXPORT extern void TF_AttrBuilderSetType(TF_AttrBuilder* builder,
                                                 const char* attr_name,
                                                 TF_DataType value);
TF_CAPI_EXPORT extern void TF_AttrBuilderSetTypeList(TF_AttrBuilder* builder,
                                                     const char* attr_name,
                                                     const TF_DataType* values,
                                                     int num_values);

// Checks the tensorflow::NodeDef built via the methods above to see if it can
// run on device_type.
TF_CAPI_EXPORT extern void TF_AttrBuilderCheckCanRunOnDevice(
    TF_AttrBuilder* builder, const char* device_type, TF_Status* status);

// For argument number input_index, fetch the corresponding number_attr that
// needs to be updated with the argument length of the input list.
// Returns nullptr if there is any problem like op_name is not found, or the
// argument does not support this attribute type.
TF_CAPI_EXPORT extern const char* TF_GetNumberAttrForOpListInput(
    const char* op_name, int input_index, TF_Status* status);

// Returns 1 if the op is stateful, 0 otherwise. The return value is undefined
// if the status is not ok.
TF_CAPI_EXPORT extern int TF_OpIsStateful(const char* op_type,
                                          TF_Status* status);

// Platform specific initialization routine. Very few platforms actually require
// this to be called.
TF_CAPI_EXPORT void TF_InitMain(const char* usage, int* argc, char*** argv);

// Platform-specific implementation to return an unused port. (This should used
// in tests only.)
TF_CAPI_EXPORT int TF_PickUnusedPortOrDie(void);

// Fast path method that makes constructing a single scalar tensor require less
// overhead and copies.
TF_CAPI_EXPORT extern TFE_TensorHandle* TFE_NewTensorHandleFromScalar(
    TF_DataType data_type, void* data, size_t len, TF_Status* status);

// Specify the server_def that enables collective ops.
// This is different to the above function in that it doesn't create remote
// contexts, and remotely executing ops is not possible. It just enables
// communication for collective ops.
TF_CAPI_EXPORT extern void TFE_EnableCollectiveOps(TFE_Context* ctx,
                                                   const void* proto,
                                                   size_t proto_len,
                                                   TF_Status* status);

// Information about the shape of a Tensor and its type.
struct TF_ShapeAndType {
  // Number of dimensions. -1 indicates unknown rank.
  int num_dims;
  // Array of dimensions. -1 indicates unknown dim.
  int64_t* dims;
  // The data type. May be 0 to denote unknown type.
  TF_DataType dtype;
};

typedef struct TF_ShapeAndType TF_ShapeAndType;

// A list of TF_ShapeAndType elements..
struct TF_ShapeAndTypeList {
  int num_items;
  TF_ShapeAndType* items;
};
typedef struct TF_ShapeAndTypeList TF_ShapeAndTypeList;

// API for manipulating TF_ShapeAndTypeList objects.
//
TF_CAPI_EXPORT extern TF_ShapeAndTypeList* TF_NewShapeAndTypeList(
    int num_shapes);
TF_CAPI_EXPORT extern void TF_ShapeAndTypeListSetShape(
    TF_ShapeAndTypeList* shape_list, int index, const int64_t* dims,
    int num_dims);
TF_CAPI_EXPORT extern void TF_ShapeAndTypeListSetUnknownShape(
    TF_ShapeAndTypeList* shape_list, int index);
TF_CAPI_EXPORT extern void TF_ShapeAndTypeListSetDtype(
    TF_ShapeAndTypeList* shape_list, int index, TF_DataType dtype);
TF_CAPI_EXPORT extern void TF_DeleteShapeAndTypeList(
    TF_ShapeAndTypeList* shape_list);
TF_CAPI_EXPORT extern void TF_DeleteShapeAndTypeListArray(
    TF_ShapeAndTypeList** shape_list_array, int num_items);

// Infer shapes for the given `op`. The arguments mimic the arguments of the
// `shape_inference::InferenceContext` constructor. Note the following:
//   - The inputs of the `op` are not used for shape inference. So, it is
//     OK to not have the inputs properly set in `op`. See `input_tensors`
//     if you want shape inference to consider the input tensors of the
//     op for shape inference.
//   - The types need not be set in `input_shapes` as it is not used.
//   - The number of `input_tensors` should be the same as the number of items
//     in `input_shapes`.
//
// The results are returned in `output_shapes` and
// `output_resource_shapes_and_types`. The caller is responsible for freeing the
// memory in these buffers by calling `TF_DeleteShapeAndTypeList`.
TF_CAPI_EXPORT extern void TFE_InferShapes(
    TFE_Op* op, TF_ShapeAndTypeList* input_shapes, TF_Tensor** input_tensors,
    TF_ShapeAndTypeList* input_tensor_as_shapes,
    TF_ShapeAndTypeList** input_resource_shapes_and_types,
    TF_ShapeAndTypeList** output_shapes,
    TF_ShapeAndTypeList*** output_resource_shapes_and_types, TF_Status* status);

TF_CAPI_EXPORT extern void
TF_ImportGraphDefOptionsSetValidateColocationConstraints(
    TF_ImportGraphDefOptions* opts, unsigned char enable);

#ifdef __cplusplus
} /* end extern "C" */
#endif

#endif  // TENSORFLOW_C_C_API_EXPERIMENTAL_H_
