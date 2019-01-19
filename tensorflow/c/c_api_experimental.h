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

// Creates a stack of data set + iterator nodes, currently hard-coded to return
// a sequence of 3 float values <42.0, 43.0, 44.0> over 3 calls. On success,
// returns the IteratorGetNext node, which caller can run or feed into an node.
//
// TODO(hongm): Extend the API to allow customization of the nodes created.
TF_CAPI_EXPORT extern TF_Operation* TF_MakeFakeIteratorGetNextWithDatasets(
    TF_Graph* graph, TF_Status* status);

// Similar to the above API, except that the returned iterator reads the
// file based dataset from `file_path`.
// If `is_mnist` is 0, the dataset corresponds to ImageNet.
// The iterators outputs 2 tensors:
// - A float tensor of shape `batch_size` X 784 when `is_mnist` is non-zero, or
// `batch_size` X 224 X 224 X 3 otherwise.
// - An int32 tensor of shape `batch_size`
// TODO(hongm): Extend the API to allow customization of the nodes created.
TF_CAPI_EXPORT extern TF_Operation* TF_MakeFileBasedIteratorGetNextWithDatasets(
    TF_Graph* graph, const char* file_path, int batch_size,
    unsigned char is_mnist, TF_Status* status);

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

// Prints `handle` in a human readable format to standard output for debugging.
TF_CAPI_EXPORT extern void TFE_TensorHandlePrintDebugString(
    TFE_TensorHandle* handle);

TF_CAPI_EXPORT extern void TFE_OpPrintDebugString(TFE_Op* op);

typedef struct TFE_ExecuteOpNotification TFE_ExecuteOpNotification;

// Allows invoking a kernel asynchronously, and explicitly returns a
// notification that can be waited upon. This always executes the kernel in a
// new thread.
// 1. `retvals` and `num_retvals` can only be consumed after
// `TFE_ExecuteOp` returns successfully. They shouldn't be used
// if the return is unsuccessful
// 2. These new APIs cannot be used together with the TFE context level async
// support.
TF_CAPI_EXPORT extern TFE_ExecuteOpNotification* TFE_ExecuteOpInNewThread(
    TFE_Op* op, TFE_TensorHandle** retvals, int* num_retvals,
    TF_Status* status);

// Waits to complete the op execution, and cleans up the notification.
// Errors reported by op execution are set in `status`.
TF_CAPI_EXPORT extern void TFE_ExecuteOpNotificationWaitAndDelete(
    TFE_ExecuteOpNotification* notification, TF_Status* status);

TF_CAPI_EXPORT extern void TF_MakeInternalErrorStatus(TF_Status* status,
                                                      const char* errMsg);

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
    TF_DataType dtype, void* scalar, size_t len);

// Specify the server_def that enables collective ops.
// This is different to the above function in that it doesn't create remote
// contexts, and remotely executing ops is not possible. It just enables
// communication for collective ops.
TF_CAPI_EXPORT extern void TFE_EnableCollectiveOps(TFE_Context* ctx,
                                                   const void* proto,
                                                   size_t proto_len,
                                                   TF_Status* status);

// Create a symbolic tensor from the input graph node.
TF_CAPI_EXPORT extern TFE_TensorHandle* TFE_NewTensorHandleFromTFOutput(
    TF_Output t, TF_DataType data_type);

// Returns 0 if the input tensor handle represents a symbolic tensor (i.e., a
// graph node). Otherwise returns non-0.
TF_CAPI_EXPORT extern unsigned char TFE_TensorHandleIsConcrete(
    TFE_TensorHandle* handle);

// If `handle` is a symbolic tensor, return the corresponding graph node
// represented by TF_Output. Otherwise, return an error status.
TF_CAPI_EXPORT extern TF_Output TFE_GetTFOutputFromTensorHandle(
    TFE_TensorHandle* handle, TF_Status* status);

typedef struct TFE_TraceContext TFE_TraceContext;

// A trace context contains a trace graph, to which TFE_AddEagerOpToGraph()
// calls add graph nodes as a way to symbolically execute the eager ops.
//
// It also contains a hash map from concrete input tensors to symbolic
// tensors. That map will be used to create input tensors to the trace graph.
TF_CAPI_EXPORT extern TFE_TraceContext* TFE_NewTraceContext(TF_Graph* graph);

TF_CAPI_EXPORT extern void TFE_DeleteTraceContext(TFE_TraceContext* trace_ctx);

// Symbolically executes `op`, by adding a corresponding node to the graph
// associated with `trace_ctx`. This graph node outputs a set of symbolic
// tensors in `retvals` and `num_retvals`.
TF_CAPI_EXPORT extern void TFE_AddEagerOpToGraph(TFE_Op* op,
                                                 TFE_TraceContext* trace_ctx,
                                                 TFE_TensorHandle** retvals,
                                                 int* num_retvals,
                                                 TF_Status* status);

// Finalizes the trace graph and its inputs, and returns the number of inputs.
// After this call, the next two APIs can be called to iterate over the input
// tensors.
TF_CAPI_EXPORT extern int TFE_FinalizeInputTensorsFromTraceContext(
    TFE_TraceContext* trace_ctx);

TF_CAPI_EXPORT extern TF_Output TFE_GetInputGraphNodeFromTraceContext(
    TFE_TraceContext* trace_ctx, unsigned int idx);

// Each input tensor should be consumed at most once.
TF_CAPI_EXPORT extern TFE_TensorHandle*
TFE_ConsumeInputConcreteTensorFromTraceContext(TFE_TraceContext* trace_ctx,
                                               unsigned int idx);

#ifdef __cplusplus
} /* end extern "C" */
#endif

#endif  // TENSORFLOW_C_C_API_EXPERIMENTAL_H_
