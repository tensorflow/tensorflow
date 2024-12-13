// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_C_LITERT_DISPATCH_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_C_LITERT_DISPATCH_H_

#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>

#include "tensorflow/lite/experimental/litert/c/litert_any.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_event.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer_requirements.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// /////////////////////////////////////////////////////////////////////////////
// Basic Execution API
// /////////////////////////////////////////////////////////////////////////////

LITERT_DEFINE_HANDLE(LiteRtDispatchDeviceContext);
LITERT_DEFINE_HANDLE(LiteRtDispatchInvocationContext);

typedef uint64_t LiteRtTensorBufferHandle;

typedef enum LiteRtDispatchCapabilities {
  kLiteRtDispatchCapabilitiesNone = 0,
  kLiteRtDispatchCapabilitiesBasic = 1,  // The vendor supports the Basic API
  kLiteRtDispatchCapabilitiesAsync = 2,  // The vendor supports the Async API
  kLiteRtDispatchCapabilitiesGraph = 4,  // The vendor supports the Graph API
} LiteRtDispatchCapabilities;

// Types of executable that can run on the HW accelerators.
typedef enum LiteRtDispatchExecutableType {
  kLiteRtDispatchExecutableTypeUnknown = 0,
  kLiteRtDispatchExecutableTypeDspLibrary = 1,  // DSP library
  kLiteRtDispatchExecutableTypeMlModel = 2,     // Vendor-specific ML model
} LiteRtDispatchExecutableType;

typedef struct LiteRtDispatchOption {
  const char* name;
  LiteRtAny value;
} LiteRtDispatchOption;

// This option can be used to specify a directory from where to load shared
// libraries.
static const char* kDispatchOptionSharedLibraryDir = "shared_library_dir";

// Initialize the Dispatch API runtime.
//
// This function should be called before calling any other Dispatch API
// functions.
LiteRtStatus LiteRtDispatchInitialize(const LiteRtDispatchOption* options,
                                      int num_options);

// Return the version of the Dispatch API runtime.
LiteRtStatus LiteRtDispatchGetApiVersion(LiteRtApiVersion* api_version);

// Return the vendor id of the Dispatch API runtime.
//
// This function returns a pointer to a statically allocated string that is the
// ID of vendor providing the Dispatch API runtime.
LiteRtStatus LiteRtDispatchGetVendorId(const char** vendor_id);

// Return the build ID of the Dispatch API runtime.
//
// This function returns a pointer to a statically allocated string that is the
// ID of the Dispatch API runtime build.
LiteRtStatus LiteRtDispatchGetBuildId(const char** build_id);

// Return the capabilities supported by the Dispatch API runtime as a set of the
// values specified in LiteRtDispatchCapabilities.
LiteRtStatus LiteRtDispatchGetCapabilities(int* capabilities);

// Create a `LiteRtDispatchDeviceContext` object.
//
// The returned object is used to talk with the underlying HW. The caller owns
// the memory associated with the context and should call
// LiteRtDispatchDeviceContextDestroy() to release it. Return NULL in case of
// error.
LiteRtStatus LiteRtDispatchDeviceContextCreate(
    LiteRtDispatchDeviceContext* device_context);

// Release a `LiteRtDispatchDeviceContext` object.
//
// The given context should be release only after releasing all associated
// objects.
LiteRtStatus LiteRtDispatchDeviceContextDestroy(
    LiteRtDispatchDeviceContext device_context);

// Given a tensor type for an invocation context input, obtain the attributes
// the HW requires for the associated tensor buffer. The returned
// `tensor_buffer_requirements` object is owned by the caller.
LiteRtStatus LiteRtDispatchGetInputRequirements(
    LiteRtDispatchInvocationContext invocation_context, int input_index,
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferRequirements* tensor_buffer_requirements);

// Given a tensor type for an invocation context output, obtain the attributes
// the HW requires for the associated tensor buffer. The returned
// `tensor_buffer_requirements` object is owned by the caller.
LiteRtStatus LiteRtDispatchGetOutputRequirements(
    LiteRtDispatchInvocationContext invocation_context, int output_index,
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferRequirements* tensor_buffer_requirements);

// Registers a buffer with the given device context.
// Note: The memory backing the buffer should be valid until
// `LiteRtDispatchUnregisterTensorBuffer` is called.
LiteRtStatus LiteRtDispatchRegisterTensorBuffer(
    LiteRtDispatchDeviceContext device_context,
    LiteRtTensorBuffer tensor_buffer,
    LiteRtTensorBufferHandle* tensor_buffer_handle);

// Unregisters the registered buffer associated with the given
// `LiteRtTensorBufferHandle`.
// Note: The registered `LiteRtTensorBufferHandle` is supposed to be
// unregistered with this function before the associated `ThrContext` is deleted
// by calling `LiteRtDispatchDeviceContextDestroy`.
LiteRtStatus LiteRtDispatchUnregisterTensorBuffer(
    LiteRtDispatchDeviceContext device_context,
    LiteRtTensorBufferHandle tensor_buffer_handle);

// Create an invocation context to run a given function from a given
// executable. Parameter `function_name` is required if the provided executable
// includes multiple functions.
LiteRtStatus LiteRtDispatchInvocationContextCreate(
    LiteRtDispatchDeviceContext device_context,
    LiteRtDispatchExecutableType exec_type, const void* exec_bytecode_ptr,
    size_t exec_bytecode_size, const char* function_name, int num_inputs,
    int num_outputs, LiteRtDispatchInvocationContext* invocation_context);

LiteRtStatus LiteRtDispatchInvocationContextDestroy(
    LiteRtDispatchInvocationContext invocation_context);

LiteRtStatus LiteRtDispatchAttachInput(
    LiteRtDispatchInvocationContext invocation_context, int graph_input_index,
    LiteRtTensorBufferHandle tensor_buffer_handle);

LiteRtStatus LiteRtDispatchAttachOutput(
    LiteRtDispatchInvocationContext invocation_context, int graph_output_index,
    LiteRtTensorBufferHandle tensor_buffer_handle);

LiteRtStatus LiteRtDispatchDetachInput(
    LiteRtDispatchInvocationContext invocation_context, int graph_input_index,
    LiteRtTensorBufferHandle tensor_buffer_handle);

LiteRtStatus LiteRtDispatchDetachOutput(
    LiteRtDispatchInvocationContext invocation_context, int graph_output_index,
    LiteRtTensorBufferHandle tensor_buffer_handle);

LiteRtStatus LiteRtDispatchInvoke(
    LiteRtDispatchInvocationContext invocation_context);

// /////////////////////////////////////////////////////////////////////////////
// Async Execution API
// /////////////////////////////////////////////////////////////////////////////

LiteRtStatus LiteRtDispatchAttachInputEvent(
    LiteRtDispatchInvocationContext invocation_context, int graph_input_index,
    LiteRtEvent input_event);

LiteRtStatus LiteRtDispatchInvokeAsync(
    LiteRtDispatchInvocationContext invocation_context, int num_output_events,
    LiteRtEvent* output_events);

// /////////////////////////////////////////////////////////////////////////////
// Graph Execution API
// /////////////////////////////////////////////////////////////////////////////

typedef uint64_t LiteRtDispatchNodeId;
typedef uint64_t LiteRtDispatchEdgeId;
typedef uint64_t LiteRtDispatchExecutableHandle;

LITERT_DEFINE_HANDLE(LiteRtDispatchGraph);

// Types of graph nodes.
typedef enum LiteRtDispatchNodeType {
  kLiteRtDispatchNodeTypeUnknown = 0,
  kLiteRtDispatchNodeTypeDsp =
      1,  // Can execute both ML models and Dsp libraries
  kLiteRtDispatchNodeTypeNpu = 2,  // Can execute only ML models
} LiteRtDispatchNodeType;

LiteRtStatus LiteRtDispatchGraphCreate(
    LiteRtDispatchDeviceContext device_context, LiteRtDispatchGraph** graph);

LiteRtStatus LiteRtDispatchGraphDestroy(LiteRtDispatchGraph* graph);

// Add a compute node to a given graph. Parameter node_id should be unique to
// the graph.
LiteRtStatus LiteRtDispatchAddNode(LiteRtDispatchGraph* graph,
                                   LiteRtDispatchNodeId node_id,
                                   LiteRtDispatchNodeType node_type);

// Add an edge a given graph. Parameter edge_id should be unique to the graph.
LiteRtStatus LiteRtDispatchAddEdge(LiteRtDispatchGraph* graph,
                                   LiteRtDispatchEdgeId edge_id);

// Connect a given node's input.
LiteRtStatus LiteRtDispatchConnectNodeInput(LiteRtDispatchGraph* graph,
                                            LiteRtDispatchNodeId node_id,
                                            int input_index,
                                            LiteRtDispatchEdgeId edge_id);

// Connect a given node's output.
LiteRtStatus LiteRtDispatchConnectNodeOutput(LiteRtDispatchGraph* graph,
                                             LiteRtDispatchNodeId node_id,
                                             int output_index,
                                             LiteRtDispatchEdgeId edge_id);

// Connect a given graph's input.
LiteRtStatus LiteRtDispatchConnectGraphInput(LiteRtDispatchGraph* graph,
                                             int input_index,
                                             LiteRtDispatchEdgeId edge_id);

// Connect a given graph's output.
LiteRtStatus LiteRtDispatchConnectGraphOutput(LiteRtDispatchGraph* graph,
                                              int output_index,
                                              LiteRtDispatchEdgeId edge_id);

LiteRtStatus LiteRtDispatchLoadExecutable(
    LiteRtDispatchDeviceContext device_context,
    LiteRtDispatchExecutableType type, const void* bytecode,
    size_t bytecode_size, LiteRtDispatchExecutableHandle* exec_handle);

LiteRtStatus LiteRtDispatchUnloadExecutable(
    LiteRtDispatchDeviceContext device_context,
    LiteRtDispatchExecutableHandle exec_handle);

// Assign an executable function to a graph node. Parameter `function_name` is
// mandatory if the given executable includes multiple functions.
LiteRtStatus LiteRtDispatchAssignNodeFunction(
    LiteRtDispatchGraph* graph, LiteRtDispatchNodeId node_id,
    LiteRtDispatchExecutableHandle exec_handle, const char* function_name);

// Add an annotation to an entire graph.
LiteRtStatus LiteRtDispatchAnnotateGraph(LiteRtDispatchGraph* graph,
                                         const char* key, const char* value);

// Add an annotation to a specified node.
LiteRtStatus LiteRtDispatchAnnotateNode(LiteRtDispatchGraph* graph,
                                        LiteRtDispatchNodeId node_id,
                                        const char* key, const char* value);

// Add an annotation to a specified edge.
LiteRtStatus LiteRtDispatchAnnotateEdge(LiteRtDispatchGraph* graph,
                                        LiteRtDispatchEdgeId edge_id,
                                        const char* key, const char* value);

LiteRtStatus LiteRtDispatchInvocationContextCreateFromGraph(
    LiteRtDispatchDeviceContext device_context, LiteRtDispatchGraph* graph,
    LiteRtDispatchInvocationContext* invocation_context);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_C_LITERT_DISPATCH_H_
