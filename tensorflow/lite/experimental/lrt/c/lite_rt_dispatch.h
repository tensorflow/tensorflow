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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LRT_C_LITE_RT_DISPATCH_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LRT_C_LITE_RT_DISPATCH_H_

#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>

#include "tensorflow/lite/experimental/lrt/c/lite_rt_common.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_event.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_model.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_tensor_buffer.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_tensor_buffer_requirements.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// /////////////////////////////////////////////////////////////////////////////
// Basic Execution API
// /////////////////////////////////////////////////////////////////////////////

#define LRT_DISPATCH_API_VERSION_MAJOR 0
#define LRT_DISPATCH_API_VERSION_MINOR 1
#define LRT_DISPATCH_API_VERSION_PATCH 0

LITE_RT_DEFINE_HANDLE(LrtDispatchDeviceContext);
LITE_RT_DEFINE_HANDLE(LrtDispatchInvocationContext);

typedef uint64_t LrtTensorBufferHandle;

typedef struct LrtDispatchApiVersion {
  int major;
  int minor;
  int patch;
} LrtDispatchApiVersion;

typedef enum LrtDispatchCapabilities {
  kLrtDispatchCapabilitiesNone = 0,
  kLrtDispatchCapabilitiesBasic = 1,  // The vendor supports the Basic API
  kLrtDispatchCapabilitiesAsync = 2,  // The vendor supports the Async API
  kLrtDispatchCapabilitiesGraph = 4,  // The vendor supports the Graph API
} LrtDispatchCapabilities;

// Types of executable that can run on the HW accelerators.
typedef enum LrtDispatchExecutableType {
  kLrtDispatchExecutableTypeUnknown = 0,
  kLrtDispatchExecutableTypeDspLibrary = 1,  // DSP library
  kLrtDispatchExecutableTypeMlModel = 2,     // Vendor-specific ML model
} LrtDispatchExecutableType;

// Initialize the Dispatch API runtime.
//
// This function should be called before calling any other Dispatch API
// functions. Parameter shared_lib_path is optional and can be NULL.
LrtStatus LrtDispatchInitialize(const char* shared_lib_path);

// Return the version of the Dispatch API runtime.
LrtStatus LrtDispatchGetApiVersion(LrtDispatchApiVersion* api_version);

// Return the vendor id of the Dispatch API runtime.
//
// This function returns a pointer to a statically allocated string that is the
// ID of vendor providing the Dispatch API runtime.
LrtStatus LrtDispatchGetVendorId(const char** vendor_id);

// Return the build ID of the Dispatch API runtime.
//
// This function returns a pointer to a statically allocated string that is the
// ID of the Dispatch API runtime build.
LrtStatus LrtDispatchGetBuildId(const char** build_id);

// Return the capabilities supported by the Dispatch API runtime as a set of the
// values specified in LrtDispatchCapabilities.
LrtStatus LrtDispatchGetCapabilities(int* capabilities);

// Create a `LrtDispatchDeviceContext` object.
//
// The returned object is used to talk with the underlying HW. The caller owns
// the memory associated with the context and should call
// LrtDispatchDeviceContextDestroy() to release it. Return NULL in case of
// error.
LrtStatus LrtDispatchDeviceContextCreate(
    LrtDispatchDeviceContext* device_context);

// Release a `LrtDispatchDeviceContext` object.
//
// The given context should be release only after releasing all associated
// objects.
LrtStatus LrtDispatchDeviceContextDestroy(
    LrtDispatchDeviceContext device_context);

// Given a tensor type for an invocation context input, obtain the attributes
// the HW requires for the associated tensor buffer. The returned
// `tensor_buffer_requirements` object is owned by the caller.
LrtStatus LrtDispatchGetInputRequirements(
    LrtDispatchInvocationContext invocation_context, int input_index,
    const LrtRankedTensorType* tensor_type,
    LrtTensorBufferRequirements* tensor_buffer_requirements);

// Given a tensor type for an invocation context output, obtain the attributes
// the HW requires for the associated tensor buffer. The returned
// `tensor_buffer_requirements` object is owned by the caller.
LrtStatus LrtDispatchGetOutputRequirements(
    LrtDispatchInvocationContext invocation_context, int output_index,
    const LrtRankedTensorType* tensor_type,
    LrtTensorBufferRequirements* tensor_buffer_requirements);

// Registers a buffer with the given device context.
// Note: The memory backing the buffer should be valid until
// `LrtDispatchUnregisterTensorBuffer` is called.
LrtStatus LrtDispatchRegisterTensorBuffer(
    LrtDispatchDeviceContext device_context, LrtTensorBuffer tensor_buffer,
    LrtTensorBufferHandle* tensor_buffer_handle);

// Unregisters the registered buffer associated with the given
// `LrtTensorBufferHandle`.
// Note: The registered `LrtTensorBufferHandle` is supposed to be unregistered
// with this function before the associated `ThrContext` is deleted by calling
// `LrtDispatchDeviceContextDestroy`.
LrtStatus LrtDispatchUnregisterTensorBuffer(
    LrtDispatchDeviceContext device_context,
    LrtTensorBufferHandle tensor_buffer_handle);

// Create an invocation context to run a given function from a given
// executable. Parameter `function_name` is required if the provided executable
// includes multiple functions.
LrtStatus LrtDispatchInvocationContextCreate(
    LrtDispatchDeviceContext device_context,
    LrtDispatchExecutableType exec_type, const void* exec_bytecode_ptr,
    size_t exec_bytecode_size, const char* function_name, int num_inputs,
    int num_outputs, LrtDispatchInvocationContext* invocation_context);

LrtStatus LrtDispatchInvocationContextDestroy(
    LrtDispatchInvocationContext invocation_context);

LrtStatus LrtDispatchAttachInput(
    LrtDispatchInvocationContext invocation_context, int graph_input_index,
    LrtTensorBufferHandle tensor_buffer_handle);

LrtStatus LrtDispatchAttachOutput(
    LrtDispatchInvocationContext invocation_context, int graph_output_index,
    LrtTensorBufferHandle tensor_buffer_handle);

LrtStatus LrtDispatchDetachInput(
    LrtDispatchInvocationContext invocation_context, int graph_input_index,
    LrtTensorBufferHandle tensor_buffer_handle);

LrtStatus LrtDispatchDetachOutput(
    LrtDispatchInvocationContext invocation_context, int graph_output_index,
    LrtTensorBufferHandle tensor_buffer_handle);

LrtStatus LrtDispatchInvoke(LrtDispatchInvocationContext invocation_context);

// /////////////////////////////////////////////////////////////////////////////
// Async Execution API
// /////////////////////////////////////////////////////////////////////////////

LrtStatus LrtDispatchAttachInputEvent(
    LrtDispatchInvocationContext invocation_context, int graph_input_index,
    LrtEvent input_event);

LrtStatus LrtDispatchInvokeAsync(
    LrtDispatchInvocationContext invocation_context, int num_output_events,
    LrtEvent* output_events);

// /////////////////////////////////////////////////////////////////////////////
// Graph Execution API
// /////////////////////////////////////////////////////////////////////////////

typedef uint64_t LrtDispatchNodeId;
typedef uint64_t LrtDispatchEdgeId;
typedef uint64_t LrtDispatchExecutableHandle;

LITE_RT_DEFINE_HANDLE(LrtDispatchGraph);

// Types of graph nodes.
typedef enum LrtDispatchNodeType {
  kLrtDispatchNodeTypeUnknown = 0,
  kLrtDispatchNodeTypeDsp = 1,  // Can execute both ML models and Dsp libraries
  kLrtDispatchNodeTypeNpu = 2,  // Can execute only ML models
} LrtDispatchNodeType;

LrtStatus LrtDispatchGraphCreate(LrtDispatchDeviceContext device_context,
                                 LrtDispatchGraph** graph);

LrtStatus LrtDispatchGraphDestroy(LrtDispatchGraph* graph);

// Add a compute node to a given graph. Parameter node_id should be unique to
// the graph.
LrtStatus LrtDispatchAddNode(LrtDispatchGraph* graph, LrtDispatchNodeId node_id,
                             LrtDispatchNodeType node_type);

// Add an edge a given graph. Parameter edge_id should be unique to the graph.
LrtStatus LrtDispatchAddEdge(LrtDispatchGraph* graph,
                             LrtDispatchEdgeId edge_id);

// Connect a given node's input.
LrtStatus LrtDispatchConnectNodeInput(LrtDispatchGraph* graph,
                                      LrtDispatchNodeId node_id,
                                      int input_index,
                                      LrtDispatchEdgeId edge_id);

// Connect a given node's output.
LrtStatus LrtDispatchConnectNodeOutput(LrtDispatchGraph* graph,
                                       LrtDispatchNodeId node_id,
                                       int output_index,
                                       LrtDispatchEdgeId edge_id);

// Connect a given graph's input.
LrtStatus LrtDispatchConnectGraphInput(LrtDispatchGraph* graph, int input_index,
                                       LrtDispatchEdgeId edge_id);

// Connect a given graph's output.
LrtStatus LrtDispatchConnectGraphOutput(LrtDispatchGraph* graph,
                                        int output_index,
                                        LrtDispatchEdgeId edge_id);

LrtStatus LrtDispatchLoadExecutable(LrtDispatchDeviceContext device_context,
                                    LrtDispatchExecutableType type,
                                    const void* bytecode, size_t bytecode_size,
                                    LrtDispatchExecutableHandle* exec_handle);

LrtStatus LrtDispatchUnloadExecutable(LrtDispatchDeviceContext device_context,
                                      LrtDispatchExecutableHandle exec_handle);

// Assign an executable function to a graph node. Parameter `function_name` is
// mandatory if the given executable includes multiple functions.
LrtStatus LrtDispatchAssignNodeFunction(LrtDispatchGraph* graph,
                                        LrtDispatchNodeId node_id,
                                        LrtDispatchExecutableHandle exec_handle,
                                        const char* function_name);

// Add an annotation to an entire graph.
LrtStatus LrtDispatchAnnotateGraph(LrtDispatchGraph* graph, const char* key,
                                   const char* value);

// Add an annotation to a specified node.
LrtStatus LrtDispatchAnnotateNode(LrtDispatchGraph* graph,
                                  LrtDispatchNodeId node_id, const char* key,
                                  const char* value);

// Add an annotation to a specified edge.
LrtStatus LrtDispatchAnnotateEdge(LrtDispatchGraph* graph,
                                  LrtDispatchEdgeId edge_id, const char* key,
                                  const char* value);

LrtStatus LrtDispatchInvocationContextCreateFromGraph(
    LrtDispatchDeviceContext device_context, LrtDispatchGraph* graph,
    LrtDispatchInvocationContext* invocation_context);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LRT_C_LITE_RT_DISPATCH_H_
