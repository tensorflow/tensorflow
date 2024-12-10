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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_C_LITERT_DISPATCH_API_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_C_LITERT_DISPATCH_API_H_

#include <stddef.h>

#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_event.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer_requirements.h"
#include "tensorflow/lite/experimental/litert/vendors/c/litert_dispatch.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// /////////////////////////////////////////////////////////////////////////////

typedef LiteRtStatus (*LiteRtDispatchInitializeT)(
    const LiteRtDispatchOption* options, int num_options);

typedef LiteRtStatus (*LiteRtDispatchGetVendorIdT)(const char** vendor_id);

typedef LiteRtStatus (*LiteRtDispatchGetBuildIdT)(const char** build_id);

typedef LiteRtStatus (*LiteRtDispatchGetCapabilitiesT)(int* capabilities);

typedef LiteRtStatus (*LiteRtDispatchDeviceContextCreateT)(
    LiteRtDispatchDeviceContext* device_context);

typedef LiteRtStatus (*LiteRtDispatchDeviceContextDestroyT)(
    LiteRtDispatchDeviceContext device_context);

typedef LiteRtStatus (*LiteRtDispatchGetInputRequirementsT)(
    LiteRtDispatchInvocationContext invocation_context, int input_index,
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferRequirements* tensor_buffer_requirements);

typedef LiteRtStatus (*LiteRtDispatchGetOutputRequirementsT)(
    LiteRtDispatchInvocationContext invocation_context, int output_index,
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferRequirements* tensor_buffer_requirements);

typedef LiteRtStatus (*LiteRtDispatchRegisterTensorBufferT)(
    LiteRtDispatchDeviceContext device_context,
    LiteRtTensorBuffer tensor_buffer,
    LiteRtTensorBufferHandle* tensor_buffer_handle);

typedef LiteRtStatus (*LiteRtDispatchUnregisterTensorBufferT)(
    LiteRtDispatchDeviceContext device_context,
    LiteRtTensorBufferHandle handle);

typedef LiteRtStatus (*LiteRtDispatchInvocationContextCreateT)(
    LiteRtDispatchDeviceContext device_context,
    LiteRtDispatchExecutableType exec_type, const void* exec_bytecode,
    size_t exec_bytecode_size, const char* function_name, int num_inputs,
    int num_outputs, LiteRtDispatchInvocationContext* invocation_context);

typedef LiteRtStatus (*LiteRtDispatchInvocationContextDestroyT)(
    LiteRtDispatchInvocationContext invocation_context);

typedef LiteRtStatus (*LiteRtDispatchAttachInputT)(
    LiteRtDispatchInvocationContext invocation_context, int graph_input_index,
    LiteRtTensorBufferHandle tensor_buffer_handle);

typedef LiteRtStatus (*LiteRtDispatchAttachOutputT)(
    LiteRtDispatchInvocationContext invocation_context, int graph_output_index,
    LiteRtTensorBufferHandle tensor_buffer_handle);

typedef LiteRtStatus (*LiteRtDispatchDetachInputT)(
    LiteRtDispatchInvocationContext invocation_context, int graph_input_index,
    LiteRtTensorBufferHandle tensor_buffer_handle);

typedef LiteRtStatus (*LiteRtDispatchDetachOutputT)(
    LiteRtDispatchInvocationContext invocation_context, int graph_output_index,
    LiteRtTensorBufferHandle tensor_buffer_handle);

typedef LiteRtStatus (*LiteRtDispatchInvokeT)(
    LiteRtDispatchInvocationContext invocation_context);

typedef struct LiteRtDispatchInterface {
  LiteRtDispatchInitializeT initialize;
  LiteRtDispatchGetVendorIdT get_vendor_id;
  LiteRtDispatchGetBuildIdT get_build_id;
  LiteRtDispatchGetCapabilitiesT get_capabilities;
  LiteRtDispatchDeviceContextCreateT device_context_create;
  LiteRtDispatchDeviceContextDestroyT device_context_destroy;
  LiteRtDispatchGetInputRequirementsT get_input_requirements;
  LiteRtDispatchGetOutputRequirementsT get_output_requirements;
  LiteRtDispatchRegisterTensorBufferT register_tensor_buffer;
  LiteRtDispatchUnregisterTensorBufferT unregister_tensor_buffer;
  LiteRtDispatchInvocationContextCreateT invocation_context_create;
  LiteRtDispatchInvocationContextDestroyT invocation_context_destroy;
  LiteRtDispatchAttachInputT attach_input;
  LiteRtDispatchAttachOutputT attach_output;
  LiteRtDispatchDetachInputT detach_input;
  LiteRtDispatchDetachOutputT detach_output;
  LiteRtDispatchInvokeT invoke;
} LiteRtDispatchInterface;

// /////////////////////////////////////////////////////////////////////////////

typedef LiteRtStatus (*LiteRtDispatchAttachInputEventT)(
    LiteRtDispatchInvocationContext invocation_context, int graph_input_index,
    LiteRtEvent input_event);

typedef LiteRtStatus (*LiteRtDispatchInvokeAsyncT)(
    LiteRtDispatchInvocationContext invocation_context, int num_output_events,
    LiteRtEvent* output_events);

typedef struct LiteRtDispatchAsyncInterface {
  LiteRtDispatchAttachInputEventT attach_input_event;
  LiteRtDispatchInvokeAsyncT invoke_async;
} LiteRtDispatchAsyncInterface;

// /////////////////////////////////////////////////////////////////////////////

typedef LiteRtStatus (*LiteRtDispatchGraphCreateT)(
    LiteRtDispatchDeviceContext device_context, LiteRtDispatchGraph* graph);

typedef LiteRtStatus (*LiteRtDispatchGraphDestroyT)(LiteRtDispatchGraph graph);

typedef LiteRtStatus (*LiteRtDispatchAddNodeT)(
    LiteRtDispatchGraph graph, LiteRtDispatchNodeId node_id,
    LiteRtDispatchNodeType node_type);

typedef LiteRtStatus (*LiteRtDispatchAddEdgeT)(LiteRtDispatchGraph graph,
                                               LiteRtDispatchEdgeId edge_id);

typedef LiteRtStatus (*LiteRtDispatchConnectNodeInputT)(
    LiteRtDispatchGraph graph, LiteRtDispatchNodeId node_id, int input_index,
    LiteRtDispatchEdgeId edge_id);

typedef LiteRtStatus (*LiteRtDispatchConnectNodeOutputT)(
    LiteRtDispatchGraph graph, LiteRtDispatchNodeId node_id, int output_index,
    LiteRtDispatchEdgeId edge_id);

typedef LiteRtStatus (*LiteRtDispatchConnectGraphInputT)(
    LiteRtDispatchGraph graph, int input_index, LiteRtDispatchEdgeId edge_id);

typedef LiteRtStatus (*LiteRtDispatchConnectGraphOutputT)(
    LiteRtDispatchGraph graph, int output_index, LiteRtDispatchEdgeId edge_id);

typedef LiteRtStatus (*LiteRtDispatchLoadExecutableT)(
    LiteRtDispatchDeviceContext device_context,
    LiteRtDispatchExecutableType type, const void* bytecode_ptr,
    size_t bytecode_size, LiteRtDispatchExecutableHandle* exec_handle);

typedef LiteRtStatus (*LiteRtDispatchUnloadExecutableT)(
    LiteRtDispatchDeviceContext device_context,
    LiteRtDispatchExecutableHandle exec_handle);

typedef LiteRtStatus (*LiteRtDispatchAssignNodeFunctionT)(
    LiteRtDispatchGraph graph, LiteRtDispatchNodeId node_id,
    LiteRtDispatchExecutableHandle exec_handle, const char* function_name);

typedef LiteRtStatus (*LiteRtDispatchInvocationContextCreateFromGraphT)(
    LiteRtDispatchDeviceContext device_context, LiteRtDispatchGraph graph,
    LiteRtDispatchInvocationContext* invocation_context);

typedef LiteRtStatus (*LiteRtDispatchAnnotateGraphT)(LiteRtDispatchGraph graph,
                                                     const char* key,
                                                     const char* value);

typedef LiteRtStatus (*LiteRtDispatchAnnotateNodeT)(
    LiteRtDispatchGraph graph, LiteRtDispatchNodeId node_id, const char* key,
    const char* value);

typedef LiteRtStatus (*LiteRtDispatchAnnotateEdgeT)(
    LiteRtDispatchGraph graph, LiteRtDispatchEdgeId edge_id, const char* key,
    const char* value);

typedef struct LiteRtDispatchGraphInterface {
  LiteRtDispatchGraphCreateT graph_create;
  LiteRtDispatchGraphDestroyT graph_destroy;
  LiteRtDispatchAddNodeT add_node;
  LiteRtDispatchAddEdgeT add_edge;
  LiteRtDispatchConnectNodeInputT connect_node_input;
  LiteRtDispatchConnectNodeOutputT connect_node_output;
  LiteRtDispatchConnectGraphInputT connect_graph_input;
  LiteRtDispatchConnectGraphOutputT connect_graph_output;
  LiteRtDispatchLoadExecutableT load_executable;
  LiteRtDispatchUnloadExecutableT unload_executable;
  LiteRtDispatchAssignNodeFunctionT assign_node_function;
  LiteRtDispatchAnnotateGraphT annotate_graph;
  LiteRtDispatchAnnotateNodeT annotate_node;
  LiteRtDispatchAnnotateEdgeT annotate_edge;
  LiteRtDispatchInvocationContextCreateFromGraphT
      invocation_context_create_from_graph;
} LiteRtDispatchGraphInterface;

// /////////////////////////////////////////////////////////////////////////////

// FIXME See Vulkan and OpenCL extensions.
typedef struct LiteRtDispatchApi {
  LiteRtApiVersion version;
  LiteRtDispatchInterface* interface;
  LiteRtDispatchAsyncInterface* async_interface;
  LiteRtDispatchGraphInterface* graph_interface;
} LiteRtDispatchApi;

LiteRtStatus LiteRtDispatchGetApi(LiteRtDispatchApi* api);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_C_LITERT_DISPATCH_API_H_
