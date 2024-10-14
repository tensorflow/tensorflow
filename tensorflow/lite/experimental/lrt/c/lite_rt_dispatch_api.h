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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LRT_C_LITE_RT_DISPATCH_API_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LRT_C_LITE_RT_DISPATCH_API_H_

#include <stdint.h>

#include "tensorflow/lite/experimental/lrt/c/lite_rt_common.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_dispatch.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_event.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_model.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_tensor_buffer.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_tensor_buffer_requirements.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// /////////////////////////////////////////////////////////////////////////////

typedef LrtStatus (*LrtDispatchInitializeT)();

typedef LrtStatus (*LrtDispatchGetVendorIdT)(const char** vendor_id);

typedef LrtStatus (*LrtDispatchGetBuildIdT)(const char** build_id);

typedef LrtStatus (*LrtDispatchGetCapabilitiesT)(int* capabilities);

typedef LrtStatus (*LrtDispatchDeviceContextCreateT)(
    LrtDispatchDeviceContext* device_context);

typedef LrtStatus (*LrtDispatchDeviceContextDestroyT)(
    LrtDispatchDeviceContext device_context);

typedef LrtStatus (*LrtDispatchGetInputRequirementsT)(
    LrtDispatchInvocationContext invocation_context, int input_index,
    const LrtRankedTensorType* tensor_type,
    LrtTensorBufferRequirements* tensor_buffer_requirements);

typedef LrtStatus (*LrtDispatchGetOutputRequirementsT)(
    LrtDispatchInvocationContext invocation_context, int output_index,
    const LrtRankedTensorType* tensor_type,
    LrtTensorBufferRequirements* tensor_buffer_requirements);

typedef LrtStatus (*LrtDispatchRegisterTensorBufferT)(
    LrtDispatchDeviceContext device_context, LrtTensorBuffer tensor_buffer,
    LrtTensorBufferHandle* tensor_buffer_handle);

typedef LrtStatus (*LrtDispatchUnregisterTensorBufferT)(
    LrtDispatchDeviceContext device_context, LrtTensorBufferHandle handle);

typedef LrtStatus (*LrtDispatchInvocationContextCreateT)(
    LrtDispatchDeviceContext device_context,
    LrtDispatchExecutableType exec_type, const void* exec_bytecode,
    size_t exec_bytecode_size, const char* function_name, int num_inputs,
    int num_outputs, LrtDispatchInvocationContext* invocation_context);

typedef LrtStatus (*LrtDispatchInvocationContextDestroyT)(
    LrtDispatchInvocationContext invocation_context);

typedef LrtStatus (*LrtDispatchAttachInputT)(
    LrtDispatchInvocationContext invocation_context, int graph_input_index,
    LrtTensorBufferHandle tensor_buffer_handle);

typedef LrtStatus (*LrtDispatchAttachOutputT)(
    LrtDispatchInvocationContext invocation_context, int graph_output_index,
    LrtTensorBufferHandle tensor_buffer_handle);

typedef LrtStatus (*LrtDispatchDetachInputT)(
    LrtDispatchInvocationContext invocation_context, int graph_input_index,
    LrtTensorBufferHandle tensor_buffer_handle);

typedef LrtStatus (*LrtDispatchDetachOutputT)(
    LrtDispatchInvocationContext invocation_context, int graph_output_index,
    LrtTensorBufferHandle tensor_buffer_handle);

typedef LrtStatus (*LrtDispatchInvokeT)(
    LrtDispatchInvocationContext invocation_context);

typedef struct LrtDispatchInterface {
  LrtDispatchInitializeT initialize;
  LrtDispatchGetVendorIdT get_vendor_id;
  LrtDispatchGetBuildIdT get_build_id;
  LrtDispatchGetCapabilitiesT get_capabilities;
  LrtDispatchDeviceContextCreateT device_context_create;
  LrtDispatchDeviceContextDestroyT device_context_destroy;
  LrtDispatchGetInputRequirementsT get_input_requirements;
  LrtDispatchGetOutputRequirementsT get_output_requirements;
  LrtDispatchRegisterTensorBufferT register_tensor_buffer;
  LrtDispatchUnregisterTensorBufferT unregister_tensor_buffer;
  LrtDispatchInvocationContextCreateT invocation_context_create;
  LrtDispatchInvocationContextDestroyT invocation_context_destroy;
  LrtDispatchAttachInputT attach_input;
  LrtDispatchAttachOutputT attach_output;
  LrtDispatchDetachInputT detach_input;
  LrtDispatchDetachOutputT detach_output;
  LrtDispatchInvokeT invoke;
} LrtDispatchInterface;

// /////////////////////////////////////////////////////////////////////////////

typedef LrtStatus (*LrtDispatchAttachInputEventT)(
    LrtDispatchInvocationContext invocation_context, int graph_input_index,
    LrtEvent input_event);

typedef LrtStatus (*LrtDispatchInvokeAsyncT)(
    LrtDispatchInvocationContext invocation_context, int num_output_events,
    LrtEvent* output_events);

typedef struct LrtDispatchAsyncInterface {
  LrtDispatchAttachInputEventT attach_input_event;
  LrtDispatchInvokeAsyncT invoke_async;
} LrtDispatchAsyncInterface;

// /////////////////////////////////////////////////////////////////////////////

typedef LrtStatus (*LrtDispatchGraphCreateT)(
    LrtDispatchDeviceContext device_context, LrtDispatchGraph* graph);

typedef LrtStatus (*LrtDispatchGraphDestroyT)(LrtDispatchGraph graph);

typedef LrtStatus (*LrtDispatchAddNodeT)(LrtDispatchGraph graph,
                                         LrtDispatchNodeId node_id,
                                         LrtDispatchNodeType node_type);

typedef LrtStatus (*LrtDispatchAddEdgeT)(LrtDispatchGraph graph,
                                         LrtDispatchEdgeId edge_id);

typedef LrtStatus (*LrtDispatchConnectNodeInputT)(LrtDispatchGraph graph,
                                                  LrtDispatchNodeId node_id,
                                                  int input_index,
                                                  LrtDispatchEdgeId edge_id);

typedef LrtStatus (*LrtDispatchConnectNodeOutputT)(LrtDispatchGraph graph,
                                                   LrtDispatchNodeId node_id,
                                                   int output_index,
                                                   LrtDispatchEdgeId edge_id);

typedef LrtStatus (*LrtDispatchConnectGraphInputT)(LrtDispatchGraph graph,
                                                   int input_index,
                                                   LrtDispatchEdgeId edge_id);

typedef LrtStatus (*LrtDispatchConnectGraphOutputT)(LrtDispatchGraph graph,
                                                    int output_index,
                                                    LrtDispatchEdgeId edge_id);

typedef LrtStatus (*LrtDispatchLoadExecutableT)(
    LrtDispatchDeviceContext device_context, LrtDispatchExecutableType type,
    const void* bytecode_ptr, size_t bytecode_size,
    LrtDispatchExecutableHandle* exec_handle);

typedef LrtStatus (*LrtDispatchUnloadExecutableT)(
    LrtDispatchDeviceContext device_context,
    LrtDispatchExecutableHandle exec_handle);

typedef LrtStatus (*LrtDispatchAssignNodeFunctionT)(
    LrtDispatchGraph graph, LrtDispatchNodeId node_id,
    LrtDispatchExecutableHandle exec_handle, const char* function_name);

typedef LrtStatus (*LrtDispatchInvocationContextCreateFromGraphT)(
    LrtDispatchDeviceContext device_context, LrtDispatchGraph graph,
    LrtDispatchInvocationContext* invocation_context);

typedef LrtStatus (*LrtDispatchAnnotateGraphT)(LrtDispatchGraph graph,
                                               const char* key,
                                               const char* value);

typedef LrtStatus (*LrtDispatchAnnotateNodeT)(LrtDispatchGraph graph,
                                              LrtDispatchNodeId node_id,
                                              const char* key,
                                              const char* value);

typedef LrtStatus (*LrtDispatchAnnotateEdgeT)(LrtDispatchGraph graph,
                                              LrtDispatchEdgeId edge_id,
                                              const char* key,
                                              const char* value);

typedef struct LrtDispatchGraphInterface {
  LrtDispatchGraphCreateT graph_create;
  LrtDispatchGraphDestroyT graph_destroy;
  LrtDispatchAddNodeT add_node;
  LrtDispatchAddEdgeT add_edge;
  LrtDispatchConnectNodeInputT connect_node_input;
  LrtDispatchConnectNodeOutputT connect_node_output;
  LrtDispatchConnectGraphInputT connect_graph_input;
  LrtDispatchConnectGraphOutputT connect_graph_output;
  LrtDispatchLoadExecutableT load_executable;
  LrtDispatchUnloadExecutableT unload_executable;
  LrtDispatchAssignNodeFunctionT assign_node_function;
  LrtDispatchAnnotateGraphT annotate_graph;
  LrtDispatchAnnotateNodeT annotate_node;
  LrtDispatchAnnotateEdgeT annotate_edge;
  LrtDispatchInvocationContextCreateFromGraphT
      invocation_context_create_from_graph;
} LrtDispatchGraphInterface;

// /////////////////////////////////////////////////////////////////////////////

// FIXME See Vulkan and OpenCL extensions.
typedef struct LrtDispatchApi {
  LrtDispatchApiVersion version;
  LrtDispatchInterface* interface;
  LrtDispatchAsyncInterface* async_interface;
  LrtDispatchGraphInterface* graph_interface;
} LrtDispatchApi;

LrtStatus LrtDispatchGetApi(LrtDispatchApi* api);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LRT_C_LITE_RT_DISPATCH_API_H_
