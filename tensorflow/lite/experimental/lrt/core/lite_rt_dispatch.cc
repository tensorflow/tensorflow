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

#include "tensorflow/lite/experimental/lrt/c/lite_rt_dispatch.h"

#include <dlfcn.h>
#include <fcntl.h>

#include <cstddef>

#include "tensorflow/lite/experimental/lrt/c/lite_rt_common.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_dispatch_api.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_event.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_model.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_tensor_buffer.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_tensor_buffer_requirements.h"
#include "tensorflow/lite/experimental/lrt/core/logging.h"

#define INVOKE_FUNC(function, ...)                              \
  if (!TheApi.interface) {                                      \
    LITE_RT_LOG(LRT_ERROR, "Dispatch API interface not found"); \
    return kLrtStatusErrorRuntimeFailure;                       \
  }                                                             \
  if (!TheApi.interface->function) {                            \
    LITE_RT_LOG(LRT_ERROR, #function " not found");             \
    return kLrtStatusErrorRuntimeFailure;                       \
  }                                                             \
  return TheApi.interface->function(__VA_ARGS__);

#define INVOKE_ASYNC_FUNC(function, ...)                              \
  if (!TheApi.async_interface) {                                      \
    LITE_RT_LOG(LRT_ERROR, "Dispatch API async interface not found"); \
    return kLrtStatusErrorRuntimeFailure;                             \
  }                                                                   \
  if (!TheApi.async_interface->function) {                            \
    LITE_RT_LOG(LRT_ERROR, #function " not found");                   \
    return kLrtStatusErrorRuntimeFailure;                             \
  }                                                                   \
  return TheApi.async_interface->function(__VA_ARGS__);

#define INVOKE_GRAPH_FUNC(function, ...)                              \
  if (!TheApi.graph_interface) {                                      \
    LITE_RT_LOG(LRT_ERROR, "Dispatch API graoh interface not found"); \
    return kLrtStatusErrorRuntimeFailure;                             \
  }                                                                   \
  if (!TheApi.graph_interface->function) {                            \
    LITE_RT_LOG(LRT_ERROR, #function " not found");                   \
    return kLrtStatusErrorRuntimeFailure;                             \
  }                                                                   \
  return TheApi.graph_interface->function(__VA_ARGS__);

namespace {

constexpr const char* kSharedLibPath = "libLrtDispatch.so";

LrtDispatchApi TheApi = {
    /*.version=*/{/*.major=*/0, /*.minor=*/0, /*.patch=*/0},
    /*.interface=*/nullptr,
    /*.async_interface=*/nullptr,
    /*.graph_interface=*/nullptr,
};

}  // namespace

// /////////////////////////////////////////////////////////////////////////////
// Basic Execution API
// /////////////////////////////////////////////////////////////////////////////

LrtStatus LrtDispatchInitialize(const char* shared_lib_path) {
  if (!shared_lib_path) {
    shared_lib_path = kSharedLibPath;
  }

  void* lib_handle = ::dlopen(shared_lib_path, RTLD_NOW | RTLD_LOCAL);
  if (!lib_handle) {
    LITE_RT_LOG(LRT_ERROR, "Failed to load dispatch library: %s", ::dlerror());
    return kLrtStatusErrorRuntimeFailure;
  }

  using LrtDispatchGetApi_t = LrtStatus (*)(LrtDispatchApi*);
  auto LrtDispatchGetApi = reinterpret_cast<LrtDispatchGetApi_t>(
      ::dlsym(lib_handle, "LrtDispatchGetApi"));
  if (!LrtDispatchGetApi) {
    LITE_RT_LOG(LRT_ERROR, "LrtDispatchGetApi not found");
    return kLrtStatusErrorRuntimeFailure;
  }

  if (auto status = LrtDispatchGetApi(&TheApi); status != kLrtStatusOk) {
    return status;
  }

  if (!(TheApi.version.major == LRT_DISPATCH_API_VERSION_MAJOR &&
        TheApi.version.minor <= LRT_DISPATCH_API_VERSION_MINOR)) {
    LITE_RT_LOG(LRT_ERROR,
                "Dispatch API runtime is too old, found version %d.%d.%d and "
                "expected at least version %d.%d.%d",
                TheApi.version.major, TheApi.version.minor,
                TheApi.version.patch, LRT_DISPATCH_API_VERSION_MAJOR,
                LRT_DISPATCH_API_VERSION_MINOR, LRT_DISPATCH_API_VERSION_PATCH);
    return kLrtStatusErrorRuntimeFailure;
  }

  INVOKE_FUNC(initialize);
}

LrtStatus LrtDispatchGetApiVersion(LrtDispatchApiVersion* api_version) {
  if (!api_version) {
    LITE_RT_LOG(LRT_ERROR, "Null input");
    return kLrtStatusErrorInvalidArgument;
  }
  *api_version = TheApi.version;
  return kLrtStatusOk;
}

LrtStatus LrtDispatchGetVendorId(const char** vendor_id) {
  if (!vendor_id) {
    LITE_RT_LOG(LRT_ERROR, "Null input");
    return kLrtStatusErrorInvalidArgument;
  }
  INVOKE_FUNC(get_vendor_id, vendor_id);
}

LrtStatus LrtDispatchGetBuildId(const char** build_id) {
  if (!build_id) {
    LITE_RT_LOG(LRT_ERROR, "Null input");
    return kLrtStatusErrorInvalidArgument;
  }
  INVOKE_FUNC(get_build_id, build_id);
}

LrtStatus LrtDispatchGetCapabilities(int* capabilities) {
  if (!capabilities) {
    LITE_RT_LOG(LRT_ERROR, "Null input");
    return kLrtStatusErrorInvalidArgument;
  }
  INVOKE_FUNC(get_capabilities, capabilities);
}

LrtStatus LrtDispatchDeviceContextCreate(
    LrtDispatchDeviceContext* device_context) {
  if (!device_context) {
    LITE_RT_LOG(LRT_ERROR, "Null input");
    return kLrtStatusErrorInvalidArgument;
  }
  INVOKE_FUNC(device_context_create, device_context);
}

LrtStatus LrtDispatchDeviceContextDestroy(
    LrtDispatchDeviceContext device_context) {
  if (!device_context) {
    LITE_RT_LOG(LRT_ERROR, "Null input");
    return kLrtStatusErrorInvalidArgument;
  }
  INVOKE_FUNC(device_context_destroy, device_context);
}

LrtStatus LrtDispatchGetInputRequirements(
    LrtDispatchInvocationContext invocation_context, int input_index,
    const LrtRankedTensorType* tensor_type,
    LrtTensorBufferRequirements* tensor_buffer_requirements) {
  if (!invocation_context || !tensor_type || !tensor_buffer_requirements) {
    LITE_RT_LOG(LRT_ERROR, "Null input");
    return kLrtStatusErrorInvalidArgument;
  }
  INVOKE_FUNC(get_input_requirements, invocation_context, input_index,
              tensor_type, tensor_buffer_requirements);
}

LrtStatus LrtDispatchGetOutputRequirements(
    LrtDispatchInvocationContext invocation_context, int output_index,
    const LrtRankedTensorType* tensor_type,
    LrtTensorBufferRequirements* tensor_buffer_requirements) {
  if (!invocation_context || !tensor_type || !tensor_buffer_requirements) {
    LITE_RT_LOG(LRT_ERROR, "Null input");
    return kLrtStatusErrorInvalidArgument;
  }
  INVOKE_FUNC(get_output_requirements, invocation_context, output_index,
              tensor_type, tensor_buffer_requirements);
}

LrtStatus LrtDispatchRegisterTensorBuffer(
    LrtDispatchDeviceContext device_context, LrtTensorBuffer tensor_buffer,
    LrtTensorBufferHandle* tensor_buffer_handle) {
  if (!device_context || !tensor_buffer || !tensor_buffer_handle) {
    LITE_RT_LOG(LRT_ERROR, "Null input");
    return kLrtStatusErrorInvalidArgument;
  }
  INVOKE_FUNC(register_tensor_buffer, device_context, tensor_buffer,
              tensor_buffer_handle);
}

LrtStatus LrtDispatchUnregisterTensorBuffer(
    LrtDispatchDeviceContext device_context,
    LrtTensorBufferHandle tensor_buffer_handle) {
  if (!device_context) {
    LITE_RT_LOG(LRT_ERROR, "Null input");
    return kLrtStatusErrorInvalidArgument;
  }
  INVOKE_FUNC(unregister_tensor_buffer, device_context, tensor_buffer_handle);
}

LrtStatus LrtDispatchInvocationContextCreate(
    LrtDispatchDeviceContext device_context,
    LrtDispatchExecutableType exec_type, const void* exec_bytecode_ptr,
    size_t exec_bytecode_size, const char* function_name, int num_inputs,
    int num_outputs, LrtDispatchInvocationContext* invocation_context) {
  if (!device_context || !exec_bytecode_ptr || !invocation_context) {
    LITE_RT_LOG(LRT_ERROR, "Null input");
    return kLrtStatusErrorInvalidArgument;
  }
  INVOKE_FUNC(invocation_context_create, device_context, exec_type,
              exec_bytecode_ptr, exec_bytecode_size, function_name, num_inputs,
              num_outputs, invocation_context);
}

LrtStatus LrtDispatchInvocationContextDestroy(
    LrtDispatchInvocationContext invocation_context) {
  if (!invocation_context) {
    LITE_RT_LOG(LRT_ERROR, "Null input");
    return kLrtStatusErrorInvalidArgument;
  }
  INVOKE_FUNC(invocation_context_destroy, invocation_context);
}

LrtStatus LrtDispatchAttachInput(
    LrtDispatchInvocationContext invocation_context, int graph_input_index,
    LrtTensorBufferHandle tensor_buffer_handle) {
  if (!invocation_context) {
    LITE_RT_LOG(LRT_ERROR, "Null input");
    return kLrtStatusErrorInvalidArgument;
  }
  INVOKE_FUNC(attach_input, invocation_context, graph_input_index,
              tensor_buffer_handle);
}

LrtStatus LrtDispatchAttachOutput(
    LrtDispatchInvocationContext invocation_context, int graph_output_index,
    LrtTensorBufferHandle tensor_buffer_handle) {
  if (!invocation_context) {
    LITE_RT_LOG(LRT_ERROR, "Null input");
    return kLrtStatusErrorInvalidArgument;
  }
  if (!TheApi.interface) {
    LITE_RT_LOG(LRT_ERROR, "Dispatch API interface not found");
    return kLrtStatusErrorRuntimeFailure;
  }
  if (!TheApi.interface->attach_output) {
    LITE_RT_LOG(LRT_ERROR, "attach_output_tensor_buffer not found");
    return kLrtStatusErrorRuntimeFailure;
  }
  INVOKE_FUNC(attach_output, invocation_context, graph_output_index,
              tensor_buffer_handle);
}

LrtStatus LrtDispatchDetachInput(
    LrtDispatchInvocationContext invocation_context, int graph_input_index,
    LrtTensorBufferHandle tensor_buffer_handle) {
  if (!invocation_context) {
    LITE_RT_LOG(LRT_ERROR, "Null input");
    return kLrtStatusErrorInvalidArgument;
  }
  INVOKE_FUNC(detach_input, invocation_context, graph_input_index,
              tensor_buffer_handle);
}

LrtStatus LrtDispatchDetachOutput(
    LrtDispatchInvocationContext invocation_context, int graph_output_index,
    LrtTensorBufferHandle tensor_buffer_handle) {
  if (!invocation_context) {
    LITE_RT_LOG(LRT_ERROR, "Null input");
    return kLrtStatusErrorInvalidArgument;
  }
  INVOKE_FUNC(detach_output, invocation_context, graph_output_index,
              tensor_buffer_handle);
}

LrtStatus LrtDispatchInvoke(LrtDispatchInvocationContext invocation_context) {
  if (!invocation_context) {
    LITE_RT_LOG(LRT_ERROR, "Null input");
    return kLrtStatusErrorInvalidArgument;
  }
  INVOKE_FUNC(invoke, invocation_context);
}

// /////////////////////////////////////////////////////////////////////////////
// Async Execution API
// /////////////////////////////////////////////////////////////////////////////

LrtStatus LrtDispatchAttachInputEvent(
    LrtDispatchInvocationContext invocation_context, int graph_input_index,
    LrtEvent input_event) {
  if (!invocation_context || !input_event) {
    LITE_RT_LOG(LRT_ERROR, "Null input");
    return kLrtStatusErrorInvalidArgument;
  }
  INVOKE_ASYNC_FUNC(attach_input_event, invocation_context, graph_input_index,
                    input_event);
}

LrtStatus LrtDispatchInvokeAsync(
    LrtDispatchInvocationContext invocation_context, int num_output_events,
    LrtEvent* output_events) {
  if (!invocation_context || !output_events) {
    LITE_RT_LOG(LRT_ERROR, "Null input");
    return kLrtStatusErrorInvalidArgument;
  }
  INVOKE_ASYNC_FUNC(invoke_async, invocation_context, num_output_events,
                    output_events);
}

// /////////////////////////////////////////////////////////////////////////////
// Graph Execution API
// /////////////////////////////////////////////////////////////////////////////

LrtStatus LrtDispatchGraphCreate(LrtDispatchDeviceContext device_context,
                                 LrtDispatchGraph* graph) {
  if (!device_context || !graph) {
    LITE_RT_LOG(LRT_ERROR, "Null input");
    return kLrtStatusErrorInvalidArgument;
  }
  INVOKE_GRAPH_FUNC(graph_create, device_context, graph);
}

LrtStatus LrtDispatchGraphDestroy(LrtDispatchGraph graph) {
  if (!graph) {
    LITE_RT_LOG(LRT_ERROR, "Null input");
    return kLrtStatusErrorInvalidArgument;
  }
  INVOKE_GRAPH_FUNC(graph_destroy, graph);
}

LrtStatus LrtDispatchAddNode(LrtDispatchGraph graph, LrtDispatchNodeId node_id,
                             LrtDispatchNodeType node_type) {
  if (!graph) {
    LITE_RT_LOG(LRT_ERROR, "Null input");
    return kLrtStatusErrorInvalidArgument;
  }
  INVOKE_GRAPH_FUNC(add_node, graph, node_id, node_type);
}

LrtStatus LrtDispatchAddEdge(LrtDispatchGraph graph,
                             LrtDispatchEdgeId edge_id) {
  if (!graph) {
    LITE_RT_LOG(LRT_ERROR, "Null input");
    return kLrtStatusErrorInvalidArgument;
  }
  INVOKE_GRAPH_FUNC(add_edge, graph, edge_id);
}

LrtStatus LrtDispatchConnectNodeInput(LrtDispatchGraph graph,
                                      LrtDispatchNodeId node_id,
                                      int input_index,
                                      LrtDispatchEdgeId edge_id) {
  if (!graph) {
    LITE_RT_LOG(LRT_ERROR, "Null input");
    return kLrtStatusErrorInvalidArgument;
  }
  INVOKE_GRAPH_FUNC(connect_node_input, graph, node_id, input_index, edge_id);
}

LrtStatus LrtDispatchConnectNodeOutput(LrtDispatchGraph graph,
                                       LrtDispatchNodeId node_id,
                                       int output_index,
                                       LrtDispatchEdgeId edge_id) {
  if (!graph) {
    LITE_RT_LOG(LRT_ERROR, "Null input");
    return kLrtStatusErrorInvalidArgument;
  }
  INVOKE_GRAPH_FUNC(connect_node_output, graph, node_id, output_index, edge_id);
}

LrtStatus LrtDispatchConnectGraphInput(LrtDispatchGraph graph, int input_index,
                                       LrtDispatchEdgeId edge_id) {
  if (!graph) {
    LITE_RT_LOG(LRT_ERROR, "Null input");
    return kLrtStatusErrorInvalidArgument;
  }
  INVOKE_GRAPH_FUNC(connect_graph_input, graph, input_index, edge_id);
}

LrtStatus LrtDispatchConnectGraphOutput(LrtDispatchGraph graph,
                                        int output_index,
                                        LrtDispatchEdgeId edge_id) {
  if (!graph) {
    LITE_RT_LOG(LRT_ERROR, "Null input");
    return kLrtStatusErrorInvalidArgument;
  }
  INVOKE_GRAPH_FUNC(connect_graph_output, graph, output_index, edge_id);
}

LrtStatus LrtDispatchLoadExecutable(LrtDispatchDeviceContext device_context,
                                    LrtDispatchExecutableType type,
                                    const void* bytecode, size_t bytecode_size,
                                    LrtDispatchExecutableHandle* exec_handle) {
  if (!device_context || !bytecode || !exec_handle) {
    LITE_RT_LOG(LRT_ERROR, "Null input");
    return kLrtStatusErrorInvalidArgument;
  }
  if (!TheApi.graph_interface) {
    LITE_RT_LOG(LRT_ERROR, "Dispatch API graph interface not found");
    return kLrtStatusErrorRuntimeFailure;
  }
  if (!TheApi.graph_interface->load_executable) {
    LITE_RT_LOG(LRT_ERROR, "load_executable not found");
    return kLrtStatusErrorRuntimeFailure;
  }
  INVOKE_GRAPH_FUNC(load_executable, device_context, type, bytecode,
                    bytecode_size, exec_handle);
}

LrtStatus LrtDispatchUnloadExecutable(LrtDispatchDeviceContext device_context,
                                      LrtDispatchExecutableHandle exec_handle) {
  if (!device_context) {
    LITE_RT_LOG(LRT_ERROR, "Null input");
    return kLrtStatusErrorInvalidArgument;
  }
  INVOKE_GRAPH_FUNC(unload_executable, device_context, exec_handle);
}

LrtStatus LrtDispatchAssignNodeFunction(LrtDispatchGraph graph,
                                        LrtDispatchNodeId node_id,
                                        LrtDispatchExecutableHandle exec_handle,
                                        const char* function_name) {
  if (!graph) {
    LITE_RT_LOG(LRT_ERROR, "Null input");
    return kLrtStatusErrorInvalidArgument;
  }
  INVOKE_GRAPH_FUNC(assign_node_function, graph, node_id, exec_handle,
                    function_name);
}

LrtStatus LrtDispatchAnnotateGraph(LrtDispatchGraph graph, const char* key,
                                   const char* value) {
  if (!graph) {
    LITE_RT_LOG(LRT_ERROR, "Null input");
    return kLrtStatusErrorInvalidArgument;
  }
  INVOKE_GRAPH_FUNC(annotate_graph, graph, key, value);
}

LrtStatus LrtDispatchAnnotateNode(LrtDispatchGraph graph,
                                  LrtDispatchNodeId node_id, const char* key,
                                  const char* value) {
  if (!graph) {
    LITE_RT_LOG(LRT_ERROR, "Null input");
    return kLrtStatusErrorInvalidArgument;
  }
  INVOKE_GRAPH_FUNC(annotate_node, graph, node_id, key, value);
}

LrtStatus LrtDispatchAnnotateEdge(LrtDispatchGraph graph,
                                  LrtDispatchEdgeId edge_id, const char* key,
                                  const char* value) {
  if (!graph) {
    LITE_RT_LOG(LRT_ERROR, "Null input");
    return kLrtStatusErrorInvalidArgument;
  }
  INVOKE_GRAPH_FUNC(annotate_edge, graph, edge_id, key, value);
}

LrtStatus LrtDispatchInvocationContextCreateFromGraph(
    LrtDispatchDeviceContext device_context, LrtDispatchGraph graph,
    LrtDispatchInvocationContext* invocation_context) {
  if (!device_context || !graph || !invocation_context) {
    LITE_RT_LOG(LRT_ERROR, "Null input");
    return kLrtStatusErrorInvalidArgument;
  }
  INVOKE_GRAPH_FUNC(invocation_context_create_from_graph, device_context, graph,
                    invocation_context);
}
