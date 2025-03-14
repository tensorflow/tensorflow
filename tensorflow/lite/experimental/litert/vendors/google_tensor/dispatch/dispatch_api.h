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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_GOOGLE_TENSOR_DISPATCH_DISPATCH_API_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_GOOGLE_TENSOR_DISPATCH_DISPATCH_API_H_

#include <cstddef>

#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/vendors/c/litert_dispatch.h"

namespace litert::google_tensor {

LiteRtStatus GraphCreate(LiteRtDispatchDeviceContext device_context,
                         LiteRtDispatchGraph* graph);
LiteRtStatus GraphDestroy(LiteRtDispatchGraph graph);
LiteRtStatus AddNode(LiteRtDispatchGraph graph, LiteRtDispatchNodeId node_id,
                     LiteRtDispatchNodeType node_type);
LiteRtStatus AddEdge(LiteRtDispatchGraph graph, LiteRtDispatchEdgeId edge_id);
LiteRtStatus ConnectNodeInput(LiteRtDispatchGraph graph,
                              LiteRtDispatchNodeId node_id, int input_index,
                              LiteRtDispatchEdgeId edge_id);
LiteRtStatus ConnectNodeOutput(LiteRtDispatchGraph graph,
                               LiteRtDispatchNodeId node_id, int output_index,
                               LiteRtDispatchEdgeId edge_id);
LiteRtStatus ConnectGraphInput(LiteRtDispatchGraph graph, int input_index,
                               LiteRtDispatchEdgeId edge_id);
LiteRtStatus ConnectGraphOutput(LiteRtDispatchGraph graph, int output_index,
                                LiteRtDispatchEdgeId edge_id);
LiteRtStatus LoadExecutable(LiteRtDispatchDeviceContext device_context,
                            LiteRtDispatchExecutableType type,
                            const void* bytecode, size_t bytecode_size,
                            LiteRtDispatchExecutableHandle* exec_handle);
LiteRtStatus UnloadExecutable(LiteRtDispatchDeviceContext device_context,
                              LiteRtDispatchExecutableHandle exec_handle);
LiteRtStatus AssignNodeFunction(LiteRtDispatchGraph graph,
                                LiteRtDispatchNodeId node_id,
                                LiteRtDispatchExecutableHandle exec_handle,
                                const char* function_name);
LiteRtStatus AnnotateGraph(LiteRtDispatchGraph graph, const char* key,
                           const char* value);
LiteRtStatus AnnotateNode(LiteRtDispatchGraph graph,
                          LiteRtDispatchNodeId node_id, const char* key,
                          const char* value);
LiteRtStatus AnnotateEdge(LiteRtDispatchGraph graph,
                          LiteRtDispatchEdgeId edge_id, const char* key,
                          const char* value);
LiteRtStatus InvocationContextCreateFromGraph(
    LiteRtDispatchDeviceContext device_context, LiteRtDispatchGraph graph,
    LiteRtDispatchInvocationContext* invocation_context);

}  // namespace litert::google_tensor

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_GOOGLE_TENSOR_DISPATCH_DISPATCH_API_H_
