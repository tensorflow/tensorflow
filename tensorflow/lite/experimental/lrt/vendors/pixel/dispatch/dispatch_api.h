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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LRT_VENDORS_PIXEL_DISPATCH_DISPATCH_API_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LRT_VENDORS_PIXEL_DISPATCH_DISPATCH_API_H_

#include "tensorflow/lite/experimental/lrt/c/lite_rt_dispatch.h"

namespace lrt {
namespace pixel {

LrtStatus GraphCreate(LrtDispatchDeviceContext device_context,
                      LrtDispatchGraph* graph);
LrtStatus GraphDestroy(LrtDispatchGraph graph);
LrtStatus AddNode(LrtDispatchGraph graph, LrtDispatchNodeId node_id,
                  LrtDispatchNodeType node_type);
LrtStatus AddEdge(LrtDispatchGraph graph, LrtDispatchEdgeId edge_id);
LrtStatus ConnectNodeInput(LrtDispatchGraph graph, LrtDispatchNodeId node_id,
                           int input_index, LrtDispatchEdgeId edge_id);
LrtStatus ConnectNodeOutput(LrtDispatchGraph graph, LrtDispatchNodeId node_id,
                            int output_index, LrtDispatchEdgeId edge_id);
LrtStatus ConnectGraphInput(LrtDispatchGraph graph, int input_index,
                            LrtDispatchEdgeId edge_id);
LrtStatus ConnectGraphOutput(LrtDispatchGraph graph, int output_index,
                             LrtDispatchEdgeId edge_id);
LrtStatus LoadExecutable(LrtDispatchDeviceContext device_context,
                         LrtDispatchExecutableType type, const void* bytecode,
                         size_t bytecode_size,
                         LrtDispatchExecutableHandle* exec_handle);
LrtStatus UnloadExecutable(LrtDispatchDeviceContext device_context,
                           LrtDispatchExecutableHandle exec_handle);
LrtStatus AssignNodeFunction(LrtDispatchGraph graph, LrtDispatchNodeId node_id,
                             LrtDispatchExecutableHandle exec_handle,
                             const char* function_name);
LrtStatus AnnotateGraph(LrtDispatchGraph graph, const char* key,
                        const char* value);
LrtStatus AnnotateNode(LrtDispatchGraph graph, LrtDispatchNodeId node_id,
                       const char* key, const char* value);
LrtStatus AnnotateEdge(LrtDispatchGraph graph, LrtDispatchEdgeId edge_id,
                       const char* key, const char* value);
LrtStatus InvocationContextCreateFromGraph(
    LrtDispatchDeviceContext device_context, LrtDispatchGraph graph,
    LrtDispatchInvocationContext* invocation_context);

}  // namespace pixel
}  // namespace lrt

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LRT_VENDORS_PIXEL_DISPATCH_DISPATCH_API_H_
