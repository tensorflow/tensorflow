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

#include "tensorflow/lite/experimental/litert/vendors/google_tensor/dispatch/litert_dispatch_invocation_context.h"

#include <cstddef>

#include "absl/strings/string_view.h"
#include "third_party/odml/infra/southbound/sb_api.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_event.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer_requirements.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/core/util/tensor_type_util.h"
#include "tensorflow/lite/experimental/litert/vendors/c/litert_dispatch.h"
#include "tensorflow/lite/experimental/litert/vendors/google_tensor/dispatch/litert_dispatch_device_context.h"
#include "tensorflow/lite/experimental/litert/vendors/google_tensor/dispatch/litert_dispatch_graph.h"
#include "tensorflow/lite/experimental/litert/vendors/google_tensor/dispatch/litert_dispatch_metrics.h"
#include "tensorflow/lite/experimental/litert/vendors/google_tensor/dispatch/southbound.h"

using litert::Error;
using litert::Expected;
using litert::Unexpected;

extern absl::string_view ThrEdgeIdStr(LiteRtDispatchEdgeId edge_id);

namespace {

constexpr const size_t kEdgeTpuPadding = 64;

template <class X, class Align>
inline constexpr auto Pad(X x, Align align) {
  return ((x + align - 1) / align) * align;
}

}  // namespace

litert::Expected<LiteRtDispatchInvocationContextT::Ptr>
LiteRtDispatchInvocationContextT::CreateFromBytecode(
    const litert::google_tensor::Southbound& southbound,
    LiteRtDispatchDeviceContext device_context,
    LiteRtDispatchExecutableType exec_type,
    const LiteRtMemBuffer* exec_bytecode_buffer, const char* function_name,
    int num_inputs, int num_outputs) {
  auto graph = device_context->CreateGraph();
  if (!graph) {
    return graph.Error();
  }

  LiteRtDispatchNodeId node_id = 0;
  LiteRtDispatchNodeType node_type;
  switch (exec_type) {
    case kLiteRtDispatchExecutableTypeDspLibrary:
      node_type = kLiteRtDispatchNodeTypeDsp;
      break;
    case kLiteRtDispatchExecutableTypeMlModel:
      node_type = kLiteRtDispatchNodeTypeNpu;
      break;
    default:
      LITERT_LOG(LITERT_ERROR, "Unexpected executable type: %d", exec_type);
      return Error(kLiteRtStatusErrorInvalidArgument,
                   "Unexpected executable type");
  }

  if (auto status = (*graph)->AddNode(node_id, node_type); !status) {
    return status.Error();
  }

  auto exec_handle =
      device_context->LoadExecutable(exec_type, exec_bytecode_buffer);
  if (!exec_handle) {
    return exec_handle.Error();
  }

  if (auto status =
          (*graph)->AssignNodeFunction(node_id, *exec_handle, function_name);
      !status) {
    return status.Error();
  }

  LiteRtDispatchEdgeId next_edge_id = 0;

  for (auto input_index = 0; input_index < num_inputs; ++input_index) {
    LiteRtDispatchEdgeId edge_id = next_edge_id++;
    if (auto status = (*graph)->AddEdge(edge_id); !status) {
      return status.Error();
    }
    if (auto status = (*graph)->ConnectGraphInput(input_index, edge_id);
        !status) {
      return status.Error();
    }
    if (auto status = (*graph)->ConnectNodeInput(node_id, input_index, edge_id);
        !status) {
      return status.Error();
    }
  }

  for (auto output_index = 0; output_index < num_outputs; ++output_index) {
    LiteRtDispatchEdgeId edge_id = next_edge_id++;
    if (auto status = (*graph)->AddEdge(edge_id); !status) {
      return status.Error();
    }
    if (auto status =
            (*graph)->ConnectNodeOutput(node_id, output_index, edge_id);
        !status) {
      return status.Error();
    }
    if (auto status = (*graph)->ConnectGraphOutput(output_index, edge_id);
        !status) {
      return status.Error();
    }
  }

  auto invocation_context = CreateFromGraph(southbound, device_context, *graph);
  if (!invocation_context) {
    return invocation_context.Error();
  }

  (*invocation_context)->AttachExecutable(*exec_handle);

  return invocation_context;
}

litert::Expected<LiteRtDispatchInvocationContextT::Ptr>
LiteRtDispatchInvocationContextT::CreateFromGraph(
    const litert::google_tensor::Southbound& southbound,
    LiteRtDispatchDeviceContext device_context, LiteRtDispatchGraph graph) {
  auto thr_invocation_context_get = southbound.api().thr_invocation_context_get;
  if (!thr_invocation_context_get) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "thr_invocation_context_get not found");
  }

  ThrGraph* thr_graph = graph->thr_graph();
  auto thr_icontext =
      thr_invocation_context_get(thr_graph, device_context->thr_context());
  if (!thr_icontext) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "thr_invocation_context_get failed");
  }

  device_context->add_graph(thr_graph);
  return Ptr(new LiteRtDispatchInvocationContextT(southbound, thr_icontext,
                                                  device_context, graph));
}

LiteRtDispatchInvocationContextT::~LiteRtDispatchInvocationContextT() {
  auto thr_invocation_context_delete =
      southbound_.api().thr_invocation_context_delete;
  if (!thr_invocation_context_delete) {
    LITERT_LOG(LITERT_ERROR, "thr_invocation_context_delete not found");
  } else {
    ThrGraph* thr_graph = graph_->thr_graph();
    if (auto status =
            thr_invocation_context_delete(thr_graph, thr_invocation_context_);
        status != kThrStatusSuccess) {
      LITERT_LOG(LITERT_ERROR, "thr_invocation_context_delete failed: %d",
                 status);
    }
  }

  if (exec_handle_) {
    device_context_->UnloadExecutable(*exec_handle_);
  }
}

namespace {

Expected<LiteRtTensorBufferRequirements> GetTensorBufferRequirements(
    const LiteRtRankedTensorType& tensor_type) {
  auto* tensor_strides = tensor_type.layout.strides;
  if (tensor_strides != nullptr) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Tensor strides are not supported on GoogleTensor");
  }

  LiteRtTensorBufferType supported_tensor_buffer_types[] = {
      kLiteRtTensorBufferTypeAhwb,
  };
  int num_supported_tensor_buffer_types =
      sizeof(supported_tensor_buffer_types) /
      sizeof(supported_tensor_buffer_types[0]);

  auto buffer_size = litert::internal::GetNumPackedBytes(tensor_type);
  if (!buffer_size) {
    return Unexpected(buffer_size.Error());
  }

  size_t padded_buffer_size = Pad(*buffer_size, kEdgeTpuPadding);

  LiteRtTensorBufferRequirements requirements;
  if (auto status = LiteRtCreateTensorBufferRequirements(
          num_supported_tensor_buffer_types, supported_tensor_buffer_types,
          padded_buffer_size, /*num_strides=*/0, /*strides=*/nullptr,
          &requirements);
      status != kLiteRtStatusOk) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Failed to create tensor buffer requirements");
  }

  return requirements;
}
}  // namespace

Expected<LiteRtTensorBufferRequirements>
LiteRtDispatchInvocationContextT::GetInputRequirements(
    int input_index, const LiteRtRankedTensorType& tensor_type) {
  return GetTensorBufferRequirements(tensor_type);
}

Expected<LiteRtTensorBufferRequirements>
LiteRtDispatchInvocationContextT::GetOutputRequirements(
    int output_index, const LiteRtRankedTensorType& tensor_type) {
  return GetTensorBufferRequirements(tensor_type);
}

namespace {

litert::Expected<void> AttachBufferHelper(
    const litert::google_tensor::Southbound& southbound,
    LiteRtDispatchInvocationContext invocation_context,
    LiteRtDispatchEdgeId edge_id,
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  auto thr_invocation_context_attach_buffer =
      southbound.api().thr_invocation_context_attach_buffer;
  if (!thr_invocation_context_attach_buffer) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "thr_invocation_context_attach_buffer not found");
  }

  ThrInvocationContext* thr_icontext =
      invocation_context->thr_invocation_context();
  ThrContext* thr_context = invocation_context->device_context()->thr_context();
  auto thr_edge_id = ThrEdgeIdStr(edge_id);
  ThrBufferHandle thr_buffer_handle = tensor_buffer_handle;
  if (auto status = thr_invocation_context_attach_buffer(
          thr_icontext, thr_context, thr_edge_id.data(), thr_buffer_handle);
      status != kThrStatusSuccess) {
    LITERT_LOG(LITERT_ERROR, "thr_invocation_context_attach_buffer failed: %d",
               status);
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "thr_invocation_context_attach_buffer failed");
  }

  return {};
}

}  // namespace

litert::Expected<void> LiteRtDispatchInvocationContextT::AttachInput(
    int graph_input_index, LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (auto result = graph_->InputEdge(graph_input_index); result) {
    auto edge_id = *result;
    return AttachBufferHelper(southbound_, this, edge_id, tensor_buffer_handle);
  } else {
    return result.Error();
  }
}

litert::Expected<void> LiteRtDispatchInvocationContextT::AttachOutput(
    int graph_output_index, LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (auto result = graph_->OutputEdge(graph_output_index); result) {
    auto edge_id = *result;
    return AttachBufferHelper(southbound_, this, edge_id, tensor_buffer_handle);
  } else {
    return result.Error();
  }
}

namespace {

litert::Expected<void> DetachTensorBufferHelper(
    const litert::google_tensor::Southbound& southbound,
    LiteRtDispatchInvocationContext invocation_context,
    LiteRtDispatchEdgeId edge_id,
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  auto thr_invocation_context_detach_buffer =
      southbound.api().thr_invocation_context_detach_buffer;
  if (!thr_invocation_context_detach_buffer) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "thr_invocation_context_detach_buffer not found");
  }

  ThrInvocationContext* thr_icontext =
      invocation_context->thr_invocation_context();
  ThrContext* thr_context = invocation_context->device_context()->thr_context();
  auto thr_edge_id = ThrEdgeIdStr(edge_id);
  ThrBufferHandle thr_buffer_handle = tensor_buffer_handle;
  if (auto status = thr_invocation_context_detach_buffer(
          thr_icontext, thr_context, thr_edge_id.data(), thr_buffer_handle);
      status != kThrStatusSuccess) {
    LITERT_LOG(LITERT_ERROR, "thr_invocation_context_detach_buffer failed: %d",
               status);
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "thr_invocation_context_detach_buffer failed");
  }

  return {};
}

}  // namespace

litert::Expected<void> LiteRtDispatchInvocationContextT::DetachInput(
    int graph_input_index, LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (auto result = graph_->InputEdge(graph_input_index); result) {
    auto edge_id = *result;
    return DetachTensorBufferHelper(southbound_, this, edge_id,
                                    tensor_buffer_handle);
  } else {
    return result.Error();
  }
}

litert::Expected<void> LiteRtDispatchInvocationContextT::DetachOutput(
    int graph_output_index, LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (auto result = graph_->OutputEdge(graph_output_index); result) {
    auto edge_id = *result;
    return DetachTensorBufferHelper(southbound_, this, edge_id,
                                    tensor_buffer_handle);
  } else {
    return result.Error();
  }
}

namespace {

litert::Expected<void> PrepareForInvoke(
    const litert::google_tensor::Southbound& southbound,
    LiteRtDispatchInvocationContext invocation_context,
    bool create_output_sync_fence) {
  auto thr_invocation_context_prepare_for_invoke =
      southbound.api().thr_invocation_context_prepare_for_invoke;
  if (!thr_invocation_context_prepare_for_invoke) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "thr_invocation_context_prepare_for_invoke not found");
  }

  ThrInvocationContext* thr_icontext =
      invocation_context->thr_invocation_context();
  if (auto status = thr_invocation_context_prepare_for_invoke(
          thr_icontext, create_output_sync_fence);
      status != kThrStatusSuccess) {
    LITERT_LOG(LITERT_ERROR,
               "thr_invocation_context_prepare_for_invoke failed: %d", status);
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "thr_invocation_context_prepare_for_invoke failed");
  }

  return {};
}

litert::Expected<void> InvokeOnce(
    const litert::google_tensor::Southbound& southbound,
    LiteRtDispatchInvocationContext invocation_context) {
  auto thr_invocation_context_invoke_once =
      southbound.api().thr_invocation_context_invoke_once;
  if (!thr_invocation_context_invoke_once) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "thr_invocation_context_invoke_once not found");
  }

  ThrInvocationContext* thr_icontext =
      invocation_context->thr_invocation_context();
  if (auto status = thr_invocation_context_invoke_once(thr_icontext);
      status != kThrStatusSuccess) {
    LITERT_LOG(LITERT_ERROR, "thr_invocation_context_invoke_once failed: %d",
               status);
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "thr_invocation_context_invoke_once failed");
  }

  return {};
}

litert::Expected<void> Wait(
    const litert::google_tensor::Southbound& southbound,
    LiteRtDispatchInvocationContext invocation_context) {
  auto thr_invocation_context_wait =
      southbound.api().thr_invocation_context_wait;
  if (!thr_invocation_context_wait) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "thr_invocation_context_wait not found");
  }

  ThrInvocationContext* thr_icontext =
      invocation_context->thr_invocation_context();
  if (auto status = thr_invocation_context_wait(thr_icontext);
      status != kThrStatusSuccess) {
    LITERT_LOG(LITERT_ERROR, "thr_invocation_context_wait failed: %d", status);
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "thr_invocation_context_wait failed");
  }

  return {};
}

}  // namespace

litert::Expected<void> LiteRtDispatchInvocationContextT::Invoke() {
  if (auto result = PrepareForInvoke(southbound_, this,
                                     /*create_output_sync_fence=*/false);
      !result) {
    return result.Error();
  }
  if (auto result = InvokeOnce(southbound_, this); !result) {
    return result.Error();
  }
  return Wait(southbound_, this);
}

litert::Expected<void> LiteRtDispatchInvocationContextT::AttachInputEvent(
    int graph_input_index, LiteRtEvent input_event) {
  int input_fence_fd;
  if (auto status = LiteRtGetEventSyncFenceFd(input_event, &input_fence_fd);
      status != kLiteRtStatusOk) {
    return Error(status, "Failed to get sync fence fd from event");
  }

  auto edge = graph_->InputEdge(graph_input_index);
  if (!edge) {
    LITERT_LOG(LITERT_ERROR, "Unexpected graph input index: %d",
               graph_input_index);
    return edge.Error();
  }
  auto edge_id = *edge;

  auto thr_invocation_context_attach_input_buffer_sync_fence =
      southbound_.api().thr_invocation_context_attach_input_buffer_sync_fence;
  if (!thr_invocation_context_attach_input_buffer_sync_fence) {
    return Error(
        kLiteRtStatusErrorRuntimeFailure,
        "thr_invocation_context_attach_input_buffer_sync_fence not found");
  }

  auto thr_edge_id = ThrEdgeIdStr(edge_id);
  if (auto status = thr_invocation_context_attach_input_buffer_sync_fence(
          thr_invocation_context_, thr_edge_id.data(), input_fence_fd);
      status != kThrStatusSuccess) {
    LITERT_LOG(
        LITERT_ERROR,
        "thr_invocation_context_attach_input_buffer_sync_fence failed: %d",
        status);
    return Error(
        kLiteRtStatusErrorRuntimeFailure,
        "thr_invocation_context_attach_input_buffer_sync_fence failed");
  }

  input_sync_fences_[thr_edge_id.data()] = input_fence_fd;
  return {};
}

namespace {

litert::Expected<void> GetOutputEvent(
    const litert::google_tensor::Southbound& southbound,
    LiteRtDispatchInvocationContext invocation_context, int graph_output_index,
    LiteRtEvent* output_event) {
  auto edge = invocation_context->graph()->OutputEdge(graph_output_index);
  if (!edge) {
    LITERT_LOG(LITERT_ERROR, "Unexpected graph output index: %d",
               graph_output_index);
    return edge.Error();
  }
  auto edge_id = *edge;

  auto thr_invocation_context_get_output_buffer_sync_fence =
      southbound.api().thr_invocation_context_get_output_buffer_sync_fence;
  if (!thr_invocation_context_get_output_buffer_sync_fence) {
    return Error(
        kLiteRtStatusErrorRuntimeFailure,
        "thr_invocation_context_get_output_buffer_sync_fence not found");
  }

  ThrInvocationContext* thr_icontext =
      invocation_context->thr_invocation_context();
  auto thr_edge_id = ThrEdgeIdStr(edge_id);
  int output_fence_fd;
  if (auto status = thr_invocation_context_get_output_buffer_sync_fence(
          thr_icontext, thr_edge_id.data(), &output_fence_fd);
      status != kThrStatusSuccess) {
    LITERT_LOG(LITERT_ERROR,
               "thr_invocation_context_get_output_buffer_sync_fence failed: %d",
               status);
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "thr_invocation_context_get_output_buffer_sync_fence failed");
  }

  if (auto status = LiteRtCreateEventFromSyncFenceFd(
          output_fence_fd, /*owns_fd=*/false, output_event);
      status != kLiteRtStatusOk) {
    return Error(status, "Failed to create event from sync fence fd");
  }

  return {};
}

}  // namespace

litert::Expected<void> LiteRtDispatchInvocationContextT::InvokeAsync(
    int num_output_events, LiteRtEvent* output_events) {
  if (num_output_events != graph_->NumOutputs()) {
    LITERT_LOG(LITERT_ERROR, "Unexpected number of output events: %d",
               num_output_events);
    return Error(kLiteRtStatusErrorInvalidArgument,
                 "Unexpected number of output events");
  }

  if (auto status = PrepareForInvoke(southbound_, this,
                                     /*create_output_sync_fence=*/true);
      !status) {
    return status.Error();
  }

  if (auto status = InvokeOnce(southbound_, this); !status) {
    return status.Error();
  }

  // Deatach input fences.
  auto thr_invocation_context_detach_input_buffer_sync_fence =
      southbound_.api().thr_invocation_context_detach_input_buffer_sync_fence;
  if (!thr_invocation_context_detach_input_buffer_sync_fence) {
    return Error(
        kLiteRtStatusErrorRuntimeFailure,
        "thr_invocation_context_detach_input_buffer_sync_fence not found");
  }
  for (const auto& p : input_sync_fences_) {
    const auto& thr_edge_id = p.first;
    auto input_fence_fd = p.second;
    if (auto status = thr_invocation_context_detach_input_buffer_sync_fence(
            thr_invocation_context_, thr_edge_id.data(), input_fence_fd);
        status != kThrStatusSuccess) {
      return Error(
          kLiteRtStatusErrorRuntimeFailure,
          "thr_invocation_context_deatch_input_buffer_sync_fence failed");
    }
  }
  input_sync_fences_.clear();

  // Extract output events.
  for (auto graph_output_index = 0; graph_output_index < num_output_events;
       ++graph_output_index) {
    if (auto status = GetOutputEvent(southbound_, this, graph_output_index,
                                     &output_events[graph_output_index]);
        !status) {
      LITERT_LOG(LITERT_ERROR, "Failed to get event for output %d",
                 graph_output_index);
      return status.Error();
    }
  }

  return {};
}

litert::Expected<void> LiteRtDispatchInvocationContextT::StartMetricsCollection(
    int detail_level) {
  auto thr_invocation_context_start_metrics_collection =
      southbound_.api().thr_invocation_context_start_metrics_collection;
  if (!thr_invocation_context_start_metrics_collection) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "thr_invocation_context_start_metrics_collection not found");
  }
  if (auto status = thr_invocation_context_start_metrics_collection(
          thr_invocation_context_, detail_level);
      status != kThrStatusSuccess) {
    LITERT_LOG(LITERT_ERROR,
               "thr_invocation_context_start_metrics_collection failed: %d",
               status);
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "thr_invocation_context_start_metrics_collection failed");
  }
  return {};
}

litert::Expected<void> LiteRtDispatchInvocationContextT::StopMetricsCollection(
    LiteRtDispatchMetrics* metrics) {
  auto thr_invocation_context_stop_metrics_collection =
      southbound_.api().thr_invocation_context_stop_metrics_collection;
  if (!thr_invocation_context_stop_metrics_collection) {
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "thr_invocation_context_stop_metrics_collection not found");
  }
  ThrInvocationMetrics thr_metrics{.version = 0};
  if (auto status = thr_invocation_context_stop_metrics_collection(
          thr_invocation_context_, &thr_metrics);
      status != kThrStatusSuccess) {
    LITERT_LOG(LITERT_ERROR,
               "thr_invocation_context_stop_metrics_collection failed: %d",
               status);
    *metrics = new LiteRtDispatchMetricsT(/*num_metrics=*/0,
                                          /*metric_names=*/nullptr,
                                          /*metric_values=*/nullptr);
    return Error(kLiteRtStatusErrorRuntimeFailure,
                 "thr_invocation_context_stop_metrics_collection failed");
  }
  *metrics = new LiteRtDispatchMetricsT(thr_metrics.num_metrics,
                                        thr_metrics.metric_keys,
                                        thr_metrics.metric_values);
  return {};
}
