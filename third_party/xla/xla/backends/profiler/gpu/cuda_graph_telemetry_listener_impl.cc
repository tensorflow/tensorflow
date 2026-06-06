/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/profiler/gpu/cuda_graph_telemetry_listener_impl.h"

#include <cstddef>

#include "absl/strings/string_view.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/gpu/gpu_command_buffer.h"
#include "xla/stream_executor/gpu/gpu_command_buffer_listener.h"
#include "xla/tsl/platform/logging.h"

#if CUDA_VERSION >= 13010
#include "xla/backends/profiler/gpu/cuda_graph_annotation_registry.h"
#include "xla/debug_options_flags.h"
#include "xla/stream_executor/gpu/scoped_command_buffer_annotation.h"
#endif

namespace xla {
namespace profiler {

bool CudaGraphTelemetryListenerImpl::IsEnabled() const {
  static bool enabled = []() {
#if CUDA_VERSION >= 13010
    bool val =
        GetDebugOptionsFromFlags().xla_gpu_enable_cuda_graphs_telemetry();
    LOG(INFO) << "CUDA Graphs telemetry is "
              << (val ? "enabled" : "disabled by flag");
    return val;
#else
    LOG(INFO) << "CUDA Graphs telemetry is disabled (requires CUDA 13.1+, "
                 "compiled with "
              << CUDA_VERSION << ")";
    return false;
#endif
  }();
  return enabled;
}

void CudaGraphTelemetryListenerImpl::OnRegisterNodeAnnotation(
    void* graph, stream_executor::gpu::GpuCommandBuffer::GraphNodeHandle node,
    absl::string_view annotation) {
  if (!IsEnabled()) {
    return;
  }
#if CUDA_VERSION >= 13010
  CudaGraphAnnotationRegistry::RegisterNodeAnnotation(
      reinterpret_cast<CUgraph>(graph), reinterpret_cast<CUgraphNode>(node),
      annotation);
#endif
}

void CudaGraphTelemetryListenerImpl::OnRegisterGraphSize(void* graph,
                                                         size_t size) {
  if (!IsEnabled()) {
    return;
  }
#if CUDA_VERSION >= 13010
  CudaGraphAnnotationRegistry::RegisterGraphSize(
      reinterpret_cast<CUgraph>(graph), size);
#endif
}

void CudaGraphTelemetryListenerImpl::OnRegisterChildGraph(
    void* parent_graph, void* child_graph,
    stream_executor::gpu::GpuCommandBuffer::GraphNodeHandle child_node,
    bool is_conditional) {
  if (!IsEnabled()) {
    return;
  }
#if CUDA_VERSION >= 13010
  CudaGraphAnnotationRegistry::RegisterChildGraph(
      reinterpret_cast<CUgraph>(parent_graph),
      reinterpret_cast<CUgraph>(child_graph),
      reinterpret_cast<CUgraphNode>(child_node), is_conditional);
#endif
}

void CudaGraphTelemetryListenerImpl::OnRegisterGraphExec(void* graph_exec,
                                                         void* graph) {
  if (!IsEnabled()) {
    return;
  }
#if CUDA_VERSION >= 13010
  CudaGraphAnnotationRegistry::RegisterGraphExec(
      reinterpret_cast<CUgraphExec>(graph_exec),
      reinterpret_cast<CUgraph>(graph));
#endif
}

void CudaGraphTelemetryListenerImpl::OnUnregisterGraphExec(void* graph_exec) {
  if (!IsEnabled()) {
    return;
  }
#if CUDA_VERSION >= 13010
  CudaGraphAnnotationRegistry::UnregisterGraphExec(
      reinterpret_cast<CUgraphExec>(graph_exec));
#endif
}

void CudaGraphTelemetryListenerImpl::OnUnregisterGraphAnnotations(void* graph) {
  if (!IsEnabled()) {
    return;
  }
#if CUDA_VERSION >= 13010
  CudaGraphAnnotationRegistry::UnregisterGraphAnnotations(
      reinterpret_cast<CUgraph>(graph));
#endif
}

void RegisterCudaGraphTelemetryListener() {
  static auto* listener = new CudaGraphTelemetryListenerImpl();
  stream_executor::gpu::RegisterGpuCommandBufferListener(listener);
}

static bool InitModule() {
  RegisterCudaGraphTelemetryListener();
  return true;
}

static bool dummy = InitModule();

}  // namespace profiler
}  // namespace xla
