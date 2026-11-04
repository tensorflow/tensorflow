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

// CUDA 13.2-13.3 telemetry. CUPTI records there carry only the executable
// graph's `graphId` and a synthesized `graphNodeId = (exec_id << 32) |
// ordinal`. The ordinal numbers each graph's direct nodes first, then recurses
// depth-first into child and conditional body graphs. Child-graph and cuDNN
// updates don't re-project, so those nodes keep their instantiation-time
// annotations.
//
// TODO(b/483450622): Delete once CUDA 13.4 is the minimum for telemetry.

#ifndef XLA_STREAM_EXECUTOR_CUDA_CUDA_COMMAND_BUFFER_TELEMETRY_PRE_13_4_H_
#define XLA_STREAM_EXECUTOR_CUDA_CUDA_COMMAND_BUFFER_TELEMETRY_PRE_13_4_H_

#include <cstdint>
#include <optional>

#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/cuda/cuda_command_buffer_telemetry.h"

namespace stream_executor::gpu::command_buffer_telemetry {

// Returns `cuGraphExecGetId(exec)`, or `std::nullopt` if `exec` is null or the
// call fails. Always `std::nullopt` with CUDA headers older than 13.2, where
// BUILD selects cuda_command_buffer_telemetry_pre_13_4_stub.cc.
std::optional<uint32_t> GetGraphExecId(CUgraphExec exec);

// Projects template graph node annotations from `root_graph` (and its nested
// child / conditional graphs) into `CudaGraphAnnotationRegistry` keyed by
// `exec`'s `cuGraphExecGetId` and synthesized CUPTI `graphNodeId` in
// `KeySpace::kPre134ExecGraph`. Makes CUDA calls, so callers check
// `CudaCommandBuffer::ShouldRecordAnnotations()` first. Returns the exec entry
// if this call created it: the caller then owns it and must unregister it when
// it destroys or re-instantiates `exec`.
std::optional<TrackedGraph> RegisterPre134ExecGraphNodes(CUgraph root_graph,
                                                         CUgraphExec exec);

}  // namespace stream_executor::gpu::command_buffer_telemetry

#endif  // XLA_STREAM_EXECUTOR_CUDA_CUDA_COMMAND_BUFFER_TELEMETRY_PRE_13_4_H_
