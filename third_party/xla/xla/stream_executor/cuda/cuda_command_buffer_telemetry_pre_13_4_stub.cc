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

// Selected by BUILD instead of cuda_command_buffer_telemetry_pre_13_4.cc with
// CUDA headers older than 13.2, which lack `cuGraphExecGetId`. There
// `CudaCommandBuffer::SupportsNodeAnnotations()` is false, so these functions
// are never reached; they only keep the library linkable.

#include <cstdint>
#include <optional>

#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/cuda/cuda_command_buffer_telemetry_pre_13_4.h"

namespace stream_executor::gpu::command_buffer_telemetry {

std::optional<uint32_t> GetGraphExecId(CUgraphExec /*exec*/) {
  return std::nullopt;
}

void RegisterPre134ExecGraphNodes(CUgraph /*root_graph*/,
                                  CUgraphExec /*exec*/) {}

}  // namespace stream_executor::gpu::command_buffer_telemetry
