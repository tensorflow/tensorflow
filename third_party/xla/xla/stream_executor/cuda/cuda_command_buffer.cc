/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/stream_executor/cuda/cuda_command_buffer.h"

#include <memory>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/cuda/cuda_status.h"
#include "xla/stream_executor/gpu/gpu_executor.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace stream_executor::gpu {
namespace {
absl::StatusOr<CUgraph> CreateGraph() {
  VLOG(2) << "Create new CUDA graph";
  CUgraph graph = nullptr;
  TF_RETURN_IF_ERROR(cuda::ToStatus(cuGraphCreate(&graph, /*flags=*/0),
                                    "Failed to create CUDA graph"));
  VLOG(2) << "Created CUDA graph " << graph;
  return graph;
}
}  // namespace

absl::StatusOr<std::unique_ptr<CudaCommandBuffer>> CudaCommandBuffer::Create(
    Mode mode, GpuExecutor* parent) {
  TF_ASSIGN_OR_RETURN(CUgraph graph, CreateGraph());
  return std::make_unique<CudaCommandBuffer>(mode, parent, graph,
                                             /*is_owned_graph=*/true);
}

}  // namespace stream_executor::gpu
