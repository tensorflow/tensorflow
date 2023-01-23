/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/stream_executor/cuda/cuda_graph.h"

#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_stream.h"

namespace stream_executor {
namespace gpu {

void CudaGraphSupport::DestroyGraph::operator()(cudaGraph_t graph) {
  cudaError_t err = cudaGraphDestroy(graph);
  CHECK(err == cudaSuccess)
      << "Failed to destroy CUDA graph: " << cudaGetErrorString(err);
}

void CudaGraphSupport::DestroyGraphExec::operator()(cudaGraphExec_t instance) {
  cudaError_t err = cudaGraphExecDestroy(instance);
  CHECK(err == cudaSuccess)
      << "Failed to destroy CUDA graph instance: " << cudaGetErrorString(err);
}

}  // namespace gpu
}  // namespace stream_executor
