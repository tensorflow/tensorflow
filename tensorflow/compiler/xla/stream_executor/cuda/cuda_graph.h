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

#ifndef TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_CUDA_CUDA_GRAPH_H_
#define TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_CUDA_CUDA_GRAPH_H_

#include <cstdint>
#include <functional>
#include <memory>

#include "absl/functional/any_invocable.h"
#include "third_party/gpus/cuda/include/driver_types.h"
#include "tensorflow/compiler/xla/stream_executor/stream.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace stream_executor {
namespace gpu {

class CudaGraphSupport {
 public:
  // Deleters for CUDA graph and graph exec instance that check the returned
  // status and terminate if it's not `cudaSuccess`.
  struct DestroyGraph {
    void operator()(cudaGraph_t);
  };
  struct DestroyGraphExec {
    void operator()(cudaGraphExec_t);
  };
};

//===----------------------------------------------------------------------===//
// RAII helpers for CUDA graph types.
//===----------------------------------------------------------------------===//

class OwnedCudaGraph
    : public std::unique_ptr<std::remove_pointer_t<cudaGraph_t>,
                             CudaGraphSupport::DestroyGraph> {
  // Bring std::unique_ptr constructors in scope.
  using std::unique_ptr<std::remove_pointer_t<cudaGraph_t>,
                        CudaGraphSupport::DestroyGraph>::unique_ptr;
};

class OwnedCudaGraphExec
    : public std::unique_ptr<std::remove_pointer_t<cudaGraphExec_t>,
                             CudaGraphSupport::DestroyGraphExec> {
  // Bring std::unique_ptr constructors in scope.
  using std::unique_ptr<std::remove_pointer_t<cudaGraphExec_t>,
                        CudaGraphSupport::DestroyGraphExec>::unique_ptr;

 public:
  // Updates executable graph instance with a newly captured graph. Returns an
  // error if the new graph is not compatible (see `cudaGraphExecUpdate`).
  tsl::Status Update(OwnedCudaGraph graph);

  // Launches captured graph on a given stream.
  tsl::Status Launch(stream_executor::Stream* stream);

 private:
  uint64_t num_updates_ = 0;
  uint64_t num_launches_ = 0;
};

//===----------------------------------------------------------------------===//
// CUDA Graph Helpers.
//===----------------------------------------------------------------------===//

// Captures all operations added to a `stream` by the `capture` function into
// the cuda graph instance.
tsl::StatusOr<OwnedCudaGraph> CaptureCudaGraph(
    stream_executor::Stream* stream, absl::AnyInvocable<tsl::Status()> capture,
    cudaStreamCaptureMode mode = cudaStreamCaptureModeThreadLocal);

// Instantiates a captured cuda graph instance into a cuda graph executable.
tsl::StatusOr<OwnedCudaGraphExec> InstantiateCudaGraph(OwnedCudaGraph graph);

}  // namespace gpu
}  // namespace stream_executor

#endif  // TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_CUDA_CUDA_GRAPH_H_
