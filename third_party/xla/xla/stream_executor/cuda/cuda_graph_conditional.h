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

#ifndef XLA_STREAM_EXECUTOR_CUDA_CUDA_GRAPH_CONDITIONAL_H_
#define XLA_STREAM_EXECUTOR_CUDA_CUDA_GRAPH_CONDITIONAL_H_

#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/gpu/graph_conditional.h"

namespace stream_executor::gpu {

class CudaGraphConditional final : public GraphConditional {
#if CUDA_VERSION >= 12030
 public:
  explicit CudaGraphConditional(CUgraphConditionalHandle handle)
      : handle_(handle) {}

  CUgraphConditionalHandle handle() const override { return handle_; }

 private:
  CUgraphConditionalHandle handle_;
#endif  // CUDA_VERSION >= 12030
};

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_CUDA_CUDA_GRAPH_CONDITIONAL_H_
