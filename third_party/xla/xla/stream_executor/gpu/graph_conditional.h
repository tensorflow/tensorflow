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

#ifndef XLA_STREAM_EXECUTOR_GPU_GRAPH_CONDITIONAL_H_
#define XLA_STREAM_EXECUTOR_GPU_GRAPH_CONDITIONAL_H_

#include "absl/types/span.h"
#include "xla/stream_executor/gpu/gpu_types.h"

namespace stream_executor::gpu {

// Represents a backend specific conditional handle in a command buffer.
// It's an implementation detail of GpuCommandBuffer and should not be used
// outside of it and its subclasses.
class GraphConditional {
 public:
  virtual ~GraphConditional() = default;

  // Returns a handle to the conditional node. This will go away once we
  // migrate all GpuDriver calls into subclasses.
  virtual GpuGraphConditionalHandle handle() const = 0;
};

using GraphConditionals = absl::Span<GraphConditional* const>;

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_GRAPH_CONDITIONAL_H_
