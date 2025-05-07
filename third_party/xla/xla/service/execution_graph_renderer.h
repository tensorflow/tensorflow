/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_EXECUTION_GRAPH_RENDERER_H_
#define XLA_SERVICE_EXECUTION_GRAPH_RENDERER_H_

#include <memory>
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/runtime/execution_graph.h"

namespace xla {

class ExecutionGraphRenderer {
 public:
  ExecutionGraphRenderer() = default;
  virtual ~ExecutionGraphRenderer() = default;

  // Generates a string representation for the given execution graph and thunk
  // sequence which can be published to a URL using `PublishGraph`.
  virtual std::string GenerateGraphAsString(
      const ExecutionGraph& execution_graph,
      const cpu::ThunkSequence& thunk_sequence) = 0;

  // Publishes the generated graph.
  virtual absl::StatusOr<std::string> PublishGraph(
      absl::string_view graph_as_string) = 0;
};

// Returns the registered renderer for execution graphs.
ExecutionGraphRenderer* GetExecutionGraphRenderer();

// Registers a renderer for execution graphs.
void RegisterExecutionGraphRenderer(
    std::unique_ptr<ExecutionGraphRenderer> renderer);

}  // namespace xla

#endif  // XLA_SERVICE_EXECUTION_GRAPH_RENDERER_H_
