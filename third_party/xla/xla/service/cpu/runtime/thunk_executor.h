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

#ifndef XLA_SERVICE_CPU_RUNTIME_THUNK_EXECUTOR_H_
#define XLA_SERVICE_CPU_RUNTIME_THUNK_EXECUTOR_H_

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/service/cpu/runtime/thunk.h"

namespace xla::cpu {

// A dataflow-style (run when ready) executor for a ThunkSequence that depends
// on buffer uses to build a DAG defining execution order. At run time executes
// thunks concurrently in a given thread pool.
class ThunkExecutor {
 public:
  using NodeId = int64_t;

  static constexpr NodeId kInvalidNodeId = std::numeric_limits<NodeId>::min();

  ThunkExecutor(ThunkExecutor&&) = default;
  ThunkExecutor& operator=(ThunkExecutor&&) = default;

  static absl::StatusOr<ThunkExecutor> Create(ThunkSequence thunk_sequence);

  // NodeDef defines an execution order for all thunks in a sequence.
  struct NodeDef {
    NodeId id = kInvalidNodeId;
    std::vector<NodeId> out_edges;
    std::vector<NodeId> in_edges;
  };

  absl::Span<const NodeDef> nodes_defs() const { return nodes_defs_; }
  absl::Span<const NodeId> source() const { return source_; }
  absl::Span<const NodeId> sink() const { return sink_; }

  std::string ToString() const;

 private:
  ThunkExecutor(ThunkSequence thunk_sequence, std::vector<NodeDef> nodes_defs);

  ThunkSequence thunk_sequence_;
  std::vector<NodeDef> nodes_defs_;

  std::vector<NodeId> source_;
  std::vector<NodeId> sink_;
};

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_RUNTIME_THUNK_EXECUTOR_H_
