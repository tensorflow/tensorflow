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

#ifndef XLA_BACKENDS_GPU_RUNTIME_EXECUTION_STREAM_ID_H_
#define XLA_BACKENDS_GPU_RUNTIME_EXECUTION_STREAM_ID_H_

#include <cstdint>
#include <variant>

#include "absl/strings/str_format.h"
#include "xla/tsl/lib/gtl/int_type.h"

namespace xla::gpu {

// XLA:GPU executes compiled GPU executables on the "main" stream passed via
// `ExecuteParams` to the `ThunkSequence`. It requires that all side effects
// must be visible on this stream before the XLA GPU executable starts running,
// and XLA guarantees that when execution completes all side effects will be
// visible on the same stream.
//
// However, during execution XLA can choose to run on multiple streams, and XLA
// guarantees that they are all properly synchronized with the main stream
// before the executable returns. See below for the kinds of additional streams
// XLA might use at run time.
//
// Execution on multiple streams is implemented with structured concurrency
// using `AsyncStartThunk` and `AsyncDoneThunk`:
//
//   1. `AsyncStartThunk` records an event on the current stream (waits for all
//      prior work completion) and launches nested thunks on a dedicated async
//      execution stream.
//
//   2. `AsyncDoneThunk` synchronizes the async execution stream back to the
//      parent stream, which is always the same stream that was used to start
//      the async work.
//
// In general, XLA's goal is to overlap as much communication with computation
// (or in general thunks that use different hardware resources) to get better
// utilization of available hardware resources.
//
// IMPORTANT: Don't confuse this with multi-streaming inside a command buffer.
// Conceptually a command buffer is always launched on one stream, and based on
// the DAG structure it can decide to run things concurrently. However, this is
// not exposed to the XLA runtime layer; from the XLA runtime perspective
// everything runs on a single stream.

// Additional computation streams that can be used for launching compute-bound
// kernels. These kinds of streams are used for launching async fusions and
// async calls.
TSL_LIB_GTL_DEFINE_INT_TYPE(ComputationStreamId, uint64_t);

// Additional streams for launching communication operations.
TSL_LIB_GTL_DEFINE_INT_TYPE(CommunicationStreamId, uint64_t);

// Identifies an additional execution stream used by a thunk. This is either a
// computation stream (for async fusions/calls) or a communication stream (for
// collective operations). Default execution stream is not a computation or
// communication stream, and it's always implicitly available. ExecutionStreamId
// always identifies one of the additional streams used at run time.
class ExecutionStreamId {
 public:
  // Implicit conversions from strongly-typed stream ids for convenience.
  // NOLINTNEXTLINE(google-explicit-constructor)
  ExecutionStreamId(ComputationStreamId id) : id_(id) {}
  // NOLINTNEXTLINE(google-explicit-constructor)
  ExecutionStreamId(CommunicationStreamId id) : id_(id) {}

  bool is_computation() const {
    return std::holds_alternative<ComputationStreamId>(id_);
  }

  bool is_communication() const {
    return std::holds_alternative<CommunicationStreamId>(id_);
  }

  ComputationStreamId computation_id() const {
    return std::get<ComputationStreamId>(id_);
  }

  CommunicationStreamId communication_id() const {
    return std::get<CommunicationStreamId>(id_);
  }

  friend bool operator==(const ExecutionStreamId& a,
                         const ExecutionStreamId& b) {
    return a.id_ == b.id_;
  }

  friend bool operator!=(const ExecutionStreamId& a,
                         const ExecutionStreamId& b) {
    return !(a == b);
  }

  template <typename H>
  friend H AbslHashValue(H h, const ExecutionStreamId& id) {
    return H::combine(std::move(h), id.id_);
  }

 private:
  std::variant<ComputationStreamId, CommunicationStreamId> id_;
};

template <typename Sink>
void AbslStringify(Sink& sink, ComputationStreamId id) {
  absl::Format(&sink, "computation_stream[%v]", id.value());
}

template <typename Sink>
void AbslStringify(Sink& sink, CommunicationStreamId id) {
  absl::Format(&sink, "communication_stream[%v]", id.value());
}

template <typename Sink>
void AbslStringify(Sink& sink, const ExecutionStreamId& id) {
  if (id.is_computation()) {
    absl::Format(&sink, "%v", id.computation_id());
  } else {
    absl::Format(&sink, "%v", id.communication_id());
  }
}

// Number of async compute execution streams.
inline constexpr int kDefaultNumComputeStreams = 4;

// Number of async communication streams when multi-streaming is enabled.
inline constexpr int kDefaultNumCommunicationStreams = 2;

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_EXECUTION_STREAM_ID_H_
