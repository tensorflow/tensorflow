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

#ifndef XLA_BACKENDS_GPU_RUNTIME_ASYNC_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_ASYNC_THUNK_H_

#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/runtime/async_execution.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/execution_stream_id.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/backends/gpu/runtime/thunk_executor.h"
#include "xla/stream_executor/command_buffer.h"

namespace xla::gpu {

//===-----------------------------------------------------------------------===/
// AsyncStartThunk
//===-----------------------------------------------------------------------===/

// AsyncStartThunk starts a new execution scope and executes a nested thunk
// sequence on a separate stream. Before executing nested thunks it adds a
// dependency from the compute stream to an async stream, so that all nested
// thunks correctly observe effects of all previously launched thunks.
//
// All nested thunks do not need to know anything about the async execution
// scope, they only must correctly synchronize all launched operations with a
// compute stream that is given to them. AsyncStartThunk together with an
// AsyncDoneThunk allows XLA runtime to express structured concurrency that is
// correct by construction, i.e. it allows to express nested async execution
// scopes.
class AsyncStartThunk : public Thunk {
 public:
  AsyncStartThunk(ThunkInfo thunk_info, ExecutionStreamId execution_stream_id,
                  ThunkSequence thunks);

  // Constructor that shares an existing AsyncExecution with another
  // AsyncStartThunk. Used for pipelined send/recv where multiple operations
  // share the same async stream.
  AsyncStartThunk(ThunkInfo thunk_info, ExecutionStreamId execution_stream_id,
                  ThunkSequence thunks,
                  std::shared_ptr<AsyncExecution> async_execution);

  absl::Status Prepare(const PrepareParams& params) override;
  absl::Status Initialize(const InitializeParams& params) override;
  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

  ExecutionStreamId execution_stream_id() const { return execution_stream_id_; }

  const ThunkSequence& thunks() const { return executor_.thunks(); }

  absl::StatusOr<ThunkProto> ToProto() const override;
  static absl::StatusOr<std::unique_ptr<AsyncStartThunk>> FromProto(
      ThunkInfo thunk_info, const AsyncStartThunkProto& proto,
      const Deserializer& deserializer, AsyncExecutionMap& async_executions);

  AsyncExecutionId async_execution_id() const;
  std::shared_ptr<AsyncExecution> async_execution() const;

  std::string ToString(int indent) const override;

 protected:
  absl::Status WalkNested(Walker callback) override;
  absl::Status TransformNested(Transformer callback) override;

 private:
  ExecutionStreamId execution_stream_id_;
  ThunkExecutor executor_;
  std::shared_ptr<AsyncExecution> async_execution_;
};

//===-----------------------------------------------------------------------===/
// AsyncDoneThunk
//===-----------------------------------------------------------------------===/

// AsyncDoneThunk completes an asynchronous execution scope started by a
// corresponding AsyncStartThunk (for pipelined async operations async execution
// scope owned by a canonical start operation). It synchronizes the compute
// stream with the async operation's completion event.
//
// Also implements the Command interface so it can be recorded directly into
// command buffers as an empty dependency node when needed.
class AsyncDoneThunk : public Command {
 public:
  AsyncDoneThunk(ThunkInfo thunk_info,
                 std::shared_ptr<AsyncExecution> async_execution);

  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

  absl::StatusOr<const se::CommandBuffer::Command*> Record(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer) override;

  // AsyncDoneThunk touches no buffers; this is explicit rather than relying on
  // the base class default.
  BufferUses buffer_uses() const override { return {}; }

  absl::StatusOr<ThunkProto> ToProto() const override;
  static absl::StatusOr<std::unique_ptr<AsyncDoneThunk>> FromProto(
      ThunkInfo thunk_info, const AsyncDoneThunkProto& proto,
      AsyncExecutionMap& async_executions);

  AsyncExecutionId async_execution_id() const;
  std::shared_ptr<AsyncExecution> async_execution() const;

  std::string ToString(int indent) const override;

 private:
  std::shared_ptr<AsyncExecution> async_execution_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_ASYNC_THUNK_H_
