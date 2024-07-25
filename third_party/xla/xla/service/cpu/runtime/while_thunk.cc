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

#include "xla/service/cpu/runtime/while_thunk.h"

#include <functional>
#include <memory>
#include <utility>

#include "absl/base/optimization.h"
#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/runtime/thunk.h"
#include "xla/service/cpu/runtime/thunk_executor.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla::cpu {

absl::StatusOr<std::unique_ptr<WhileThunk>> WhileThunk::Create(
    Info info, BufferAllocation::Slice cond_buffer, ThunkSequence cond_sequence,
    ThunkSequence body_sequence) {
  TF_ASSIGN_OR_RETURN(ThunkExecutor cond_executor,
                      ThunkExecutor::Create(std::move(cond_sequence)));
  TF_ASSIGN_OR_RETURN(ThunkExecutor body_executor,
                      ThunkExecutor::Create(std::move(body_sequence)));
  return absl::WrapUnique(new WhileThunk(std::move(info), cond_buffer,
                                         std::move(cond_executor),
                                         std::move(body_executor)));
}

WhileThunk::WhileThunk(Info info, BufferAllocation::Slice cond_buffer,
                       ThunkExecutor cond_executor, ThunkExecutor body_executor)
    : Thunk(Kind::kWhile, std::move(info)),
      cond_buffer_(cond_buffer),
      cond_executor_(std::move(cond_executor)),
      body_executor_(std::move(body_executor)) {}

tsl::AsyncValueRef<WhileThunk::ExecuteEvent> WhileThunk::ExecuteAsync(
    const ExecuteParams& params, tsl::AsyncValueRef<ExecuteEvent> dependency,
    bool* condition) {
  auto event = tsl::MakeConstructedAsyncValueRef<ExecuteEvent>();

  // Allocate while loop iteration function on heap so we can detach its life
  // time from the caller stack.
  auto loop_fn = std::make_shared<std::function<void(absl::Status)>>();
  *loop_fn = [this, condition, &params, event,
              loop = loop_fn.get()](absl::Status status) {
    // Dependency completed with an error. Forward it to the result event.
    if (ABSL_PREDICT_FALSE(!status.ok())) {
      event.SetError(std::move(status));
      return;
    }

    while (*condition) {
      auto body_event = body_executor_.Execute(params);
      auto cond_event = body_event.FlatMap([this, &params](ExecuteEvent) {
        return cond_executor_.Execute(params);
      });

      // Immediately forward error to the caller.
      if (ABSL_PREDICT_FALSE(cond_event.IsError())) {
        event.SetError(cond_event.GetError());
        return;
      }

      // If we don't know yet wether we should execute the next iteration or
      // not, attach `AndThen` continuation to the `cond_event`.
      if (!cond_event.IsAvailable()) {
        cond_event.AndThen(
            [loop](absl::Status status) { (*loop)(std::move(status)); });
        return;
      }

      // At this point `*condition` should have been updated and we may continue
      // executing the while loop in the current thread.
      DCHECK(cond_event.IsAvailable());
    }

    // Successfully completed while loop iterations.
    event.SetStateConcrete();
  };

  // Kick-off loop execution once dependency event is available.
  dependency.AndThen(*loop_fn);

  // Keep `loop_fn` alive until the end of the while loop execution.
  event.AndThen([loop_fn = std::move(loop_fn)]() {});

  return event;
}

tsl::AsyncValueRef<Thunk::ExecuteEvent> WhileThunk::Execute(
    const ExecuteParams& params) {
  tsl::profiler::TraceMe trace([&] { return TraceMeEncode(); });

  TF_ASSIGN_OR_RETURN(
      se::DeviceMemoryBase cond_data,
      params.buffer_allocations->GetDeviceAddress(cond_buffer_));

  bool* condition = reinterpret_cast<bool*>(cond_data.opaque());

  // Execute `cond` thunk sequence to initialize the loop condition.
  auto init_event = cond_executor_.Execute(params);

  // Immediately forward error to the caller.
  if (ABSL_PREDICT_FALSE(init_event.IsError())) {
    return init_event.GetError();
  }

  // If we don't know if we should continue or not, switch to async execution
  // mode using `init_event` as a dependency.
  if (ABSL_PREDICT_FALSE(!init_event.IsAvailable())) {
    return ExecuteAsync(params, std::move(init_event), condition);
  }

  while (*condition) {
    auto body_event = body_executor_.Execute(params);
    auto cond_event = body_event.FlatMap([this, &params](ExecuteEvent) {
      return cond_executor_.Execute(params);
    });

    // Immediately forward error to the caller.
    if (ABSL_PREDICT_FALSE(cond_event.IsError())) {
      return cond_event.GetError();
    }

    // If we don't know if we should continue or not, switch to async execution
    // mode using `cond_event` as a dependency.
    if (ABSL_PREDICT_FALSE(!cond_event.IsAvailable())) {
      return ExecuteAsync(params, std::move(cond_event), condition);
    }

    // At this point `*condition` should have been updated and we may continue
    // executing the while loop in the current thread.
    DCHECK(cond_event.IsAvailable());
  }

  // Successfully completed while loop iterations.
  return OkExecuteEvent();
}

WhileThunk::BufferUses WhileThunk::buffer_uses() const {
  BufferUses buffer_uses = {{cond_buffer_, BufferUse::kWrite}};

  BufferUses cond_uses = cond_executor_.buffer_uses();
  buffer_uses.insert(buffer_uses.end(), cond_uses.begin(), cond_uses.end());

  BufferUses body_uses = body_executor_.buffer_uses();
  buffer_uses.insert(buffer_uses.end(), body_uses.begin(), body_uses.end());

  return buffer_uses;
}

WhileThunk::ResourceUses WhileThunk::resource_uses() const {
  ResourceUses resource_uses;

  ResourceUses cond_uses = cond_executor_.resource_uses();
  resource_uses.insert(resource_uses.end(), cond_uses.begin(), cond_uses.end());

  ResourceUses body_uses = body_executor_.resource_uses();
  resource_uses.insert(resource_uses.end(), body_uses.begin(), body_uses.end());

  return resource_uses;
}

}  // namespace xla::cpu
