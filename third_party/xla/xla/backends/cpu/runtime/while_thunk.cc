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

#include "xla/backends/cpu/runtime/while_thunk.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xla/backends/cpu/runtime/buffer_allocations.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/thunk_executor.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::cpu {

absl::StatusOr<std::unique_ptr<WhileThunk>> WhileThunk::Create(
    Info info, BufferAllocation::Slice cond_buffer, ThunkSequence cond_sequence,
    ThunkSequence body_sequence, std::optional<int64_t> trip_count) {
  TF_ASSIGN_OR_RETURN(ThunkExecutor cond_executor,
                      ThunkExecutor::Create(std::move(cond_sequence)));
  TF_ASSIGN_OR_RETURN(ThunkExecutor body_executor,
                      ThunkExecutor::Create(std::move(body_sequence)));
  return absl::WrapUnique(new WhileThunk(std::move(info), cond_buffer,
                                         std::move(cond_executor),
                                         std::move(body_executor), trip_count));
}

WhileThunk::WhileThunk(Info info, BufferAllocation::Slice cond_buffer,
                       ThunkExecutor cond_executor, ThunkExecutor body_executor,
                       std::optional<int64_t> trip_count)
    : Thunk(Kind::kWhile, std::move(info)),
      cond_buffer_(cond_buffer),
      cond_executor_(std::move(cond_executor)),
      body_executor_(std::move(body_executor)),
      trip_count_(trip_count) {}

tsl::AsyncValueRef<Thunk::ExecuteEvent> WhileThunk::Execute(
    const ExecuteParams& params) {
  VLOG(3) << absl::StreamFormat(
      "While: #trip_count=%s",
      trip_count_.has_value() ? absl::StrCat(*trip_count_) : "unknown");

  // Most of the while loops in XLA have statically known trip count.
  if (ABSL_PREDICT_TRUE(trip_count_.has_value())) {
    return ExecuteForLoop(params, *trip_count_);
  }

  const BufferAllocations* allocations = params.buffer_allocations;

  se::DeviceMemoryBase cond_data;
  if (ShouldCheckBufferSlices()) {
    TF_ASSIGN_OR_RETURN(cond_data, allocations->GetDeviceAddress(cond_buffer_));
  } else {
    cond_data = allocations->GetDeviceAddressUnchecked(cond_buffer_);
  }

  bool* condition = reinterpret_cast<bool*>(cond_data.opaque());
  return ExecuteWhileLoop(params, condition);
}

tsl::AsyncValueRef<WhileThunk::ExecuteEvent> WhileThunk::ExecuteForLoop(
    const ExecuteParams& params, int64_t trip_count) {
  for (int64_t loop_counter = 0; loop_counter < trip_count; ++loop_counter) {
    auto body_event = body_executor_.Execute(params);

    // If loop iteration has not completed yet, switch to async execution mode
    // using `body_event` as a dependency and continue the loop iteration
    // starting from `loop_counter + 1`.
    if (ABSL_PREDICT_FALSE(!body_event.IsAvailable())) {
      return ExecuteAsyncForLoop(params, std::move(body_event),
                                 loop_counter + 1, trip_count);
    }

    if (ABSL_PREDICT_FALSE(body_event.IsError())) {
      return body_event.GetError();
    }

    DCHECK(body_event.IsConcrete());
  }

  // Successfully completed `trip_count` while loop iterations.
  return OkExecuteEvent();
}

tsl::AsyncValueRef<WhileThunk::ExecuteEvent> WhileThunk::ExecuteWhileLoop(
    const ExecuteParams& params, bool* condition) {
  // Execute `cond` thunk sequence to initialize the loop condition.
  auto init_event = cond_executor_.Execute(params);

  // If we don't know if we should continue or not, switch to async execution
  // mode using `init_event` as a dependency.
  if (ABSL_PREDICT_FALSE(!init_event.IsAvailable())) {
    return ExecuteAsyncWhileLoop(params, std::move(init_event), condition);
  }

  // Immediately forward error to the caller.
  if (ABSL_PREDICT_FALSE(init_event.IsError())) {
    return init_event.GetError();
  }

  DCHECK(init_event.IsConcrete());

  while (*condition) {
    auto body_event = body_executor_.Execute(params);
    auto cond_event = body_event.FlatMap([this, &params](ExecuteEvent) {
      return cond_executor_.Execute(params);
    });

    // If loop iteration has not completed yet, switch to async execution mode
    // using `cond_event` as a dependency and maybe continue the loop
    // iteration (if `condition` is still true).
    if (ABSL_PREDICT_FALSE(!cond_event.IsAvailable())) {
      return ExecuteAsyncWhileLoop(params, std::move(cond_event), condition);
    }

    // Immediately forward error to the caller.
    if (ABSL_PREDICT_FALSE(cond_event.IsError())) {
      return cond_event.GetError();
    }

    // At this point `*condition` should have been updated and we may continue
    // executing the while loop in the current thread.
    DCHECK(cond_event.IsConcrete());
  }

  // Successfully completed while loop iterations.
  return OkExecuteEvent();
}

tsl::AsyncValueRef<WhileThunk::ExecuteEvent> WhileThunk::ExecuteAsyncForLoop(
    const ExecuteParams& params, tsl::AsyncValueRef<ExecuteEvent> dependency,
    int64_t loop_counter, int64_t trip_count) {
  auto event = tsl::MakeConstructedAsyncValueRef<ExecuteEvent>();

  // Allocate while loop iteration function on heap so we can detach its life
  // time from the caller stack.
  auto loop_fn = std::make_shared<std::function<void(int64_t, absl::Status)>>();
  *loop_fn = [this, trip_count, &params, event, loop = loop_fn.get()](
                 int64_t loop_counter, absl::Status status) {
    // Dependency completed with an error. Forward it to the result event.
    if (ABSL_PREDICT_FALSE(!status.ok())) {
      event.SetError(std::move(status));
      return;
    }

    for (; loop_counter < trip_count; ++loop_counter) {
      auto body_event = body_executor_.Execute(params);

      // If loop iteration has not completed yet, continue execution
      // asynchronously starting from `loop_counter + 1`.
      if (!body_event.IsAvailable()) {
        body_event.AndThen([loop, loop_counter](absl::Status status) {
          (*loop)(loop_counter + 1, std::move(status));
        });
        return;
      }

      // Immediately forward error to the caller.
      if (ABSL_PREDICT_FALSE(body_event.IsError())) {
        event.SetError(body_event.GetError());
        return;
      }

      DCHECK(body_event.IsConcrete());
    }

    // Successfully completed `trip_count` while loop iterations.
    event.SetStateConcrete();
  };

  // Kick-off loop execution once dependency event is available.
  dependency.AndThen([loop_counter, loop = loop_fn.get()](absl::Status status) {
    (*loop)(loop_counter, std::move(status));
  });

  // Keep `loop_fn` alive until the end of the while loop execution.
  event.AndThen([loop_fn = std::move(loop_fn)]() {});

  return event;
}

tsl::AsyncValueRef<WhileThunk::ExecuteEvent> WhileThunk::ExecuteAsyncWhileLoop(
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

      // If loop iteration has not completed yet, continue execution
      // asynchronously (if `condition` is still true when it becomes ready).
      if (!cond_event.IsAvailable()) {
        cond_event.AndThen(
            [loop](absl::Status status) { (*loop)(std::move(status)); });
        return;
      }

      // Immediately forward error to the caller.
      if (ABSL_PREDICT_FALSE(cond_event.IsError())) {
        event.SetError(cond_event.GetError());
        return;
      }

      // At this point `*condition` should have been updated and we may continue
      // executing the while loop in the current thread.
      DCHECK(cond_event.IsConcrete());
    }

    // Successfully completed while loop iterations.
    event.SetStateConcrete();
  };

  // Kick-off loop execution once dependency event is available.
  dependency.AndThen([loop = loop_fn.get()](absl::Status status) {
    (*loop)(std::move(status));
  });

  // Keep `loop_fn` alive until the end of the while loop execution.
  event.AndThen([loop_fn = std::move(loop_fn)]() {});

  return event;
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

std::vector<std::pair<std::string, const ThunkSequence*>>
WhileThunk::nested_thunks() const {
  std::string maybe_trip_count_info =
      trip_count_.has_value() ? absl::StrCat(" trip_count=", *trip_count_) : "";
  return {{absl::StrCat(info().op_name, "-while-condition"),
           &cond_executor_.thunk_sequence()},
          {absl::StrCat(info().op_name, "-while-body", maybe_trip_count_info),
           &body_executor_.thunk_sequence()}};
}

}  // namespace xla::cpu
