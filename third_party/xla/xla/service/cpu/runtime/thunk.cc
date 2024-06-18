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

#include "xla/service/cpu/runtime/thunk.h"

#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>

#include "xla/executable_run_options.h"
#include "xla/service/global_device_id.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "tsl/platform/logging.h"
#include "tsl/profiler/lib/traceme.h"
#include "tsl/profiler/lib/traceme_encode.h"

namespace xla::cpu {

std::string_view Thunk::KindToString(Kind kind) {
  switch (kind) {
    case Kind::kAllReduce:
      return "all-reduce";
    case Kind::kCall:
      return "call";
    case Kind::kCopy:
      return "copy";
    case Kind::kConditional:
      return "conditional";
    case Kind::kDot:
      return "dot";
    case Kind::kFft:
      return "fft";
    case Kind::kInfeed:
      return "infeed";
    case Kind::kRngGetAndUpdateState:
      return "rng-get-and-update-state";
    case Kind::kKernel:
      return "kernel";
    case Kind::kOutfeed:
      return "outfeed";
    case Kind::kWhile:
      return "while";
  }
}

absl::StatusOr<Thunk::CollectiveExecuteParams>
Thunk::CollectiveExecuteParams::Create(
    const ExecutableRunOptions* run_options) {
  // Device ordinal must be set by caller and passed in run options, if not,
  // we use the device ordinal from the parent StreamExecutor.
  int32_t device_ordinal =
      run_options->device_ordinal() >= 0
          ? run_options->device_ordinal()
          : run_options->stream()->parent()->device_ordinal();

  return CollectiveExecuteParams{run_options->run_id(), device_ordinal,
                                 GlobalDeviceId(run_options->device_ordinal()),
                                 run_options->device_assignment()};
}

Thunk::CollectiveExecuteParams::CollectiveExecuteParams(
    RunId run_id, int64_t local_device_ordinal, GlobalDeviceId global_device_id,
    const DeviceAssignment* device_assignment)
    : run_id(run_id),
      local_device_ordinal(local_device_ordinal),
      global_device_id(global_device_id),
      device_assignment(device_assignment) {}

tsl::AsyncValueRef<Thunk::ExecuteEvent> Thunk::OkExecuteEvent() {
  static tsl::AsyncValueOwningRef<ExecuteEvent>* event = [] {
    auto* storage = new tsl::internal::AsyncValueStorage<ExecuteEvent>();
    return new tsl::AsyncValueOwningRef<ExecuteEvent>(
        tsl::MakeAvailableAsyncValueRef<ExecuteEvent>(*storage));
  }();
  return event->AsRef();
}

// Encodes thunk info into the TraceMe compatible format.
std::string Thunk::TraceMeEncode() const {
  return tsl::profiler::TraceMeEncode(info_.op_name,
                                      {{"hlo_op", info_.op_name},
                                       {"hlo_module", info_.module_name},
                                       {"program_id", info_.module_id}});
}

std::ostream& operator<<(std::ostream& os, Thunk::Kind kind) {
  os << Thunk::KindToString(kind);
  return os;
}

ThunkSequence::ThunkSequence(std::unique_ptr<Thunk> thunk) {
  push_back(std::move(thunk));
}

void ThunkSequence::Append(ThunkSequence other) {
  reserve(size() + other.size());
  for (auto& thunk : other) {
    push_back(std::move(thunk));
  }
}

ThunkSequence::BufferUses ThunkSequence::buffer_uses() const {
  BufferUses buffer_uses;
  for (auto& thunk : *this) {
    BufferUses uses = thunk->buffer_uses();
    buffer_uses.insert(buffer_uses.end(), uses.begin(), uses.end());
  }
  return buffer_uses;
}

}  // namespace xla::cpu
