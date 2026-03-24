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

#include "xla/backends/cpu/runtime/thunk.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <utility>

#include "absl/base/no_destructor.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/cpu/collectives/cpu_collectives.h"
#include "xla/backends/cpu/collectives/in_process_collectives.h"
#include "xla/backends/cpu/runtime/ynnpack/ynn_interop.h"
#include "xla/backends/cpu/runtime/ynnpack/ynn_threadpool.h"
#include "xla/executable_run_options.h"
#include "xla/runtime/device_id.h"
#include "xla/service/cpu/cpu_executable_run_options.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/profiler/lib/traceme.h"
#include "tsl/profiler/lib/traceme_encode.h"

namespace xla::cpu {

// Ok execute event allocated with the static storage duration.
static tsl::internal::AsyncValueStorage<Thunk::ExecuteEvent> ok_storage;
absl::NoDestructor<tsl::AsyncValueOwningRef<Thunk::ExecuteEvent>>
    Thunk::ok_event_(tsl::MakeAvailableAsyncValueRef<ExecuteEvent>(ok_storage));

absl::string_view Thunk::KindToString(Kind kind) {
  switch (kind) {
    case Kind::kCall:
      return "call";
    case Kind::kCollective:
      return "collective";
    case Kind::kConditional:
      return "conditional";
    case Kind::kConvolution:
      return "convolution";
    case Kind::kCopy:
      return "copy";
    case Kind::kCustomCall:
      return "custom-call";
    case Kind::kDot:
      return "dot";
    case Kind::kFft:
      return "fft";
    case Kind::kInfeed:
      return "infeed";
    case Kind::kKernel:
      return "kernel";
    case Kind::kOutfeed:
      return "outfeed";
    case Kind::kPartitionId:
      return "partition-id";
    case Kind::kReplicaId:
      return "replica-id";
    case Kind::kRngGetAndUpdateState:
      return "rng-get-and-update-state";
    case Kind::kSort:
      return "sort";
    case Kind::kTopK:
      return "topk";
    case Kind::kWhile:
      return "while";
    case Kind::kYnnFusion:
      return "ynn-fusion";
    case Kind::kOneDnnFusion:
      return "onednn-fusion";
  }
}

Thunk::Thunk(Kind kind, Info info) : kind_(kind), info_(std::move(info)) {}

absl::StatusOr<Thunk::CollectiveExecuteParams>
Thunk::CollectiveExecuteParams::Create(
    const ExecutableRunOptions* run_options) {
  // Device ordinal must be set by caller and passed in run options, if not,
  // we use the device ordinal from the parent StreamExecutor.
  int32_t device_ordinal =
      run_options->device_ordinal() >= 0
          ? run_options->device_ordinal()
          : run_options->stream()->parent()->device_ordinal();

  // Default implementation of a collectives interface that can execute
  // collective operations within the same process.
  static CpuCollectives* const in_process_collectives =
      new InProcessCollectives();

  // If CPU executable run options are set, use the collectives interface
  // provided by the executable run options if it is set. Otherwise, use the
  // in-process collectives interface.
  const CpuExecutableRunOptions* cpu_run_options =
      run_options->cpu_executable_run_options();
  CpuCollectives* collectives =
      cpu_run_options && cpu_run_options->collectives()
          ? cpu_run_options->collectives()
          : in_process_collectives;

  return CollectiveExecuteParams{run_options->run_id(), device_ordinal,
                                 GlobalDeviceId(run_options->device_ordinal()),
                                 run_options->device_assignment(), collectives};
}

Thunk::CollectiveExecuteParams::CollectiveExecuteParams(
    RunId run_id, int64_t local_device_ordinal, GlobalDeviceId global_device_id,
    const DeviceAssignment* device_assignment, CpuCollectives* collectives)
    : run_id(run_id),
      local_device_ordinal(local_device_ordinal),
      global_device_id(global_device_id),
      device_assignment(device_assignment),
      collectives(collectives) {}

absl::StatusOr<Thunk::CustomCallExecuteParams>
Thunk::CustomCallExecuteParams::Create(
    const ExecutableRunOptions* run_options) {
  // Device ordinal must be set by caller and passed in run options, if not,
  // we use the device ordinal from the parent StreamExecutor.
  int32_t device_ordinal =
      run_options->device_ordinal() >= 0
          ? run_options->device_ordinal()
          : run_options->stream()->parent()->device_ordinal();

  return CustomCallExecuteParams{run_options->run_id(), device_ordinal,
                                 run_options->intra_op_thread_pool(),
                                 run_options->ffi_execution_context()};
}

Thunk::CustomCallExecuteParams::CustomCallExecuteParams(
    RunId run_id, int32_t device_ordinal,
    const Eigen::ThreadPoolDevice* intra_op_thread_pool,
    const ffi::ExecutionContext* ffi_execution_context)
    : run_id(run_id),
      device_ordinal(device_ordinal),
      intra_op_thread_pool(intra_op_thread_pool),
      ffi_execution_context(ffi_execution_context) {}

absl::StatusOr<Thunk::YnnParams> Thunk::YnnParams::Create(
    const ExecutableRunOptions* run_options) {
  TF_ASSIGN_OR_RETURN(YnnThreadpool threadpool,
                      CreateYnnThreadpool(run_options->intra_op_thread_pool()));
  return YnnParams(std::move(threadpool));
}

Thunk::YnnParams::YnnParams(YnnThreadpool threadpool)
    : threadpool(std::move(threadpool)) {}

Thunk::ExecuteSession::ExecuteSession(int64_t max_workers,
                                      int64_t split_threshold)
    : lock_(std::make_shared<std::nullopt_t>(std::nullopt)),
      max_workers_(max_workers),
      split_threshold_(split_threshold) {}

// Encodes thunk info into the TraceMe compatible format.
std::string Thunk::TraceMeEncode(int64_t run_id, int64_t device_ordinal) const {
  return tsl::profiler::TraceMeEncode(info_.op_name,
                                      {{"hlo_op", info_.op_name},
                                       {"hlo_module", info_.module_name},
                                       {"program_id", info_.module_id},
                                       {"run_id", run_id},
                                       {"device_ordinal", device_ordinal}});
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

ThunkSequence::ResourceUses ThunkSequence::resource_uses() const {
  ResourceUses resource_uses;
  for (auto& thunk : *this) {
    ResourceUses uses = thunk->resource_uses();
    resource_uses.insert(resource_uses.end(), uses.begin(), uses.end());
  }
  return resource_uses;
}

static void ForEach(const ThunkSequence& sequence,
                    absl::FunctionRef<void(const Thunk&)> fn) {
  for (auto& thunk : sequence) {
    fn(*thunk);
    for (auto& [name, nested] : thunk->nested_thunks()) {
      ForEach(*nested, fn);
    }
  }
}

static absl::Status ForEach(const ThunkSequence& sequence,
                            absl::FunctionRef<absl::Status(const Thunk&)> fn) {
  for (auto& thunk : sequence) {
    TF_RETURN_IF_ERROR(fn(*thunk));
    for (auto& [name, nested] : thunk->nested_thunks()) {
      TF_RETURN_IF_ERROR(ForEach(*nested, fn));
    }
  }
  return absl::OkStatus();
}

void ThunkSequence::ForEach(absl::FunctionRef<void(const Thunk&)> fn) const {
  xla::cpu::ForEach(*this, fn);
}

absl::Status ThunkSequence::ForEachWithStatus(
    absl::FunctionRef<absl::Status(const Thunk&)> fn) const {
  return xla::cpu::ForEach(*this, fn);
}

}  // namespace xla::cpu
