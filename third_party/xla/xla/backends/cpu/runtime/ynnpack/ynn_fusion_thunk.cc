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

#include "xla/backends/cpu/runtime/ynnpack/ynn_fusion_thunk.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <ostream>
#include <utility>
#include <vector>

#include "ynnpack/include/ynnpack.h"
#include "absl/algorithm/container.h"
#include "absl/base/no_destructor.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/bind_front.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/ynnpack/ynn_interop.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/runtime/buffer_use.h"
#include "xla/stream_executor/device_address.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::cpu {

// YNNPACK executable instantiated for the fusion operation.
struct YnnFusionThunk::YnnExecutable {
  tsl::AsyncValueRef<YnnFusionThunk::ExecuteEvent> Invoke(
      const YnnThreadpool& threadpool,
      absl::Span<se::DeviceAddressBase> arguments,
      absl::Span<se::DeviceAddressBase> results,
      absl::FunctionRef<bool(size_t)> is_captured_argument);

  // Resets YNNPACK runtime and subgraph.
  absl::Status Reset();

  YnnSubgraph subgraph = nullptr;
  YnnRuntime runtime = nullptr;

  // TODO(ezhulenev): Today we rely on device memory as an identity of the
  // captured argument, and this is not correct as we can have multiple
  // arguments allocated to the heap address. This is work in progress, and will
  // be migrated to a buffer identity passed to XLA by the client (PjRt).
  std::vector<se::DeviceAddressBase> captured_arguments;
};

namespace {
struct YnnExternalValue {
  uint32_t id;
  void* data;
};
}  // namespace

static enum ynn_status set_external_values(
    ynn_runtime_t runtime, absl::Span<const YnnExternalValue> external_values) {
  for (const auto& [id, data] : external_values) {
    enum ynn_status status = ynn_set_external_value_data(runtime, id, data);
    if (status != ynn_status_success) {
      return status;
    }
  }
  return ynn_status_success;
}

tsl::AsyncValueRef<YnnFusionThunk::ExecuteEvent>
YnnFusionThunk::YnnExecutable::Invoke(
    const YnnThreadpool& threadpool,
    absl::Span<se::DeviceAddressBase> arguments,
    absl::Span<se::DeviceAddressBase> results,
    absl::FunctionRef<bool(size_t)> is_captured_argument) {
  // Create external values for all arguments and results.
  absl::InlinedVector<YnnExternalValue, 8> external_values;
  external_values.reserve(arguments.size() + results.size());

  // External tensor id for arguments and results.
  uint32_t id = 0;

  for (const se::DeviceAddressBase& argument : arguments) {
    YnnExternalValue value{id++, argument.opaque()};
    if (!is_captured_argument(value.id)) {
      external_values.push_back(value);
    }
  }

  for (const se::DeviceAddressBase& result : results) {
    YnnExternalValue value{id++, result.opaque()};
    external_values.push_back(value);
  }

  DCHECK_NE(runtime.get(), nullptr) << "YNNPACK runtime is not initialized";

  YNN_RETURN_IF_ERROR(set_external_values(runtime.get(), external_values));

  // Update threadpool used by the YNNPACK runtime.
  YNN_RETURN_IF_ERROR(ynn_update_runtime_with_threadpool(
      runtime.get(), reinterpret_cast<ynn_threadpool_t>(threadpool.get())));

  // Execute YNNPACK runtime in the caller thread.
  YNN_RETURN_IF_ERROR(ynn_invoke_runtime(runtime.get()));
  return OkExecuteEvent();
}

absl::Status YnnFusionThunk::YnnExecutable::Reset() {
  runtime.reset();
  subgraph.reset();
  return absl::OkStatus();
}

absl::StatusOr<YnnFusionThunk::YnnExecutable>
YnnFusionThunk::CreateYnnExecutable(
    const YnnThreadpool& threadpool,
    absl::Span<const se::DeviceAddressBase> arguments_buffers) {
  bool capturing = !captured_arguments_ids_.empty();
  VLOG(3) << absl::StreamFormat(
      "Create %s YNN executable for `%s` operation: num_created=%d",
      capturing ? "capturing" : "pooled", info().op_name,
      capturing ? num_capturing_created_.fetch_add(1)
                : ynn_executable_pool_.num_created());

  YnnExecutable executable;

  // Keep track of the arguments captured by value.
  executable.captured_arguments = CaptureArguments(arguments_buffers);

  if (builder_) {
    TF_ASSIGN_OR_RETURN(executable.subgraph, builder_(arguments_, results_));
  } else {
    TF_ASSIGN_OR_RETURN(
        executable.subgraph,
        capturing_builder_(arguments_, results_, arguments_buffers));
  }

  TF_ASSIGN_OR_RETURN(
      executable.runtime, CreateYnnRuntime([&](ynn_runtime_t* runtime) {
        uint32_t ynn_flags = 0;
        return ynn_create_runtime(
            executable.subgraph.get(),
            reinterpret_cast<ynn_threadpool_t>(threadpool.get()), ynn_flags,
            runtime);
      }));
  YNN_RETURN_IF_ERROR(ynn_reshape_runtime(executable.runtime.get()));

  return {std::move(executable)};
}

absl::Status YnnFusionThunk::UpdateYnnExecutable(
    const YnnThreadpool& threadpool, YnnExecutable& executable,
    absl::Span<const se::DeviceAddressBase> arguments_buffers) {
  DCHECK(capturing_builder_) << "YNN executable is not capturing arguments";
  DCHECK_EQ(executable.captured_arguments.size(),
            captured_arguments_ids_.size())
      << "Unexpected number of captured arguments";

  // If all arguments captured by value are the same as the last execution,
  // we can reuse the YNN executable.
  auto capture_arguments = CaptureArguments(arguments_buffers);
  if (executable.captured_arguments == capture_arguments) {
    VLOG(3) << absl::StreamFormat("Reuse YNN executable for `%s` operation",
                                  info().op_name);
    return absl::OkStatus();
  }

  VLOG(3) << absl::StreamFormat("Update YNN executable for `%s` operation",
                                info().op_name);

  TF_RETURN_IF_ERROR(executable.Reset());

  // Keep track of the updated arguments captured by value.
  executable.captured_arguments = std::move(capture_arguments);

  TF_ASSIGN_OR_RETURN(
      executable.subgraph,
      capturing_builder_(arguments_, results_, arguments_buffers));

  TF_ASSIGN_OR_RETURN(
      executable.runtime, CreateYnnRuntime([&](ynn_runtime_t* runtime) {
        uint32_t ynn_flags = 0;
        return ynn_create_runtime(
            executable.subgraph.get(),
            reinterpret_cast<ynn_threadpool_t>(threadpool.get()), ynn_flags,
            runtime);
      }));
  YNN_RETURN_IF_ERROR(ynn_reshape_runtime(executable.runtime.get()));

  return absl::OkStatus();
}

std::vector<se::DeviceAddressBase> YnnFusionThunk::CaptureArguments(
    absl::Span<const se::DeviceAddressBase> arguments_buffers) {
  std::vector<se::DeviceAddressBase> captured_arguments_ids;
  captured_arguments_ids.reserve(captured_arguments_ids_.size());
  for (int64_t i = 0; i < captured_arguments_ids_.size(); ++i) {
    int32_t arg_index = captured_arguments_ids_[i];
    captured_arguments_ids.push_back(arguments_buffers[arg_index]);
  }
  return captured_arguments_ids;
}

absl::StatusOr<std::unique_ptr<YnnFusionThunk>> YnnFusionThunk::Create(
    Options options, Info info, const HloInstruction* hlo,
    std::vector<Argument> arguments, std::vector<Result> results,
    Builder builder) {
  return absl::WrapUnique(new YnnFusionThunk(
      std::move(options), std::move(info), hlo, std::move(arguments),
      std::move(results), std::move(builder)));
}

absl::StatusOr<std::unique_ptr<YnnFusionThunk>> YnnFusionThunk::Create(
    Options options, Info info, const HloInstruction* hlo,
    std::vector<Argument> arguments, std::vector<Result> results,
    CapturingBuilder capturing_builder,
    absl::Span<const int64_t> captured_arguments_ids) {
  return absl::WrapUnique(
      new YnnFusionThunk(std::move(options), std::move(info), hlo,
                         std::move(arguments), std::move(results),
                         std::move(capturing_builder), captured_arguments_ids));
}

YnnFusionThunk::YnnFusionThunk(Options options, Info info,
                               const HloInstruction* hlo,
                               std::vector<Argument> arguments,
                               std::vector<Result> results, Builder builder)
    : Thunk(Kind::kYnnFusion, std::move(info)),
      options_(std::move(options)),
      hlo_(hlo),
      arguments_(std::move(arguments)),
      results_(std::move(results)),
      builder_(std::move(builder)),
      ynn_executable_pool_(
          absl::bind_front(&YnnFusionThunk::CreateYnnExecutable, this)) {}

YnnFusionThunk::YnnFusionThunk(Options options, Info info,
                               const HloInstruction* hlo,
                               std::vector<Argument> arguments,
                               std::vector<Result> results,
                               CapturingBuilder capturing_builder,
                               absl::Span<const int64_t> captured_arguments_ids)
    : Thunk(Kind::kYnnFusion, std::move(info)),
      options_(std::move(options)),
      hlo_(hlo),
      arguments_(std::move(arguments)),
      results_(std::move(results)),
      capturing_builder_(std::move(capturing_builder)),
      captured_arguments_ids_(captured_arguments_ids.begin(),
                              captured_arguments_ids.end()),
      ynn_executable_pool_(
          absl::bind_front(&YnnFusionThunk::CreateYnnExecutable, this)) {}

YnnFusionThunk::~YnnFusionThunk() = default;

YnnFusionThunk::BufferUses YnnFusionThunk::buffer_uses() const {
  BufferUses buffer_uses;
  for (const Argument& argument : arguments_) {
    buffer_uses.push_back(BufferUse::Read(argument.slice, argument.shape));
  }
  for (const Result& result : results_) {
    buffer_uses.push_back(BufferUse::Write(result.slice, result.shape));
  }

  return buffer_uses;
}

const YnnThreadpool& GetYnnThreadpool(const Thunk::ExecuteParams& params) {
  static absl::NoDestructor<YnnThreadpool> no_threadpool(nullptr);
  return params.ynn_params ? params.ynn_params->threadpool : *no_threadpool;
}

tsl::AsyncValueRef<YnnFusionThunk::ExecuteEvent> YnnFusionThunk::Execute(
    const ExecuteParams& params) {
  VLOG(3) << absl::StreamFormat("YNN %s `%s`: %s", fusion_kind(),
                                info().op_name, fusion_description());

  if (VLOG_IS_ON(3) && has_fusion_details()) {
    for (auto& detail : fusion_details()) {
      VLOG(3) << detail;
    }
  }

  // Resolve device memory for arguments.
  absl::InlinedVector<se::DeviceAddressBase, 8> arguments_buffers;
  arguments_buffers.resize(arguments_.size());
  for (size_t i = 0; i < arguments_.size(); ++i) {
    Argument& argument = arguments_[i];

    TF_ASSIGN_OR_RETURN(
        arguments_buffers[i],
        params.buffer_allocations->GetDeviceAddress(argument.slice));

    VLOG(3) << absl::StreamFormat("  %s: %s in slice %s (%p)", argument_name(i),
                                  argument.shape.ToString(true),
                                  argument.slice.ToString(),
                                  arguments_buffers[i].opaque());
  }

  // Resolve device memory for results.
  absl::InlinedVector<se::DeviceAddressBase, 4> results_buffers;
  results_buffers.resize(results_.size());
  for (size_t i = 0; i < results_.size(); ++i) {
    Result& result = results_[i];

    TF_ASSIGN_OR_RETURN(
        results_buffers[i],
        params.buffer_allocations->GetDeviceAddress(results_[i].slice));

    VLOG(3) << absl::StreamFormat("  %s: %s in slice %s (%p)", result_name(i),
                                  result.shape.ToString(true),
                                  result.slice.ToString(),
                                  results_buffers[i].opaque());
  }

  DCHECK(builder_ || capturing_builder_) << "One of the builders must be set.";

  auto invoke = [&](typename YnnExecutablePool::BorrowedObject executable) {
    auto executed = executable->Invoke(
        GetYnnThreadpool(params), absl::MakeSpan(arguments_buffers),
        absl::MakeSpan(results_buffers), [&](size_t id) {
          return absl::c_linear_search(captured_arguments_ids_, id);
        });

    // Do not return executable to the pool until the execution is done.
    executed.AndThen([executable = std::move(executable)] {});
    return executed;
  };

  // Borrow YnnExecutable from the pool.
  TF_ASSIGN_OR_RETURN(auto executable,
                      ynn_executable_pool_.GetOrCreate(GetYnnThreadpool(params),
                                                       arguments_buffers));

  // If YNN graph doesn't capture any of the arguments by value, we can execute
  // XnnExecutable immediately.
  if (captured_arguments_ids_.empty()) {
    return invoke(std::move(executable));
  }

  // Otherwise reset YnnExecutable to capture new arguments buffers.
  TF_RETURN_IF_ERROR(UpdateYnnExecutable(GetYnnThreadpool(params), *executable,
                                         arguments_buffers));
  return invoke(std::move(executable));
}

}  // namespace xla::cpu
