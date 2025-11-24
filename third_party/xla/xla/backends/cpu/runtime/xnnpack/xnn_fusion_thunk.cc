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

#include "xla/backends/cpu/runtime/xnnpack/xnn_fusion_thunk.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <ostream>
#include <utility>
#include <vector>

#include "experimental.h"  // xnnpack
#include "xnnpack.h"
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
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/xnnpack/xnn_interop.h"
#include "xla/runtime/buffer_use.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::cpu {

absl::string_view XnnFusionThunk::XnnFusionKindToString(XnnFusionKind kind) {
  switch (kind) {
    case XnnFusionKind::kFusion:
      return "xnn-fusion";
    case XnnFusionKind::kDot:
      return "xnn-dot";
    case XnnFusionKind::kConvolution:
      return "xnn-convolution";
  }
}

std::ostream& operator<<(std::ostream& os, XnnFusionThunk::XnnFusionKind kind) {
  return os << XnnFusionThunk::XnnFusionKindToString(kind);
}

// XNNPACK executable instantiated for the fusion operation.
struct XnnFusionThunk::XnnExecutable {
  tsl::AsyncValueRef<XnnFusionThunk::ExecuteEvent> Invoke(
      const XnnThreadpool& threadpool,
      absl::Span<se::DeviceMemoryBase> arguments,
      absl::Span<se::DeviceMemoryBase> results,
      absl::FunctionRef<bool(size_t)> is_captured_argument);

  // Resets XNNPACK runtime and subgraph.
  absl::Status Reset();

  XnnSubgraph subgraph = nullptr;
  XnnRuntime runtime = nullptr;

  // TODO(ezhulenev): Today we rely on device memory as an identity of the
  // captured argument, and this is not correct as we can have multiple
  // arguments allocated to the heap address. This is work in progress, and will
  // be migrated to a buffer identity passed to XLA by the client (PjRt).
  std::vector<se::DeviceMemoryBase> captured_arguments;
};

tsl::AsyncValueRef<XnnFusionThunk::ExecuteEvent>
XnnFusionThunk::XnnExecutable::Invoke(
    const XnnThreadpool& threadpool, absl::Span<se::DeviceMemoryBase> arguments,
    absl::Span<se::DeviceMemoryBase> results,
    absl::FunctionRef<bool(size_t)> is_captured_argument) {
  // Create external values for all arguments and results.
  absl::InlinedVector<xnn_external_value, 8> external_values;
  external_values.reserve(arguments.size() + results.size());

  // External tensor id for arguments and results.
  uint32_t id = 0;

  for (const se::DeviceMemoryBase& argument : arguments) {
    xnn_external_value value{id++, argument.opaque()};
    if (!is_captured_argument(value.id)) {
      external_values.push_back(value);
    }
  }

  for (const se::DeviceMemoryBase& result : results) {
    xnn_external_value value{id++, result.opaque()};
    external_values.push_back(value);
  }

  DCHECK_NE(runtime.get(), nullptr) << "XNNPACK runtime is not initialized";
  XNN_RETURN_IF_ERROR(xnn_setup_runtime_v2(
      runtime.get(), external_values.size(), external_values.data()));

  // Update threadpool used by the XNNPACK runtime.
  xnn_update_runtime_with_threadpool(runtime.get(), threadpool.get());

  // Execute XNNPACK runtime in the caller thread.
  XNN_RETURN_IF_ERROR(xnn_invoke_runtime(runtime.get()));
  return OkExecuteEvent();
}

absl::Status XnnFusionThunk::XnnExecutable::Reset() {
  runtime.reset();
  subgraph.reset();
  return absl::OkStatus();
}

absl::StatusOr<XnnFusionThunk::XnnExecutable>
XnnFusionThunk::CreateXnnExecutable(
    const XnnThreadpool& threadpool,
    absl::Span<const se::DeviceMemoryBase> arguments_buffers) {
  bool capturing = !captured_arguments_ids_.empty();
  VLOG(3) << absl::StreamFormat(
      "Create %s XNN executable for `%s` operation: num_created=%d",
      capturing ? "capturing" : "pooled", info().op_name,
      capturing ? num_capturing_created_.fetch_add(1)
                : xnn_executable_pool_.num_created());

  XnnExecutable executable;

  // Keep track of the arguments captured by value.
  executable.captured_arguments = CaptureArguments(arguments_buffers);

  if (builder_) {
    TF_ASSIGN_OR_RETURN(executable.subgraph, builder_(arguments_, results_));
  } else {
    TF_ASSIGN_OR_RETURN(
        executable.subgraph,
        capturing_builder_(arguments_, results_, arguments_buffers));
  }

  uint32_t flags = XNN_FLAG_SLINKY_ENABLED | XNN_FLAG_SLINKY_STATIC_BOUNDS |
                   XNN_FLAG_DONT_SPIN_WORKERS;

  TF_ASSIGN_OR_RETURN(
      executable.runtime, CreateXnnRuntime([&](xnn_runtime_t* runtime) {
        return xnn_create_runtime_with_threadpool(
            executable.subgraph.get(), /*weights_cache=*/nullptr,
            threadpool.get(), flags, runtime);
      }));
  XNN_RETURN_IF_ERROR(xnn_reshape_runtime(executable.runtime.get()));

  return {std::move(executable)};
}

absl::Status XnnFusionThunk::UpdateXnnExecutable(
    const XnnThreadpool& threadpool, XnnExecutable& executable,
    absl::Span<const se::DeviceMemoryBase> arguments_buffers) {
  DCHECK(capturing_builder_) << "XNN executable is not capturing arguments";
  DCHECK_EQ(executable.captured_arguments.size(),
            captured_arguments_ids_.size())
      << "Unexpected number of captured arguments";

  // If all arguments captured by value are the same as the last execution,
  // we can reuse the XNN executable.
  auto capture_arguments = CaptureArguments(arguments_buffers);
  if (executable.captured_arguments == capture_arguments) {
    VLOG(3) << absl::StreamFormat("Reuse XNN executable for `%s` operation",
                                  info().op_name);
    return absl::OkStatus();
  }

  VLOG(3) << absl::StreamFormat("Update XNN executable for `%s` operation",
                                info().op_name);

  TF_RETURN_IF_ERROR(executable.Reset());

  // Keep track of the updated arguments captured by value.
  executable.captured_arguments = std::move(capture_arguments);

  TF_ASSIGN_OR_RETURN(
      executable.subgraph,
      capturing_builder_(arguments_, results_, arguments_buffers));

  uint32_t flags = XNN_FLAG_SLINKY_ENABLED | XNN_FLAG_SLINKY_STATIC_BOUNDS |
                   XNN_FLAG_DONT_SPIN_WORKERS;

  TF_ASSIGN_OR_RETURN(
      executable.runtime, CreateXnnRuntime([&](xnn_runtime_t* runtime) {
        return xnn_create_runtime_with_threadpool(
            executable.subgraph.get(), /*weights_cache=*/nullptr,
            threadpool.get(), flags, runtime);
      }));
  XNN_RETURN_IF_ERROR(xnn_reshape_runtime(executable.runtime.get()));

  return absl::OkStatus();
}

std::vector<se::DeviceMemoryBase> XnnFusionThunk::CaptureArguments(
    absl::Span<const se::DeviceMemoryBase> arguments_buffers) {
  std::vector<se::DeviceMemoryBase> captured_arguments_ids;
  captured_arguments_ids.reserve(captured_arguments_ids_.size());
  for (int64_t i = 0; i < captured_arguments_ids_.size(); ++i) {
    int32_t arg_index = captured_arguments_ids_[i];
    captured_arguments_ids.push_back(arguments_buffers[arg_index]);
  }
  return captured_arguments_ids;
}

absl::StatusOr<std::unique_ptr<XnnFusionThunk>> XnnFusionThunk::Create(
    Options options, Info info, std::vector<Argument> arguments,
    std::vector<Result> results, Builder builder) {
  TF_RETURN_IF_ERROR(InitializeXnnPack());

  return absl::WrapUnique(new XnnFusionThunk(
      XnnFusionKind::kFusion, std::move(options), std::move(info),
      std::move(arguments), std::move(results), std::move(builder)));
}

absl::StatusOr<std::unique_ptr<XnnFusionThunk>> XnnFusionThunk::Create(
    Options options, Info info, std::vector<Argument> arguments,
    std::vector<Result> results, CapturingBuilder capturing_builder,
    absl::Span<const int64_t> captured_arguments_ids) {
  TF_RETURN_IF_ERROR(InitializeXnnPack());

  return absl::WrapUnique(new XnnFusionThunk(
      XnnFusionKind::kFusion, std::move(options), std::move(info),
      std::move(arguments), std::move(results), std::move(capturing_builder),
      captured_arguments_ids));
}

XnnFusionThunk::XnnFusionThunk(XnnFusionKind kind, Options options, Info info,
                               std::vector<Argument> arguments,
                               std::vector<Result> results, Builder builder)
    : Thunk(Kind::kXnnFusion, std::move(info)),
      xnn_fusion_kind_(kind),
      options_(std::move(options)),
      arguments_(std::move(arguments)),
      results_(std::move(results)),
      builder_(std::move(builder)),
      xnn_executable_pool_(
          absl::bind_front(&XnnFusionThunk::CreateXnnExecutable, this)) {}

XnnFusionThunk::XnnFusionThunk(XnnFusionKind kind, Options options, Info info,
                               std::vector<Argument> arguments,
                               std::vector<Result> results,
                               CapturingBuilder capturing_builder,
                               absl::Span<const int64_t> captured_arguments_ids)
    : Thunk(Kind::kXnnFusion, std::move(info)),
      xnn_fusion_kind_(kind),
      options_(std::move(options)),
      arguments_(std::move(arguments)),
      results_(std::move(results)),
      capturing_builder_(std::move(capturing_builder)),
      captured_arguments_ids_(captured_arguments_ids.begin(),
                              captured_arguments_ids.end()),
      xnn_executable_pool_(
          absl::bind_front(&XnnFusionThunk::CreateXnnExecutable, this)) {}

XnnFusionThunk::~XnnFusionThunk() = default;

XnnFusionThunk::BufferUses XnnFusionThunk::buffer_uses() const {
  BufferUses buffer_uses;
  for (const Argument& argument : arguments_) {
    buffer_uses.push_back(BufferUse::Read(argument.slice, argument.shape));
  }
  for (const Result& result : results_) {
    buffer_uses.push_back(BufferUse::Write(result.slice, result.shape));
  }

  return buffer_uses;
}

const XnnThreadpool& GetXnnThreadpool(const Thunk::ExecuteParams& params) {
  static absl::NoDestructor<XnnThreadpool> no_threadpool(nullptr);
  return params.xnn_params ? params.xnn_params->threadpool : *no_threadpool;
}

tsl::AsyncValueRef<XnnFusionThunk::ExecuteEvent> XnnFusionThunk::Execute(
    const ExecuteParams& params) {
  VLOG(3) << absl::StreamFormat("XNN %s `%s`: %s", fusion_kind(),
                                info().op_name, fusion_description());

  if (VLOG_IS_ON(3) && has_fusion_details()) {
    for (auto& detail : fusion_details()) {
      VLOG(3) << detail;
    }
  }

  // Resolve device memory for arguments.
  absl::InlinedVector<se::DeviceMemoryBase, 8> arguments_buffers;
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
  absl::InlinedVector<se::DeviceMemoryBase, 4> results_buffers;
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

  auto invoke = [&](typename XnnExecutablePool::BorrowedObject executable) {
    auto executed = executable->Invoke(
        GetXnnThreadpool(params), absl::MakeSpan(arguments_buffers),
        absl::MakeSpan(results_buffers), [&](size_t id) {
          return absl::c_linear_search(captured_arguments_ids_, id);
        });

    // Do not return executable to the pool until the execution is done.
    executed.AndThen([executable = std::move(executable)] {});
    return executed;
  };

  // Borrow XnnExecutable from the pool.
  TF_ASSIGN_OR_RETURN(auto executable,
                      xnn_executable_pool_.GetOrCreate(GetXnnThreadpool(params),
                                                       arguments_buffers));

  // If XNN graph doesn't capture any of the arguments by value, we can execute
  // XnnExecutable immediately.
  if (captured_arguments_ids_.empty()) {
    return invoke(std::move(executable));
  }

  // Otherwise reset XnnExecutable to capture new arguments buffers.
  TF_RETURN_IF_ERROR(UpdateXnnExecutable(GetXnnThreadpool(params), *executable,
                                         arguments_buffers));
  return invoke(std::move(executable));
}

}  // namespace xla::cpu
