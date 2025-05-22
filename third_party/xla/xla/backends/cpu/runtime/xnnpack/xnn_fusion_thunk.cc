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

#include "xnnpack.h"
#include "absl/base/optimization.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "pthreadpool.h"
#include "xla/backends/cpu/runtime/parallel_loop_runner.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/xnnpack/xnn_interop.h"
#include "xla/backends/cpu/runtime/xnnpack/xnn_threadpool.h"
#include "xla/runtime/buffer_use.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::cpu {
namespace {

enum class ParallelizationMode { kInline, kParallelLoopRunner };

template <typename Sink>
void AbslStringify(Sink& sink, ParallelizationMode m) {
  switch (m) {
    case ParallelizationMode::kInline:
      sink.Append("kInline");
      break;
    case ParallelizationMode::kParallelLoopRunner:
      sink.Append("kParallelLoopRunner");
      break;
  }
}

}  // namespace

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

// XNNPACK runtime instantiated for the fusion operation.
struct XnnFusionThunk::XnnRuntime {
  XnnRuntime() = default;
  ~XnnRuntime() { Destroy(); }

  XnnRuntime(XnnRuntime&&);
  XnnRuntime& operator=(XnnRuntime&&);

  tsl::AsyncValueRef<XnnFusionThunk::ExecuteEvent> Invoke(
      const Eigen::ThreadPoolDevice* device,
      absl::Span<se::DeviceMemoryBase> arguments,
      absl::Span<se::DeviceMemoryBase> results,
      absl::FunctionRef<bool(size_t)> is_captured_argument);

  void Destroy();

  std::unique_ptr<ParallelLoopRunner> runner;
  pthreadpool_t threadpool = nullptr;

  xnn_subgraph_t subgraph = nullptr;
  xnn_workspace_t workspace = nullptr;
  xnn_runtime_t runtime = nullptr;
};

XnnFusionThunk::XnnRuntime::XnnRuntime(XnnRuntime&& other) {
  *this = std::move(other);
}

auto XnnFusionThunk::XnnRuntime::operator=(XnnRuntime&& other) -> XnnRuntime& {
  Destroy();

  threadpool = other.threadpool;
  subgraph = other.subgraph;
  workspace = other.workspace;
  runtime = other.runtime;

  other.threadpool = nullptr;
  other.subgraph = nullptr;
  other.workspace = nullptr;
  other.runtime = nullptr;

  runner = std::move(other.runner);
  return *this;
}

tsl::AsyncValueRef<XnnFusionThunk::ExecuteEvent>
XnnFusionThunk::XnnRuntime::Invoke(
    const Eigen::ThreadPoolDevice* device,
    absl::Span<se::DeviceMemoryBase> arguments,
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

  DCHECK_NE(runtime, nullptr) << "XNNPACK runtime is not initialized";
  XNN_RETURN_IF_ERROR(xnn_setup_runtime_v2(runtime, external_values.size(),
                                           external_values.data()));

  // Execute XNNPACK runtime using a parallel loop runner.
  if (runner) {
    runner->set_device(device);
    XNN_RETURN_IF_ERROR(xnn_invoke_runtime(runtime));
    return runner->ResetDoneEvent();
  }

  // Execute XNNPACK runtime in the caller thread.
  XNN_RETURN_IF_ERROR(xnn_invoke_runtime(runtime));
  return OkExecuteEventSingleton();
}

void XnnFusionThunk::XnnRuntime::Destroy() {
  if (runtime != nullptr) {
    XNN_LOG_IF_ERROR(xnn_delete_runtime(runtime));
  }
  if (subgraph != nullptr) {
    XNN_LOG_IF_ERROR(xnn_delete_subgraph(subgraph));
  }
  if (workspace != nullptr) {
    XNN_LOG_IF_ERROR(xnn_release_workspace(workspace));
  }
  if (threadpool) {
    DestroyCustomPthreadpool(threadpool);
  }
}

absl::StatusOr<XnnFusionThunk::XnnRuntime> XnnFusionThunk::CreateXnnRuntime(
    const Eigen::ThreadPoolDevice* device, bool capturing,
    absl::FunctionRef<absl::StatusOr<xnn_subgraph_t>()> builder) {
  ParallelizationMode parallelization_mode =
      options_.use_threadpool && device
          ? ParallelizationMode::kParallelLoopRunner
          : ParallelizationMode::kInline;

  VLOG(3) << absl::StreamFormat(
      "Create %s XNN runtime for `%s` operation: num_created=%d, "
      "parallelization_mode=%v",
      capturing ? "capturing" : "pooled", info().op_name,
      capturing ? num_capturing_created_.fetch_add(1)
                : xnn_runtime_pool_.num_created(),
      parallelization_mode);

  XnnRuntime runtime;

  // Construct XNNPACK subgraph using user-provided builder function.
  TF_ASSIGN_OR_RETURN(runtime.subgraph, builder());

  // Configure XNNPACK runtime thread pool if parallelization is enabled.
  if (parallelization_mode == ParallelizationMode::kParallelLoopRunner) {
    runtime.runner = std::make_unique<ParallelLoopRunner>(device);
    runtime.threadpool = CreateCustomPthreadpool(runtime.runner.get());
  }

  XNN_RETURN_IF_ERROR(xnn_create_workspace(&runtime.workspace));

  XNN_RETURN_IF_ERROR(
      xnn_create_runtime_v4(runtime.subgraph, nullptr, runtime.workspace,
                            runtime.threadpool, 0, &runtime.runtime));

  XNN_RETURN_IF_ERROR(xnn_reshape_runtime(runtime.runtime));

  return {std::move(runtime)};
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
    absl::Span<const int64_t> captured_arguments) {
  TF_RETURN_IF_ERROR(InitializeXnnPack());

  return absl::WrapUnique(new XnnFusionThunk(
      XnnFusionKind::kFusion, std::move(options), std::move(info),
      std::move(arguments), std::move(results), std::move(capturing_builder),
      captured_arguments));
}

XnnFusionThunk::XnnFusionThunk(XnnFusionKind kind, Options options, Info info,
                               std::vector<Argument> arguments,
                               std::vector<Result> results, Builder builder)
    : Thunk(Kind::kXnnFusion, std::move(info)),
      options_(std::move(options)),
      arguments_(std::move(arguments)),
      results_(std::move(results)),
      builder_(std::move(builder)),
      xnn_fusion_kind_(kind),
      xnn_runtime_pool_([this](const Eigen::ThreadPoolDevice* device) {
        return CreateXnnRuntime(device, /*capturing=*/false, [this] {
          return builder_(arguments_, results_);
        });
      }) {}

XnnFusionThunk::XnnFusionThunk(XnnFusionKind kind, Options options, Info info,
                               std::vector<Argument> arguments,
                               std::vector<Result> results,
                               CapturingBuilder capturing_builder,
                               absl::Span<const int64_t> captured_arguments)
    : Thunk(Kind::kXnnFusion, std::move(info)),
      options_(std::move(options)),
      arguments_(std::move(arguments)),
      results_(std::move(results)),
      capturing_builder_(std::move(capturing_builder)),
      xnn_fusion_kind_(kind),
      captured_arguments_(captured_arguments.begin(), captured_arguments.end()),
      xnn_runtime_pool_(nullptr) {}

XnnFusionThunk::~XnnFusionThunk() = default;

XnnFusionThunk::BufferUses XnnFusionThunk::buffer_uses() const {
  BufferUses buffer_uses;
  for (const Argument& argument : arguments_) {
    buffer_uses.push_back(BufferUse::Read(argument.slice));
  }
  for (const Result& result : results_) {
    buffer_uses.push_back(BufferUse::Write(result.slice));
  }
  return buffer_uses;
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

  auto invoke = [&](XnnRuntime& runtime) {
    return runtime.Invoke(
        params.intra_op_threadpool, absl::MakeSpan(arguments_buffers),
        absl::MakeSpan(results_buffers),
        [&](size_t id) { return captured_arguments_.contains(id); });
  };

  const Eigen::ThreadPoolDevice* device = params.intra_op_threadpool;

  if (ABSL_PREDICT_TRUE(builder_)) {
    // Borrow XNNPACK runtime from the pool.
    TF_ASSIGN_OR_RETURN(auto runtime, xnn_runtime_pool_.GetOrCreate(device));
    auto executed = invoke(*runtime);

    // Do not return runtime to the pool until the execution is done.
    executed.AndThen([runtime = std::move(runtime)] {});
    return executed;
  }

  // Create XNNPACK runtime for capturing graphs.
  TF_ASSIGN_OR_RETURN(
      auto runtime, CreateXnnRuntime(device, /*capturing=*/true, [&, this] {
        return capturing_builder_(arguments_, results_, arguments_buffers);
      }));
  return invoke(runtime);
}

}  // namespace xla::cpu
