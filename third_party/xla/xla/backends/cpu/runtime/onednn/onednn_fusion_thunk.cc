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

#include "xla/backends/cpu/runtime/onednn/onednn_fusion_thunk.h"

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "oneapi/dnnl/dnnl_common.hpp"
#include "oneapi/dnnl/dnnl_graph.hpp"
#include "oneapi/dnnl/dnnl_threadpool.hpp"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/onednn_fusion.h"
#include "xla/backends/cpu/runtime/onednn/onednn_threadpool.h"
#include "xla/backends/cpu/runtime/parallel_loop_runner.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/runtime/buffer_use.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::cpu {

// oneDNN runtime instantiated for the fusion operation.
struct OneDnnFusionThunk::OneDnnRuntime {
  OneDnnRuntime(OneDnnFusion fusion, const Eigen::ThreadPoolDevice* device);

  OneDnnRuntime(OneDnnRuntime&&) = default;
  OneDnnRuntime& operator=(OneDnnRuntime&&) = default;

  absl::Status Invoke(const Eigen::ThreadPoolDevice* device,
                      absl::Span<se::DeviceMemoryBase> arguments,
                      absl::Span<se::DeviceMemoryBase> results);

  OneDnnFusion fusion;

  std::unique_ptr<OneDnnThreadPool> threadpool;

  dnnl::engine engine;
  dnnl::stream stream;
  std::vector<dnnl::graph::compiled_partition> compiled_partitions;
};

OneDnnFusionThunk::OneDnnRuntime::OneDnnRuntime(
    OneDnnFusion fusion, const Eigen::ThreadPoolDevice* device)
    : fusion(std::move(fusion)),
      threadpool(std::make_unique<OneDnnThreadPool>(device)),
      engine(dnnl::engine::kind::cpu, 0),
      stream(dnnl::threadpool_interop::make_stream(engine, threadpool.get())) {}

absl::Status OneDnnFusionThunk::OneDnnRuntime::Invoke(
    const Eigen::ThreadPoolDevice* device,
    absl::Span<se::DeviceMemoryBase> arguments,
    absl::Span<se::DeviceMemoryBase> results) {
  // Number of arguments and results must match the number of logical tensors.
  TF_RET_CHECK(arguments.size() == fusion.parameters.size())
      << "Arguments size mismatch";
  TF_RET_CHECK(results.size() == fusion.results.size())
      << "Results size mismatch";

  // Update the threadpool device.
  threadpool->set_device(device);

  // Create tensors for arguments.
  std::vector<dnnl::graph::tensor> argument_data;
  argument_data.reserve(arguments.size());

  for (size_t i = 0; i < arguments.size(); ++i) {
    argument_data.emplace_back(fusion.parameters[i], engine,
                               arguments[i].opaque());
  }

  // Create tensors for results.
  std::vector<dnnl::graph::tensor> result_data;
  result_data.reserve(results.size());

  for (size_t i = 0; i < results.size(); ++i) {
    result_data.emplace_back(fusion.results[i], engine, results[i].opaque());
  }

  for (const auto& partition : compiled_partitions) {
    partition.execute(stream, argument_data, result_data);
  }

  return absl::OkStatus();
}

absl::StatusOr<OneDnnFusionThunk::OneDnnRuntime>
OneDnnFusionThunk::CreateOneDnnRuntime(
    const Eigen::ThreadPoolDevice* device,
    absl::FunctionRef<absl::StatusOr<OneDnnFusion>()> builder) {
  VLOG(3) << absl::StreamFormat(
      "Create oneDNN runtime for `%s` operation: num_created=%d",
      info().op_name, onednn_runtime_pool_.num_created());

  // Construct oneDNN fusion using user-provided builder function.
  TF_ASSIGN_OR_RETURN(OneDnnFusion fusion, builder());

  OneDnnRuntime runtime(std::move(fusion), device);

  // Compile constructed graph for given engine.
  for (const auto& partition : runtime.fusion.graph.get_partitions()) {
    runtime.compiled_partitions.push_back(partition.compile(
        runtime.fusion.parameters, runtime.fusion.results, runtime.engine));
  }

  return {std::move(runtime)};
}

absl::StatusOr<std::unique_ptr<OneDnnFusionThunk>> OneDnnFusionThunk::Create(
    Info info, std::vector<Argument> arguments, std::vector<Result> results,
    Builder builder) {
  return absl::WrapUnique(
      new OneDnnFusionThunk(std::move(info), std::move(arguments),
                            std::move(results), std::move(builder)));
}

OneDnnFusionThunk::OneDnnFusionThunk(Info info, std::vector<Argument> arguments,
                                     std::vector<Result> results,
                                     Builder builder)
    : Thunk(Kind::kOneDnnFusion, std::move(info)),
      arguments_(std::move(arguments)),
      results_(std::move(results)),
      builder_(std::move(builder)),
      onednn_runtime_pool_([this](const Eigen::ThreadPoolDevice* device) {
        return CreateOneDnnRuntime(
            device, [this] { return builder_(arguments_, results_); });
      }) {}

OneDnnFusionThunk::~OneDnnFusionThunk() = default;

OneDnnFusionThunk::BufferUses OneDnnFusionThunk::buffer_uses() const {
  BufferUses buffer_uses;
  for (const Argument& argument : arguments_) {
    buffer_uses.push_back(BufferUse::Read(argument.slice));
  }
  for (const Result& result : results_) {
    buffer_uses.push_back(BufferUse::Write(result.slice));
  }
  return buffer_uses;
}

tsl::AsyncValueRef<OneDnnFusionThunk::ExecuteEvent> OneDnnFusionThunk::Execute(
    const ExecuteParams& params) {
  VLOG(3) << absl::StreamFormat("oneDNN %s `%s`: %s", fusion_kind(),
                                info().op_name, fusion_description());

  if (VLOG_IS_ON(3) && has_fusion_details()) {
    for (auto& detail : fusion_details()) VLOG(3) << detail;
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

  const Eigen::ThreadPoolDevice* device = params.intra_op_threadpool;

  // Borrow oneDNN runtime from the pool.
  TF_ASSIGN_OR_RETURN(auto runtime, onednn_runtime_pool_.GetOrCreate(device));
  TF_RETURN_IF_ERROR(runtime->Invoke(params.intra_op_threadpool,
                                     absl::MakeSpan(arguments_buffers),
                                     absl::MakeSpan(results_buffers)));

  return OkExecuteEvent();
}

}  // namespace xla::cpu
