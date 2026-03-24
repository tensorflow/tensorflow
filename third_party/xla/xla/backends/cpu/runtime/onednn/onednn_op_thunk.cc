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

#include "xla/backends/cpu/runtime/onednn/onednn_op_thunk.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/call_once.h"
#include "absl/base/dynamic_annotations.h"
#include "absl/log/check.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "Eigen/ThreadPool"
#include "oneapi/dnnl/dnnl_common.hpp"
#include "oneapi/dnnl/dnnl_threadpool.hpp"
#include "xla/backends/cpu/runtime/onednn/onednn_threadpool.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/cpu/onednn_convolution.h"
#include "xla/service/cpu/onednn_layer_norm.h"
#include "xla/service/cpu/onednn_matmul.h"
#include "xla/service/cpu/onednn_memory_util.h"
#include "xla/service/cpu/onednn_softmax.h"
#include "xla/stream_executor/device_address.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::cpu {

// oneDNN runtime instantiated for the oneDNN operation.
struct OneDnnOpThunk::OneDnnRuntime {
  explicit OneDnnRuntime(Eigen::ThreadPoolInterface* thread_pool);

  OneDnnRuntime(OneDnnRuntime&&) = default;
  OneDnnRuntime& operator=(OneDnnRuntime&&) = default;

  tsl::AsyncValueRef<OneDnnOpThunk::ExecuteEvent> Invoke(
      Eigen::ThreadPoolInterface* thread_pool,
      absl::Span<MemrefInfoHandler> arguments,
      absl::Span<MemrefInfoHandler> results,
      const OneDnnOpThunk::OneDnnOpConfig& config, const std::string& target);

  std::unique_ptr<OneDnnThreadPool> threadpool;

  dnnl::engine cpu_engine;
  dnnl::stream onednn_stream;
  // We initialize the resources struct here to default values, so that we can
  // keep the primitive and memory objects alive for the duration of the
  // runtime. Otherwise, they would be destroyed as soon as we exit the
  // ExecuteOneDnn<primitive> method. This is a requirement of
  // oneDNN library's asynchronous execution model.
  OneDnnResources resources;
};

OneDnnOpThunk::OneDnnRuntime::OneDnnRuntime(
    Eigen::ThreadPoolInterface* thread_pool)
    : threadpool(
          std::make_unique<OneDnnThreadPool>(thread_pool, /*is_async=*/true)),
      cpu_engine(dnnl::engine::kind::cpu, 0),
      onednn_stream(
          dnnl::threadpool_interop::make_stream(cpu_engine, threadpool.get())),
      resources() {}

tsl::AsyncValueRef<OneDnnOpThunk::ExecuteEvent>
OneDnnOpThunk::OneDnnRuntime::Invoke(
    Eigen::ThreadPoolInterface* thread_pool,
    absl::Span<MemrefInfoHandler> arguments,
    absl::Span<MemrefInfoHandler> results,
    const OneDnnOpThunk::OneDnnOpConfig& config, const std::string& target) {
  // Update threadpool
  threadpool->set_thread_pool(thread_pool);

  // TODO(intel-tf): Add support for more oneDNN operations as needed.
  static absl::once_flag log_once_flag;
  absl::call_once(log_once_flag, [&] {
    VLOG(0) << absl::StreamFormat(
        "Executing oneDNN thunk with target `%s`: num_args=%d, num_results=%d",
        target, arguments.size(), results.size());
  });

  if (target == "__onednn$matmul") {
    const auto& matmul_config = std::get<OneDnnMatMulConfig>(config);
    ExecuteOneDnnMatMul(arguments, results, matmul_config, cpu_engine,
                        onednn_stream, resources);
  } else if (target == "__onednn$convolution") {
    const auto& conv_config = std::get<OneDnnConvolutionConfig>(config);
    ExecuteOneDnnConvolution(arguments, results, conv_config, cpu_engine,
                             onednn_stream, resources);
  } else if (target == "__onednn$layernorm") {
    const auto& ln_config = std::get<OneDnnNormConfig>(config);
    ExecuteOneDnnLayerNorm(arguments, results, ln_config, cpu_engine,
                           onednn_stream, resources);
  } else if (target == "__onednn$softmax") {
    const auto& softmax_config = std::get<OneDnnSoftmaxConfig>(config);
    ExecuteOneDnnSoftmax(arguments, results, softmax_config, cpu_engine,
                         onednn_stream, resources);
  } else {
    return absl::InvalidArgumentError(
        absl::StrFormat("Unsupported oneDNN operation target: `%s`", target));
  }

  return threadpool->done_event();
}

absl::StatusOr<std::unique_ptr<OneDnnOpThunk>> OneDnnOpThunk::Create(
    const std::string& custom_call_target, Info info, OpBuffers buffers,
    OneDnnOpConfig config) {
  // Update custom_call op_name with target
  info.op_name = absl::StrCat(info.op_name, custom_call_target);
  return absl::WrapUnique(new OneDnnOpThunk(std::move(custom_call_target),
                                            std::move(info), std::move(buffers),
                                            std::move(config)));
}

OneDnnOpThunk::OneDnnOpThunk(const std::string& custom_call_target, Info info,
                             OpBuffers buffers, OneDnnOpConfig config)
    : Thunk(Thunk::Kind::kCustomCall, std::move(info)),
      op_buffers_(std::move(buffers)),
      config_(std::move(config)),
      target_(custom_call_target) {}

OneDnnOpThunk::~OneDnnOpThunk() = default;

OneDnnOpThunk::BufferUses OneDnnOpThunk::buffer_uses() const {
  BufferUses buffer_uses;
  for (int i = 0; i < op_buffers_.arguments_buffers.size(); i++) {
    buffer_uses.emplace_back(BufferUse::Read(op_buffers_.arguments_buffers[i],
                                             op_buffers_.arguments_shapes[i]));
  }
  for (int i = 0; i < op_buffers_.results_buffers.size(); i++) {
    buffer_uses.emplace_back(BufferUse::Write(op_buffers_.results_buffers[i],
                                              op_buffers_.results_shapes[i]));
  }
  return buffer_uses;
}

tsl::AsyncValueRef<OneDnnOpThunk::ExecuteEvent> OneDnnOpThunk::Execute(
    const ExecuteParams& params) {
  Eigen::ThreadPoolInterface* thread_pool =
      params.intra_op_threadpool->getPool();
  DCHECK(thread_pool != nullptr) << "Thread pool must not be null";

  // Create oneDNN runtime for the operation.
  auto runtime = std::make_unique<OneDnnRuntime>(thread_pool);

  // Resolve device memory for arguments.
  int64_t num_operands = op_buffers_.arguments_shapes.size();
  runtime->resources.arg_memrefs.reserve(num_operands);
  for (size_t i = 0; i < num_operands; ++i) {
    const auto& shape = op_buffers_.arguments_shapes[i];
    TF_ASSIGN_OR_RETURN(se::DeviceAddressBase arg,
                        params.buffer_allocations->GetDeviceAddress(
                            op_buffers_.arguments_buffers[i]));

    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(arg.opaque(), arg.size());
    VLOG(3) << absl::StreamFormat(
        "  arg: %s (%p)", op_buffers_.arguments_shapes[i].ToString(true),
        arg.opaque());

    auto memref = CreateMemrefFromShape(shape, arg.opaque());
    runtime->resources.arg_memrefs.push_back(std::move(memref));
  }

  // Resolve device memory for results.
  int64_t num_results = op_buffers_.results_shapes.size();
  runtime->resources.result_memrefs.reserve(num_results);
  for (size_t i = 0; i < num_results; ++i) {
    const auto& shape = op_buffers_.results_shapes[i];
    TF_ASSIGN_OR_RETURN(se::DeviceAddressBase res,
                        params.buffer_allocations->GetDeviceAddress(
                            op_buffers_.results_buffers[i]));

    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(res.opaque(), res.size());
    VLOG(3) << absl::StreamFormat("  res: %s (%p)",
                                  op_buffers_.results_shapes[i].ToString(true),
                                  res.opaque());

    auto memref = CreateMemrefFromShape(shape, res.opaque());
    runtime->resources.result_memrefs.push_back(std::move(memref));
  }

  auto executed = runtime->Invoke(
      thread_pool, absl::MakeSpan(runtime->resources.arg_memrefs),
      absl::MakeSpan(runtime->resources.result_memrefs), config_, target_);

  // Do not return runtime to the pool until the execution is done.
  executed.AndThen([runtime = std::move(runtime)]() {
    // runtime will be destroyed here when going out of scope.
    VLOG(3) << "OneDnnOpThunk execution completed and destroying runtime now.";
  });

  return executed;
}

}  // namespace xla::cpu
