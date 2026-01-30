// Copyright 2023 The OpenXLA Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "xla/python/ifrt_proxy/client/executable.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "llvm/Support/Casting.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/layout.h"
#include "xla/pjrt/host_callback.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/pjrt/profiling/device_time_measurement.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/array_spec.h"
#include "xla/python/ifrt/attribute_map.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/host_callback.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/user_context.h"
#include "xla/python/ifrt/user_context_status_util.h"
#include "xla/python/ifrt_proxy/client/array.h"
#include "xla/python/ifrt_proxy/client/host_buffer.h"
#include "xla/python/ifrt_proxy/client/rpc_helper.h"
#include "xla/python/ifrt_proxy/common/ifrt_service.pb.h"
#include "xla/python/ifrt_proxy/common/types.h"
#include "xla/python/ifrt_proxy/common/versions.h"
#include "xla/python/pjrt_ifrt/pjrt_host_callback.h"
#include "xla/shape_util.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status_to_from_proto.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/cpu_info.h"
#include "tsl/platform/mem.h"
#include "tsl/platform/protobuf.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla {
namespace ifrt {
namespace proxy {

namespace {

// Returns the thread options to use for the global pool.
tsl::ThreadOptions GetThreadOptions() {
  tsl::ThreadOptions thread_options;
  // Ensure the threads' stack is large enough for arbitrary Python code.
  thread_options.stack_size = 2 * 1024 * 1024;  // 2 MiB
  return thread_options;
}

// Returns the global pool used for host callback RPCs and executions.
tsl::thread::ThreadPool* GetGlobalThreadPool() {
  static tsl::thread::ThreadPool* global_pool = new tsl::thread::ThreadPool(
      tsl::Env::Default(), GetThreadOptions(), "XLAIFRTProxy",
      std::min(16, tsl::port::MaxParallelism()));
  return global_pool;
}

// Locally executes the loaded host callback with given operand buffer from the
// IFRT proxy server and returns a result buffer to be sent back.
absl::StatusOr<absl::Cord> ExecuteLoadedHostCallback(
    xla::ifrt::LoadedHostCallback* loaded_host_callback,
    absl::Cord operand_buffer) {
#if defined(PLATFORM_GOOGLE)
  auto* pjrt_host_callback =
      llvm::dyn_cast<PjRtHostSendAndRecvLoadedHostCallback>(
          loaded_host_callback);
  if (pjrt_host_callback == nullptr) {
    return absl::UnimplementedError(
        "Non-PjRt host callbacks cannot be executed");
  }
  const xla::HostCallback& xla_host_callback =
      pjrt_host_callback->host_callback();

  // The following allocates both operands and results using `aligned_alloc` in
  // order to (loosely) emulate the XLA implementation where host callbacks are
  // often called with aligned operand/result buffers. While this may not be
  // strictly necessary for some callbacks, this reduces the chances of proxied
  // callbacks behaving differently on a best-effort basis.
  constexpr int kAlignment = 32;

  struct Deleter {
    void operator()(void* p) { tsl::port::AlignedFree(p); }
  };

  std::vector<std::unique_ptr<char, Deleter>> operands;
  operands.reserve(xla_host_callback.operands.size());
  std::vector<void*> operand_ptrs;
  operand_ptrs.reserve(xla_host_callback.operands.size());

  absl::CordReader reader(operand_buffer);
  for (const auto& spec : xla_host_callback.operands) {
    const int64_t size = xla::ShapeUtil::ByteSizeOf(spec.shape);
    void* p = tsl::port::AlignedMalloc(
        size, static_cast<std::align_val_t>(kAlignment));
    CHECK(p != nullptr);
    std::unique_ptr<char, Deleter> buffer(reinterpret_cast<char*>(p));

    if (reader.Available() < size) {
      return absl::InternalError(absl::StrCat(
          "Buffer overflow while reading host callback execution operands; ",
          "range: [", reader.Position(), ", ", reader.Position() + size, "), ",
          "buffer size: ", operand_buffer.size()));
    }
    reader.ReadN(size, buffer.get());

    operand_ptrs.push_back(buffer.get());
    operands.push_back(std::move(buffer));
  }
  if (reader.Available() > 0) {
    return absl::InternalError(absl::StrCat(
        "Host callback execution did not consume the entire operand buffer; "
        "size: ",
        operand_buffer.size(), "; consumed: ", reader.Available()));
  }

  absl::Cord result_buffer;
  std::vector<void*> result_ptrs;
  result_ptrs.reserve(xla_host_callback.results.size());

  for (const auto& spec : xla_host_callback.results) {
    const int64_t size = xla::ShapeUtil::ByteSizeOf(spec.shape);
    void* data = tsl::port::AlignedMalloc(
        size, static_cast<std::align_val_t>(kAlignment));
    CHECK(data != nullptr);

    result_ptrs.push_back(data);
    result_buffer.AppendExternalMemory(
        absl::string_view(reinterpret_cast<char*>(data), size), data,
        &tsl::port::AlignedFree);
  }

  TF_RETURN_IF_ERROR(
      xla_host_callback.callback(result_ptrs.data(), operand_ptrs.data()));

  return result_buffer;
#else
  return absl::UnimplementedError("ExecuteLoadedHostCallback is unsupported.");
#endif
}

// Same as `ExecuteLoadedHostCallback`, except that it uses host buffer store to
// retrieve operands and store results.
absl::StatusOr<uint64_t> PrepareAndExecuteLoadedHostCallback(
    RpcHelper* rpc_helper, xla::ifrt::LoadedHostCallback* loaded_host_callback,
    uint64_t operand_handle) {
  ClientHostBufferStore* host_buffer_store =
      rpc_helper->host_buffer_store().get();
  TF_ASSIGN_OR_RETURN(absl::Cord operands,
                      host_buffer_store->Lookup(operand_handle).Await());
  absl::Cleanup cleanup = [&]() {
    host_buffer_store->Delete(operand_handle).OnReady([](absl::Status status) {
      if (!status.ok()) {
        LOG(ERROR) << "Failed to delete host callback operands: " << status;
      }
    });
  };

  TF_ASSIGN_OR_RETURN(
      absl::Cord results,
      ExecuteLoadedHostCallback(loaded_host_callback, std::move(operands)));

  const uint64_t result_handle = rpc_helper->NextHandle();
  TF_RETURN_IF_ERROR(host_buffer_store->Store(result_handle, results).Await());
  return result_handle;
}

// Bundles together the state needed to poll the state of a single host
// callback.
struct LoadedHostCallbackPollingState {
  std::shared_ptr<RpcHelper> rpc_helper;
  uint64_t handle;
  tsl::RCReference<xla::ifrt::LoadedHostCallback> loaded_host_callback;
};

// Handles a single poll response from the server, executes the loaded host
// callback, and returns the result. Must run in a non-RPC response processing
// thread and have a sufficient stack size for the host callback execution.
void OnLoadedHostCallbackPollResponse(
    const LoadedHostCallbackPollingState& state, uint64_t operand_handle,
    std::shared_ptr<LoadedHostCallbackPollResponse> response) {
  auto ret_req = std::make_unique<LoadedHostCallbackReturnRequest>();
  ret_req->set_host_callback_execution_handle(
      response->host_callback_execution_handle());

  absl::StatusOr<uint64_t> result_handle = PrepareAndExecuteLoadedHostCallback(
      state.rpc_helper.get(), state.loaded_host_callback.get(), operand_handle);
  if (result_handle.ok()) {
    ret_req->set_result_host_buffer_handle(*result_handle);
  } else {
    *ret_req->mutable_error() = tsl::StatusToProto(result_handle.status());
  }

  state.rpc_helper->LoadedHostCallbackReturn(std::move(ret_req))
      .OnReady(
          [](absl::StatusOr<std::shared_ptr<LoadedHostCallbackReturnResponse>>
                 response) {
            if (!response.ok()) {
              LOG(ERROR) << "Failed to return host callback results: "
                         << response.status();
            }
          });
}

// Polls the state of a single host callback once asynchronously. When the poll
// response is received, the host callback is executed asynchronously and the
// polling is scheduled again, unless the host callback is destructed from the
// server.
void LoadedHostCallbackPoll(LoadedHostCallbackPollingState state) {
  const uint64_t operand_handle = state.rpc_helper->NextHandle();

  auto poll_req = std::make_unique<LoadedHostCallbackPollRequest>();
  poll_req->set_loaded_host_callback_handle(state.handle);
  poll_req->set_operand_host_buffer_handle(operand_handle);

  auto response_future =
      state.rpc_helper->LoadedHostCallbackPoll(std::move(poll_req));
  response_future.OnReady(
      [state = std::move(state), operand_handle](
          absl::StatusOr<std::shared_ptr<LoadedHostCallbackPollResponse>>
              response) mutable {
        GetGlobalThreadPool()->Schedule(
            [state = std::move(state), operand_handle,
             response = std::move(response)]() mutable {
              if (!response.ok()) {
                LOG_EVERY_N_SEC(ERROR, 60)
                    << "Failed to poll host callback execution: "
                    << response.status();
              } else if (!(*response)->has_host_callback_execution_handle()) {
                // The host callback is destructed from the server.
                return;
              } else {
                OnLoadedHostCallbackPollResponse(state, operand_handle,
                                                 *std::move(response));
              }
              LoadedHostCallbackPoll(std::move(state));
            });
      });
};

}  // namespace

// OutputSpecCache caches the output specification of the
// LoadedExecutableExecuteResponse received first from the server; this
// information should be unchanged across multiple invocations of
// `LoadedExecutable::Execute`, and so can be used to optimize further
// executions.
class LoadedExecutable::OutputSpecCache {
 public:
  explicit OutputSpecCache(LoadedExecutable* absl_nonnull parent)
      : parent_(parent) {}

  // Returns the cached output spec if already cached, and std::nullopt if not.
  std::optional<absl::Span<const ArraySpec>> Retrieve() {
    absl::MutexLock l(mu_);
    if (!data_.has_value()) {
      return std::nullopt;
    }
    return data_.value();
  }

  // If data has not already been cached, derives and caches the output spec
  // from the given `outputs` parameter. If data has already been cached,
  // returns OK status.
  absl::Status Cache(const tsl::protobuf::RepeatedPtrField<
                     LoadedExecutableExecuteResponse_Output>& outputs) {
    {
      absl::MutexLock l(mu_);
      if (data_.has_value()) {
        return absl::OkStatus();
      }
    }
    std::vector<ArraySpec> data;
    for (const auto& output : outputs) {
      TF_ASSIGN_OR_RETURN(auto dtype, DType::FromProto(output.dtype()));
      TF_ASSIGN_OR_RETURN(auto shape, Shape::FromProto(output.shape()));
      TF_ASSIGN_OR_RETURN(
          auto sharding,
          Sharding::FromProto(parent_->client(), output.sharding()));
      data.push_back(ArraySpec{/*dtype=*/dtype, /*shape=*/std::move(shape),
                               /*sharding=*/std::move(sharding)});
    }
    {
      absl::MutexLock l(mu_);
      if (!data_.has_value()) {
        data_.emplace(std::move(data));
      }
    }
    return absl::OkStatus();
  }

 private:
  absl::Mutex mu_;
  std::optional<std::vector<ArraySpec>> data_ ABSL_GUARDED_BY(mu_);
  LoadedExecutable* const parent_;
};

LoadedExecutable::LoadedExecutable(
    xla::ifrt::Client* client, std::shared_ptr<RpcHelper> rpc_helper,
    uint64_t handle, std::string name, int num_devices,
    std::optional<DeviceListRef> devices,
    std::vector<xla::ifrt::Device*> addressable_devices,
    absl::StatusOr<std::optional<std::string>> fingerprint,
    std::vector<tsl::RCReference<xla::ifrt::LoadedHostCallback>>
        loaded_host_callbacks,
    std::vector<uint64_t> loaded_host_callback_handles)
    : client_(client),
      rpc_helper_(std::move(rpc_helper)),
      handle_(handle),
      name_(std::move(name)),
      num_devices_(num_devices),
      devices_(devices),
      addressable_devices_(std::move(addressable_devices)),
      fingerprint_(std::move(fingerprint)),
      user_context_(xla::ifrt::UserContextScope::current()),
      output_spec_cache_(
          std::make_unique<LoadedExecutable::OutputSpecCache>(this)) {
  // Start host callback pollers.
  CHECK_EQ(loaded_host_callbacks.size(), loaded_host_callback_handles.size());
  if (!loaded_host_callbacks.empty()) {
    // Note: individual host callbacks may live longer than the executable as
    // the destruction of an IFRT executable is not required to block until all
    // in-flight executions are complete.
    for (int i = 0; i < loaded_host_callbacks.size(); ++i) {
      LoadedHostCallbackPoll(LoadedHostCallbackPollingState{
          rpc_helper_, loaded_host_callback_handles[i],
          loaded_host_callbacks[i]});
    }
  }

  tsl::profiler::TraceMe traceme_ifrt_entrypoint(
      "IfrtProxyEntrypointLoadedExecutableCreate");
  // Asynchronously fetch shardings. Since users of `LoadedExecutable` typically
  // require sharding information to invoke the executable, it is beneficial to
  // eagerly schedule this fetch since, in some implementations, it may take a
  // long time for sharding information to be available.

  auto [promise, future] = tsl::MakePromise<std::shared_ptr<Metadata>>();
  metadata_future_ = std::move(future);

  auto req = std::make_unique<LoadedExecutableMetadataRequest>();
  req->set_loaded_executable_handle(handle_);

  auto on_done = [promise = std::move(promise)](
                     absl::StatusOr<
                         std::shared_ptr<LoadedExecutableMetadataResponse>>
                         response) mutable {
    if (!response.ok()) {
      LOG(ERROR) << "LoadedExecutableMetadata: Got " << response.status();
      promise.Set(response.status());
      return;
    }

    auto info = std::make_shared<Metadata>();

    if (response.value()->has_parameter_shardings()) {
      const auto& p = response.value()->parameter_shardings().shardings();
      info->parameter_shardings.emplace(p.begin(), p.end());
    }
    if (response.value()->has_output_shardings()) {
      const auto& o = response.value()->output_shardings().shardings();
      info->output_shardings.emplace(o.begin(), o.end());
    }

    auto parse_layouts =
        [](const LoadedExecutableMetadataResponse::LayoutList& list)
        -> absl::StatusOr<std::vector<std::shared_ptr<const xla::PjRtLayout>>> {
      std::vector<std::shared_ptr<const xla::PjRtLayout>> layouts;
      layouts.reserve(list.layouts_size());
      for (const auto& layout_proto : list.layouts()) {
        TF_ASSIGN_OR_RETURN(xla::Layout layout,
                            xla::Layout::FromProto(layout_proto));
        layouts.push_back(std::make_shared<xla::PjRtLayout>(std::move(layout)));
      }
      return layouts;
    };

    if (response.value()->has_parameter_layouts_list()) {
      info->parameter_layouts =
          parse_layouts(response.value()->parameter_layouts_list());
    } else if (response.value()->has_parameter_layouts_error()) {
      info->parameter_layouts = xla::ifrt::ReattachUserContextRefs(
          tsl::StatusFromProto(response.value()->parameter_layouts_error()));
    } else {
      info->parameter_layouts = absl::UnimplementedError(
          "IFRT Proxy server did not return parameter layouts");
    }
    if (response.value()->has_output_layouts_list()) {
      info->output_layouts =
          parse_layouts(response.value()->output_layouts_list());
    } else if (response.value()->has_output_layouts_error()) {
      info->output_layouts = xla::ifrt::ReattachUserContextRefs(
          tsl::StatusFromProto(response.value()->output_layouts_error()));
    } else {
      info->output_layouts = absl::UnimplementedError(
          "IFRT Proxy server did not return output layouts");
    }

    if (response.value()->has_compiled_memory_stats()) {
      info->compiled_memory_stats = xla::CompiledMemoryStats::FromProto(
          response.value()->compiled_memory_stats());
    } else if (response.value()->has_compiled_memory_stats_error()) {
      info->compiled_memory_stats =
          xla::ifrt::ReattachUserContextRefs(tsl::StatusFromProto(
              response.value()->compiled_memory_stats_error()));
    } else {
      info->compiled_memory_stats = absl::UnimplementedError(
          "IFRT Proxy server did not return compiled memory stats");
    }

    info->size_of_generated_code_in_bytes =
        response.value()->size_of_generated_code_in_bytes();

    if (const absl::Status s =
            xla::ifrt::ReattachUserContextRefs(tsl::StatusFromProto(
                response.value()->output_memory_kinds().status()));
        !s.ok()) {
      info->output_memory_kinds = s;
    } else {
      std::vector<std::vector<absl::string_view>> output_memory_kinds;
      for (const auto& list :
           response.value()->output_memory_kinds().memory_kind_lists()) {
        std::vector<absl::string_view> kinds;
        kinds.reserve(list.memory_kinds_size());
        for (const absl::string_view kind : list.memory_kinds()) {
          const auto it = info->memory_kinds.insert(std::string(kind)).first;
          kinds.push_back(*it);
        }
        output_memory_kinds.push_back(std::move(kinds));
      }
      info->output_memory_kinds = std::move(output_memory_kinds);
    }

    if (response.value()->has_donated_input_indices()) {
      info->donatable_input_indices =
          std::vector<int>(response.value()
                               ->donated_input_indices()
                               .donated_input_indices()
                               .begin(),
                           response.value()
                               ->donated_input_indices()
                               .donated_input_indices()
                               .end());
      info->donatable_input_indices_set =
          absl::flat_hash_set<int>(info->donatable_input_indices->begin(),
                                   info->donatable_input_indices->end());
    } else if (response.value()->has_donated_input_indices_error()) {
      info->donatable_input_indices =
          xla::ifrt::ReattachUserContextRefs(tsl::StatusFromProto(
              response.value()->donated_input_indices_error()));
    } else {
      info->donatable_input_indices = absl::UnimplementedError(
          "IFRT Proxy server did not return donated input indices");
    }

    promise.Set(std::move(info));
  };
  rpc_helper_->LoadedExecutableMetadata(std::move(req))
      .OnReady(std::move(on_done));
}

LoadedExecutable::~LoadedExecutable() {
  tsl::profiler::TraceMe traceme_ifrt_entrypoint(
      "IfrtProxyEntrypointLoadedExecutableDestruct");

  auto req = std::make_unique<LoadedExecutableDestructRequest>();
  req->set_loaded_executable_handle(handle_);

  rpc_helper_->LoadedExecutableDestruct(std::move(req))
      .OnReady(
          [](absl::StatusOr<std::shared_ptr<LoadedExecutableDestructResponse>>
                 response) {
            if (!response.ok()) {
              LOG(ERROR) << "Failed to destroy `LoadedExecutable`: "
                         << response.status();
            }
          });
}

xla::ifrt::Client* LoadedExecutable::client() const { return client_; }

absl::string_view LoadedExecutable::name() const { return name_; }

absl::StatusOr<std::optional<std::string>> LoadedExecutable::Fingerprint()
    const {
  return fingerprint_;
}

absl::StatusOr<std::string> LoadedExecutable::Serialize() const {
  return absl::UnimplementedError(
      "IFRT service executable does not support `Serialize` since the "
      "underlying serialization format is not stable");
}

int LoadedExecutable::num_devices() const { return num_devices_; }

int64_t LoadedExecutable::SizeOfGeneratedCodeInBytes() const {
  tsl::profiler::TraceMe traceme_ifrt_entrypoint(
      "IfrtProxyEntrypointLoadedExecutableSizeOfGeneratedCodeInBytes");
  auto info = metadata_future_.Await();
  if (!info.ok()) {
    LOG(ERROR) << "SizeOfGeneratedCodeInBytes: " << info.status();
    return 0;
  }
  return (*info)->size_of_generated_code_in_bytes;
}

absl::StatusOr<CompiledMemoryStats> LoadedExecutable::GetCompiledMemoryStats()
    const {
  tsl::profiler::TraceMe traceme_ifrt_entrypoint(
      "IfrtProxyEntrypointLoadedExecutableGetCompiledMemoryStats");
  auto info = metadata_future_.Await();
  if (!info.ok()) {
    return info.status();
  }
  return (*info)->compiled_memory_stats;
}

std::optional<std::vector<OpSharding>> LoadedExecutable::GetParameterShardings()
    const {
  tsl::profiler::TraceMe traceme_ifrt_entrypoint(
      "IfrtProxyEntrypointLoadedExecutableGetParameterShardings");
  auto info = metadata_future_.Await();
  if (!info.ok()) {
    return std::nullopt;
  }
  return (*info)->parameter_shardings;
}

absl::StatusOr<absl::Span<const int>>
LoadedExecutable::GetDonatableInputIndices() const {
  tsl::profiler::TraceMe traceme_ifrt_entrypoint(
      "IfrtProxyEntrypointLoadedExecutableDonatableInputIndices");
  TF_ASSIGN_OR_RETURN(auto info, metadata_future_.Await());
  return info->donatable_input_indices;
}

std::optional<std::vector<OpSharding>> LoadedExecutable::GetOutputShardings()
    const {
  tsl::profiler::TraceMe traceme_ifrt_entrypoint(
      "IfrtProxyEntrypointLoadedExecutableGetOutputShardings");
  auto info = metadata_future_.Await();
  if (!info.ok()) {
    return std::nullopt;
  }
  return (*info)->output_shardings;
}

absl::StatusOr<std::vector<std::shared_ptr<const xla::PjRtLayout>>>
LoadedExecutable::GetParameterLayouts() const {
  tsl::profiler::TraceMe traceme_ifrt_entrypoint(
      "IfrtProxyEntrypointLoadedExecutableGetParameterLayouts");
  TF_ASSIGN_OR_RETURN(auto info, metadata_future_.Await());
  return info->parameter_layouts;
}

absl::StatusOr<std::vector<std::shared_ptr<const xla::PjRtLayout>>>
LoadedExecutable::GetOutputLayouts() const {
  tsl::profiler::TraceMe traceme_ifrt_entrypoint(
      "IfrtProxyEntrypointLoadedExecutableGetOutputLayouts");
  TF_ASSIGN_OR_RETURN(auto info, metadata_future_.Await());
  return info->output_layouts;
}

absl::StatusOr<std::vector<std::vector<absl::string_view>>>
LoadedExecutable::GetOutputMemoryKinds() const {
  tsl::profiler::TraceMe traceme_ifrt_entrypoint(
      "IfrtProxyEntrypointLoadedExecutableGetOutputMemoryKinds");
  TF_ASSIGN_OR_RETURN(auto info, metadata_future_.Await());
  return info->output_memory_kinds;
}

absl::StatusOr<std::vector<std::shared_ptr<HloModule>>>
LoadedExecutable::GetHloModules() const {
  return absl::UnimplementedError(
      "IFRT service does not support LoadedExecutable::GetHloModules() since "
      "HloModule does not provide stable serialization");
}

absl::StatusOr<xla::ifrt::AttributeMap> LoadedExecutable::GetCostAnalysis()
    const {
  if (rpc_helper_->protocol_version() <
      protocol_version::kLoadedExecutableGetCostAnalysis) {
    return absl::UnimplementedError(
        "LoadedExecutable::GetCostAnalysis() is unimplemented by IFRT proxy");
  }

  absl::MutexLock l(cost_analysis_mu_);
  if (!cost_analysis_response_.has_value()) {
    auto req = std::make_unique<LoadedExecutableCostAnalysisRequest>();
    req->set_loaded_executable_handle(handle_);

    absl::StatusOr<std::shared_ptr<LoadedExecutableCostAnalysisResponse>>
        response =
            rpc_helper_->LoadedExecutableCostAnalysis(std::move(req)).Await();

    if (!response.ok()) {
      // Connection-related error, so log the error.
      LOG(ERROR) << "LoadedExecutableCostAnalysis: Got " << response.status();
      cost_analysis_response_ = response.status();
    }
    if (response.ok() && response.value()->has_attributes()) {
      cost_analysis_response_ =
          AttributeMap::FromProto(response.value()->attributes());
    } else {
      cost_analysis_response_ = xla::ifrt::ReattachUserContextRefs(
          tsl::StatusFromProto(response.value()->status()));
    }
  }
  return *cost_analysis_response_;
}

absl::StatusOr<std::string> LoadedExecutable::GetHumanReadableProgramText()
    const {
  if (rpc_helper_->protocol_version() <
      protocol_version::kLoadedExecutableGetHumanReadableProgramText) {
    return absl::UnimplementedError(
        "LoadedExecutable::GetHumanReadableProgramText() is unimplemented by "
        "IFRT proxy");
  }

  absl::MutexLock l(human_readable_program_text_mu_);
  if (!human_readable_program_text_.has_value()) {
    auto req =
        std::make_unique<LoadedExecutableHumanReadableProgramTextRequest>();
    req->set_loaded_executable_handle(handle_);

    absl::StatusOr<
        std::shared_ptr<LoadedExecutableHumanReadableProgramTextResponse>>
        response =
            rpc_helper_
                ->LoadedExecutableHumanReadableProgramText(std::move(req))
                .Await();

    if (!response.ok()) {
      // Connection-related error, so log the error.
      LOG(ERROR) << "LoadedExecutableHumanReadableProgramText: Got "
                 << response.status();
      human_readable_program_text_ = response.status();
    } else if ((*response)->has_human_readable_program_text()) {
      human_readable_program_text_ = (*response)->human_readable_program_text();
    } else {
      human_readable_program_text_ = xla::ifrt::ReattachUserContextRefs(
          tsl::StatusFromProto((*response)->status()));
    }
  }
  return *human_readable_program_text_;
}

absl::StatusOr<xla::ifrt::LoadedExecutable::ExecuteResult>
LoadedExecutable::Execute(absl::Span<xla::ifrt::ArrayRef> args,
                          const ExecuteOptions& options,
                          std::optional<xla::ifrt::DeviceListRef> devices) {
  tsl::profiler::TraceMe traceme_ifrt_entrypoint(
      "IfrtProxyEntrypointLoadedExecutableExecute");
  auto req = std::make_unique<LoadedExecutableExecuteRequest>();
  req->set_loaded_executable_handle(handle_);

  TF_ASSIGN_OR_RETURN(auto info, metadata_future_.Await());
  for (int i = 0; i < args.size(); ++i) {
    xla::ifrt::ArrayRef& arg = args[i];
    auto* array = llvm::dyn_cast_or_null<Array>(arg.get());
    if (array == nullptr) {
      return absl::InvalidArgumentError(
          "Invalid IFRT array type provided to `LoadedExecutable::Execute`");
    }
    if (options.non_donatable_input_indices.contains(i)) {
      TF_ASSIGN_OR_RETURN(ArrayHandle handle,
                          array->GetHandle(ArrayCopySemantics::kAlwaysCopy));
      req->add_args_handles(handle.handle);
    } else if (!info->donatable_input_indices_set.has_value()) {
      TF_ASSIGN_OR_RETURN(ArrayHandle handle,
                          array->GetHandleUnknownIfBeingDonated());
      req->add_args_handles(handle.handle);
    } else if (info->donatable_input_indices_set->contains(i)) {
      TF_ASSIGN_OR_RETURN(ArrayHandle handle,
                          array->GetHandle(ArrayCopySemantics::kDonateInput));
      req->add_args_handles(handle.handle);
    } else {
      TF_ASSIGN_OR_RETURN(ArrayHandle handle,
                          array->GetHandle(ArrayCopySemantics::kAlwaysCopy));
      req->add_args_handles(handle.handle);
    }
  }
  TF_RETURN_IF_ERROR(options.ToProto(*req->mutable_execute_options(),
                                     rpc_helper_->ifrt_serdes_version()));
  if (devices.has_value()) {
    for (const auto* device : (*devices)->devices()) {
      req->add_device_ids(device->Id().value());
    }
  }

  std::optional<uint64_t> device_time_key = xla::GetDeviceTimeMeasurementKey();
  if (device_time_key.has_value()) {
    // An active device time measurement requires the server to respond with
    // measured device times after the execution is complete.
    req->mutable_execute_options()->set_fill_status(true);
  }

  // Starting version 6, the server populates the status future only if it was
  // explicitly requested via `options.fill_status`.
  const bool result_needs_exec_status = req->execute_options().fill_status();

  // The client generates handles if the protocol version is sufficiently newer,
  // and we've already seen at least one response from an execute (and thus know
  // the number of handles to generate).
  const bool client_generated_handles =
      output_spec_cache_->Retrieve().has_value();

  xla::ifrt::LoadedExecutable::ExecuteResult result;
  // TODO(hyeontaek): `GetOutputLayouts()` uses a concrete layout for a
  // default layout. This will change as proper IFRT layout support is fleshed
  // out. While the code here using `layouts` will automatically benefit from
  // the semantics change for `GetOutputLayouts()`, we would have a slightly
  // inconsistent state here until the change happens where output arrays use a
  // concrete layout for a default layout. This will not cause an issue for the
  // time being when the user always uses concrete layouts, but we would need to
  // resolve this issue before the user begins to use `nullptr` default layouts
  // without resolving it to a concrete layout.
  absl::StatusOr<std::vector<std::shared_ptr<const xla::PjRtLayout>>> layouts =
      GetOutputLayouts();

  if (client_generated_handles) {
    auto output_specs = *output_spec_cache_->Retrieve();
    if (layouts.ok() && layouts->size() != output_specs.size()) {
      return absl::InternalError(absl::StrCat(
          "Mismatch between output specs and layouts: ", output_specs.size(),
          " vs ", layouts->size()));
    }
    for (int i = 0; i < output_specs.size(); ++i) {
      const auto& output_spec = output_specs[i];
      uint64_t handle = rpc_helper_->NextHandle();
      if (layouts.ok()) {
        result.outputs.push_back(tsl::MakeRef<Array>(
            client(), rpc_helper_, output_spec.dtype, output_spec.shape,
            output_spec.sharding, ArrayHandle{handle},
            /*layout=*/std::move((*layouts)[i])));
      } else {
        result.outputs.push_back(tsl::MakeRef<Array>(
            client(), rpc_helper_, output_spec.dtype, output_spec.shape,
            output_spec.sharding, ArrayHandle{handle}, /*layout=*/nullptr));
      }
      req->add_result_array_handle(handle);
    }
    uint64_t status_handle = rpc_helper_->NextHandle();
    if (result_needs_exec_status) {
      req->set_result_status_handle(status_handle);
    }
    rpc_helper_->LoadedExecutableExecute(std::move(req));
    if (result_needs_exec_status) {
      // Note that the RPCs within `FetchExecuteResult` need to be sent after
      // `LoadedExecutableExecute` above, or the server will not recognize the
      // handle being sent.
      tsl::Future<> status = FetchExecuteResult(status_handle, device_time_key);
      if (options.fill_status) {
        result.status = std::move(status);
      }
    }

    return result;
  }

  TF_ASSIGN_OR_RETURN(
      std::shared_ptr<LoadedExecutableExecuteResponse> response,
      rpc_helper_->LoadedExecutableExecute(std::move(req)).Await());
  auto status = output_spec_cache_->Cache(response->outputs());
  if (!status.ok()) {
    // Handles in `response` need to be destructed remotely.
    for (const auto& output : response->outputs()) {
      Array::Destruct(rpc_helper_.get(), ArrayHandle{output.array_handle()});
    }
    if (result_needs_exec_status) {
      // `FetchExecuteResult` deletes the server-side future handle.
      FetchExecuteResult(response->status_handle(), device_time_key);
    }
    return status;
  }
  absl::Span<const ArraySpec> output_specs = *output_spec_cache_->Retrieve();
  if (layouts.ok() && layouts->size() != output_specs.size()) {
    return absl::InternalError(absl::StrCat(
        "Mismatch between output specs and layouts: ", output_specs.size(),
        " vs ", layouts->size()));
  }
  for (int i = 0; i < output_specs.size(); ++i) {
    const auto& output_spec = output_specs[i];
    if (layouts.ok()) {
      result.outputs.push_back(tsl::MakeRef<Array>(
          client(), rpc_helper_, output_spec.dtype, output_spec.shape,
          output_spec.sharding,
          ArrayHandle{response->outputs()[i].array_handle()},
          /*layout=*/std::move((*layouts)[i])));
    } else {
      result.outputs.push_back(tsl::MakeRef<Array>(
          client(), rpc_helper_, output_spec.dtype, output_spec.shape,
          output_spec.sharding,
          ArrayHandle{response->outputs()[i].array_handle()},
          /*layout=*/nullptr));
    }
  }
  if (result_needs_exec_status) {
    tsl::Future<> status =
        FetchExecuteResult(response->status_handle(), device_time_key);
    if (options.fill_status) {
      result.status = std::move(status);
    }
  } else {
    CHECK_EQ(response->status_handle(), 0);
  }

  return result;
}

std::optional<DeviceListRef> LoadedExecutable::devices() const {
  return devices_;
}

absl::Span<xla::ifrt::Device* const> LoadedExecutable::addressable_devices()
    const {
  return addressable_devices_;
}

tsl::Future<> LoadedExecutable::FetchExecuteResult(
    uint64_t status_handle, std::optional<uint64_t> device_time_key) {
  if (rpc_helper_->protocol_version() < protocol_version::kExecuteResult) {
    return rpc_helper_->CheckFuture(status_handle);
  }
  auto req = std::make_unique<LoadedExecutableFetchExecuteResultRequest>();
  req->set_result_status_handle(status_handle);

  using RespT = std::shared_ptr<LoadedExecutableFetchExecuteResultResponse>;

  tsl::Future<RespT> result =
      rpc_helper_->LoadedExecutableFetchExecuteResult(std::move(req));

  if (device_time_key.has_value()) {
    result.OnReady([device_time_key](const absl::StatusOr<RespT>& resp) {
      if (!resp.ok()) {
        LOG_EVERY_N_SEC(ERROR, 60)
            << "Device time measurement was requested but failed to retrieve "
               "the execution result: "
            << resp.status();
        return;
      }

      for (const auto& [device_type_name, duration] : (*resp)->device_time()) {
        xla::DeviceTimeMeasurement::DeviceType device_type;
        if (device_type_name == "tpu") {
          device_type = xla::DeviceTimeMeasurement::DeviceType::kTpu;
        } else if (device_type_name == "gpu") {
          device_type = xla::DeviceTimeMeasurement::DeviceType::kGpu;
        } else {
          device_type = xla::DeviceTimeMeasurement::DeviceType::kUnknown;
        }
        if (device_type != xla::DeviceTimeMeasurement::DeviceType::kUnknown) {
          xla::RecordDeviceTimeMeasurement(
              *device_time_key, absl::Microseconds(duration), device_type);
        }
      }
    });
  }

  return result.GetReadyFuture();
}

char LoadedExecutable::ID = 0;  // NOLINT

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
