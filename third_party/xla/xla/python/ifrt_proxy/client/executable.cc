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
#include "absl/types/span.h"
#include "llvm/Support/Casting.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/layout.h"
#include "xla/pjrt/host_callback.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/array_spec.h"
#include "xla/python/ifrt/attribute_map.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/future.h"
#include "xla/python/ifrt/host_callback.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt_proxy/client/array.h"
#include "xla/python/ifrt_proxy/client/host_buffer.h"
#include "xla/python/ifrt_proxy/client/rpc_helper.h"
#include "xla/python/ifrt_proxy/common/ifrt_service.pb.h"
#include "xla/python/ifrt_proxy/common/types.h"
#include "xla/python/ifrt_proxy/common/versions.h"
#include "xla/python/pjrt_ifrt/pjrt_host_callback.h"
#include "xla/shape_util.h"
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
    void* p = tsl::port::AlignedMalloc(size, kAlignment);
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
    void* data = tsl::port::AlignedMalloc(size, kAlignment);
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

}  // namespace

// OutputSpecCache caches the output specification of the
// LoadedExecutableExecuteResponse received first from the server; this
// information should be unchanged across multiple invocations of
// `LoadedExecutable::Execute`, and so can be used to optimize further
// executions.
class LoadedExecutable::OutputSpecCache {
 public:
  explicit OutputSpecCache(absl::Nonnull<LoadedExecutable*> parent)
      : parent_(parent) {}

  // Returns the cached output spec if already cached, and std::nullopt if not.
  std::optional<absl::Span<const ArraySpec>> Retrieve() {
    absl::MutexLock l(&mu_);
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
      absl::MutexLock l(&mu_);
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
      absl::MutexLock l(&mu_);
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
    std::vector<xla::ifrt::Device*> addressable_devices,
    absl::StatusOr<std::optional<std::string>> fingerprint,
    Future<> ready_future,
    std::vector<tsl::RCReference<xla::ifrt::LoadedHostCallback>>
        loaded_host_callbacks,
    std::vector<uint64_t> loaded_host_callback_handles)
    : client_(client),
      rpc_helper_(std::move(rpc_helper)),
      handle_(handle),
      name_(std::move(name)),
      num_devices_(num_devices),
      addressable_devices_(std::move(addressable_devices)),
      fingerprint_(std::move(fingerprint)),
      ready_future_(std::move(ready_future)),
      output_spec_cache_(
          std::make_unique<LoadedExecutable::OutputSpecCache>(this)) {
  // Start host callback pollers.
  CHECK_EQ(loaded_host_callbacks.size(), loaded_host_callback_handles.size());
  if (!loaded_host_callbacks.empty()) {
    for (int i = 0; i < loaded_host_callbacks.size(); ++i) {
      PollLoadedHostCallback(loaded_host_callback_handles[i],
                             loaded_host_callbacks[i]);
    }
  }

  tsl::profiler::TraceMe traceme_ifrt_entrypoint(

      "IfrtProxyEntrypointLoadedExecutableCreate");
  // Asynchronously fetch shardings. Since users of `LoadedExecutable` typically
  // require sharding information to invoke the executable, it is beneficial to
  // eagerly schedule this fetch since, in some implementations, it may take a
  // long time for sharding information to be available.

  auto promise = Future<std::shared_ptr<Metadata>>::CreatePromise();
  metadata_future_ = Future<std::shared_ptr<Metadata>>(promise);

  auto req = std::make_unique<LoadedExecutableMetadataRequest>();
  req->set_loaded_executable_handle(handle_);

  auto on_done =
      [promise](
          absl::StatusOr<std::shared_ptr<LoadedExecutableMetadataResponse>>
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
            [](const LoadedExecutableMetadataResponse::LayoutList& list) {
              std::vector<std::shared_ptr<const xla::PjRtLayout>> layouts;
              layouts.reserve(list.layouts_size());
              for (const auto& layout : list.layouts()) {
                layouts.push_back(std::make_shared<xla::PjRtLayout>(
                    xla::Layout::CreateFromProto(layout)));
              }
              return layouts;
            };

        if (response.value()->has_parameter_layouts_list()) {
          info->parameter_layouts =
              parse_layouts(response.value()->parameter_layouts_list());
        } else if (response.value()->has_parameter_layouts_error()) {
          info->parameter_layouts =
              tsl::StatusFromProto(response.value()->parameter_layouts_error());
        } else {
          info->parameter_layouts = absl::UnimplementedError(
              "IFRT Proxy server did not return parameter layouts");
        }
        if (response.value()->has_output_layouts_list()) {
          info->output_layouts =
              parse_layouts(response.value()->output_layouts_list());
        } else if (response.value()->has_output_layouts_error()) {
          info->output_layouts =
              tsl::StatusFromProto(response.value()->output_layouts_error());
        } else {
          info->output_layouts = absl::UnimplementedError(
              "IFRT Proxy server did not return output layouts");
        }

        if (const absl::Status s = tsl::StatusFromProto(
                response.value()->output_memory_kinds().status());
            !s.ok()) {
          info->output_memory_kinds = s;
        } else {
          std::vector<std::vector<absl::string_view>> output_memory_kinds;
          for (const auto& list :
               response.value()->output_memory_kinds().memory_kind_lists()) {
            std::vector<absl::string_view> kinds;
            kinds.reserve(list.memory_kinds_size());
            for (const absl::string_view kind : list.memory_kinds()) {
              const auto it =
                  info->memory_kinds.insert(std::string(kind)).first;
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
          info->donatable_input_indices = tsl::StatusFromProto(
              response.value()->donated_input_indices_error());
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

Future<> LoadedExecutable::GetReadyFuture() const { return ready_future_; }

int LoadedExecutable::num_devices() const { return num_devices_; }

int64_t LoadedExecutable::SizeOfGeneratedCodeInBytes() const {
  LOG(FATAL) << "Unimplemented";
}

absl::StatusOr<CompiledMemoryStats> LoadedExecutable::GetCompiledMemoryStats()
    const {
  return absl::UnimplementedError("Unimplemented");
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
  return absl::UnimplementedError("Unimplemented");
}

absl::StatusOr<xla::ifrt::LoadedExecutable::ExecuteResult>
LoadedExecutable::Execute(absl::Span<tsl::RCReference<xla::ifrt::Array>> args,
                          const ExecuteOptions& options,
                          std::optional<xla::ifrt::DeviceListRef> devices) {
  tsl::profiler::TraceMe traceme_ifrt_entrypoint(
      "IfrtProxyEntrypointLoadedExecutableExecute");
  auto req = std::make_unique<LoadedExecutableExecuteRequest>();
  req->set_loaded_executable_handle(handle_);

  TF_ASSIGN_OR_RETURN(auto info, metadata_future_.Await());
  for (int i = 0; i < args.size(); ++i) {
    tsl::RCReference<xla::ifrt::Array>& arg = args[i];
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
  TF_ASSIGN_OR_RETURN(*req->mutable_execute_options(), options.ToProto());
  if (devices.has_value()) {
    for (const auto* device : (*devices)->devices()) {
      req->add_device_ids(device->Id().value());
    }
  }

  // Starting version 6, the server populates the status future only if it was
  // explicitly requested via `options.fill_status`.
  const bool result_needs_exec_status =
      rpc_helper_->version().protocol_version() < 6 || options.fill_status;

  // The client generates handles if the protocol version is sufficiently newer,
  // and we've already seen at least one response from an execute (and thus know
  // the number of handles to generate).
  const bool client_generated_handles =
      (rpc_helper_->version().protocol_version() >=
       protocol_version::kClientHandlesExecutableOptimization) &&
      output_spec_cache_->Retrieve().has_value();

  xla::ifrt::LoadedExecutable::ExecuteResult result;
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
      // Note that `CheckFuture` needs to be sent after
      // `LoadedExecutableExecute` above, or the server will not recognize the
      // handle being sent.
      result.status = rpc_helper_->CheckFuture(status_handle);
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
      // `CheckFuture` deletes the server-side future handle.
      rpc_helper_->CheckFuture(response->status_handle());
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
    result.status = rpc_helper_->CheckFuture(response->status_handle());
  } else {
    CHECK_EQ(response->status_handle(), 0);
  }

  return result;
}

Future<> LoadedExecutable::Delete() {
  tsl::profiler::TraceMe traceme_ifrt_entrypoint(
      "IfrtProxyEntrypointLoadedExecutableDelete");
  auto req = std::make_unique<LoadedExecutableDeleteRequest>();
  req->set_loaded_executable_handle(handle_);

  auto promise = Future<>::CreatePromise();
  Future<> result(promise);

  rpc_helper_->LoadedExecutableDelete(std::move(req))
      .OnReady(
          [promise = std::move(promise), rpc_helper = rpc_helper_](
              absl::StatusOr<std::shared_ptr<LoadedExecutableDeleteResponse>>
                  response) mutable {
            if (!response.ok()) {
              promise.Set(response.status());
              return;
            }
            rpc_helper->CheckFuture((*response)->future_handle())
                .OnReady([promise = std::move(promise)](
                             absl::Status s) mutable { promise.Set(s); });
          });
  return result;
}

bool LoadedExecutable::IsDeleted() const {
  tsl::profiler::TraceMe traceme_ifrt_entrypoint(
      "IfrtProxyEntrypointLoadedExecutableIsDeleted");
  auto req = std::make_unique<LoadedExecutableIsDeletedRequest>();
  req->set_loaded_executable_handle(handle_);

  absl::StatusOr<std::shared_ptr<LoadedExecutableIsDeletedResponse>> response =
      rpc_helper_->LoadedExecutableIsDeleted(std::move(req)).Await();
  if (!response.ok()) {
    LOG(ERROR) << "Failed to query the deletion status of `LoadedExecutable`: "
               << response.status();
    return false;
  }
  return (*response)->is_deleted();
}

absl::Span<xla::ifrt::Device* const> LoadedExecutable::addressable_devices()
    const {
  return addressable_devices_;
}

namespace {

static tsl::ThreadOptions GetThreadOptions() {
  tsl::ThreadOptions thread_options;
  // Ensure the threads' stack is large enough for arbitrary Python code.
  thread_options.stack_size = 2 * 1024 * 1024;  // 2 MiB
  return thread_options;
}

}  // namespace

void LoadedExecutable::PollLoadedHostCallback(
    uint64_t handle,
    tsl::RCReference<xla::ifrt::LoadedHostCallback> loaded_host_callback) {
  // Note: individual host callbacks may live longer than the executable as the
  // destruction of an IFRT executable is not required to block until all
  // in-flight executions are complete. Therefore, the following lambda must not
  // capture `this` and is scheduled on the default thread pool.
  auto f = [rpc_helper = rpc_helper_, handle,
            loaded_host_callback = std::move(loaded_host_callback)]() {
    while (true) {
      const uint64_t operand_handle = rpc_helper->NextHandle();

      auto poll_req = std::make_unique<LoadedHostCallbackPollRequest>();
      poll_req->set_loaded_host_callback_handle(handle);
      poll_req->set_operand_host_buffer_handle(operand_handle);
      auto response =
          rpc_helper->LoadedHostCallbackPoll(std::move(poll_req)).Await();

      if (!response.ok()) {
        LOG_EVERY_N_SEC(ERROR, 60)
            << "Failed to poll host callback execution: " << response.status();
        continue;
      }

      if (!(*response)->has_host_callback_execution_handle()) {
        // The host callback is destructed from the server.
        break;
      }

      auto ret_req = std::make_unique<LoadedHostCallbackReturnRequest>();
      ret_req->set_host_callback_execution_handle(
          (*response)->host_callback_execution_handle());

      absl::StatusOr<uint64_t> result_handle =
          PrepareAndExecuteLoadedHostCallback(
              rpc_helper.get(), loaded_host_callback.get(), operand_handle);
      if (result_handle.ok()) {
        ret_req->set_result_host_buffer_handle(*result_handle);
      } else {
        *ret_req->mutable_error() = tsl::StatusToProto(result_handle.status());
      }

      rpc_helper->LoadedHostCallbackReturn(std::move(ret_req))
          .OnReady([](absl::StatusOr<
                       std::shared_ptr<LoadedHostCallbackReturnResponse>>
                          response) {
            if (!response.ok()) {
              LOG(ERROR) << "Failed to return host callback results: "
                         << response.status();
            }
          });
    }
  };

  static auto* const global_pool = new tsl::thread::ThreadPool(
      tsl::Env::Default(), GetThreadOptions(), "XLAIFRTProxy",
      std::min(16, tsl::port::MaxParallelism()));
  global_pool->Schedule(std::move(f));
}

char LoadedExecutable::ID = 0;  // NOLINT

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
