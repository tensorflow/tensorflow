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

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/bind_front.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/Support/Casting.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/layout.h"
#include "xla/pjrt/host_callback.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
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
#include "xla/python/pjrt_ifrt/pjrt_host_callback.h"
#include "xla/shape_util.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/cpu_info.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status_to_from_proto.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/threadpool.h"

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
    void operator()(void* p) { free(p); }
  };

  std::vector<std::unique_ptr<char, Deleter>> operands;
  operands.reserve(xla_host_callback.operands.size());
  std::vector<void*> operand_ptrs;
  operand_ptrs.reserve(xla_host_callback.operands.size());

  absl::CordReader reader(operand_buffer);
  for (const auto& spec : xla_host_callback.operands) {
    const int64_t size = xla::ShapeUtil::ByteSizeOf(spec.shape);
    void* p;
    CHECK_EQ(posix_memalign(&p, kAlignment, size), 0);
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
    void* data;
    CHECK_EQ(posix_memalign(&data, kAlignment, size), 0);

    result_ptrs.push_back(data);
    result_buffer.AppendExternalMemory(
        absl::string_view(reinterpret_cast<char*>(data), size), data, &free);
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
    ClientHostBufferStore* host_buffer_store,
    xla::ifrt::LoadedHostCallback* loaded_host_callback,
    uint64_t operand_handle) {
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

  const uint64_t result_handle = host_buffer_store->NextHandle();
  TF_RETURN_IF_ERROR(host_buffer_store->Store(result_handle, results).Await());
  return result_handle;
}

}  // namespace

LoadedExecutable::LoadedExecutable(
    xla::ifrt::Client* client, std::shared_ptr<RpcHelper> rpc_helper,
    uint64_t handle, std::string name, int num_devices,
    std::vector<xla::ifrt::LoadedExecutable::LogicalDeviceIds>
        addressable_device_logical_device_ids,
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
      addressable_device_logical_device_ids_(
          std::move(addressable_device_logical_device_ids)),
      addressable_devices_(std::move(addressable_devices)),
      fingerprint_(std::move(fingerprint)),
      ready_future_(std::move(ready_future)) {
  // Start host callback pollers.
  CHECK_EQ(loaded_host_callbacks.size(), loaded_host_callback_handles.size());
  if (!loaded_host_callbacks.empty()) {
    for (int i = 0; i < loaded_host_callbacks.size(); ++i) {
      PollLoadedHostCallback(loaded_host_callback_handles[i],
                             loaded_host_callbacks[i]);
    }
  }

  // Asynchronously fetch shardings. Since users of `LoadedExecutable` typically
  // require sharding information to invoke the executable, it is beneficial to
  // eagerly schedule this fetch since, in some implementations, it may take a
  // long time for sharding information to be available.

  auto promise =
      Future<absl::StatusOr<std::shared_ptr<Metadata>>>::CreatePromise();
  metadata_future_ = Future<absl::StatusOr<std::shared_ptr<Metadata>>>(promise);

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
              std::vector<xla::Layout> layouts;
              layouts.reserve(list.layouts_size());
              for (const auto& layout : list.layouts()) {
                layouts.push_back(xla::Layout::CreateFromProto(layout));
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

        promise.Set(std::move(info));
      };
  rpc_helper_->LoadedExecutableMetadata(std::move(req))
      .OnReady(std::move(on_done));
}

LoadedExecutable::~LoadedExecutable() {
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
  auto info = metadata_future_.Await();
  if (!info.ok()) {
    return std::nullopt;
  }
  return (*info)->parameter_shardings;
}

std::optional<std::vector<OpSharding>> LoadedExecutable::GetOutputShardings()
    const {
  auto info = metadata_future_.Await();
  if (!info.ok()) {
    return std::nullopt;
  }
  return (*info)->output_shardings;
}

absl::StatusOr<std::vector<std::unique_ptr<Layout>>>
LoadedExecutable::GetParameterLayouts() const {
  TF_ASSIGN_OR_RETURN(auto info, metadata_future_.Await());
  TF_RETURN_IF_ERROR(info->parameter_layouts.status());

  std::vector<std::unique_ptr<Layout>> result;
  result.reserve(info->parameter_layouts->size());
  for (const xla::Layout& layout : *info->parameter_layouts) {
    result.push_back(std::make_unique<xla::PjRtXlaLayout>(layout));
  }
  return result;
}

absl::StatusOr<std::vector<std::unique_ptr<Layout>>>
LoadedExecutable::GetOutputLayouts() const {
  TF_ASSIGN_OR_RETURN(auto info, metadata_future_.Await());
  TF_RETURN_IF_ERROR(info->output_layouts.status());

  std::vector<std::unique_ptr<Layout>> result;
  result.reserve(info->output_layouts->size());
  for (const xla::Layout& layout : *info->output_layouts) {
    result.push_back(std::make_unique<xla::PjRtXlaLayout>(layout));
  }
  return result;
}

absl::StatusOr<std::vector<std::vector<absl::string_view>>>
LoadedExecutable::GetOutputMemoryKinds() const {
  TF_ASSIGN_OR_RETURN(auto info, metadata_future_.Await());
  return info->output_memory_kinds;
}

absl::StatusOr<std::vector<std::shared_ptr<HloModule>>>
LoadedExecutable::GetHloModules() const {
  return absl::UnimplementedError(
      "IFRT service does not support LoadedExecutable::GetHloModules() since "
      "HloModule does not provide stable serialization");
}

absl::StatusOr<
    absl::flat_hash_map<std::string, xla::ifrt::Executable::CostAnalysisValue>>
LoadedExecutable::GetCostAnalysis() const {
  return absl::UnimplementedError("Unimplemented");
}

absl::StatusOr<xla::ifrt::LoadedExecutable::ExecuteResult>
LoadedExecutable::Execute(absl::Span<tsl::RCReference<xla::ifrt::Array>> args,
                          const ExecuteOptions& options,
                          std::optional<xla::ifrt::DeviceList> devices) {
  auto req = std::make_unique<LoadedExecutableExecuteRequest>();
  req->set_loaded_executable_handle(handle_);
  for (const auto& arg : args) {
    auto* array = llvm::dyn_cast_or_null<Array>(arg.get());
    if (array == nullptr) {
      return absl::InvalidArgumentError(
          "Invalid IFRT array type provided to `LoadedExecutable::Execute`");
    }
    req->add_args_handles(array->handle().handle);
  }
  TF_ASSIGN_OR_RETURN(*req->mutable_execute_options(), options.ToProto());
  if (devices.has_value()) {
    for (const auto* device : *devices) {
      req->add_device_ids(device->Id().value());
    }
  }

  TF_ASSIGN_OR_RETURN(
      std::shared_ptr<LoadedExecutableExecuteResponse> response,
      rpc_helper_->LoadedExecutableExecute(std::move(req)).Await());

  // NOTE: All future and array handles in `response` must have an owner
  // locally, or be requested to be destructed remotely, before returning.

  xla::ifrt::LoadedExecutable::ExecuteResult result;

  // Populate the execution status future. `CheckFuture` deletes the server-side
  // futures after its completion.
  result.status = rpc_helper_->CheckFuture(response->status_handle());

  // Create output arrays. The cleanup logic ensures that all handles are
  // properly cleaned up on early return.
  absl::Cleanup cleanup = [&]() {
    int index = result.outputs.size();
    result.outputs.clear();  // Cleaned up by `~Array()`.

    for (; index < response->outputs_size(); ++index) {
      Array::Destruct(rpc_helper_.get(),
                      ArrayHandle{response->outputs(index).array_handle()});
    }
  };
  const auto lookup_device = absl::bind_front(&Client::LookupDevice, client());
  for (const auto& output : response->outputs()) {
    TF_ASSIGN_OR_RETURN(DType dtype, DType::FromProto(output.dtype()));
    TF_ASSIGN_OR_RETURN(Shape shape, Shape::FromProto(output.shape()));
    TF_ASSIGN_OR_RETURN(auto sharding,
                        Sharding::FromProto(lookup_device, output.sharding()));
    result.outputs.push_back(tsl::MakeRef<Array>(
        client(), rpc_helper_, dtype, std::move(shape), std::move(sharding),
        ArrayHandle{output.array_handle()}));
  }
  std::move(cleanup).Cancel();

  return result;
}

Future<> LoadedExecutable::Delete() {
  auto req = std::make_unique<LoadedExecutableDeleteRequest>();
  req->set_loaded_executable_handle(handle_);

  absl::StatusOr<std::shared_ptr<LoadedExecutableDeleteResponse>> response =
      rpc_helper_->LoadedExecutableDelete(std::move(req)).Await();
  if (!response.ok()) {
    return Future<>(response.status());
  }
  return rpc_helper_->CheckFuture((*response)->future_handle());
}

bool LoadedExecutable::IsDeleted() const {
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

absl::Span<const LoadedExecutable::LogicalDeviceIds>
LoadedExecutable::addressable_device_logical_ids() const {
  return addressable_device_logical_device_ids_;
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
      const uint64_t operand_handle =
          rpc_helper->host_buffer_store()->NextHandle();

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
              rpc_helper->host_buffer_store().get(), loaded_host_callback.get(),
              operand_handle);
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

  static auto* global_pool = new tsl::thread::ThreadPool(
      tsl::Env::Default(), GetThreadOptions(), "XLAIFRTProxy",
      std::min(16, tsl::port::MaxParallelism()));
  global_pool->Schedule(std::move(f));
}

char LoadedExecutable::ID = 0;  // NOLINT

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
