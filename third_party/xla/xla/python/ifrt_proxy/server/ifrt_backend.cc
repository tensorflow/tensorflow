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

#include "xla/python/ifrt_proxy/server/ifrt_backend.h"

#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/bind_front.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "llvm/Support/Casting.h"
#include "xla/layout.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/compiler.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/future.h"
#include "xla/python/ifrt/host_callback.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/program.h"
#include "xla/python/ifrt/program_serdes.h"
#include "xla/python/ifrt/remap_plan.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt_proxy/common/array_util.h"
#include "xla/python/ifrt_proxy/common/ifrt_service.pb.h"
#include "xla/python/ifrt_proxy/common/proto_util.h"
#include "xla/python/ifrt_proxy/common/types.h"
#include "xla/python/ifrt_proxy/common/types.pb.h"
#include "xla/python/ifrt_proxy/server/host_buffer.h"
#include "xla/python/ifrt_proxy/server/host_callback.h"
#include "xla/python/ifrt_proxy/server/version.h"
#include "xla/python/pjrt_ifrt/xla_compiler.h"
#include "xla/status_macros.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status_to_from_proto.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/threadpool.h"

namespace xla {
namespace ifrt {
namespace proxy {

IfrtBackend::IfrtBackend(IfrtProxyVersion version, uint64_t session_id,
                         std::shared_ptr<xla::ifrt::Client> ifrt_client,
                         std::shared_ptr<HostBufferStore> host_buffer_store)
    : version_(std::move(version)),
      session_id_(session_id),
      client_(std::move(ifrt_client)),
      host_buffer_store_(std::move(host_buffer_store)),
      compile_thread_pool_(
          tsl::Env::Default(),
          []() {
            tsl::ThreadOptions options;
            // Use a larger stack size since XLA often requires larger stacks
            // for compilation.
            options.stack_size = 240 * 1024;
            return options;
          }(),
          "IfrtBackend",
          // TODO(b/282757875): Consider making this configurable.
          /*num_threads=*/32) {}

absl::StatusOr<std::unique_ptr<IfrtBackend>> IfrtBackend::Create(
    IfrtProxyVersion version, uint64_t session_id,
    std::shared_ptr<xla::ifrt::Client> ifrt_client,
    std::shared_ptr<HostBufferStore> host_buffer_store) {
  if (ifrt_client == nullptr) {
    return absl::InvalidArgumentError("ifrt_client cannot be a nullptr.");
  }
  if (version.protocol_version() < kServerMinVersion ||
      version.protocol_version() > kServerMaxVersion) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Protocol version ", version.protocol_version(),
        " is unsupported by IFRT Proxy server; supported versions: [",
        kServerMinVersion, ",", kServerMaxVersion, "]"));
  }
  return absl::WrapUnique<IfrtBackend>(
      new IfrtBackend(std::move(version), session_id, std::move(ifrt_client),
                      std::move(host_buffer_store)));
}

IfrtBackend::~IfrtBackend() {
  // Cancel all in-flight host callback executions.
  {
    absl::MutexLock lock(&host_callback_queues_mutex_);
    for (const auto& [key, queue] : host_callback_queues_) {
      queue->Close();
    }
  }
  absl::flat_hash_map<uint64_t, RemoteLoadedHostCallbackQueue::ExecutionRequest>
      host_callback_executions;
  {
    absl::MutexLock lock(&host_callback_executions_mutex_);
    host_callback_executions.swap(host_callback_executions_);
  }
  for (auto& [handle, execution_request] : host_callback_executions) {
    std::move(execution_request)
        .status.Set(absl::CancelledError("IFRT backend has shut down"));
  }

  // Wait until all async work from `AsyncExecute` finishes execution.
  {
    auto done = [this]() ABSL_SHARED_LOCKS_REQUIRED(in_flight_count_mutex_) {
      return in_flight_count_ == 0;
    };
    absl::MutexLock lock(&in_flight_count_mutex_, absl::Condition(&done));
  }
}

Future<BackendInterface::Response> IfrtBackend::Process(
    std::unique_ptr<IfrtRequest> request) {
  switch (request->request_case()) {
    case IfrtRequest::RequestCase::kInitRequest:
      return Future<Response>(HandleInit(std::move(request)));
    case IfrtRequest::RequestCase::kCheckFutureRequest:
      return HandleCheckFutureRequest(std::move(request));
    case IfrtRequest::RequestCase::kMakeArrayFromHostBufferRequest:
      return Future<Response>(
          HandleMakeArrayFromHostBufferRequest(std::move(request)));
    case IfrtRequest::RequestCase::kAssembleArrayFromSingleDeviceArraysRequest:
      return Future<Response>(
          HandleAssembleArrayFromSingleDeviceArraysRequest(std::move(request)));
    case IfrtRequest::RequestCase::kRemapArraysRequest:
      return Future<Response>(HandleRemapArraysRequest(std::move(request)));
    case IfrtRequest::RequestCase::kCopyToHostBufferRequest:
      return HandleCopyToHostBufferRequest(std::move(request));
    case IfrtRequest::RequestCase::kDisassembleIntoSingleDeviceArraysRequest:
      return Future<Response>(
          HandleDisassembleIntoSingleDeviceArraysRequest(std::move(request)));
    case IfrtRequest::RequestCase::kCheckArrayReadyRequest:
      return Future<Response>(HandleCheckArrayReadyRequest(std::move(request)));
    case IfrtRequest::RequestCase::kReshardRequest:
      return Future<Response>(HandleReshardRequest(std::move(request)));
    case IfrtRequest::RequestCase::kFullyReplicatedShardRequest:
      return Future<Response>(
          HandleFullyReplicatedShardRequest(std::move(request)));
    case IfrtRequest::RequestCase::kDeleteArrayRequest:
      return Future<Response>(HandleDeleteArrayRequest(std::move(request)));
    case IfrtRequest::RequestCase::kIsArrayDeletedRequest:
      return Future<Response>(HandleIsArrayDeletedRequest(std::move(request)));
    case IfrtRequest::RequestCase::kDestructArrayRequest:
      return Future<Response>(HandleDestructArrayRequest(std::move(request)));
    case IfrtRequest::RequestCase::kCompileRequest:
      return Future<Response>(HandleCompileRequest(std::move(request)));
    case IfrtRequest::RequestCase::kLoadedExecutableMetadataRequest:
      return HandleLoadedExecutableMetadataRequest(std::move(request));
    case IfrtRequest::RequestCase::kLoadedExecutableExecuteRequest:
      return Future<Response>(
          HandleLoadedExecutableExecuteRequest(std::move(request)));
    case IfrtRequest::RequestCase::kLoadedExecutableDeleteRequest:
      return Future<Response>(
          HandleLoadedExecutableDeleteRequest(std::move(request)));
    case IfrtRequest::RequestCase::kLoadedExecutableIsDeletedRequest:
      return Future<Response>(
          HandleLoadedExecutableIsDeletedRequest(std::move(request)));
    case IfrtRequest::RequestCase::kLoadedExecutableDestructRequest:
      return Future<Response>(
          HandleLoadedExecutableDestructRequest(std::move(request)));
    case IfrtRequest::RequestCase::kLoadedHostCallbackPollRequest:
      return HandleLoadedHostCallbackPollRequest(std::move(request));
    case IfrtRequest::RequestCase::kLoadedHostCallbackReturnRequest:
      return Future<Response>(
          HandleLoadedHostCallbackReturnRequest(std::move(request)));
    case IfrtRequest::RequestCase::kGetDefaultDeviceAssignmentRequest:
      return Future<Response>(
          HandleGetDefaultDeviceAssignmentRequest(std::move(request)));
    default:
      return Future<Response>(absl::UnimplementedError(absl::StrCat(
          "Got unimplemented request type: ", request->request_case())));
  }
}

uint64_t IfrtBackend::HandleGenerator::New() {
  absl::MutexLock lock(&mu_);
  return current_++;
}

void IfrtBackend::HandleGenerator::BulkNew(absl::Span<uint64_t> handles) {
  absl::MutexLock lock(&mu_);
  std::iota(handles.begin(), handles.end(), current_);
  current_ += handles.size();
}

Future<BackendInterface::Response> IfrtBackend::AsyncExecute(
    std::function<absl::StatusOr<Response>()> handle_fn,
    tsl::thread::ThreadPool* thread_pool) {
  {
    absl::MutexLock lock(&in_flight_count_mutex_);
    ++in_flight_count_;
  }
  auto promise = Future<Response>::CreatePromise();
  auto f = [this, promise, handle_fn = std::move(handle_fn)]() mutable {
    promise.Set(handle_fn());
    {
      absl::MutexLock lock(&in_flight_count_mutex_);
      --in_flight_count_;
    }
  };
  if (thread_pool != nullptr) {
    thread_pool->Schedule(std::move(f));
  } else {
    tsl::Env::Default()->SchedClosure(std::move(f));
  }
  return Future<Response>(std::move(promise));
}

/////////////////////////////////////////////////////////////////////////////
//
// Handlers for individual request types
//

absl::StatusOr<BackendInterface::Response> IfrtBackend::HandleInit(
    std::unique_ptr<IfrtRequest> request) {
  std::unique_ptr<IfrtResponse> response =
      NewIfrtResponse(request->request_metadata().op_id());
  auto* init_resp = response->mutable_init_response();
  init_resp->set_session_id(session_id_);
  init_resp->set_platform_name(AsProtoStringData(client_->platform_name()));
  init_resp->set_platform_version(
      AsProtoStringData(client_->platform_version()));
  init_resp->set_platform_id(client_->platform_id());
  init_resp->set_runtime_type(AsProtoStringData(client_->runtime_type()));
  init_resp->set_process_index(client_->process_index());

  for (auto* device : client_->devices()) {
    InitResponse::Device* d = init_resp->add_devices();
    d->set_id(device->Id().value());
    d->set_device_kind(AsProtoStringData(device->Kind()));
    if (auto default_memory = device->DefaultMemory(); default_memory.ok()) {
      d->set_default_memory_id((*default_memory)->Id().value());
    }
    for (const auto* memory : device->Memories()) {
      d->add_memory_ids(memory->Id().value());
    }
    d->set_debug_string(AsProtoStringData(device->DebugString()));
    d->set_to_string(AsProtoStringData(device->ToString()));
    for (const auto& [name, attr] : device->Attributes()) {
      TF_ASSIGN_OR_RETURN((*d->mutable_attributes())[name],
                          ToVariantProto(attr));
    }
  }
  for (auto* addressable_device : client_->addressable_devices()) {
    init_resp->add_addressable_device_ids(addressable_device->Id().value());
  }

  absl::flat_hash_map<int, xla::ifrt::Memory*> memories;
  for (auto* device : client_->devices()) {
    for (xla::ifrt::Memory* memory : device->Memories()) {
      const auto [it, inserted] =
          memories.insert({memory->Id().value(), memory});
      if (!inserted && it->second != memory) {
        return absl::FailedPreconditionError(absl::StrCat(
            "Two memories cannot have the same id: ", memory->ToString(),
            " vs. ", it->second->ToString()));
      }
    }
  }
  for (const auto& [id, memory] : memories) {
    auto* m = init_resp->add_memories();
    m->set_id(id);
    m->set_memory_space_kind(AsProtoStringData(*memory->Kind().memory_kind()));
    for (const auto* device : memory->Devices()) {
      m->add_device_ids(device->Id().value());
    }
    m->set_debug_string(AsProtoStringData(memory->DebugString()));
    m->set_to_string(AsProtoStringData(memory->ToString()));
  }

  return response;
}

Future<BackendInterface::Response> IfrtBackend::HandleCheckFutureRequest(
    std::unique_ptr<IfrtRequest> request) {
  const CheckFutureRequest& check_request = request->check_future_request();

  Future<> future;
  {
    absl::MutexLock lock(&futures_mutex_);
    const auto it = futures_.find(check_request.future_handle());
    if (it == futures_.end()) {
      return Future<Response>(absl::NotFoundError(absl::StrCat(
          "Unknown future handle: ", check_request.future_handle())));
    }
    future = std::move(it->second);
    futures_.erase(it);
  }

  auto promise = Future<BackendInterface::Response>::CreatePromise();
  // With PjRtFuture, the `Future` needs to be owned by one or more owners until
  // `OnReady()`'s lambda gets executed. So, capture a copy of `future` in the
  // lambda, making the lambda itself an owner of `future`.
  future.OnReady([op_id = request->request_metadata().op_id(), promise,
                  hold = future](absl::Status status) mutable {
    if (!status.ok()) {
      promise.Set(std::move(status));
      return;
    }
    auto ifrt_resp = NewIfrtResponse(op_id);
    ifrt_resp->mutable_check_future_response();
    promise.Set(std::move(ifrt_resp));
  });

  return Future<BackendInterface::Response>(std::move(promise));
}

absl::StatusOr<BackendInterface::Response>
IfrtBackend::HandleMakeArrayFromHostBufferRequest(
    std::unique_ptr<IfrtRequest> request) {
  if (!request->has_make_array_from_host_buffer_request()) {
    return absl::InternalError(
        "MakeArrayFromHostBuffer got an IfrtRequest with no "
        "MakeArrayFromHostBufferRequest in it.");
  }
  auto* make_array_request =
      request->mutable_make_array_from_host_buffer_request();

  TF_ASSIGN_OR_RETURN(
      auto sharding, Sharding::FromProto(
                         absl::bind_front(&Client::LookupDevice, client_.get()),
                         make_array_request->sharding()));

  const auto byte_strides = [&]() -> std::optional<std::vector<int64_t>> {
    if (!make_array_request->has_byte_strides()) return std::nullopt;
    return FromByteStridesProto(make_array_request->byte_strides());
  }();
  TF_ASSIGN_OR_RETURN(const auto shape,
                      Shape::FromProto(make_array_request->shape()));
  TF_ASSIGN_OR_RETURN(const auto dtype,
                      DType::FromProto(make_array_request->dtype()));

  const uint64_t host_buffer_handle = make_array_request->host_buffer_handle();
  absl::Cleanup cleanup = [&] {
    CHECK_OK(host_buffer_store_->Delete(host_buffer_handle));
  };
  TF_ASSIGN_OR_RETURN(std::shared_ptr<const std::string> host_buffer,
                      host_buffer_store_->Lookup(host_buffer_handle));
  std::move(cleanup).Invoke();

  TF_ASSIGN_OR_RETURN(const auto mem_region,
                      ArrayMemRegion::FromMinimalMemRegion(
                          *host_buffer, dtype, shape, byte_strides));

  TF_ASSIGN_OR_RETURN(
      auto array,
      client_->MakeArrayFromHostBuffer(
          mem_region.zeroth_element(), dtype, std::move(shape),
          std::move(byte_strides), std::move(sharding),
          xla::ifrt::Client::HostBufferSemantics::
              kImmutableUntilTransferCompletes,
          [hold = std::move(host_buffer)]() mutable { hold.reset(); }));

  // TODO(b/282757875): Consider merging the handle_generator with the
  // arrays_.
  uint64_t handle = handle_generator_.New();
  {
    absl::MutexLock lock(&arrays_mutex_);
    arrays_.insert({handle, std::move(array)});
  }

  std::unique_ptr<IfrtResponse> response =
      NewIfrtResponse(request->request_metadata().op_id());
  auto* make_array_resp =
      response->mutable_make_array_from_host_buffer_response();
  make_array_resp->set_array_handle(handle);

  return response;
}

absl::StatusOr<BackendInterface::Response>
IfrtBackend::HandleAssembleArrayFromSingleDeviceArraysRequest(
    std::unique_ptr<IfrtRequest> request) {
  const auto& assemble_request =
      request->assemble_array_from_single_device_arrays_request();

  std::vector<tsl::RCReference<xla::ifrt::Array>> arrays;
  {
    absl::ReaderMutexLock lock(&arrays_mutex_);
    for (const uint64_t handle :
         assemble_request.single_device_array_handles()) {
      TF_ASSIGN_OR_RETURN(arrays.emplace_back(), GetArrayLocked(handle));
    }
  }

  TF_ASSIGN_OR_RETURN(Shape shape, Shape::FromProto(assemble_request.shape()));
  TF_ASSIGN_OR_RETURN(
      auto sharding, Sharding::FromProto(
                         absl::bind_front(&Client::LookupDevice, client_.get()),
                         assemble_request.sharding()));
  TF_ASSIGN_OR_RETURN(auto semantics, FromArrayCopySemanticsProto(
                                          assemble_request.copy_semantics()));

  TF_ASSIGN_OR_RETURN(auto array, client_->AssembleArrayFromSingleDeviceArrays(
                                      std::move(shape), std::move(sharding),
                                      absl::MakeSpan(arrays), semantics));

  auto ifrt_resp = NewIfrtResponse(request->request_metadata().op_id());

  uint64_t handle = handle_generator_.New();
  ifrt_resp->mutable_assemble_array_from_single_device_arrays_response()
      ->set_array_handle(handle);
  {
    absl::MutexLock lock(&arrays_mutex_);
    arrays_.insert({handle, std::move(array)});
  }

  return ifrt_resp;
}

absl::StatusOr<BackendInterface::Response>
IfrtBackend::HandleRemapArraysRequest(std::unique_ptr<IfrtRequest> request) {
  const auto& remap_request = request->remap_arrays_request();

  std::vector<tsl::RCReference<xla::ifrt::Array>> arrays;
  {
    absl::ReaderMutexLock lock(&arrays_mutex_);
    for (const uint64_t handle : remap_request.array_handles()) {
      TF_ASSIGN_OR_RETURN(arrays.emplace_back(), GetArrayLocked(handle));
    }
  }

  TF_ASSIGN_OR_RETURN(
      RemapPlan plan,
      RemapPlan::FromProto(
          absl::bind_front(&Client::LookupDevice, client_.get()),
          remap_request.plan()));
  TF_ASSIGN_OR_RETURN(auto semantics, FromArrayCopySemanticsProto(
                                          remap_request.copy_semantics()));

  TF_ASSIGN_OR_RETURN(
      auto out_arrays,
      client_->RemapArrays(plan, absl::MakeSpan(arrays), semantics));

  // Set up an IfrtResponse with pre-allocated space for the right number of
  // single device array handles.
  int64_t num_arrays = out_arrays.size();
  auto response = NewIfrtResponse(request->request_metadata().op_id());

  // Pre-allocate space in the response proto and fill it in with bulk allocated
  // new handles.
  auto* handles =
      response->mutable_remap_arrays_response()->mutable_array_handles();
  handles->Reserve(num_arrays);
  uint64_t* handles_buf = handles->AddNAlreadyReserved(num_arrays);
  handle_generator_.BulkNew(absl::MakeSpan(handles_buf, num_arrays));

  // Install the newly created arrays into the arrays_.
  {
    absl::MutexLock lock(&arrays_mutex_);
    for (int i = 0; i < num_arrays; ++i) {
      arrays_.insert({handles_buf[i], out_arrays[i]});
    }
  }

  return response;
}

Future<BackendInterface::Response> IfrtBackend::HandleCopyToHostBufferRequest(
    std::unique_ptr<IfrtRequest> request) {
  const CopyToHostBufferRequest& copy_to_host =
      request->copy_to_host_buffer_request();

  auto array = GetArray(copy_to_host.array_handle());
  if (!array.ok()) {
    return Future<Response>(array.status());
  }

  // Determine the size and allocate the host buffer.
  // TODO(b/282757875): We may need to redo this to account for byte_strides,
  // padding, and alignment requirements.
  std::optional<int> element_size = (*array)->dtype().byte_size();
  if (element_size == std::nullopt) {
    return Future<Response>(
        absl::InternalError("Array element size is unknown."));
  }
  int64_t host_buffer_size =
      (*array)->shape().num_elements() * element_size.value();
  // Use `std::unique_ptr<std::string>` for pointer stability.
  auto host_buffer = std::make_unique<std::string>();
  host_buffer->resize(host_buffer_size);

  const auto byte_strides = [&]() -> std::optional<std::vector<int64_t>> {
    if (!copy_to_host.has_byte_strides()) {
      return std::nullopt;
    }
    return FromByteStridesProto(copy_to_host.byte_strides());
  }();
  const auto mem_region = ArrayMemRegion::FromMinimalMemRegion(
      absl::string_view(*host_buffer), (*array)->dtype(), (*array)->shape(),
      byte_strides);
  if (!mem_region.ok()) {
    return Future<Response>(mem_region.status());
  }

  // TODO(b/282757875): Consider other ArrayCopySemantics.
  Future<> copy_status =
      (*array)->CopyToHostBuffer(mem_region->zeroth_element(), byte_strides,
                                 ArrayCopySemantics::kAlwaysCopy);

  auto resp_promise = Future<BackendInterface::Response>::CreatePromise();
  Future<BackendInterface::Response> resp_future(resp_promise);
  auto on_ready = [this, op_id = request->request_metadata().op_id(),
                   host_buffer = std::move(host_buffer),
                   host_buffer_handle = copy_to_host.host_buffer_handle()](
                      absl::Status status) mutable
      -> absl::StatusOr<std::unique_ptr<IfrtResponse>> {
    TF_RETURN_IF_ERROR(status);

    TF_RETURN_IF_ERROR(
        host_buffer_store_->Store(host_buffer_handle, *std::move(host_buffer)));

    std::unique_ptr<IfrtResponse> response = NewIfrtResponse(op_id);
    response->mutable_copy_to_host_buffer_response();
    return response;
  };
  copy_status.OnReady(
      [promise = std::move(resp_promise), on_ready = std::move(on_ready)](
          absl::Status status) mutable { promise.Set(on_ready(status)); });

  return resp_future;
}

absl::StatusOr<BackendInterface::Response>
IfrtBackend::HandleDisassembleIntoSingleDeviceArraysRequest(
    std::unique_ptr<IfrtRequest> request) {
  TF_ASSIGN_OR_RETURN(
      auto array,
      GetArray(request->disassemble_into_single_device_arrays_request()
                   .array_handle()));

  // TODO(b/282757875): Consider other ArrayCopySemantics.
  TF_ASSIGN_OR_RETURN(auto single_device_arrays,
                      array->DisassembleIntoSingleDeviceArrays(
                          xla::ifrt::ArrayCopySemantics::kAlwaysCopy));

  // Set up an IfrtResponse with pre-allocated space for the right number of
  // single device array handles.
  int64_t num_arrays = single_device_arrays.size();
  auto response = NewIfrtResponse(request->request_metadata().op_id());

  // Pre-allocate space in the response proto and fill it in with bulk allocated
  // new handles.
  auto* handles =
      response->mutable_disassemble_into_single_device_arrays_response()
          ->mutable_single_device_array_handles();
  handles->Reserve(num_arrays);
  uint64_t* handles_buf = handles->AddNAlreadyReserved(num_arrays);
  handle_generator_.BulkNew(absl::MakeSpan(handles_buf, num_arrays));

  // Install the newly created arrays into the arrays_.
  {
    absl::MutexLock lock(&arrays_mutex_);
    for (int i = 0; i < num_arrays; ++i) {
      arrays_.insert({handles_buf[i], single_device_arrays[i]});
    }
  }

  return response;
}

Future<BackendInterface::Response> IfrtBackend::HandleCheckArrayReadyRequest(
    std::unique_ptr<IfrtRequest> request) {
  auto array = GetArray(request->check_array_ready_request().array_handle());
  if (!array.ok()) {
    return Future<Response>(array.status());
  }

  auto ifrt_response_promise =
      Future<BackendInterface::Response>::CreatePromise();
  Future<BackendInterface::Response> ifrt_response_future(
      ifrt_response_promise);

  (*array)->GetReadyFuture().OnReady(
      [op_id = request->request_metadata().op_id(),
       promise = std::move(ifrt_response_promise)](
          absl::Status status) mutable -> void {
        if (!status.ok()) {
          promise.Set(std::move(status));
          return;
        }
        auto ifrt_response = NewIfrtResponse(op_id);
        ifrt_response->mutable_check_array_ready_response();
        promise.Set(std::move(ifrt_response));
      });
  return ifrt_response_future;
}

absl::StatusOr<BackendInterface::Response> IfrtBackend::HandleReshardRequest(
    std::unique_ptr<IfrtRequest> request) {
  const auto& reshard_request = request->reshard_request();
  TF_ASSIGN_OR_RETURN(auto array, GetArray(reshard_request.array_handle()));
  TF_ASSIGN_OR_RETURN(
      std::shared_ptr<const Sharding> sharding,
      Sharding::FromProto(
          absl::bind_front(&Client::LookupDevice, client_.get()),
          reshard_request.sharding()));
  TF_ASSIGN_OR_RETURN(auto semantics, FromArrayCopySemanticsProto(
                                          reshard_request.copy_semantics()));

  TF_ASSIGN_OR_RETURN(auto resharded_array,
                      array->Reshard(sharding, semantics));

  uint64_t resharded_array_handle = handle_generator_.New();
  {
    absl::MutexLock lock(&arrays_mutex_);
    arrays_.insert({resharded_array_handle, std::move(resharded_array)});
  }

  auto ifrt_resp = NewIfrtResponse(request->request_metadata().op_id());
  ifrt_resp->mutable_reshard_response()->set_array_handle(
      resharded_array_handle);
  return ifrt_resp;
}

absl::StatusOr<BackendInterface::Response>
IfrtBackend::HandleFullyReplicatedShardRequest(
    std::unique_ptr<IfrtRequest> request) {
  const auto& fully_replicated_shard_request =
      request->fully_replicated_shard_request();
  TF_ASSIGN_OR_RETURN(auto array,
                      GetArray(fully_replicated_shard_request.array_handle()));
  TF_ASSIGN_OR_RETURN(auto semantics,
                      FromArrayCopySemanticsProto(
                          fully_replicated_shard_request.copy_semantics()));

  // Here we are making the assumption that the `FullyReplicatedShard` returns
  // the Array corresponding to the first device in the sharding - as needed by
  // the proxy client for making the SingleDeviceSharding corresponding to the
  // newly created array. Revisit this when IFRT supports: (1) an inexpensive
  // way to derive a SingleDeviceSharding from a fully replicated Array's
  // sharding and (2) A generalized Reshard API that allows the user to request
  // an Array to be made out of a specific single shard.
  TF_ASSIGN_OR_RETURN(auto new_array, array->FullyReplicatedShard(semantics));

  uint64_t new_array_handle = handle_generator_.New();
  {
    absl::MutexLock lock(&arrays_mutex_);
    arrays_.insert({new_array_handle, std::move(new_array)});
  }
  auto ifrt_resp = NewIfrtResponse(request->request_metadata().op_id());
  ifrt_resp->mutable_fully_replicated_shard_response()->set_array_handle(
      new_array_handle);
  return ifrt_resp;
}

absl::StatusOr<BackendInterface::Response>
IfrtBackend::HandleDeleteArrayRequest(std::unique_ptr<IfrtRequest> request) {
  TF_ASSIGN_OR_RETURN(auto array,
                      GetArray(request->delete_array_request().array_handle()));

  auto deletion_future = array->Delete();
  uint64_t future_handle = handle_generator_.New();
  {
    absl::MutexLock lock(&futures_mutex_);
    futures_.insert({future_handle, std::move(deletion_future)});
  }

  auto ifrt_resp = NewIfrtResponse(request->request_metadata().op_id());
  ifrt_resp->mutable_delete_array_response()->set_deletion_future_handle(
      future_handle);
  return ifrt_resp;
}

absl::StatusOr<BackendInterface::Response>
IfrtBackend::HandleIsArrayDeletedRequest(std::unique_ptr<IfrtRequest> request) {
  TF_ASSIGN_OR_RETURN(
      auto array, GetArray(request->is_array_deleted_request().array_handle()));

  auto ifrt_resp = NewIfrtResponse(request->request_metadata().op_id());
  ifrt_resp->mutable_is_array_deleted_response()->set_deleted(
      array->IsDeleted());
  return ifrt_resp;
}

absl::StatusOr<BackendInterface::Response>
IfrtBackend::HandleDestructArrayRequest(std::unique_ptr<IfrtRequest> request) {
  {
    absl::MutexLock lock(&arrays_mutex_);
    bool deleted =
        arrays_.erase(request->destruct_array_request().array_handle());
    if (!deleted) {
      return absl::NotFoundError(
          absl::StrCat("Unknown array handle: ",
                       request->destruct_array_request().array_handle()));
    }
  }
  auto ifrt_resp = NewIfrtResponse(request->request_metadata().op_id());

  // Currently DestructArrayResponse is an empty message, but proxy clients may
  // rely on its presence for correct demuxing.
  ifrt_resp->mutable_destruct_array_response();
  return ifrt_resp;
}

Future<BackendInterface::Response> IfrtBackend::HandleCompileRequest(
    std::unique_ptr<IfrtRequest> request) {
  // Perform compilation on a thread pool in order to (1) avoid blocking the RPC
  // thread during compilation and (2) run compilation with bigger stacks (often
  // necessary for XLA).
  auto f = [this, request = std::shared_ptr<IfrtRequest>(
                      std::move(request))]() -> absl::StatusOr<Response> {
    const CompileRequest& compile_request = request->compile_request();

    auto deserialize_program_options =
        std::make_unique<DeserializeProgramOptions>(
            absl::bind_front(&Client::LookupDevice, client_.get()));
    TF_ASSIGN_OR_RETURN(
        auto program,
        Deserialize<xla::ifrt::Program>(
            compile_request.program(), std::move(deserialize_program_options)));
    TF_ASSIGN_OR_RETURN(auto options, Deserialize<xla::ifrt::CompileOptions>(
                                          compile_request.compile_options(),
                                          /*options=*/nullptr));

    // Deserialize host callbacks. IFRT proxy currently allows only one type of
    // host callbacks from the client (`RemoteLoadedHostCallback`) and this is
    // serialized out of band into its own field in the request proto.
    std::vector<std::shared_ptr<RemoteLoadedHostCallbackQueue>>
        host_callback_queues;
    {
      std::vector<tsl::RCReference<xla::ifrt::LoadedHostCallback>>
          loaded_host_callbacks;
      for (int i = 0; i < compile_request.host_callbacks_size(); ++i) {
        host_callback_queues.emplace_back(
            std::make_shared<RemoteLoadedHostCallbackQueue>());
        TF_ASSIGN_OR_RETURN(
            loaded_host_callbacks.emplace_back(),
            RemoteLoadedHostCallback::CreateFromSerialized(
                client_.get(), compile_request.host_callbacks(i),
                host_callback_queues.back()));
      }
      if (!loaded_host_callbacks.empty()) {
        if (auto xla_options =
                llvm::dyn_cast<xla::ifrt::XlaCompileOptions>(options.get())) {
          xla_options->loaded_host_callbacks = std::move(loaded_host_callbacks);
        } else {
          return absl::UnimplementedError(
              "Host callbacks are supported only for XLA-like IFRT "
              "implementations using `xla::ifrt::XlaCompileOptions`");
        }
      }
    }

    TF_ASSIGN_OR_RETURN(auto executable,
                        client_->GetDefaultCompiler()->Compile(
                            std::move(program), std::move(options)));

    std::unique_ptr<IfrtResponse> ifrt_resp =
        NewIfrtResponse(request->request_metadata().op_id());
    auto* compile_resp = ifrt_resp->mutable_compile_response();

    uint64_t handle = handle_generator_.New();
    compile_resp->set_loaded_executable_handle(handle);

    std::vector<uint64_t> host_callback_handles(host_callback_queues.size());
    handle_generator_.BulkNew(absl::MakeSpan(host_callback_handles));
    compile_resp->mutable_loaded_host_callback_handles()->Add(
        host_callback_handles.begin(), host_callback_handles.end());

    // Populate executable metadata.
    compile_resp->set_name(AsProtoStringData(executable->name()));
    compile_resp->set_num_devices(executable->num_devices());
    for (const auto& logical_device_id :
         executable->addressable_device_logical_ids()) {
      LogicalDeviceIds* proto =
          compile_resp->add_addressable_device_logical_ids();
      proto->set_replica(logical_device_id.replica);
      proto->set_partition(logical_device_id.partition);
    }
    for (const auto* device : executable->addressable_devices()) {
      compile_resp->add_addressable_device_ids(device->Id().value());
    }
    // TODO(b/282757875): Consider making fingerprint calculation asynchronous
    // if it is expected to take long.
    auto fingerprint = executable->Fingerprint();
    if (!fingerprint.ok()) {
      *compile_resp->mutable_fingerprint_error() =
          tsl::StatusToProto(fingerprint.status());
    } else if (fingerprint->has_value()) {
      compile_resp->set_fingerprint_value(std::move(fingerprint)->value());
    }
    // Register the ready future to `futures_`. Caller is expected to call
    // `CheckFuture` exactly once to check for its status and erase it. In
    // future, we may introduce separate mechanisms to remove futures from
    // `futures_` without checking its status for situations where futures are
    // not used.
    {
      absl::MutexLock lock(&futures_mutex_);
      compile_resp->set_ready_future_handle(handle_generator_.New());
      futures_.insert(
          {compile_resp->ready_future_handle(), executable->GetReadyFuture()});
    }

    {
      absl::MutexLock lock(&executables_mutex_);
      executables_.insert({handle, std::move(executable)});
    }
    {
      absl::MutexLock lock(&host_callback_queues_mutex_);
      for (int i = 0; i < host_callback_queues.size(); ++i) {
        host_callback_queues_.insert(
            {host_callback_handles[i], std::move(host_callback_queues[i])});
      }
    }

    return ifrt_resp;
  };
  return AsyncExecute(std::move(f), &compile_thread_pool_);
}

Future<BackendInterface::Response>
IfrtBackend::HandleLoadedExecutableMetadataRequest(
    std::unique_ptr<IfrtRequest> request) {
  // Call `GetParameterShardings` and `GetOutputShardings` on a thread pool
  // since some implementations may block until compilation completes.
  return AsyncExecute([this, request = std::shared_ptr<IfrtRequest>(std::move(
                                 request))]() -> absl::StatusOr<Response> {
    const uint64_t handle = request->loaded_executable_metadata_request()
                                .loaded_executable_handle();
    TF_ASSIGN_OR_RETURN(std::shared_ptr<xla::ifrt::LoadedExecutable> executable,
                        GetLoadedExecutable(handle));

    std::unique_ptr<IfrtResponse> ifrt_resp =
        NewIfrtResponse(request->request_metadata().op_id());
    auto* metadata_resp =
        ifrt_resp->mutable_loaded_executable_metadata_response();

    if (auto parameter_shardings = executable->GetParameterShardings();
        parameter_shardings.has_value()) {
      metadata_resp->mutable_parameter_shardings()->mutable_shardings()->Add(
          parameter_shardings->begin(), parameter_shardings->end());
    }
    if (auto output_shardings = executable->GetOutputShardings();
        output_shardings.has_value()) {
      metadata_resp->mutable_output_shardings()->mutable_shardings()->Add(
          output_shardings->begin(), output_shardings->end());
    }

    if (auto parameter_layouts = executable->GetParameterLayouts();
        parameter_layouts.ok()) {
      auto* const layouts =
          metadata_resp->mutable_parameter_layouts_list()->mutable_layouts();
      for (const std::unique_ptr<xla::PjRtLayout>& parameter_layout :
           *parameter_layouts) {
        // TODO(b/329165105): use PjRtLayout::Serialize instead
        const xla::PjRtXlaLayout* layout =
            dynamic_cast<const xla::PjRtXlaLayout*>(parameter_layout.get());
        TF_RET_CHECK(layout != nullptr)
            << "IFRT proxy only supports PjRtXlaLayout, got a different "
               "subclass";
        layouts->Add(layout->xla_layout().ToProto());
      }
    } else {
      *metadata_resp->mutable_parameter_layouts_error() =
          tsl::StatusToProto(parameter_layouts.status());
    }
    if (auto output_layouts = executable->GetOutputLayouts();
        output_layouts.ok()) {
      auto* const layouts =
          metadata_resp->mutable_output_layouts_list()->mutable_layouts();
      for (const std::unique_ptr<xla::PjRtLayout>& output_layout :
           *output_layouts) {
        // TODO(b/329165105): use PjRtLayout::Serialize instead
        const xla::PjRtXlaLayout* layout =
            dynamic_cast<const xla::PjRtXlaLayout*>(output_layout.get());
        TF_RET_CHECK(layout != nullptr)
            << "IFRT proxy only supports PjRtXlaLayout, got a different "
               "subclass";
        layouts->Add(layout->xla_layout().ToProto());
      }
    } else {
      *metadata_resp->mutable_output_layouts_error() =
          tsl::StatusToProto(output_layouts.status());
    }

    auto output_memory_kinds = executable->GetOutputMemoryKinds();
    if (output_memory_kinds.ok()) {
      for (const auto& memory_kinds : *output_memory_kinds) {
        auto* const list = metadata_resp->mutable_output_memory_kinds()
                               ->add_memory_kind_lists()
                               ->mutable_memory_kinds();
        list->Reserve(memory_kinds.size());
        list->Add(memory_kinds.begin(), memory_kinds.end());
      }
    } else {
      *metadata_resp->mutable_output_memory_kinds()->mutable_status() =
          tsl::StatusToProto(output_memory_kinds.status());
    }

    return ifrt_resp;
  });
}

absl::StatusOr<BackendInterface::Response>
IfrtBackend::HandleLoadedExecutableExecuteRequest(
    std::unique_ptr<IfrtRequest> request) {
  const LoadedExecutableExecuteRequest& execute =
      request->loaded_executable_execute_request();
  TF_ASSIGN_OR_RETURN(std::shared_ptr<xla::ifrt::LoadedExecutable> executable,
                      GetLoadedExecutable(execute.loaded_executable_handle()));

  std::vector<tsl::RCReference<xla::ifrt::Array>> args;
  args.reserve(execute.args_handles_size());
  {
    absl::ReaderMutexLock lock(&arrays_mutex_);
    for (const uint64_t handle : execute.args_handles()) {
      TF_ASSIGN_OR_RETURN(args.emplace_back(), GetArrayLocked(handle));
    }
  }

  TF_ASSIGN_OR_RETURN(auto execute_options,
                      xla::ifrt::LoadedExecutable::ExecuteOptions::FromProto(
                          execute.execute_options()));

  std::optional<DeviceList> devices;
  if (!execute.device_ids().empty()) {
    DeviceList::Devices d;
    d.reserve(execute.device_ids_size());
    for (const int32_t device_id : execute.device_ids()) {
      TF_ASSIGN_OR_RETURN(d.emplace_back(),
                          client_->LookupDevice(DeviceId(device_id)));
    }
    devices = DeviceList(std::move(d));
  }

  TF_ASSIGN_OR_RETURN(
      xla::ifrt::LoadedExecutable::ExecuteResult result,
      executable->Execute(absl::MakeSpan(args), execute_options, devices));

  auto ifrt_resp = NewIfrtResponse(request->request_metadata().op_id());
  LoadedExecutableExecuteResponse* execute_response =
      ifrt_resp->mutable_loaded_executable_execute_response();

  // Register the future to `futures_`. Caller is expected to call
  // `CheckFuture` exactly once to check for its status and erase it. In future,
  // we may introduce separate mechanisms to remove futures from `futures_`
  // without checking its status for situations where futures are not used.
  {
    absl::MutexLock lock(&futures_mutex_);
    execute_response->set_status_handle(handle_generator_.New());
    futures_.insert(
        {execute_response->status_handle(), std::move(result.status)});
  }

  // Register output arrays. At this point, we should never early return because
  // doing so will leak futures or output arrays registered so far.
  std::vector<uint64_t> output_handles(result.outputs.size());
  handle_generator_.BulkNew(absl::MakeSpan(output_handles));
  {
    absl::MutexLock lock(&arrays_mutex_);
    for (int i = 0; i < result.outputs.size(); ++i) {
      tsl::RCReference<xla::ifrt::Array>& array = result.outputs[i];

      LoadedExecutableExecuteResponse::Output* output =
          execute_response->add_outputs();
      *output->mutable_dtype() = array->dtype().ToProto();
      *output->mutable_shape() = array->shape().ToProto();
      TF_ASSIGN_OR_RETURN(*output->mutable_sharding(),
                          array->sharding().ToProto());
      output->set_array_handle(output_handles[i]);

      arrays_.insert({output_handles[i], std::move(array)});
    }
  }

  return ifrt_resp;
}

absl::StatusOr<BackendInterface::Response>
IfrtBackend::HandleLoadedExecutableDeleteRequest(
    std::unique_ptr<IfrtRequest> request) {
  const auto& del = request->loaded_executable_delete_request();
  TF_ASSIGN_OR_RETURN(std::shared_ptr<xla::ifrt::LoadedExecutable> executable,
                      GetLoadedExecutable(del.loaded_executable_handle()));

  Future<> future = executable->Delete();

  auto ifrt_resp = NewIfrtResponse(request->request_metadata().op_id());
  auto* del_response = ifrt_resp->mutable_loaded_executable_delete_response();

  {
    absl::MutexLock lock(&futures_mutex_);
    del_response->set_future_handle(handle_generator_.New());
    futures_.insert({del_response->future_handle(), std::move(future)});
  }

  return ifrt_resp;
}

absl::StatusOr<BackendInterface::Response>
IfrtBackend::HandleLoadedExecutableIsDeletedRequest(
    std::unique_ptr<IfrtRequest> request) {
  const auto& is_deleted = request->loaded_executable_is_deleted_request();
  TF_ASSIGN_OR_RETURN(
      std::shared_ptr<xla::ifrt::LoadedExecutable> executable,
      GetLoadedExecutable(is_deleted.loaded_executable_handle()));

  auto ifrt_resp = NewIfrtResponse(request->request_metadata().op_id());
  auto* is_deleted_response =
      ifrt_resp->mutable_loaded_executable_is_deleted_response();
  is_deleted_response->set_is_deleted(executable->IsDeleted());

  return ifrt_resp;
}

absl::StatusOr<BackendInterface::Response>
IfrtBackend::HandleLoadedExecutableDestructRequest(
    std::unique_ptr<IfrtRequest> request) {
  const auto& destruct = request->loaded_executable_destruct_request();

  std::shared_ptr<xla::ifrt::LoadedExecutable> executable;
  {
    absl::MutexLock lock(&executables_mutex_);
    const auto it = executables_.find(destruct.loaded_executable_handle());
    if (it == executables_.end()) {
      return absl::NotFoundError(
          absl::StrCat("Unknown loaded executable handle: ",
                       destruct.loaded_executable_handle()));
    }
    executable = std::move(it->second);
    executables_.erase(it);
  }
  executable.reset();

  // `RemoteLoadedHostCallback`'s request queue is closed when the host callback
  // objects are destroyed by the underlying IFRT implementation when there are
  // no more host callback executions to be done.

  auto ifrt_resp = NewIfrtResponse(request->request_metadata().op_id());
  ifrt_resp->mutable_loaded_executable_destruct_response();
  return ifrt_resp;
}

Future<BackendInterface::Response>
IfrtBackend::HandleLoadedHostCallbackPollRequest(
    std::unique_ptr<IfrtRequest> request) {
  return AsyncExecute([this, request = std::shared_ptr<IfrtRequest>(std::move(
                                 request))]() -> absl::StatusOr<Response> {
    const auto& poll = request->loaded_host_callback_poll_request();
    const uint64_t handle = poll.loaded_host_callback_handle();

    // Find the host callback queue associated with the given handle.
    std::shared_ptr<RemoteLoadedHostCallbackQueue> queue;
    {
      absl::MutexLock lock(&host_callback_queues_mutex_);
      auto it = host_callback_queues_.find(handle);
      if (it == host_callback_queues_.end()) {
        return absl::NotFoundError(
            absl::StrCat("Unknown loaded host callback handle: ", handle));
      }
      queue = it->second;
    }

    // Block until the host callback has any pending execution and pop its
    // execution info. May return a nullopt if the host callback has been
    // deleted by the underlying IFRT implementation.
    auto execution_request = queue->Pop();
    if (!execution_request.has_value()) {
      {
        absl::MutexLock lock(&host_callback_queues_mutex_);
        host_callback_queues_.erase(handle);
      }
      auto ifrt_resp = NewIfrtResponse(request->request_metadata().op_id());
      ifrt_resp->mutable_loaded_host_callback_poll_response();
      return ifrt_resp;
    }

    // After this point, we must fulfill the promise eventually in order to
    // avoid deadlock (`absl::Cleanup` ensures this).

    absl::Cleanup cleanup = [&] {
      std::move(execution_request)
          ->status.Set(absl::UnknownError(
              "Unable to enqueue the host callback execution"));
    };

    // Store the operands as a single contiguous buffer in the host buffer
    // store. The client retrieves it by invoking `HostBufferLookup`.
    {
      std::string buffer;
      for (const auto& operand : execution_request->operands) {
        buffer.append(static_cast<const char*>(operand.data), operand.size);
      }
      TF_RETURN_IF_ERROR(host_buffer_store_->Store(
          poll.operand_host_buffer_handle(), std::move(buffer)));
    }

    const uint64_t execution_handle = handle_generator_.New();
    {
      absl::MutexLock lock(&host_callback_executions_mutex_);
      host_callback_executions_.insert(
          {execution_handle, *std::move(execution_request)});
    }
    std::move(cleanup).Cancel();

    auto ifrt_resp = NewIfrtResponse(request->request_metadata().op_id());
    auto* poll_response =
        ifrt_resp->mutable_loaded_host_callback_poll_response();
    poll_response->set_host_callback_execution_handle(execution_handle);
    return ifrt_resp;
  });
}

absl::StatusOr<BackendInterface::Response>
IfrtBackend::HandleLoadedHostCallbackReturnRequest(
    std::unique_ptr<IfrtRequest> request) {
  const auto& ret = request->loaded_host_callback_return_request();

  RemoteLoadedHostCallbackQueue::ExecutionRequest execution_request;
  {
    absl::MutexLock lock(&host_callback_executions_mutex_);
    const auto it =
        host_callback_executions_.find(ret.host_callback_execution_handle());
    if (it == host_callback_executions_.end()) {
      return absl::NotFoundError(
          absl::StrCat("Unknown host callback execution: ",
                       ret.host_callback_execution_handle()));
    }
    execution_request = std::move(it->second);
    host_callback_executions_.erase(it);
  }
  absl::Cleanup cleanup = [&] {
    std::move(execution_request)
        .status.Set(absl::UnknownError(
            "Unable to process the host callback execution results"));
  };

  // Copy the results from the host buffer store to the preallocated result
  // buffers from `RemoteLoadedHostCallback`. Must be done before fulfilling the
  // promise since the buffers may not be alive after that.
  absl::Status status;
  if (ret.has_result_host_buffer_handle()) {
    TF_ASSIGN_OR_RETURN(
        std::shared_ptr<const std::string> buffer,
        host_buffer_store_->Lookup(ret.result_host_buffer_handle()));
    absl::Cleanup cleanup = [&] {
      CHECK_OK(host_buffer_store_->Delete(ret.result_host_buffer_handle()));
    };

    int64_t offset = 0;
    for (const auto& result : execution_request.results) {
      if (offset + result.size > buffer->size()) {
        return absl::InternalError(
            absl::StrCat("Buffer overflow while reading host callback "
                         "execution results; ",
                         "range: [", offset, ", ", offset + result.size, "), ",
                         "buffer size: ", buffer->size()));
      }
      std::memcpy(result.data, buffer->data() + offset, result.size);
      offset += result.size;
    }
    if (offset != buffer->size()) {
      return absl::InternalError(
          absl::StrCat("Host callback execution did not consume the entire "
                       "result buffer; size: ",
                       buffer->size(), "; consumed: ", offset));
    }
  } else {
    status = tsl::StatusFromProto(ret.error());
  }

  // Fulfill the result promise. This unblocks the execution of the associated
  // `RemoteLoadedHostCallback`. It is unsafe to access `execution_request`
  // after this since the buffers may not be alive.
  std::move(execution_request).status.Set(std::move(status));
  std::move(cleanup).Cancel();

  auto ifrt_resp = NewIfrtResponse(request->request_metadata().op_id());
  ifrt_resp->mutable_loaded_host_callback_return_response();
  return ifrt_resp;
}

absl::StatusOr<BackendInterface::Response>
IfrtBackend::HandleGetDefaultDeviceAssignmentRequest(
    std::unique_ptr<IfrtRequest> request) {
  const auto& get_default_device_assignment_request =
      request->get_default_device_assignment_request();
  TF_ASSIGN_OR_RETURN(
      auto assignment,
      client_->GetDefaultDeviceAssignment(
          get_default_device_assignment_request.num_replicas(),
          get_default_device_assignment_request.num_partitions()));

  auto ifrt_resp = NewIfrtResponse(request->request_metadata().op_id());

  // Currently, the xla::DeviceAssignment::Serialize does not fail. If test
  // coverage for this error is needed, consider using testing::test_value to
  // inject one.
  TF_RETURN_IF_ERROR(assignment.Serialize(
      ifrt_resp->mutable_get_default_device_assignment_response()
          ->mutable_device_assignment()));

  return ifrt_resp;
}

absl::StatusOr<std::shared_ptr<xla::ifrt::LoadedExecutable>>
IfrtBackend::GetLoadedExecutable(uint64_t handle) {
  absl::MutexLock lock(&executables_mutex_);
  auto it = executables_.find(handle);
  if (it == executables_.end()) {
    return absl::NotFoundError(
        absl::StrCat("Unknown loaded executable handle: ", handle));
  }
  return it->second;
}

absl::StatusOr<tsl::RCReference<xla::ifrt::Array>> IfrtBackend::GetArray(
    uint64_t array_handle) {
  absl::ReaderMutexLock lock(&arrays_mutex_);
  return GetArrayLocked(array_handle);
}

absl::StatusOr<tsl::RCReference<xla::ifrt::Array>> IfrtBackend::GetArrayLocked(
    uint64_t array_handle) {
  auto it = arrays_.find(array_handle);
  if (it == arrays_.end()) {
    return absl::NotFoundError(
        absl::StrCat("Unknown array handle: ", array_handle));
  }
  return it->second;
}

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
