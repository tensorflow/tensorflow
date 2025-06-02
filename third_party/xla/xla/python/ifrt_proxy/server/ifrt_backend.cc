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
#include <deque>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/bind_front.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "llvm/Support/Casting.h"
#include "xla/layout.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/array_spec.h"
#include "xla/python/ifrt/basic_device_list.h"
#include "xla/python/ifrt/compiler.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
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
#include "xla/python/ifrt/value.h"
#include "xla/python/ifrt_proxy/common/array_util.h"
#include "xla/python/ifrt_proxy/common/ifrt_service.pb.h"
#include "xla/python/ifrt_proxy/common/prof_util.h"
#include "xla/python/ifrt_proxy/common/proto_util.h"
#include "xla/python/ifrt_proxy/common/types.h"
#include "xla/python/ifrt_proxy/common/types.pb.h"
#include "xla/python/ifrt_proxy/common/versions.h"
#include "xla/python/ifrt_proxy/server/host_buffer.h"
#include "xla/python/ifrt_proxy/server/host_callback.h"
#include "xla/python/ifrt_proxy/server/version.h"
#include "xla/python/pjrt_ifrt/xla_compiler.h"
#include "xla/status_macros.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status_to_from_proto.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace ifrt {
namespace proxy {
namespace {

using IfrtArrayRef = xla::ifrt::ArrayRef;

absl::StatusOr<IfrtArrayRef> MakeStringArrayFromHostBuffer(
    Client* client, std::shared_ptr<const std::string> host_buffer, DType dtype,
    Shape shape, std::optional<absl::Span<const int64_t>> byte_strides,
    ShardingRef sharding) {
  TF_ASSIGN_OR_RETURN(std::vector<absl::Cord> string_host_buffer,
                      DeserializeStringHostBufferFromString(*host_buffer));
  const void* data = string_host_buffer.data();

  return client->MakeArrayFromHostBuffer(
      data, dtype, std::move(shape), std::move(byte_strides),
      std::move(sharding),
      xla::ifrt::Client::HostBufferSemantics::kImmutableUntilTransferCompletes,
      /*on_done_with_host_buffer=*/
      [host_buffer = std::move(host_buffer),
       string_host_buffer = std::move(string_host_buffer)]() {});
}

// Parses a `MakeArraysFromHostBufferShardsRequest::ShardIndices` proto to
// `xla::ifrt::Client::MakeArraysFromHostBufferShardsSpec::ShardIndices`.
xla::ifrt::Client::MakeArraysFromHostBufferShardsSpec::ShardIndices
ParseMakeArraysFromHostBufferShardsSpecShardIndicesProto(
    const MakeArraysFromHostBufferShardsRequest::ShardIndices&
        shard_indices_proto) {
  xla::ifrt::Client::MakeArraysFromHostBufferShardsSpec::ShardIndices
      shard_indices;
  shard_indices.reserve(shard_indices_proto.indices_size());
  for (const int shard_index : shard_indices_proto.indices()) {
    shard_indices.push_back(shard_index);
  }
  return shard_indices;
}

// Parses a `MakeArraysFromHostBufferShardsRequest::HostBuffer` proto to
// `xla::ifrt::Client::HostBuffer`. It requires a referenced host buffer handle
// to exist in `host_buffer_store`. Once this function returns, the host buffer
// may be deleted from `host_buffer_store` without affecting the returned
// `xla::ifrt::Client::HostBuffer`.
absl::StatusOr<xla::ifrt::Client::HostBuffer>
ParseMakeArraysFromHostBufferShardsSpecHostBufferProto(
    HostBufferStore* host_buffer_store,
    const MakeArraysFromHostBufferShardsRequest::HostBuffer&
        host_buffer_proto) {
  TF_ASSIGN_OR_RETURN(DType dtype, DType::FromProto(host_buffer_proto.dtype()));
  TF_ASSIGN_OR_RETURN(Shape shape, Shape::FromProto(host_buffer_proto.shape()));
  std::optional<std::vector<int64_t>> byte_strides;
  if (host_buffer_proto.has_byte_strides()) {
    byte_strides = FromByteStridesProto(host_buffer_proto.byte_strides());
  }
  TF_ASSIGN_OR_RETURN(
      std::shared_ptr<const std::string> host_buffer,
      host_buffer_store->Lookup(host_buffer_proto.host_buffer_handle(),
                                /*timeout=*/absl::InfiniteDuration()));
  const void* data;
  std::function<void()> on_done_with_host_buffer;
  if (dtype.kind() == DType::kString) {
    TF_ASSIGN_OR_RETURN(std::vector<absl::Cord> string_host_buffer,
                        DeserializeStringHostBufferFromString(*host_buffer));
    data = string_host_buffer.data();
    on_done_with_host_buffer = [host_buffer = std::move(host_buffer),
                                string_host_buffer =
                                    std::move(string_host_buffer)]() mutable {
      string_host_buffer.clear();
      host_buffer.reset();
    };
  } else {
    TF_ASSIGN_OR_RETURN(const auto mem_region,
                        ArrayMemRegion::FromMinimalMemRegion(
                            *host_buffer, dtype, shape, byte_strides));
    data = mem_region.zeroth_element();
    on_done_with_host_buffer = [host_buffer =
                                    std::move(host_buffer)]() mutable {
      host_buffer.reset();
    };
  }

  return xla::ifrt::Client::HostBuffer{data, dtype, std::move(shape),
                                       std::move(byte_strides),
                                       std::move(on_done_with_host_buffer)};
}

// Returns a string_view that is guaranteed to be valid and constant until this
// process dies.
absl::string_view GetRequestName(const IfrtRequest* req) {
  if (IfrtRequest::descriptor() == nullptr) return "unknown";
  if (req == nullptr) return "unknown";
  auto* field =
      IfrtRequest::descriptor()->FindFieldByNumber(req->request_case());
  if (field == nullptr) return "unknown";
  return field->name();
}

}  // namespace

// Maintains all arrays that will be created in the scope of an incoming
// request. Expected usage is as follows for a hypothetical `FooBarRequest`:
//
// FooBarResponse ProcessFooBarRequest(FooBarRequest req) {
//   ArrayStore::Reservation asr(req.client_generated_handles_if_present());
//   FooBarResponse result = ProcessFooBarRequestInternal(asr, req);
//   return asr.ProcessResponse(result);
// }
//
// FooBarResponse ProcessFooBarRequestInternal(ArrayStore::Reservation& asr,
//                                             FooBarRequest req) {
//   IfrtArrayRef created_arrays;
//   ...
//   FooBarResponse response;
//   response.set_array_handles(asr.Fill(std::move(created_arrays)));
//   return response;
// }
class IfrtBackend::ArrayStore::Reservation {
 public:
  // Creates a Reservation that will use `client_generated_handles` if
  // non-empty, and will generate handles (at a later point) otherwise.
  Reservation(absl::Span<const uint64_t> client_generated_handles,
              IfrtBackend::ArrayStore* parent);

  // Creates a Reservation that will use `client_generated_handle` if
  // non-zero, and will generate a handle (at a later point) otherwise.
  Reservation(uint64_t client_generated_handle,
              IfrtBackend::ArrayStore* parent);

  // Associate the provided `arrays` with the handles maintained by this
  // reservation. Returns the handles.
  //
  // Assumes that `arrays` has the same number of elements as
  // when the Reservation was constructed, if constructed with client-generated
  // handles.
  std::vector<uint64_t> Fill(absl::Span<const IfrtArrayRef> arrays);

  // Convenience wrapper of above method.
  uint64_t Fill(IfrtArrayRef array) {
    return Fill(absl::MakeConstSpan(&array, 1))[0];
  }

  // If the result contains an error, and the reservation is for
  // client-generated handles, associates those handles with the error.
  //
  // Assumes that either `Fill()` has been called and the result is OK, or
  // `Fill()` has not been called and result is an error.
  //
  // Returns the provided `result` as such.
  absl::StatusOr<Response> ProcessResponse(absl::StatusOr<Response> result);

  // Checks that `Fill()` and `ProcessResponse()` have been called as expected.
  ~Reservation() {
    absl::MutexLock l(&mu_);
    CHECK(filled_);
  }

 private:
  IfrtBackend::ArrayStore* const parent_;

  absl::Mutex mu_;
  std::vector<uint64_t> reserved_handles_ ABSL_GUARDED_BY(mu_);
  bool filled_ ABSL_GUARDED_BY(mu_) = false;
};

IfrtBackend::ArrayStore::Reservation::Reservation(
    absl::Span<const uint64_t> client_generated_handles,
    IfrtBackend::ArrayStore* parent)
    : parent_(parent),
      reserved_handles_(client_generated_handles.begin(),
                        client_generated_handles.end()) {
  for (uint64_t h : client_generated_handles) {
    // The CHECK failing is a bug at either the proxy-client or server.
    CHECK(h != 0) << "IFRT proxy client supplied '0' as a handle.";
  }
}

IfrtBackend::ArrayStore::Reservation::Reservation(
    uint64_t client_generated_handle, IfrtBackend::ArrayStore* parent)
    : parent_(parent) {
  if (client_generated_handle != 0) {
    reserved_handles_.push_back(client_generated_handle);
  }
}

absl::StatusOr<IfrtBackend::Response>
IfrtBackend::ArrayStore::Reservation::ProcessResponse(
    absl::StatusOr<Response> result) {
  if (!result.ok()) {
    absl::MutexLock l(&mu_);
    CHECK(!filled_);
    filled_ = true;
    parent_->Insert(reserved_handles_, result.status());
  }
  return result;
}

std::vector<uint64_t> IfrtBackend::ArrayStore::Reservation::Fill(
    absl::Span<const IfrtArrayRef> arrays) {
  absl::MutexLock l(&mu_);
  CHECK(!filled_);
  filled_ = true;

  if (reserved_handles_.empty()) {
    reserved_handles_.resize(arrays.size());
    parent_->handle_generator_->GenerateAtServerBulk(
        absl::MakeSpan(reserved_handles_));
  }

  parent_->Insert(reserved_handles_, arrays);

  return reserved_handles_;
}

struct IfrtBackend::LoadedExecutableWithInfo {
  explicit LoadedExecutableWithInfo(xla::ifrt::LoadedExecutableRef executable_p)
      : executable(std::move(executable_p)) {}

  absl::Mutex mu;
  // `output_spec` captures the output specification from the result of the
  // first `Execute()`, and is used to verify that further `Execute()` calls
  // do not result in a different specification.
  std::optional<std::vector<xla::ifrt::ArraySpec>> output_spec
      ABSL_GUARDED_BY(mu);
  const xla::ifrt::LoadedExecutableRef executable;

  absl::flat_hash_set<int> donatable_indices ABSL_GUARDED_BY(mu);
};

class IfrtBackend::InOrderRequestsProcessor {
  struct Entry {
    std::unique_ptr<IfrtRequest> req;
    Future<Response>::Promise promise;
    XFlowHelper xflow;
  };

 public:
  explicit InOrderRequestsProcessor(IfrtBackend* parent)
      : parent_(parent),
        thread_(tsl::Env::Default()->StartThread(
            tsl::ThreadOptions(), "ifrt_backend_reqs_processor",
            absl::bind_front(&InOrderRequestsProcessor::Loop, this))) {}

  void Shutdown(std::string reason) {
    {
      absl::MutexLock l(&mu_);
      if (shutdown_msg_.has_value()) {
        return;
      }
      shutdown_msg_ = reason;
    }

    LOG(INFO) << "IfrtBackend::InOrderRequestsProcessor being destroyed, "
                 "waiting for currently processed request to finish.";
    thread_.reset();
    std::deque<Entry> should_cancel;

    {
      absl::MutexLock l(&mu_);
      entries_.swap(should_cancel);
    }

    LOG(INFO) << "IfrtBackend::InOrderRequestsProcessor being destroyed, "
                 "cancelling remaining requests.";
    for (auto& entry : should_cancel) {
      entry.promise.Set(absl::CancelledError(reason));
    }
    LOG(INFO) << "IfrtBackend::InOrderRequestsProcessor has been destroyed.";
  }

  Future<Response> Push(std::unique_ptr<IfrtRequest> request) {
    VLOG(3) << "Enqueuing " << request->ShortDebugString();
    auto promise = Future<Response>::CreatePromise();
    Future<Response> result(promise);
    absl::MutexLock l(&mu_);
    if (shutdown_msg_.has_value()) {
      promise.Set(absl::InternalError(absl::StrCat(
          "InOrderRequestsProcessor already stopped: ", *shutdown_msg_)));
      return result;
    }
    Entry entry{/*req=*/std::move(request), /*promise=*/std::move(promise),
                XFlowHelper(GetRequestName(request.get()))};
    entry.xflow.InstantActivity<XFlowHelper::kSend>();
    entries_.push_back(std::move(entry));
    return result;
  }

  ~InOrderRequestsProcessor() {
    Shutdown("InOrderRequestsProcessor is being destroyed");
  }

 private:
  std::optional<Entry> Pop() {
    absl::MutexLock l(&mu_);
    auto cond = [&]() ABSL_SHARED_LOCKS_REQUIRED(mu_) {
      return shutdown_msg_.has_value() || !entries_.empty();
    };
    mu_.Await(absl::Condition(&cond));
    if (shutdown_msg_.has_value()) return std::nullopt;
    auto result = std::move(entries_.front());
    entries_.pop_front();
    return result;
  }

  void Loop() {
    while (auto entry = Pop()) {
      uint64_t op_id = entry->req->request_metadata().op_id();
      VLOG(3) << "Processing " << entry->req->ShortDebugString();
      int request_case = entry->req->request_case();
      auto span = entry->xflow.Span<XFlowHelper::kRecvSend>();
      parent_->ProcessInternal(std::move(entry->req))
          .OnReady([p = std::move(entry->promise),
                    xflow = std::move(entry->xflow), request_case,
                    op_id](absl::StatusOr<Response> r) mutable {
            auto span = xflow.Span<XFlowHelper::kRecv>();
            if (!r.ok()) {
              absl::string_view request_type = "REQUEST_NOT_SET";
              if (request_case != IfrtRequest::RequestCase::REQUEST_NOT_SET) {
                request_type = IfrtRequest::descriptor()
                                   ->FindFieldByNumber(request_case)
                                   ->name();
              }
              LOG(WARNING) << "Responding " << request_type << "(" << op_id
                           << "): " << r.status();
            } else {
              VLOG(3) << "Responding " << op_id << ": "
                      << (*r)->ShortDebugString();
            }
            p.Set(std::move(r));
          });
    }
  }

  absl::Mutex mu_;
  std::deque<Entry> entries_ ABSL_GUARDED_BY(mu_);
  std::optional<std::string> shutdown_msg_ ABSL_GUARDED_BY(mu_);
  IfrtBackend* const parent_;
  std::unique_ptr<tsl::Thread> thread_;
};

IfrtBackend::IfrtBackend(IfrtProxyVersion version, uint64_t session_id,
                         std::shared_ptr<xla::ifrt::Client> ifrt_client,
                         std::shared_ptr<HostBufferStore> host_buffer_store)
    : handle_generator_(this),
      version_(std::move(version)),
      session_id_(session_id),
      client_(std::move(ifrt_client)),
      host_buffer_store_(std::move(host_buffer_store)),
      array_store_(&handle_generator_),
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
          /*num_threads=*/32),
      in_order_requests_processor_(
          std::make_unique<InOrderRequestsProcessor>(this)) {}

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
  // The requests processor may be processing a request that is blocked on a
  // `HostBufferStore::Lookup()`. Shutdown the buffer store first so that any
  // blocked `Lookup`s return with an error.
  host_buffer_store_->Shutdown("IFRT backend has shut down");

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

  // Shutdown the requests processor.
  in_order_requests_processor_->Shutdown("IFRT backend has shut down");

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
  return in_order_requests_processor_->Push(std::move(request));
}

Future<BackendInterface::Response> IfrtBackend::ProcessInternal(
    std::unique_ptr<IfrtRequest> request) {
  std::optional<ArrayStore::Reservation> asr;
  switch (request->request_case()) {
    case IfrtRequest::RequestCase::kInitRequest:
      return Future<Response>(HandleInit(std::move(request)));
    case IfrtRequest::RequestCase::kCheckFutureRequest:
      return HandleCheckFutureRequest(std::move(request));
    case IfrtRequest::RequestCase::kMakeArrayFromHostBufferRequest:
      asr.emplace(request->make_array_from_host_buffer_request().array_handle(),
                  &array_store_);
      return Future<Response>(asr->ProcessResponse(
          HandleMakeArrayFromHostBufferRequest(*asr, std::move(request))));
    case IfrtRequest::RequestCase::kMakeArraysFromHostBufferShardsRequest:
      asr.emplace(request->make_arrays_from_host_buffer_shards_request()
                      .array_handles(),
                  &array_store_);
      return Future<Response>(
          asr->ProcessResponse(HandleMakeArraysFromHostBufferShardsRequest(
              *asr, std::move(request))));
    case IfrtRequest::RequestCase::kMakeErrorArraysRequest:
      asr.emplace(request->make_error_arrays_request().array_handles(),
                  &array_store_);
      return Future<Response>(asr->ProcessResponse(
          HandleMakeErrorArraysRequest(*asr, std::move(request))));
    case IfrtRequest::RequestCase::kAssembleArrayFromSingleDeviceArraysRequest:
      asr.emplace(request->assemble_array_from_single_device_arrays_request()
                      .result_handle(),
                  &array_store_);
      return Future<Response>(
          asr->ProcessResponse(HandleAssembleArrayFromSingleDeviceArraysRequest(
              *asr, std::move(request))));
    case IfrtRequest::RequestCase::kRemapArraysRequest:
      asr.emplace(request->remap_arrays_request().result_handles(),
                  &array_store_);
      return Future<Response>(asr->ProcessResponse(
          HandleRemapArraysRequest(*asr, std::move(request))));
    case IfrtRequest::RequestCase::kCopyToHostBufferRequest:
      return HandleCopyToHostBufferRequest(std::move(request));
    case IfrtRequest::RequestCase::kDisassembleIntoSingleDeviceArraysRequest:
      asr.emplace(request->disassemble_into_single_device_arrays_request()
                      .result_handles(),
                  &array_store_);
      return Future<Response>(
          asr->ProcessResponse(HandleDisassembleIntoSingleDeviceArraysRequest(
              *asr, std::move(request))));
    case IfrtRequest::RequestCase::kCheckValueReadyRequest:
      return Future<Response>(HandleCheckValueReadyRequest(std::move(request)));
    case IfrtRequest::RequestCase::kCopyArraysRequest:
      asr.emplace(request->copy_arrays_request().result_handles(),
                  &array_store_);
      return Future<Response>(asr->ProcessResponse(
          HandleCopyArraysRequest(*asr, std::move(request))));
    case IfrtRequest::RequestCase::kFullyReplicatedShardRequest:
      asr.emplace(request->fully_replicated_shard_request().result_handle(),
                  &array_store_);
      return Future<Response>(asr->ProcessResponse(
          HandleFullyReplicatedShardRequest(*asr, std::move(request))));
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
    case IfrtRequest::RequestCase::kLoadedExecutableExecuteRequest: {
      asr.emplace(
          request->loaded_executable_execute_request().result_array_handle(),
          &array_store_);
      uint64_t client_generated_status_handle =
          request->loaded_executable_execute_request().result_status_handle();
      absl::StatusOr<Response> result =
          HandleLoadedExecutableExecuteRequest(*asr, std::move(request));
      if (client_generated_status_handle != 0) {
        // Populate the handle if not already populated.
        absl::MutexLock l(&futures_mutex_);
        const bool inserted = futures_
                                  .insert({client_generated_status_handle,
                                           Future<>(result.status())})
                                  .second;
        // If `HandleLoadedExecutableExecuteRequest` returned OK, verify that
        // it already has populated status_handle.
        if (result.ok()) {
          CHECK(!inserted);
        }
      }
      return Future<Response>(asr->ProcessResponse(std::move(result)));
    }
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
    case IfrtRequest::RequestCase::kGetDefaultLayoutRequest:
      return Future<Response>(
          HandleGetDefaultLayoutRequest(std::move(request)));
    default:
      LOG(ERROR) << "Got unimplemented request type: "
                 << request->DebugString();
      return Future<Response>(absl::UnimplementedError(absl::StrCat(
          "Got unimplemented request type: ", request->request_case())));
  }
}

IfrtBackend::HandleGenerator::HandleGenerator(IfrtBackend* parent)
    : parent_(parent), current_(kServerGeneratedHandlesMinValue) {}

uint64_t IfrtBackend::HandleGenerator::GenerateAtServer() {
  absl::MutexLock lock(&mu_);
  uint64_t result = current_++;
  CHECK_GE(result, kServerGeneratedHandlesMinValue);
  return result;
}

void IfrtBackend::HandleGenerator::GenerateAtServerBulk(
    absl::Span<uint64_t> result_handles) {
  absl::MutexLock lock(&mu_);
  std::iota(result_handles.begin(), result_handles.end(), current_);
  current_ += result_handles.size();
  CHECK_GE(current_, kServerGeneratedHandlesMinValue);
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

  absl::Span<xla::ifrt::Device* const> all_devices;
  if (version_.protocol_version() < 7) {
    all_devices = client_->devices();
  } else {
    all_devices = client_->GetAllDevices();
  }
  for (auto* device : all_devices) {
    InitResponse::Device* d = init_resp->add_all_devices();
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
    if (version_.protocol_version() <= 3) {
      for (const auto& [name, attr] : device->Attributes().map()) {
        TF_ASSIGN_OR_RETURN(
            (*d->mutable_deprecated_attributes())[name],
            std::visit(
                [&](const auto& attr) { return ToVariantProto(attr.value); },
                attr));
      }
    } else {
      *d->mutable_attributes() = device->Attributes().ToProto();
    }

    if (device->IsAddressable()) {
      init_resp->add_addressable_device_ids(device->Id().value());
    }
  }
  for (auto* device : client_->devices()) {
    init_resp->add_primary_device_ids(device->Id().value());
  }

  absl::flat_hash_map<int, xla::ifrt::Memory*> memories;
  for (auto* device : all_devices) {
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
  *init_resp->mutable_client_attributes() = client_->Attributes().ToProto();

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

Future<BackendInterface::Response> IfrtBackend::HandleCheckValueReadyRequest(
    std::unique_ptr<IfrtRequest> request) {
  std::vector<xla::ifrt::ValueRef> values;
  values.reserve(request->check_value_ready_request().value_handles_size());
  for (const auto& value_handle :
       request->check_value_ready_request().value_handles()) {
    // TODO(b/261991179): IFRT Proxy currently supports Arrays as the only value
    // type, but this may be extended later to other types such as Tuples.
    absl::StatusOr<IfrtArrayRef> array = array_store_.Find(value_handle);
    if (!array.ok()) {
      return Future<Response>(array.status());
    }
    values.push_back(*std::move(array));
  }

  auto ifrt_response_promise =
      Future<BackendInterface::Response>::CreatePromise();
  Future<BackendInterface::Response> ifrt_response_future(
      ifrt_response_promise);

  client_->GetReadyFuture(values).OnReady(
      [op_id = request->request_metadata().op_id(),
       promise = std::move(ifrt_response_promise)](
          absl::Status status) mutable -> void {
        if (!status.ok()) {
          promise.Set(std::move(status));
          return;
        }
        auto ifrt_response = NewIfrtResponse(op_id);
        ifrt_response->mutable_check_value_ready_response();
        promise.Set(std::move(ifrt_response));
      });
  return ifrt_response_future;
}

absl::StatusOr<BackendInterface::Response>
IfrtBackend::HandleMakeArrayFromHostBufferRequest(
    ArrayStore::Reservation& asr, std::unique_ptr<IfrtRequest> request) {
  CHECK(request->has_make_array_from_host_buffer_request());
  auto* make_array_request =
      request->mutable_make_array_from_host_buffer_request();

  TF_ASSIGN_OR_RETURN(
      auto sharding,
      Sharding::FromProto(client_.get(), make_array_request->sharding()));

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
    host_buffer_store_->Delete(host_buffer_handle).IgnoreError();
  };
  TF_ASSIGN_OR_RETURN(
      std::shared_ptr<const std::string> host_buffer,
      host_buffer_store_->Lookup(host_buffer_handle,
                                 /*timeout=*/absl::InfiniteDuration()));
  std::move(cleanup).Invoke();

  IfrtArrayRef array;
  if (dtype.kind() == DType::kString) {
    TF_ASSIGN_OR_RETURN(array,
                        MakeStringArrayFromHostBuffer(
                            client_.get(), std::move(host_buffer), dtype, shape,
                            std::move(byte_strides), std::move(sharding)));
  } else {
    TF_ASSIGN_OR_RETURN(const auto mem_region,
                        ArrayMemRegion::FromMinimalMemRegion(
                            *host_buffer, dtype, shape, byte_strides));
    TF_ASSIGN_OR_RETURN(
        array,
        client_->MakeArrayFromHostBuffer(
            mem_region.zeroth_element(), dtype, std::move(shape),
            std::move(byte_strides), std::move(sharding),
            xla::ifrt::Client::HostBufferSemantics::
                kImmutableUntilTransferCompletes,
            [hold = std::move(host_buffer)]() mutable { hold.reset(); }));
  }

  std::unique_ptr<IfrtResponse> response =
      NewIfrtResponse(request->request_metadata().op_id());
  auto* make_array_resp =
      response->mutable_make_array_from_host_buffer_response();
  make_array_resp->set_array_handle(asr.Fill(std::move(array)));

  return response;
}

absl::StatusOr<BackendInterface::Response>
IfrtBackend::HandleMakeArraysFromHostBufferShardsRequest(
    ArrayStore::Reservation& asr, std::unique_ptr<IfrtRequest> request) {
  CHECK(request->has_make_arrays_from_host_buffer_shards_request());
  auto* make_arrays_request =
      request->mutable_make_arrays_from_host_buffer_shards_request();

  absl::Cleanup cleanup = [&] {
    for (const auto& spec : make_arrays_request->specs()) {
      for (const auto& host_buffer : spec.host_buffers()) {
        host_buffer_store_->Delete(host_buffer.host_buffer_handle())
            .IgnoreError();
      }
    }
  };

  std::vector<xla::ifrt::Client::MakeArraysFromHostBufferShardsSpec> specs;
  specs.reserve(make_arrays_request->specs_size());
  for (const auto& spec_proto : make_arrays_request->specs()) {
    xla::ifrt::Client::MakeArraysFromHostBufferShardsSpec::Buffers buffers;
    buffers.reserve(spec_proto.host_buffers_size());
    for (int buffer_idx = 0; buffer_idx < spec_proto.host_buffers_size();
         ++buffer_idx) {
      xla::ifrt::Client::MakeArraysFromHostBufferShardsSpec::ShardIndices
          shard_indices =
              ParseMakeArraysFromHostBufferShardsSpecShardIndicesProto(
                  spec_proto.addressable_shard_indices(buffer_idx));
      TF_ASSIGN_OR_RETURN(
          xla::ifrt::Client::HostBuffer host_buffer,
          ParseMakeArraysFromHostBufferShardsSpecHostBufferProto(
              host_buffer_store_.get(), spec_proto.host_buffers(buffer_idx)));
      buffers.push_back({std::move(shard_indices), std::move(host_buffer)});
    }
    TF_ASSIGN_OR_RETURN(
        auto array_spec,
        ArraySpec::FromProto(client_.get(), spec_proto.array_spec()));
    specs.push_back({std::move(buffers), std::move(array_spec)});
  }

  std::move(cleanup).Invoke();

  TF_ASSIGN_OR_RETURN(std::vector<xla::ifrt::ArrayRef> arrays,
                      client_->MakeArraysFromHostBufferShards(
                          absl::MakeSpan(specs),
                          xla::ifrt::Client::HostBufferSemantics::
                              kImmutableUntilTransferCompletes,
                          client_->CreateUserContext()));

  std::vector<uint64_t> handles;
  handles.reserve(make_arrays_request->specs_size());
  if (!make_arrays_request->array_handles().empty()) {
    TF_RET_CHECK(make_arrays_request->array_handles_size() ==
                 make_arrays_request->specs_size());
    for (uint64_t handle : make_arrays_request->array_handles()) {
      handles.push_back(handle);
    }
  } else {
    // TODO(b/282757875): Consider merging the handle_generator with the
    // arrays_.
    for (int i = 0; i < make_arrays_request->specs_size(); ++i) {
      handles.push_back(handle_generator_.GenerateAtServer());
    }
  }

  std::unique_ptr<IfrtResponse> response =
      NewIfrtResponse(request->request_metadata().op_id());
  auto* make_arrays_resp =
      response->mutable_make_arrays_from_host_buffer_shards_response();
  make_arrays_resp->mutable_array_handles()->Reserve(arrays.size());
  for (uint64_t handle : asr.Fill(arrays)) {
    make_arrays_resp->add_array_handles(handle);
  }

  return response;
}

absl::StatusOr<BackendInterface::Response>
IfrtBackend::HandleMakeErrorArraysRequest(
    ArrayStore::Reservation& asr, std::unique_ptr<IfrtRequest> request) {
  CHECK(request->has_make_error_arrays_request());
  auto* make_array_request = request->mutable_make_error_arrays_request();

  const absl::Status error = tsl::StatusFromProto(make_array_request->error());

  std::vector<xla::ifrt::ArraySpec> array_specs;
  array_specs.reserve(make_array_request->array_specs_size());
  for (const auto& array_spec_proto : make_array_request->array_specs()) {
    TF_ASSIGN_OR_RETURN(auto array_spec,
                        ArraySpec::FromProto(client_.get(), array_spec_proto));
    array_specs.push_back(std::move(array_spec));
  }

  TF_ASSIGN_OR_RETURN(std::vector<IfrtArrayRef> arrays,
                      client_->MakeErrorArrays(error, array_specs,
                                               client_->CreateUserContext()));

  std::unique_ptr<IfrtResponse> response =
      NewIfrtResponse(request->request_metadata().op_id());
  auto* make_array_resp = response->mutable_make_error_arrays_response();
  for (uint64_t handle : asr.Fill(arrays)) {
    make_array_resp->add_array_handles(handle);
  }

  return response;
}

absl::StatusOr<BackendInterface::Response>
IfrtBackend::HandleAssembleArrayFromSingleDeviceArraysRequest(
    ArrayStore::Reservation& asr, std::unique_ptr<IfrtRequest> request) {
  const auto& assemble_request =
      request->assemble_array_from_single_device_arrays_request();

  TF_ASSIGN_OR_RETURN(
      std::vector<IfrtArrayRef> arrays,
      array_store_.Find(assemble_request.single_device_array_handles()));

  TF_ASSIGN_OR_RETURN(Shape shape, Shape::FromProto(assemble_request.shape()));
  TF_ASSIGN_OR_RETURN(
      auto sharding,
      Sharding::FromProto(client_.get(), assemble_request.sharding()));
  TF_ASSIGN_OR_RETURN(
      auto array_copy_semantics,
      FromArrayCopySemanticsProto(assemble_request.copy_semantics()));
  SingleDeviceShardSemantics single_device_shard_semantics;
  if (version_.protocol_version() < 8) {
    single_device_shard_semantics = SingleDeviceShardSemantics::kAllShards;
  } else {
    TF_ASSIGN_OR_RETURN(single_device_shard_semantics,
                        FromSingleDeviceShardSemanticsProto(
                            assemble_request.single_device_shard_semantics()));
  }
  IfrtArrayRef array;
  if (version_.protocol_version() <
      protocol_version::kAssembleArrayFromSingleDeviceArraysWithDType) {
    if (arrays.empty()) {
      return absl::InvalidArgumentError(
          "AssembleArrayFromSingleDeviceArrays requires at least one array.");
    }
    TF_ASSIGN_OR_RETURN(array, client_->AssembleArrayFromSingleDeviceArrays(
                                   std::move(shape), std::move(sharding),
                                   absl::MakeSpan(arrays), array_copy_semantics,
                                   single_device_shard_semantics));
  } else {
    TF_ASSIGN_OR_RETURN(DType dtype,
                        DType::FromProto(assemble_request.dtype()));
    TF_ASSIGN_OR_RETURN(array, client_->AssembleArrayFromSingleDeviceArrays(
                                   dtype, std::move(shape), std::move(sharding),
                                   absl::MakeSpan(arrays), array_copy_semantics,
                                   single_device_shard_semantics));
  }

  auto ifrt_resp = NewIfrtResponse(request->request_metadata().op_id());

  ifrt_resp->mutable_assemble_array_from_single_device_arrays_response()
      ->set_array_handle(asr.Fill(std::move(array)));
  return ifrt_resp;
}

absl::StatusOr<BackendInterface::Response>
IfrtBackend::HandleRemapArraysRequest(ArrayStore::Reservation& asr,
                                      std::unique_ptr<IfrtRequest> request) {
  const auto& remap_request = request->remap_arrays_request();

  TF_ASSIGN_OR_RETURN(std::vector<IfrtArrayRef> arrays,
                      array_store_.Find(remap_request.array_handles()));
  TF_ASSIGN_OR_RETURN(RemapPlan plan, RemapPlan::FromProto(
                                          client_.get(), remap_request.plan()));
  TF_ASSIGN_OR_RETURN(auto semantics, FromArrayCopySemanticsProto(
                                          remap_request.copy_semantics()));

  TF_ASSIGN_OR_RETURN(
      auto out_arrays,
      client_->RemapArrays(plan, absl::MakeSpan(arrays), semantics));

  std::vector<uint64_t> response_handles = asr.Fill(out_arrays);

  auto response = NewIfrtResponse(request->request_metadata().op_id());
  response->mutable_remap_arrays_response()->mutable_array_handles()->Assign(
      response_handles.begin(), response_handles.end());

  return response;
}

Future<BackendInterface::Response>
IfrtBackend::HandleCopyToStringHostBufferRequest(
    std::unique_ptr<IfrtRequest> request) {
  const CopyToHostBufferRequest& copy_to_host =
      request->copy_to_host_buffer_request();

  absl::StatusOr<IfrtArrayRef> array =
      array_store_.Find(copy_to_host.array_handle());
  if (!array.ok()) {
    return Future<Response>(array.status());
  }

  if (copy_to_host.has_byte_strides()) {
    return Future<Response>(absl::InvalidArgumentError(
        "Byte strides are not supported for string arrays."));
  }

  // Allocate the host buffer and start the copy.
  auto host_buffer = std::make_unique<std::vector<absl::Cord>>(
      (*array)->shape().num_elements());
  Future<> copy_status = (*array)->CopyToHostBuffer(
      host_buffer->data(), /*byte_strides=*/std::nullopt,
      ArrayCopySemantics::kAlwaysCopy);

  auto resp_promise = Future<BackendInterface::Response>::CreatePromise();
  Future<BackendInterface::Response> resp_future(resp_promise);

  // Make the response proto when the copy is done.
  auto response_maker =
      [this, op_id = request->request_metadata().op_id(),
       host_buffer = std::move(host_buffer),
       host_buffer_handle =
           copy_to_host.host_buffer_handle()](absl::Status status) mutable
      -> absl::StatusOr<std::unique_ptr<IfrtResponse>> {
    TF_RETURN_IF_ERROR(status);

    TF_ASSIGN_OR_RETURN(auto serialized_string_host_buffer,
                        SerializeStringHostBuffer(*host_buffer));
    TF_RETURN_IF_ERROR(host_buffer_store_->Store(
        host_buffer_handle, std::move(*serialized_string_host_buffer)));

    std::unique_ptr<IfrtResponse> response = NewIfrtResponse(op_id);
    response->mutable_copy_to_host_buffer_response();
    return response;
  };
  copy_status.OnReady([promise = std::move(resp_promise),
                       response_maker = std::move(response_maker)](
                          absl::Status status) mutable {
    promise.Set(response_maker(status));
  });

  return resp_future;
}

Future<BackendInterface::Response> IfrtBackend::HandleCopyToHostBufferRequest(
    std::unique_ptr<IfrtRequest> request) {
  const CopyToHostBufferRequest& copy_to_host =
      request->copy_to_host_buffer_request();

  absl::StatusOr<IfrtArrayRef> array =
      array_store_.Find(copy_to_host.array_handle());
  if (!array.ok()) {
    return Future<Response>(array.status());
  }

  if ((*array)->dtype().kind() == DType::kString) {
    return HandleCopyToStringHostBufferRequest(std::move(request));
  }

  const auto byte_strides = [&]() -> std::optional<std::vector<int64_t>> {
    if (!copy_to_host.has_byte_strides()) {
      return std::nullopt;
    }
    return FromByteStridesProto(copy_to_host.byte_strides());
  }();

  // Use `ArrayMemRegion`'s factory functions to determine the size necessary
  // for the host buffer.
  const auto pseudo_mem_region = ArrayMemRegion::FromZerothElementPointer(
      /*zeroth_element=*/nullptr, (*array)->dtype(), (*array)->shape(),
      byte_strides);
  if (!pseudo_mem_region.ok()) {
    return Future<Response>(pseudo_mem_region.status());
  }

  // Use `std::unique_ptr<std::string>` for pointer stability.
  auto host_buffer = std::make_unique<std::string>();
  host_buffer->resize(pseudo_mem_region->nbytes());

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
    ArrayStore::Reservation& asr, std::unique_ptr<IfrtRequest> request) {
  const auto& disassemble_request =
      request->disassemble_into_single_device_arrays_request();
  TF_ASSIGN_OR_RETURN(IfrtArrayRef array,
                      array_store_.Find(disassemble_request.array_handle()));
  SingleDeviceShardSemantics single_device_shard_semantics;
  if (version_.protocol_version() < 8) {
    single_device_shard_semantics = SingleDeviceShardSemantics::kAllShards;
  } else {
    TF_ASSIGN_OR_RETURN(
        single_device_shard_semantics,
        FromSingleDeviceShardSemanticsProto(
            disassemble_request.single_device_shard_semantics()));
  }

  // TODO(b/282757875): Consider other ArrayCopySemantics.
  TF_ASSIGN_OR_RETURN(auto single_device_arrays,
                      array->DisassembleIntoSingleDeviceArrays(
                          xla::ifrt::ArrayCopySemantics::kAlwaysCopy,
                          single_device_shard_semantics));

  std::vector<uint64_t> response_handles =
      asr.Fill(std::move(single_device_arrays));

  auto response = NewIfrtResponse(request->request_metadata().op_id());
  response->mutable_disassemble_into_single_device_arrays_response()
      ->mutable_array_handles()
      ->Assign(response_handles.begin(), response_handles.end());
  return response;
}

absl::StatusOr<BackendInterface::Response> IfrtBackend::HandleCopyArraysRequest(
    ArrayStore::Reservation& asr, std::unique_ptr<IfrtRequest> request) {
  const auto& copy_arrays_request = request->copy_arrays_request();

  TF_ASSIGN_OR_RETURN(std::vector<IfrtArrayRef> arrays,
                      array_store_.Find(copy_arrays_request.array_handles()));

  std::optional<DeviceListRef> devices;
  if (!copy_arrays_request.device_ids().empty()) {
    BasicDeviceList::Devices ds;
    for (const auto& device_id : copy_arrays_request.device_ids()) {
      TF_ASSIGN_OR_RETURN(ds.emplace_back(),
                          client_->LookupDevice(DeviceId(device_id)));
    }
    devices.emplace(BasicDeviceList::Create(std::move(ds)));
  }
  std::optional<MemoryKind> memory_kind;
  if (copy_arrays_request.has_memory_kind()) {
    if (const absl::string_view m = copy_arrays_request.memory_kind();
        !m.empty()) {
      memory_kind.emplace(MemoryKind(m));
    } else {
      memory_kind.emplace(MemoryKind());
    }
  }
  TF_ASSIGN_OR_RETURN(
      auto semantics,
      FromArrayCopySemanticsProto(copy_arrays_request.copy_semantics()));

  TF_ASSIGN_OR_RETURN(
      auto new_arrays,
      client_->CopyArrays(absl::MakeSpan(arrays), std::move(devices),
                          memory_kind, semantics));

  std::vector<uint64_t> response_handles = asr.Fill(std::move(new_arrays));

  std::unique_ptr<IfrtResponse> ifrt_resp =
      NewIfrtResponse(request->request_metadata().op_id());
  ifrt_resp->mutable_copy_arrays_response()->mutable_array_handles()->Assign(
      response_handles.begin(), response_handles.end());
  return ifrt_resp;
}

absl::StatusOr<BackendInterface::Response>
IfrtBackend::HandleFullyReplicatedShardRequest(
    ArrayStore::Reservation& asr, std::unique_ptr<IfrtRequest> request) {
  const auto& fully_replicated_shard_request =
      request->fully_replicated_shard_request();
  TF_ASSIGN_OR_RETURN(
      IfrtArrayRef array,
      array_store_.Find(fully_replicated_shard_request.array_handle()));
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

  auto ifrt_resp = NewIfrtResponse(request->request_metadata().op_id());
  ifrt_resp->mutable_fully_replicated_shard_response()->set_array_handle(
      asr.Fill(std::move(new_array)));
  return ifrt_resp;
}

absl::StatusOr<BackendInterface::Response>
IfrtBackend::HandleDeleteArrayRequest(std::unique_ptr<IfrtRequest> request) {
  std::vector<uint64_t> bad_handles;
  std::vector<Future<>> deletion_futures;

  auto delete_handle = [&](uint64_t handle) {
    absl::StatusOr<IfrtArrayRef> array = array_store_.Find(handle);
    if (array.ok()) {
      deletion_futures.push_back(array.value()->Delete());
    } else {
      deletion_futures.push_back(Future<>(array.status()));
    }
  };

  if (request->delete_array_request().has_array_handle_deprecated()) {
    // TODO(b/296144873): After removing array_handle_deprecated(), move
    // delete_handle's definition to the single place it is used.
    delete_handle(request->delete_array_request().array_handle_deprecated());
  }

  for (auto array_handle : request->delete_array_request().array_handle()) {
    delete_handle(array_handle);
  }

  uint64_t future_handle = handle_generator_.GenerateAtServer();
  {
    absl::MutexLock lock(&futures_mutex_);
    futures_.insert({future_handle, JoinFutures(deletion_futures)});
  }

  auto ifrt_resp = NewIfrtResponse(request->request_metadata().op_id());
  ifrt_resp->mutable_delete_array_response()->set_deletion_future_handle(
      future_handle);
  return ifrt_resp;
}

absl::StatusOr<BackendInterface::Response>
IfrtBackend::HandleIsArrayDeletedRequest(std::unique_ptr<IfrtRequest> request) {
  TF_ASSIGN_OR_RETURN(
      IfrtArrayRef array,
      array_store_.Find(request->is_array_deleted_request().array_handle()));

  auto ifrt_resp = NewIfrtResponse(request->request_metadata().op_id());
  ifrt_resp->mutable_is_array_deleted_response()->set_deleted(
      array->IsDeleted());
  return ifrt_resp;
}

absl::StatusOr<BackendInterface::Response>
IfrtBackend::HandleDestructArrayRequest(std::unique_ptr<IfrtRequest> request) {
  std::vector<uint64_t> missing_handles = array_store_.EraseAndReturnMissing(
      request->destruct_array_request().array_handle());

  if (request->destruct_array_request().has_array_handle_deprecated()) {
    auto missing = array_store_.EraseAndReturnMissing(
        {request->destruct_array_request().array_handle_deprecated()});
    if (!missing.empty()) {
      CHECK_EQ(missing.size(), 1);
      missing_handles.push_back(missing[0]);
    }
  }

  if (!missing_handles.empty()) {
    return absl::NotFoundError(absl::StrCat(
        "Unknown array handle(s): ", absl::StrJoin(missing_handles, ",")));
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
        std::make_unique<DeserializeProgramOptions>(client_.get());

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

    if (auto xla_options =
            llvm::dyn_cast<xla::ifrt::XlaCompileOptions>(options.get())) {
      // TODO(emilyaf): Devices should be plumbed through or serialized to
      // support MPMD parallelism, which allows executables with empty device
      // assignments. In the meantime, devices are obtained from the device
      // assignment in compile_options.
      TF_ASSIGN_OR_RETURN(xla_options->devices,
                          xla::ifrt::GetDeviceListFromXlaCompileOptions(
                              client_.get(), xla_options->compile_options));
    }

    TF_ASSIGN_OR_RETURN(auto executable,
                        client_->GetDefaultCompiler()->CompileAndLoad(
                            std::move(program), std::move(options)));

    std::unique_ptr<IfrtResponse> ifrt_resp =
        NewIfrtResponse(request->request_metadata().op_id());
    auto* compile_resp = ifrt_resp->mutable_compile_response();

    uint64_t handle = handle_generator_.GenerateAtServer();
    compile_resp->set_loaded_executable_handle(handle);

    std::vector<uint64_t> host_callback_handles(host_callback_queues.size());
    handle_generator_.GenerateAtServerBulk(
        absl::MakeSpan(host_callback_handles));
    compile_resp->mutable_loaded_host_callback_handles()->Add(
        host_callback_handles.begin(), host_callback_handles.end());

    // Populate executable metadata.
    compile_resp->set_name(AsProtoStringData(executable->name()));
    compile_resp->set_num_devices(executable->num_devices());
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
      compile_resp->set_ready_future_handle(
          handle_generator_.GenerateAtServer());
      futures_.insert(
          {compile_resp->ready_future_handle(), executable->GetReadyFuture()});
    }

    {
      absl::MutexLock lock(&executables_mutex_);
      executables_.insert({handle, std::make_shared<LoadedExecutableWithInfo>(
                                       std::move(executable))});
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
    TF_ASSIGN_OR_RETURN(
        std::shared_ptr<LoadedExecutableWithInfo> executable_info,
        GetLoadedExecutable(handle));
    LoadedExecutable* executable = executable_info->executable.get();

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
      for (const std::shared_ptr<const xla::PjRtLayout>& parameter_layout :
           *parameter_layouts) {
        // TODO(b/329165105): use PjRtLayout::Serialize instead
        layouts->Add(parameter_layout->xla_layout().ToProto());
      }
    } else {
      *metadata_resp->mutable_parameter_layouts_error() =
          tsl::StatusToProto(parameter_layouts.status());
    }
    if (auto output_layouts = executable->GetOutputLayouts();
        output_layouts.ok()) {
      auto* const layouts =
          metadata_resp->mutable_output_layouts_list()->mutable_layouts();
      for (const std::shared_ptr<const xla::PjRtLayout>& output_layout :
           *output_layouts) {
        // TODO(b/329165105): use PjRtLayout::Serialize instead
        layouts->Add(output_layout->xla_layout().ToProto());
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

    auto donated_input_indices = executable->GetDonatableInputIndices();
    if (donated_input_indices.ok()) {
      metadata_resp->mutable_donated_input_indices()
          ->mutable_donated_input_indices()
          ->Add(donated_input_indices->begin(), donated_input_indices->end());
    } else {
      *metadata_resp->mutable_donated_input_indices_error() =
          tsl::StatusToProto(donated_input_indices.status());
    }

    return ifrt_resp;
  });
}

absl::StatusOr<BackendInterface::Response>
IfrtBackend::HandleLoadedExecutableExecuteRequest(
    ArrayStore::Reservation& asr, std::unique_ptr<IfrtRequest> request) {
  const LoadedExecutableExecuteRequest& execute =
      request->loaded_executable_execute_request();
  TF_ASSIGN_OR_RETURN(std::shared_ptr<LoadedExecutableWithInfo> executable_info,
                      GetLoadedExecutable(execute.loaded_executable_handle()));

  TF_ASSIGN_OR_RETURN(std::vector<IfrtArrayRef> args,
                      array_store_.Find(execute.args_handles()));

  TF_ASSIGN_OR_RETURN(auto execute_options,
                      xla::ifrt::LoadedExecutable::ExecuteOptions::FromProto(
                          execute.execute_options()));
  // Force the old behavior where `fill_status` was implicitly true before
  // protocol version 6. Can be cleaned up once version 6 is outside the
  // compatibility window.
  if (version_.protocol_version() < 6) {
    execute_options.fill_status = true;
  }

  if (execute.result_status_handle() != 0) {
    TF_RET_CHECK(execute_options.fill_status);
  }

  std::optional<DeviceListRef> devices;
  if (!execute.device_ids().empty()) {
    BasicDeviceList::Devices d;
    d.reserve(execute.device_ids_size());
    for (const int32_t device_id : execute.device_ids()) {
      TF_ASSIGN_OR_RETURN(d.emplace_back(),
                          client_->LookupDevice(DeviceId(device_id)));
    }
    devices = BasicDeviceList::Create(std::move(d));
  }

  TF_ASSIGN_OR_RETURN(xla::ifrt::LoadedExecutable::ExecuteResult result,
                      executable_info->executable->Execute(
                          absl::MakeSpan(args), execute_options, devices));

  // The proxy client expects (and the IFRT API implicitly guarantees) that
  // output specs of a `LoadedExecutable` remains constant across `Execute()`
  // calls. Verify that this expectation is satisfied.
  {
    absl::MutexLock l(&executable_info->mu);
    if (executable_info->output_spec.has_value()) {
      CHECK_EQ(result.outputs.size(), executable_info->output_spec->size())
          << "LoadedExecutable::Execute returned different number of outputs "
          << "across invocations";
      for (int i = 0; i < result.outputs.size(); ++i) {
        CHECK_EQ(result.outputs[i]->dtype(),
                 (*executable_info->output_spec)[i].dtype)
            << "LoadedExecutable::Execute output " << i
            << "mismatched dtype across invocations";
        CHECK_EQ(result.outputs[i]->shape(),
                 (*executable_info->output_spec)[i].shape)
            << "LoadedExecutable::Execute output " << i
            << "mismatched shape across invocations";
      }

      // Check that only donatable arguments were deleted. The following assumes
      // that there was no other concurrent operation issued that would delete
      // the array. As of March 2025, the proxy-server issues operations in
      // sequence, so this assumption is satisfied.
      for (int i = 0; i < args.size(); ++i) {
        if (execute_options.non_donatable_input_indices.contains(i) ||
            !executable_info->donatable_indices.contains(i)) {
          CHECK(!args[i]->IsDeleted());
        }
      }
    } else {
      // First `Execute()` call.
      executable_info->output_spec.emplace();
      executable_info->output_spec->reserve(result.outputs.size());
      for (const auto& output : result.outputs) {
        executable_info->output_spec->push_back(
            ArraySpec{/*dtype=*/output->dtype(), /*shape=*/output->shape(),
                      /*sharding=*/output->shared_ptr_sharding()});
      }
      executable_info->donatable_indices = [&] {
        absl::flat_hash_set<int> result;
        absl::StatusOr<absl::Span<const int>> donatable_input_indices =
            executable_info->executable->GetDonatableInputIndices();
        if (donatable_input_indices.ok()) {
          result.insert(donatable_input_indices->begin(),
                        donatable_input_indices->end());
        }
        return result;
      }();
    }
  }

  // Generate sharding protos if we will be returning them to the client in the
  // response.
  std::vector<ShardingProto> output_sharding_protos;
  if (execute.result_array_handle().empty()) {
    output_sharding_protos.reserve(result.outputs.size());
    for (int i = 0; i < result.outputs.size(); ++i) {
      TF_ASSIGN_OR_RETURN(output_sharding_protos.emplace_back(),
                          result.outputs[i]->sharding().ToProto());
    }
  }

  auto ifrt_resp = NewIfrtResponse(request->request_metadata().op_id());
  LoadedExecutableExecuteResponse* execute_response =
      ifrt_resp->mutable_loaded_executable_execute_response();

  // The code below here cannot deal with error returns: it mints handles or
  // assigns values to client-minted handles, and assumes this is done
  // atomically (as in ACID) across all handles.
  [&]() -> void {
    if (execute_options.fill_status) {
      // Caller is expected to call `CheckFuture` exactly once to check for its
      // status and erase it.
      absl::MutexLock lock(&futures_mutex_);
      uint64_t status_handle = execute.result_status_handle();
      if (status_handle == 0) {
        status_handle = handle_generator_.GenerateAtServer();
      }
      execute_response->set_status_handle(status_handle);
      futures_.insert({status_handle, std::move(result.status)});
    }

    std::vector<uint64_t> result_handles = asr.Fill(result.outputs);

    // Fill the output spec and handles in the response if the client did not
    // supply handles.
    if (execute.result_array_handle().empty()) {
      for (int i = 0; i < result.outputs.size(); ++i) {
        LoadedExecutableExecuteResponse::Output* output =
            execute_response->add_outputs();
        *output->mutable_dtype() = result.outputs[i]->dtype().ToProto();
        *output->mutable_shape() = result.outputs[i]->shape().ToProto();
        *output->mutable_sharding() = std::move(output_sharding_protos[i]);
        output->set_array_handle(result_handles[i]);
      }
    }
  }();

  return ifrt_resp;
}

// This handler will be deleted on 2025-06-06 since the underlying IFRT API is
// deprecated. An error is returned until then to gracefully handle old clients.
absl::StatusOr<BackendInterface::Response>
IfrtBackend::HandleLoadedExecutableDeleteRequest(
    std::unique_ptr<IfrtRequest> request) {
  Future<> future(absl::UnimplementedError(
      "LoadedExecutable::Delete is no longer supported"));

  auto ifrt_resp = NewIfrtResponse(request->request_metadata().op_id());
  auto* del_response = ifrt_resp->mutable_loaded_executable_delete_response();

  {
    absl::MutexLock lock(&futures_mutex_);
    del_response->set_future_handle(handle_generator_.GenerateAtServer());
    futures_.insert({del_response->future_handle(), std::move(future)});
  }

  return ifrt_resp;
}

// This handler will be deleted on 2025-06-06 since the underlying IFRT API is
// deprecated. false is returned until then to gracefully handle old clients.
absl::StatusOr<BackendInterface::Response>
IfrtBackend::HandleLoadedExecutableIsDeletedRequest(
    std::unique_ptr<IfrtRequest> request) {
  auto ifrt_resp = NewIfrtResponse(request->request_metadata().op_id());
  auto* is_deleted_response =
      ifrt_resp->mutable_loaded_executable_is_deleted_response();
  is_deleted_response->set_is_deleted(false);

  return ifrt_resp;
}

absl::StatusOr<BackendInterface::Response>
IfrtBackend::HandleLoadedExecutableDestructRequest(
    std::unique_ptr<IfrtRequest> request) {
  const auto& destruct = request->loaded_executable_destruct_request();

  std::shared_ptr<LoadedExecutableWithInfo> executable;
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

    const uint64_t execution_handle = handle_generator_.GenerateAtServer();
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
        host_buffer_store_->Lookup(ret.result_host_buffer_handle(),
                                   /*timeout=*/absl::InfiniteDuration()));
    absl::Cleanup cleanup = [&] {
      host_buffer_store_->Delete(ret.result_host_buffer_handle()).IgnoreError();
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
  assignment.Serialize(
      ifrt_resp->mutable_get_default_device_assignment_response()
          ->mutable_device_assignment());

  return ifrt_resp;
}

absl::StatusOr<BackendInterface::Response>
IfrtBackend::HandleGetDefaultLayoutRequest(
    std::unique_ptr<IfrtRequest> request) {
  const auto& get_default_layout_request =
      request->get_default_layout_request();
  TF_ASSIGN_OR_RETURN(auto dtype,
                      DType::FromProto(get_default_layout_request.dtype()));
  TF_ASSIGN_OR_RETURN(
      Device* const device,
      client_->LookupDevice(DeviceId(get_default_layout_request.device_id())));
  MemoryKind memory_kind =
      get_default_layout_request.memory_kind().empty()
          ? MemoryKind()
          : MemoryKind(get_default_layout_request.memory_kind());
  TF_ASSIGN_OR_RETURN(
      std::shared_ptr<const xla::PjRtLayout> layout,
      client_->GetDefaultLayout(dtype, get_default_layout_request.dims(),
                                device, memory_kind));

  auto ifrt_resp = NewIfrtResponse(request->request_metadata().op_id());

  *ifrt_resp->mutable_get_default_layout_response()
       ->mutable_serialized_pjrt_layout() = layout->Serialize();

  return ifrt_resp;
}

absl::StatusOr<std::shared_ptr<IfrtBackend::LoadedExecutableWithInfo>>
IfrtBackend::GetLoadedExecutable(uint64_t handle) {
  absl::MutexLock lock(&executables_mutex_);
  auto it = executables_.find(handle);
  if (it == executables_.end()) {
    return absl::NotFoundError(
        absl::StrCat("Unknown loaded executable handle: ", handle));
  }
  return it->second;
}

absl::StatusOr<IfrtArrayRef> IfrtBackend::ArrayStore::Find(uint64_t handle) {
  absl::MutexLock l(&mu_);
  auto it = arrays_.find(handle);
  if (it == arrays_.end()) {
    return absl::NotFoundError(absl::StrCat("Unknown array handle: ", handle));
  }
  return it->second;
}

absl::StatusOr<std::vector<IfrtArrayRef>> IfrtBackend::ArrayStore::Find(
    absl::Span<const uint64_t> handles) {
  std::vector<IfrtArrayRef> result;
  result.reserve(handles.size());
  absl::MutexLock l(&mu_);
  for (const uint64_t h : handles) {
    auto it = arrays_.find(h);
    if (it == arrays_.end()) {
      return absl::NotFoundError(absl::StrCat("Unknown array handle: ", h));
    }
    if (!it->second.ok()) {
      return it->second.status();
    }
    result.push_back(*it->second);
  }
  return result;
}

std::vector<uint64_t> IfrtBackend::ArrayStore::EraseAndReturnMissing(
    absl::Span<const uint64_t> handles) {
  std::vector<uint64_t> missing_handles;
  std::vector<xla::ifrt::ArrayRef> to_destruct;
  {
    absl::MutexLock l(&mu_);
    for (const uint64_t h : handles) {
      auto it = arrays_.find(h);
      if (it == arrays_.end()) {
        missing_handles.push_back(h);
        continue;
      }
      if (it->second.ok()) {
        to_destruct.push_back(*std::move(it->second));
      }
      arrays_.erase(it);
    }
  }
  return missing_handles;
}

void IfrtBackend::ArrayStore::Insert(absl::Span<const uint64_t> handles,
                                     const absl::Status& status) {
  absl::MutexLock l(&mu_);
  for (const uint64_t h : handles) {
    CHECK(arrays_.insert({h, status}).second) << h;
  }
}

void IfrtBackend::ArrayStore::Insert(
    absl::Span<const uint64_t> handles,
    absl::Span<const xla::ifrt::ArrayRef> arrays) {
  CHECK_EQ(handles.size(), arrays.size());
  absl::MutexLock l(&mu_);
  for (int i = 0; i < handles.size(); ++i) {
    CHECK(arrays_.insert({handles[i], arrays[i]}).second) << handles[i];
  }
}

IfrtBackend::ArrayStore::ArrayStore(HandleGenerator* handle_generator)
    : handle_generator_(handle_generator) {}

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
