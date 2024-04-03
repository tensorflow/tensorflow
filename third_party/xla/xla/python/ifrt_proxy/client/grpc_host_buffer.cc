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

#include "xla/python/ifrt_proxy/client/grpc_host_buffer.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include "grpcpp/client_context.h"
#include "grpcpp/support/client_callback.h"
#include "grpcpp/support/status.h"
#include "grpcpp/support/sync_stream.h"
#include "xla/pjrt/distributed/util.h"
#include "xla/python/ifrt/future.h"
#include "xla/python/ifrt_proxy/common/grpc_ifrt_service.grpc.pb.h"
#include "xla/python/ifrt_proxy/common/grpc_ifrt_service.pb.h"
#include "tsl/platform/env.h"
#include "tsl/platform/unbounded_work_queue.h"
#include "tsl/protobuf/status.pb.h"

namespace xla {
namespace ifrt {
namespace proxy {

static constexpr int64_t kChunkSize = 1024 * 1024;

GrpcClientHostBufferStore::GrpcClientHostBufferStore(
    std::shared_ptr<grpc::GrpcIfrtService::StubInterface> stub,
    IfrtProxyVersion version, uint64_t session_id)
    : stub_(std::move(stub)),
      version_(std::move(version)),
      session_id_(session_id),
      lookup_work_queue_(std::make_unique<tsl::UnboundedWorkQueue>(
          tsl::Env::Default(), "HostBufferStoreLookupsWorkQueue")) {}

GrpcClientHostBufferStore::~GrpcClientHostBufferStore() {
  LOG(INFO) << "Waiting for destruction of HostBufferStoreLookupsWorkQueue...";
  lookup_work_queue_.reset();
  LOG(INFO) << "Destructed HostBufferStoreLookupsWorkQueue.";
}

uint64_t GrpcClientHostBufferStore::NextHandle() {
  return next_handle_.fetch_add(1, std::memory_order_relaxed);
}

Future<absl::Status> GrpcClientHostBufferStore::Store(uint64_t handle,
                                                      absl::string_view data) {
  // The current implementation synchronously sends host buffer chunks. We may
  // consider making it asynchronous if the caller can leverage such asynchrony.

  GrpcHostBufferStoreMetadata metadata;
  metadata.set_session_id(session_id_);
  metadata.set_handle(handle);
  metadata.set_buffer_size(data.size());

  ::grpc::ClientContext context;
  context.AddMetadata("ifrt-proxy-grpc-host-buffer-store-metadata-bin",
                      metadata.SerializeAsString());

  GrpcHostBufferStoreResponse response;
  auto writer = stub_->HostBufferStore(&context, &response);

  for (int64_t offset = 0; offset < data.size(); offset += kChunkSize) {
    GrpcHostBufferStoreRequest request;
#if defined(PLATFORM_GOOGLE)
    request.set_alias_data(data.substr(offset, kChunkSize));
#else
    // TODO(b/325306748): Find a way to not do a memory-copy.
    request.set_data(std::string(data.substr(offset, kChunkSize)));
#endif
    writer->Write(request);
  }

  if (!writer->WritesDone()) {
    return Future<absl::Status>(
        absl::InternalError("Failed to write all host buffer chunks"));
  }

  return Future<absl::Status>(xla::FromGrpcStatus(writer->Finish()));
}

Future<absl::Status> GrpcClientHostBufferStore::Store(uint64_t handle,
                                                      const absl::Cord& data) {
  // The current implementation synchronously sends host buffer chunks. We may
  // consider making it asynchronous if the caller can leverage such asynchrony.

  GrpcHostBufferStoreMetadata metadata;
  metadata.set_session_id(session_id_);
  metadata.set_handle(handle);
  metadata.set_buffer_size(data.size());

  ::grpc::ClientContext context;
  context.AddMetadata("ifrt-proxy-grpc-host-buffer-store-metadata-bin",
                      metadata.SerializeAsString());

  GrpcHostBufferStoreResponse response;
  auto writer = stub_->HostBufferStore(&context, &response);

  for (absl::string_view chunk : data.Chunks()) {
    for (int64_t offset = 0; offset < chunk.size(); offset += kChunkSize) {
      GrpcHostBufferStoreRequest request;
#if defined(PLATFORM_GOOGLE)
      request.set_alias_data(chunk.substr(offset, kChunkSize));
#else
      // TODO(b/325306748): Find a way to not do a memory-copy.
      request.set_data(std::string(chunk.substr(offset, kChunkSize)));
#endif
      writer->Write(request);
    }
  }
  if (!writer->WritesDone()) {
    return Future<absl::Status>(
        absl::InternalError("Failed to write all host buffer chunks"));
  }

  return Future<absl::Status>(xla::FromGrpcStatus(writer->Finish()));
}

Future<absl::StatusOr<absl::Cord>> GrpcClientHostBufferStore::Lookup(
    uint64_t handle) {
  auto promise = Future<absl::StatusOr<absl::Cord>>::CreatePromise();

  lookup_work_queue_->Schedule([this, handle, promise]() mutable -> void {
    GrpcHostBufferLookupRequest request;
    request.set_handle(handle);
    request.set_session_id(session_id_);

    ::grpc::ClientContext context;

    std::unique_ptr<::grpc::ClientReaderInterface<GrpcHostBufferLookupResponse>>
        stream = stub_->HostBufferLookup(&context, request);

    absl::Cord data;
    GrpcHostBufferLookupResponse response;
    while (stream->Read(&response)) {
      data.Append(response.data());
    }

    absl::Status status = xla::FromGrpcStatus(stream->Finish());
    if (status.ok()) {
      promise.Set(std::move(data));
    } else {
      promise.Set(status);
    }
  });

  return Future<absl::StatusOr<absl::Cord>>(promise);
}

Future<absl::Status> GrpcClientHostBufferStore::Delete(uint64_t handle) {
  GrpcHostBufferDeleteRequest request;
  request.set_session_id(session_id_);
  request.set_handle(handle);

  ::grpc::ClientContext context;
  GrpcHostBufferDeleteResponse response;
  return Future<absl::Status>(xla::FromGrpcStatus(
      stub_->HostBufferDelete(&context, request, &response)));
}

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
