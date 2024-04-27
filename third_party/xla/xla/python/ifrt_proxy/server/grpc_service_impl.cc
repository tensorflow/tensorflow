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

#include "xla/python/ifrt_proxy/server/grpc_service_impl.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/cleanup/cleanup.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/status.h"
#include "grpcpp/support/sync_stream.h"
#include "xla/pjrt/distributed/util.h"
#include "xla/python/ifrt_proxy/common/grpc_ifrt_service.pb.h"
#include "xla/python/ifrt_proxy/common/proto_util.h"
#include "xla/python/ifrt_proxy/server/host_buffer.h"
#include "xla/python/ifrt_proxy/server/ifrt_session_handler.h"
#include "xla/python/ifrt_proxy/server/version.h"

namespace xla {
namespace ifrt {
namespace proxy {

::grpc::Status GrpcServiceImpl::GetVersion(::grpc::ServerContext* context,
                                           const GrpcGetVersionRequest* request,
                                           GrpcGetVersionResponse* response) {
  auto protocol_version =
      ChooseVersion(request->min_version().protocol_version(),
                    request->max_version().protocol_version());
  if (!protocol_version.ok()) {
    return xla::ToGrpcStatus(protocol_version.status());
  }
  response->mutable_version()->set_protocol_version(*protocol_version);
  return ::grpc::Status::OK;
}

::grpc::Status GrpcServiceImpl::IfrtSession(
    ::grpc::ServerContext* context,
    ::grpc::ServerReaderWriter<IfrtResponse, IfrtRequest>* stream) {
  GrpcIfrtSessionMetadata metadata;
  {
    const auto it = context->client_metadata().find(
        "ifrt-proxy-grpc-ifrt-session-metadata-bin");
    if (it == context->client_metadata().end()) {
      return ::grpc::Status(::grpc::StatusCode::INVALID_ARGUMENT,
                            "Missing metadata for GrpcIfrtService.IfrtSession: "
                            "ifrt-proxy-grpc-ifrt-session-metadata-bin");
    }
    if (!metadata.ParseFromString(AsProtoStringData(
            absl::string_view(it->second.data(), it->second.size())))) {
      return ::grpc::Status(::grpc::StatusCode::INVALID_ARGUMENT,
                            "Unable to parse GrpcIfrtSessionMetadata");
    }
  }

  const uint64_t session_id =
      next_session_id_.fetch_add(1, std::memory_order_relaxed);

  VLOG(0) << "Starting a new IFRT session with session_id=" << session_id;

  // Create a host buffer store for the session.
  auto host_buffer_store =
      std::make_shared<xla::ifrt::proxy::HostBufferStore>();
  {
    absl::MutexLock l(&host_buffer_store_mu_);
    CHECK(host_buffer_stores_.insert({session_id, host_buffer_store}).second);
  }
  absl::Cleanup cleanup = [&] {
    absl::MutexLock l(&host_buffer_store_mu_);
    CHECK_GT(host_buffer_stores_.erase(session_id), 0);
  };

  absl::Mutex writer_mu;

  auto session_handler = IfrtSessionHandler::Create(
      session_id,
      [this, version = metadata.version(),
       host_buffer_store = std::move(host_buffer_store)](uint64_t session_id) {
        return backend_factory_(version, session_id, host_buffer_store);
      });

  if (!session_handler.ok()) {
    LOG(INFO) << "Creating session " << session_id
              << " failed: " << session_handler.status();
    return xla::ToGrpcStatus(session_handler.status());
  }

  bool first_request_read = false;
  while (true) {
    auto request = std::make_unique<IfrtRequest>();
    if (!stream->Read(request.get())) {
      break;
    }
    if (!first_request_read) {
      VLOG(0) << "First request read for session " << session_id;
      first_request_read = true;
    }
    (*session_handler)
        ->NewIncomingRequest(std::move(request),
                             [&](std::shared_ptr<IfrtResponse> response) {
                               absl::MutexLock l(&writer_mu);
                               stream->Write(*response);
                             });
  }

  VLOG(0) << "Finishing IFRT session " << session_id;
  return ::grpc::Status::OK;
}

::grpc::Status GrpcServiceImpl::HostBufferStore(
    ::grpc::ServerContext* context,
    ::grpc::ServerReader<GrpcHostBufferStoreRequest>* stream,
    GrpcHostBufferStoreResponse* response) {
  const auto it = context->client_metadata().find(
      "ifrt-proxy-grpc-host-buffer-store-metadata-bin");
  if (it == context->client_metadata().end()) {
    return ::grpc::Status(
        ::grpc::StatusCode::INTERNAL,
        "Missing gRPC metadata for GrpcHostBufferService.Store");
  }

  GrpcHostBufferStoreMetadata metadata;
  if (!metadata.ParseFromString(AsProtoStringData(
          absl::string_view(it->second.data(), it->second.size())))) {
    return ::grpc::Status(::grpc::StatusCode::DATA_LOSS,
                          "Unable to parse GrpcHostBufferStoreMetadata");
  }

  std::string data;
  data.reserve(metadata.buffer_size());

  GrpcHostBufferStoreRequest request;
  while (stream->Read(&request)) {
    data.append(request.data());
  }
  if (data.size() != metadata.buffer_size()) {
    return ::grpc::Status(
        ::grpc::StatusCode::DATA_LOSS,
        absl::StrCat("Potential data loss for host buffers: expected ",
                     metadata.buffer_size(), " bytes but got ", data.size(),
                     " bytes"));
  }

  auto store = GetHostBufferStore(metadata.session_id());
  if (!store.ok()) {
    return xla::ToGrpcStatus(store.status());
  }
  return xla::ToGrpcStatus((*store)->Store(metadata.handle(), std::move(data)));
}

::grpc::Status GrpcServiceImpl::HostBufferLookup(
    ::grpc::ServerContext* context, const GrpcHostBufferLookupRequest* request,
    ::grpc::ServerWriter<GrpcHostBufferLookupResponse>* stream) {
  static constexpr int64_t kChunkSize = 1024 * 1024;

  auto store = GetHostBufferStore(request->session_id());
  if (!store.ok()) {
    return xla::ToGrpcStatus(store.status());
  }
  auto data = (*store)->Lookup(request->handle());
  if (!data.ok()) {
    return xla::ToGrpcStatus(data.status());
  }

  GrpcHostBufferLookupResponse response;
  if (!(*data)->empty()) {
    for (int64_t offset = 0; offset < (*data)->size(); offset += kChunkSize) {
#if defined(PLATFORM_GOOGLE)
      response.set_alias_data(
          absl::string_view(**data).substr(offset, kChunkSize));
#else
      // TODO(b/325306748): Find a way to not do a memory-copy.
      response.set_data((*data)->substr(offset, kChunkSize));
#endif
      stream->Write(response);
      response.Clear();
    }
  } else {
    // Send at least one response even if the buffer is empty.
    stream->Write(response);
  }

  return ::grpc::Status::OK;
}

::grpc::Status GrpcServiceImpl::HostBufferDelete(
    ::grpc::ServerContext* context, const GrpcHostBufferDeleteRequest* request,
    GrpcHostBufferDeleteResponse* response) {
  auto store = GetHostBufferStore(request->session_id());
  if (!store.ok()) {
    return xla::ToGrpcStatus(store.status());
  }
  return xla::ToGrpcStatus((*store)->Delete(request->handle()));
}

bool GrpcServiceImpl::Test_InsertHostBufferStore(
    uint64_t session_id,
    std::shared_ptr<xla::ifrt::proxy::HostBufferStore> store) {
  absl::MutexLock l(&host_buffer_store_mu_);
  return host_buffer_stores_.insert({session_id, std::move(store)}).second;
}

bool GrpcServiceImpl::Test_DeleteHostBufferStore(uint64_t session_id) {
  absl::MutexLock l(&host_buffer_store_mu_);
  return host_buffer_stores_.erase(session_id) > 0;
}

absl::StatusOr<std::shared_ptr<xla::ifrt::proxy::HostBufferStore>>
GrpcServiceImpl::GetHostBufferStore(uint64_t session_id) {
  absl::MutexLock l(&host_buffer_store_mu_);
  const auto it = host_buffer_stores_.find(session_id);
  if (it == host_buffer_stores_.end()) {
    return absl::NotFoundError(
        absl::StrCat("Session id ", session_id, " does not exist"));
  }
  return it->second;
}

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
