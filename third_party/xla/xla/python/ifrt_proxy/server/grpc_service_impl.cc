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
#include "xla/python/ifrt/attribute_map.h"
#include "xla/python/ifrt/serdes_version.h"
#include "xla/python/ifrt_proxy/common/grpc_ifrt_service.pb.h"
#include "xla/python/ifrt_proxy/common/proto_util.h"
#include "xla/python/ifrt_proxy/server/host_buffer.h"
#include "xla/python/ifrt_proxy/server/version.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla {
namespace ifrt {
namespace proxy {

::grpc::Status GrpcServiceImpl::GetVersion(::grpc::ServerContext* context,
                                           const GrpcGetVersionRequest* request,
                                           GrpcGetVersionResponse* response) {
  auto protocol_version =
      ChooseProtocolVersion(request->min_version().protocol_version(),
                            request->max_version().protocol_version());
  if (!protocol_version.ok()) {
    return xla::ToGrpcStatus(protocol_version.status());
  }
  auto ifrt_serdes_version_number = ChooseIfrtSerdesVersionNumber(
      SerDesVersionNumber(request->min_version().ifrt_serdes_version_number()),
      SerDesVersionNumber(request->max_version().ifrt_serdes_version_number()));
  if (!ifrt_serdes_version_number.ok()) {
    return xla::ToGrpcStatus(ifrt_serdes_version_number.status());
  }
  response->mutable_version()->set_protocol_version(*protocol_version);
  response->mutable_version()->set_ifrt_serdes_version_number(
      ifrt_serdes_version_number->value());
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

  LOG(INFO) << "Starting new IFRT session " << session_id << " for peer '"
            << context->peer()
            << "' with metadata=" << metadata.ShortDebugString();

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

  absl::StatusOr<AttributeMap> initialization_data =
      AttributeMap::FromProto(metadata.initialization_data());
  if (!initialization_data.ok()) {
    LOG(INFO) << "Failed to parse initialization data for session "
              << session_id << ": " << initialization_data.status();
    return xla::ToGrpcStatus(initialization_data.status());
  }

  auto backend = backend_factory_(metadata.version(), session_id,
                                  std::move(host_buffer_store),
                                  *std::move(initialization_data));
  if (!backend.ok()) {
    LOG(INFO) << "Creating IFRT backend " << session_id
              << " failed: " << backend.status();
    return xla::ToGrpcStatus(backend.status());
  }

  absl::Mutex writer_mu;
  bool first_request_read = false;
  while (true) {
    auto request = std::make_unique<IfrtRequest>();
    if (!stream->Read(request.get())) {
      break;
    }
    if (!first_request_read) {
      LOG(INFO) << "First request read for session " << session_id;
      first_request_read = true;
    }
    const uint64_t op_id = request->request_metadata().op_id();
    auto response = (*backend)->Process(std::move(request));
    response.OnReady(
        [op_id, stream,
         &writer_mu](absl::StatusOr<std::shared_ptr<IfrtResponse>> response) {
          absl::MutexLock l(&writer_mu);
          if (response.ok()) {
            stream->Write(**response);
          } else {
            stream->Write(*NewIfrtResponse(op_id, response.status()));
          }
        });
  }

  LOG(INFO) << "stream->Read() returned false for session " << session_id
            << "; waiting until all response callbacks are called.";
  backend->reset();  // Blocks until all response callbacks are called.
  LOG(INFO) << "Cleaning up host buffer store for session " << session_id;
  std::move(cleanup).Invoke();
  LOG(INFO) << "Done with IFRT session " << session_id;
  return ::grpc::Status::OK;
}

::grpc::Status GrpcServiceImpl::HostBufferStore(
    ::grpc::ServerContext* context,
    ::grpc::ServerReader<GrpcHostBufferStoreRequest>* stream,
    GrpcHostBufferStoreResponse* response) {
  tsl::profiler::TraceMe traceme("HostBufferStore");
  const auto it = context->client_metadata().find(
      "ifrt-proxy-grpc-host-buffer-store-metadata-bin");
  if (it == context->client_metadata().end()) {
    LOG(WARNING) << "Missing gRPC metadata for GrpcHostBufferService.Store";
    return ::grpc::Status(
        ::grpc::StatusCode::INTERNAL,
        "Missing gRPC metadata for GrpcHostBufferService.Store");
  }

  GrpcHostBufferStoreMetadata metadata;
  if (!metadata.ParseFromString(AsProtoStringData(
          absl::string_view(it->second.data(), it->second.size())))) {
    LOG(WARNING) << "Unable to parse GrpcHostBufferStoreMetadata";
    return ::grpc::Status(::grpc::StatusCode::DATA_LOSS,
                          "Unable to parse GrpcHostBufferStoreMetadata");
  }
  auto store = GetHostBufferStore(metadata.session_id());
  if (!store.ok()) {
    LOG(INFO) << "HostBufferStore failed to get host buffer store for session "
              << metadata.session_id() << ": " << store.status();
    return xla::ToGrpcStatus(store.status());
  }

  VLOG(3) << "HostBufferStore starting to receive data "
          << metadata.ShortDebugString();
  std::string data;
  data.reserve(metadata.buffer_size());

  GrpcHostBufferStoreRequest request;
  while (stream->Read(&request)) {
    data.append(request.data());
  }
  VLOG(3) << "HostBufferStore received all data "
          << metadata.ShortDebugString();
  if (data.size() != metadata.buffer_size()) {
    std::string error = absl::StrCat(
        "Potential data loss for host buffers: expected ",
        metadata.buffer_size(), " bytes but got ", data.size(), " bytes");
    LOG(WARNING) << "Bad request by proxy client: " << error;
    return ::grpc::Status(::grpc::StatusCode::DATA_LOSS, error);
  }

  absl::Status s = (*store)->Store(metadata.handle(), std::move(data));
  if (!s.ok()) {
    LOG(INFO) << "HostBufferStore for session " << metadata.session_id()
              << " failed to store buffer " << metadata.handle() << ": " << s;
  }
  return xla::ToGrpcStatus(std::move(s));
}

::grpc::Status GrpcServiceImpl::HostBufferLookup(
    ::grpc::ServerContext* context, const GrpcHostBufferLookupRequest* request,
    ::grpc::ServerWriter<GrpcHostBufferLookupResponse>* stream) {
  tsl::profiler::TraceMe traceme("HostBufferLookup");
  static constexpr int64_t kChunkSize = 1024 * 1024;

  auto store = GetHostBufferStore(request->session_id());
  if (!store.ok()) {
    LOG(WARNING) << "Returning '" << store.status()
                 << "' while attempting to retrieve proxy buffer store for "
                 << request->ShortDebugString();
    return xla::ToGrpcStatus(store.status());
  }
  auto data = (*store)->Lookup(request->handle());
  if (!data.ok()) {
    LOG(WARNING) << "Returning '" << data.status()
                 << "' while attempting to retrieve data for "
                 << request->ShortDebugString();
    return xla::ToGrpcStatus(data.status());
  }

  VLOG(3) << "HostBufferLookup starting to send data "
          << request->ShortDebugString();
  tsl::profiler::TraceMe trace_me_send_data([size = data.value()->size()]() {
    return tsl::profiler::TraceMeEncode("HostBufferLookup_Send",
                                        {{"size", size}});
  });
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
  VLOG(3) << "HostBufferLookup done sending data "
          << request->ShortDebugString();

  return ::grpc::Status::OK;
}

::grpc::Status GrpcServiceImpl::HostBufferDelete(
    ::grpc::ServerContext* context, const GrpcHostBufferDeleteRequest* request,
    GrpcHostBufferDeleteResponse* response) {
  tsl::profiler::TraceMe traceme("HostBufferDelete");
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
