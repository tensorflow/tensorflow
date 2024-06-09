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

#include "xla/python/ifrt_proxy/server/grpc_server.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/functional/any_invocable.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "grpc/grpc.h"
#include "grpcpp/completion_queue.h"
#include "grpcpp/grpcpp.h"
#include "grpcpp/server_builder.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt_proxy/common/grpc_credentials.h"
#include "xla/python/ifrt_proxy/common/grpc_ifrt_service.grpc.pb.h"
#include "xla/python/ifrt_proxy/server/grpc_service_impl.h"
#include "xla/python/ifrt_proxy/server/host_buffer.h"
#include "xla/python/ifrt_proxy/server/ifrt_backend.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace ifrt {
namespace proxy {

GrpcServer::~GrpcServer() {
  server_->Shutdown();
  server_->Wait();
}

absl::StatusOr<std::unique_ptr<GrpcServer>> GrpcServer::Create(
    absl::string_view address,
    std::unique_ptr<grpc::GrpcIfrtService::Service> impl) {
  if (impl == nullptr) {
    return absl::InvalidArgumentError(
        "Service implementation cannot be a nullptr.");
  }

  ::grpc::ServerBuilder builder;
  // Remove message size limit to accommodate large messages exchanged during
  // model compilation.
  builder.AddChannelArgument(GRPC_ARG_MAX_SEND_MESSAGE_LENGTH, -1);
  builder.AddChannelArgument(GRPC_ARG_MAX_RECEIVE_MESSAGE_LENGTH, -1);
  builder.RegisterService(impl.get());
  builder.AddListeningPort(std::string(address), GetServerCredentials());
  auto server = builder.BuildAndStart();
  if (server == nullptr) {
    return absl::UnavailableError(
        absl::StrCat("Failed to initialize gRPC server at address:", address));
  }

  return absl::WrapUnique<GrpcServer>(
      new GrpcServer(address, std::move(impl), std::move(server)));
}

absl::StatusOr<std::unique_ptr<GrpcServer>>
GrpcServer::CreateFromIfrtClientFactory(
    absl::string_view address,
    absl::AnyInvocable<absl::StatusOr<std::shared_ptr<xla::ifrt::Client>>()>
        backend_ifrt_client_factory) {
  if (backend_ifrt_client_factory == nullptr) {
    return absl::InvalidArgumentError(
        "backend_ifrt_client_factory cannot be nullptr.");
  }

  auto service = std::make_unique<GrpcServiceImpl>(
      [ifrt_client_factory = std::move(backend_ifrt_client_factory)](
          IfrtProxyVersion version, uint64_t session_id,
          std::shared_ptr<HostBufferStore> host_buffer_store) mutable
      -> absl::StatusOr<std::unique_ptr<BackendInterface>> {
        TF_ASSIGN_OR_RETURN(auto ifrt_client, ifrt_client_factory());
        return IfrtBackend::Create(version, session_id, std::move(ifrt_client),
                                   std::move(host_buffer_store));
      });

  return Create(address, std::move(service));
}

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
