/*
 * Copyright 2023 The OpenXLA Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef XLA_PYTHON_IFRT_PROXY_SERVER_GRPC_SERVER_H_
#define XLA_PYTHON_IFRT_PROXY_SERVER_GRPC_SERVER_H_

#include <memory>
#include <string>
#include <utility>

#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "grpcpp/server.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt_proxy/common/grpc_ifrt_service.grpc.pb.h"

namespace xla {
namespace ifrt {
namespace proxy {

// Makes and runs a gRPC server with the given implementation and address.
// Destroying this object shuts down the underlying gRPC server, and so can
// block.
class GrpcServer {
 public:
  // The address parameter must be in the standard URI format - as needed by the
  // ::grpc::ServerBuilder::AddListentingPort. See the ::grpc::ServerBuilder
  // documentation for more details.
  static absl::StatusOr<std::unique_ptr<GrpcServer>> Create(
      absl::string_view address,
      std::unique_ptr<grpc::GrpcIfrtService::Service> impl);

  static absl::StatusOr<std::unique_ptr<GrpcServer>>
  CreateFromIfrtClientFactory(
      absl::string_view address,
      absl::AnyInvocable<absl::StatusOr<std::shared_ptr<xla::ifrt::Client>>()>
          backend_ifrt_client_factory);

  // Starts shutting down the server and waits until it properly shuts down.
  ~GrpcServer();

  // Address this server is listening on.
  std::string address() const { return address_; }

  // Blocks until the server shuts down.
  void Wait() { server_->Wait(); }

 private:
  GrpcServer(absl::string_view address,
             std::unique_ptr<grpc::GrpcIfrtService::Service> impl,
             std::unique_ptr<::grpc::Server> server)
      : address_(address), impl_(std::move(impl)), server_(std::move(server)) {}

  const std::string address_;  // Address this server is listening on.

  // Make sure that impl_ outlives the server_.
  std::unique_ptr<grpc::GrpcIfrtService::Service> impl_;
  std::unique_ptr<::grpc::Server> server_;
};

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_PROXY_SERVER_GRPC_SERVER_H_
