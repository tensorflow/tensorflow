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

#ifndef XLA_PYTHON_IFRT_PROXY_SERVER_GRPC_SERVICE_IMPL_H_
#define XLA_PYTHON_IFRT_PROXY_SERVER_GRPC_SERVICE_IMPL_H_

#include <atomic>
#include <cstdint>
#include <memory>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/die_if_null.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/status.h"
#include "grpcpp/support/sync_stream.h"
#include "xla/python/ifrt_proxy/common/grpc_ifrt_service.grpc.pb.h"
#include "xla/python/ifrt_proxy/common/grpc_ifrt_service.pb.h"
#include "xla/python/ifrt_proxy/common/ifrt_service.pb.h"
#include "xla/python/ifrt_proxy/server/host_buffer.h"
#include "xla/python/ifrt_proxy/server/ifrt_backend.h"

namespace xla {
namespace ifrt {
namespace proxy {

// Implementation for `GrpcIfrtService`.
class GrpcServiceImpl : public grpc::GrpcIfrtService::Service {
 public:
  using BackendFactory =
      absl::AnyInvocable<absl::StatusOr<std::unique_ptr<BackendInterface>>(
          IfrtProxyVersion version, uint64_t session_id,
          std::shared_ptr<xla::ifrt::proxy::HostBufferStore>
              host_buffer_store)>;

  explicit GrpcServiceImpl(BackendFactory backend_factory)
      : backend_factory_(ABSL_DIE_IF_NULL(std::move(backend_factory))) {}

  ::grpc::Status GetVersion(::grpc::ServerContext* context,
                            const GrpcGetVersionRequest* request,
                            GrpcGetVersionResponse* response) override;

  ::grpc::Status IfrtSession(
      ::grpc::ServerContext* context,
      ::grpc::ServerReaderWriter<IfrtResponse, IfrtRequest>* stream) override;

  ::grpc::Status HostBufferStore(
      ::grpc::ServerContext* context,
      ::grpc::ServerReader<GrpcHostBufferStoreRequest>* stream,
      GrpcHostBufferStoreResponse* response) override;

  ::grpc::Status HostBufferLookup(
      ::grpc::ServerContext* context,
      const GrpcHostBufferLookupRequest* request,
      ::grpc::ServerWriter<GrpcHostBufferLookupResponse>* stream) override;

  ::grpc::Status HostBufferDelete(
      ::grpc::ServerContext* context,
      const GrpcHostBufferDeleteRequest* request,
      GrpcHostBufferDeleteResponse* response) override;

  // Test-only method that adds a new session in the host buffer store map.
  // Returns false if the session id already exists.
  bool Test_InsertHostBufferStore(
      uint64_t session_id,
      std::shared_ptr<xla::ifrt::proxy::HostBufferStore> store);

  // Test-only method that removes the given session id from the host buffer
  // store map. Returns false if the session id does not exist.
  bool Test_DeleteHostBufferStore(uint64_t session_id);

 private:
  absl::StatusOr<std::shared_ptr<xla::ifrt::proxy::HostBufferStore>>
  GetHostBufferStore(uint64_t session_id)
      ABSL_LOCKS_EXCLUDED(host_buffer_store_mu_);

  BackendFactory backend_factory_;
  std::atomic<uint64_t> next_session_id_ = 1;

  absl::Mutex host_buffer_store_mu_;
  absl::flat_hash_map<uint64_t,
                      std::shared_ptr<xla::ifrt::proxy::HostBufferStore>>
      host_buffer_stores_ ABSL_GUARDED_BY(host_buffer_store_mu_);
};

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_PROXY_SERVER_GRPC_SERVICE_IMPL_H_
