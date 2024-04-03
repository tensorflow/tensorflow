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

#ifndef XLA_PYTHON_IFRT_PROXY_CLIENT_GRPC_HOST_BUFFER_H_
#define XLA_PYTHON_IFRT_PROXY_CLIENT_GRPC_HOST_BUFFER_H_

#include <atomic>
#include <cstdint>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include "xla/python/ifrt/future.h"
#include "xla/python/ifrt_proxy/client/host_buffer.h"
#include "xla/python/ifrt_proxy/common/grpc_ifrt_service.grpc.pb.h"
#include "tsl/platform/unbounded_work_queue.h"

namespace xla {
namespace ifrt {
namespace proxy {

class GrpcClientHostBufferStore : public ClientHostBufferStore {
 public:
  GrpcClientHostBufferStore(
      std::shared_ptr<grpc::GrpcIfrtService::StubInterface> stub,
      IfrtProxyVersion version, uint64_t session_id);

  ~GrpcClientHostBufferStore() override;

  // Implements ClientHostBufferStore.

  uint64_t NextHandle() override;
  Future<absl::Status> Store(uint64_t handle, absl::string_view data) override;
  Future<absl::Status> Store(uint64_t handle, const absl::Cord& data) override;
  Future<absl::StatusOr<absl::Cord>> Lookup(uint64_t handle) override;
  Future<absl::Status> Delete(uint64_t handle) override;

 private:
  const std::shared_ptr<grpc::GrpcIfrtService::StubInterface> stub_;
  const IfrtProxyVersion version_;
  const uint64_t session_id_;
  std::atomic<uint64_t> next_handle_ = 0;

  // Implementation note: `lookup_work_queue_` may have closures that invoke
  // user-defined code. Each `Lookup()` call is associated with a scheduled
  // closure, and the closure is used to first perform synchronous reads of the
  // streaming RPC, and then to do `promise.Set()` for the Future returned to
  // the caller.
  std::unique_ptr<tsl::UnboundedWorkQueue> lookup_work_queue_;
};

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_PROXY_CLIENT_GRPC_HOST_BUFFER_H_
