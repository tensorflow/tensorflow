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

#ifndef XLA_PYTHON_IFRT_PROXY_CLIENT_RPC_HELPER_H_
#define XLA_PYTHON_IFRT_PROXY_CLIENT_RPC_HELPER_H_

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/python/ifrt/future.h"
#include "xla/python/ifrt_proxy/client/client_session.h"
#include "xla/python/ifrt_proxy/client/host_buffer.h"
#include "xla/python/ifrt_proxy/common/ifrt_service.pb.h"

namespace xla {
namespace ifrt {
namespace proxy {

// RpcHelper helps establish a connection with the IFRT server and perform
// logical RPCs on the connection.
//
// TODO(b/266635130): RpcHelper currently makes each logical RPC order-dependent
// on the previous RPC it was asked to make. Instead, allow users of RpcHelper
// specify the necessary dependency.
class RpcHelper {
 public:
  RpcHelper(IfrtProxyVersion version, std::shared_ptr<ClientSession> session)
      : version_(std::move(version)), session_(std::move(session)) {}

  void Disconnect();

  RpcHelper(const RpcHelper&) = delete;
  RpcHelper& operator=(const RpcHelper&) = delete;
  ~RpcHelper() { Disconnect(); }

  // IFRT Proxy version negotiated between the client and the server.
  const IfrtProxyVersion& version() const { return version_; }

  // Initializes the host buffer store for this RpcHelper instance. This must be
  // called exactly once during initialization before `host_buffer_store()` is
  // called.
  void set_host_buffer_store(
      std::shared_ptr<ClientHostBufferStore> host_buffer_store) {
    CHECK(host_buffer_store_ == nullptr);
    host_buffer_store_ = std::move(host_buffer_store);
  }

  const std::shared_ptr<ClientHostBufferStore>& host_buffer_store() const {
    return host_buffer_store_;
  }

  template <typename T>
  using ResponseFuture = Future<std::shared_ptr<T>>;

  // Wrapper function for various logical RPCs defined in ifrt_service.proto.
  // Whenever the RPC finishes, `on_done` will be called with the result or the
  // return status. `on_done` can be called with various locks held and should
  // return quickly without blocking on any event. `on_done` is guaranteed to be
  // called exactly once.
  //
  // The functions can be invoked after the connection is broken, but will
  // result in `on_done` getting called with an error (see
  // "WrapAsConnectionError" in `rpc_helper.cc`).

  ResponseFuture<InitResponse> Init(std::unique_ptr<InitRequest> req);
  ResponseFuture<GetDefaultDeviceAssignmentResponse> GetDefaultDeviceAssignment(
      std::unique_ptr<GetDefaultDeviceAssignmentRequest> req);

  ResponseFuture<CheckFutureResponse> CheckFuture(
      std::unique_ptr<CheckFutureRequest> req);

  ResponseFuture<MakeArrayFromHostBufferResponse> MakeArrayFromHostBuffer(
      std::unique_ptr<MakeArrayFromHostBufferRequest> req);
  ResponseFuture<AssembleArrayFromSingleDeviceArraysResponse>
  AssembleArrayFromSingleDeviceArrays(
      std::unique_ptr<AssembleArrayFromSingleDeviceArraysRequest> req);
  ResponseFuture<DisassembleIntoSingleDeviceArraysResponse>
  DisassembleIntoSingleDeviceArrays(
      std::unique_ptr<DisassembleIntoSingleDeviceArraysRequest> req);
  ResponseFuture<CopyToHostBufferResponse> CopyToHostBuffer(
      std::unique_ptr<CopyToHostBufferRequest> req);
  ResponseFuture<CheckArrayReadyResponse> CheckArrayReady(
      std::unique_ptr<CheckArrayReadyRequest> req);
  ResponseFuture<ReshardResponse> Reshard(std::unique_ptr<ReshardRequest> req);
  ResponseFuture<FullyReplicatedShardResponse> FullyReplicatedShard(
      std::unique_ptr<FullyReplicatedShardRequest> req);
  ResponseFuture<IsArrayDeletedResponse> IsArrayDeleted(
      std::unique_ptr<IsArrayDeletedRequest> req);
  ResponseFuture<DeleteArrayResponse> DeleteArray(
      std::unique_ptr<DeleteArrayRequest> req);
  ResponseFuture<DestructArrayResponse> DestructArray(
      std::unique_ptr<DestructArrayRequest> req);

  ResponseFuture<CompileResponse> Compile(std::unique_ptr<CompileRequest> req);

  ResponseFuture<LoadedExecutableMetadataResponse> LoadedExecutableMetadata(
      std::unique_ptr<LoadedExecutableMetadataRequest> req);
  ResponseFuture<LoadedExecutableExecuteResponse> LoadedExecutableExecute(
      std::unique_ptr<LoadedExecutableExecuteRequest> req);
  ResponseFuture<LoadedExecutableDeleteResponse> LoadedExecutableDelete(
      std::unique_ptr<LoadedExecutableDeleteRequest> req);
  ResponseFuture<LoadedExecutableIsDeletedResponse> LoadedExecutableIsDeleted(
      std::unique_ptr<LoadedExecutableIsDeletedRequest> req);
  ResponseFuture<LoadedExecutableDestructResponse> LoadedExecutableDestruct(
      std::unique_ptr<LoadedExecutableDestructRequest> req);

  ResponseFuture<LoadedHostCallbackPollResponse> LoadedHostCallbackPoll(
      std::unique_ptr<LoadedHostCallbackPollRequest> req);
  ResponseFuture<LoadedHostCallbackReturnResponse> LoadedHostCallbackReturn(
      std::unique_ptr<LoadedHostCallbackReturnRequest> req);

  // Utility functions for common functions.

  Future<> CheckFuture(uint64_t handle);

 private:
  RequestMetadata ManufactureRequestMetadata() ABSL_LOCKS_EXCLUDED(mu_);

  const IfrtProxyVersion version_;
  const std::shared_ptr<ClientSession> session_;
  std::shared_ptr<ClientHostBufferStore> host_buffer_store_;

  absl::Mutex mu_;
  uint64_t next_op_id_ ABSL_GUARDED_BY(mu_) = 1;
};

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_PROXY_CLIENT_RPC_HELPER_H_
