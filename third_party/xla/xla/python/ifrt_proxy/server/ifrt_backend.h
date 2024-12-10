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

#ifndef XLA_PYTHON_IFRT_PROXY_SERVER_IFRT_BACKEND_H_
#define XLA_PYTHON_IFRT_PROXY_SERVER_IFRT_BACKEND_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/array_spec.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/future.h"
#include "xla/python/ifrt/host_callback.h"
#include "xla/python/ifrt_proxy/common/ifrt_service.pb.h"
#include "xla/python/ifrt_proxy/server/host_buffer.h"
#include "xla/python/ifrt_proxy/server/host_callback.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "tsl/platform/threadpool.h"

namespace xla {
namespace ifrt {
namespace proxy {

// The abstract class `BackendInterface` defines the interface used by the IFRT
// service to interact with a variety of backend runtime system it can utilize.
class BackendInterface {
 public:
  virtual ~BackendInterface() = default;

  // Currently, responses (particularly those that carry buffer contents) can be
  // of non-trivial size. Once we figured out how best to move the data, we may
  // want to revise the shared_ptr below to the `IfrtResponse` proto itself.
  // Also, if and when we have a move-only Future in xla::ifrt, we may consider
  // changing it to std::unique_ptr.
  using Response = std::shared_ptr<IfrtResponse>;

  // Processes a given IFRT Request and returns a Future of an IfrtResponse.
  virtual Future<Response> Process(std::unique_ptr<IfrtRequest> request) = 0;
};

// IfrtBackend implements a backend that already has a linkable C++ client that
// conforms to the xla::ifrt API.
class IfrtBackend final : public BackendInterface {
 public:
  // Creates an returns an IfrtBackend that uses the given IFRT Client to
  // process the incoming proxy client requests. The `ifrt_client` param cannot
  // be a nullptr.
  static absl::StatusOr<std::unique_ptr<IfrtBackend>> Create(
      IfrtProxyVersion version, uint64_t session_id,
      std::shared_ptr<xla::ifrt::Client> ifrt_client,
      std::shared_ptr<HostBufferStore> host_buffer_store);

  ~IfrtBackend() override;

  // IFRT Proxy version negotiated between the client and the server.
  const IfrtProxyVersion& version() const { return version_; }

  Future<Response> Process(std::unique_ptr<IfrtRequest> request) override;

 private:
  // Generates unique handles for returning to the client. Guaranteed to return
  // handles that do not conflict with client-generated handles (via client-side
  // RpcHelper). All object types currently use this single "handle space".
  class HandleGenerator {
   public:
    explicit HandleGenerator(IfrtBackend* parent);

    // Returns the client-generated handle after performing some convenience
    // checks, provided that the client is of a protocol_version capable of
    // doing this. If the client has old protocol versions, generate a handle at
    // the server.
    absl::StatusOr<uint64_t> FromClientGenerated(uint64_t from_client);

    // Performs the same function as `FromClientGenerated` but in bulk, and
    // saves them into the provided Span.
    absl::Status FromClientGeneratedBulk(
        const tsl::protobuf::RepeatedField<uint64_t>& from_client,
        absl::Span<uint64_t> result_handles);

    uint64_t GenerateAtServer();

    void GenerateAtServerBulk(absl::Span<uint64_t> result_handles);

   private:
    IfrtBackend* const parent_;
    absl::Mutex mu_;
    uint64_t current_ ABSL_GUARDED_BY(mu_);
  };

  IfrtBackend(IfrtProxyVersion version, uint64_t session_id,
              std::shared_ptr<xla::ifrt::Client> ifrt_client,
              std::shared_ptr<HostBufferStore> host_buffer_store);

  // Executes the given function on the given thread pool and returns a future
  // that becomes ready when the function returns. If the thread pool is not
  // given, uses a default thread pool implementation that does not limit the
  // maximum number of threads.
  Future<Response> AsyncExecute(
      std::function<absl::StatusOr<Response>()> handle_fn,
      tsl::thread::ThreadPool* thread_pool = nullptr);

  Future<Response> ProcessInternal(std::unique_ptr<IfrtRequest> request);

  //////////////////////////////////////////////////////////////////////
  // Handlers for individual requests
  //

  absl::StatusOr<Response> HandleInit(std::unique_ptr<IfrtRequest> request);

  Future<Response> HandleCheckFutureRequest(
      std::unique_ptr<IfrtRequest> request);

  Future<Response> HandleCheckValueReadyRequest(
      std::unique_ptr<IfrtRequest> request);

  absl::StatusOr<Response> HandleMakeArrayFromHostBufferRequest(
      std::unique_ptr<IfrtRequest> request);
  absl::StatusOr<Response> HandleAssembleArrayFromSingleDeviceArraysRequest(
      std::unique_ptr<IfrtRequest> request);
  absl::StatusOr<Response> HandleRemapArraysRequest(
      std::unique_ptr<IfrtRequest> request);
  Future<Response> HandleCopyToHostBufferRequest(
      std::unique_ptr<IfrtRequest> request);
  absl::StatusOr<Response> HandleDisassembleIntoSingleDeviceArraysRequest(
      std::unique_ptr<IfrtRequest> request);
  absl::StatusOr<Response> HandleCopyArraysRequest(
      std::unique_ptr<IfrtRequest> request);
  absl::StatusOr<Response> HandleReshardRequest(
      std::unique_ptr<IfrtRequest> request);
  absl::StatusOr<Response> HandleFullyReplicatedShardRequest(
      std::unique_ptr<IfrtRequest> request);
  absl::StatusOr<Response> HandleDeleteArrayRequest(
      std::unique_ptr<IfrtRequest> request);
  absl::StatusOr<Response> HandleIsArrayDeletedRequest(
      std::unique_ptr<IfrtRequest> request);
  absl::StatusOr<Response> HandleDestructArrayRequest(
      std::unique_ptr<IfrtRequest> request);

  Future<Response> HandleCompileRequest(std::unique_ptr<IfrtRequest> request);

  Future<Response> HandleLoadedExecutableMetadataRequest(
      std::unique_ptr<IfrtRequest> request);
  absl::StatusOr<Response> HandleLoadedExecutableExecuteRequest(
      std::unique_ptr<IfrtRequest> request);
  absl::StatusOr<Response> HandleLoadedExecutableDeleteRequest(
      std::unique_ptr<IfrtRequest> request);
  absl::StatusOr<Response> HandleLoadedExecutableIsDeletedRequest(
      std::unique_ptr<IfrtRequest> request);
  absl::StatusOr<Response> HandleLoadedExecutableDestructRequest(
      std::unique_ptr<IfrtRequest> request);

  Future<Response> HandleLoadedHostCallbackPollRequest(
      std::unique_ptr<IfrtRequest> request);
  absl::StatusOr<Response> HandleLoadedHostCallbackReturnRequest(
      std::unique_ptr<IfrtRequest> request);

  absl::StatusOr<Response> HandleGetDefaultDeviceAssignmentRequest(
      std::unique_ptr<IfrtRequest> request);

  //////////////////////////////////////////////////////////////////////
  // Auxiliary/Helper methods for the handler methods above
  //

  Future<BackendInterface::Response> HandleCopyToStringHostBufferRequest(
      std::unique_ptr<IfrtRequest> request);

  //////////////////////////////////////////////////////////////////////
  // Convenient methods for object lookups
  //

  struct LoadedExecutableWithInfo;
  absl::StatusOr<std::shared_ptr<LoadedExecutableWithInfo>> GetLoadedExecutable(
      uint64_t handle);

  absl::StatusOr<tsl::RCReference<xla::ifrt::Array>> GetArray(uint64_t handle);
  absl::StatusOr<tsl::RCReference<xla::ifrt::Array>> GetArrayLocked(
      uint64_t handle) ABSL_SHARED_LOCKS_REQUIRED(arrays_mutex_);

  HandleGenerator handle_generator_;

  // Must not change during the life of this object.
  const IfrtProxyVersion version_;
  const uint64_t session_id_;
  const std::shared_ptr<xla::ifrt::Client> client_;
  const std::shared_ptr<HostBufferStore> host_buffer_store_;

  absl::Mutex futures_mutex_;
  absl::flat_hash_map<uint64_t, Future<>> futures_
      ABSL_GUARDED_BY(futures_mutex_);

  absl::Mutex arrays_mutex_;
  absl::flat_hash_map<uint64_t, tsl::RCReference<xla::ifrt::Array>> arrays_
      ABSL_GUARDED_BY(arrays_mutex_);

  absl::Mutex executables_mutex_;
  absl::flat_hash_map<uint64_t, std::shared_ptr<LoadedExecutableWithInfo>>
      executables_ ABSL_GUARDED_BY(executables_mutex_);

  absl::Mutex host_callback_queues_mutex_;
  absl::flat_hash_map<uint64_t, std::shared_ptr<RemoteLoadedHostCallbackQueue>>
      host_callback_queues_ ABSL_GUARDED_BY(host_callback_queues_mutex_);

  absl::Mutex host_callback_executions_mutex_;
  absl::flat_hash_map<uint64_t, RemoteLoadedHostCallbackQueue::ExecutionRequest>
      host_callback_executions_
          ABSL_GUARDED_BY(host_callback_executions_mutex_);

  absl::Mutex in_flight_count_mutex_;
  int64_t in_flight_count_ ABSL_GUARDED_BY(in_flight_count_mutex_) = 0;

  // Use a separate thread pool for compilation as XLA compilation often
  // requires a bigger stack.
  tsl::thread::ThreadPool compile_thread_pool_;

  class InOrderRequestsProcessor;
  std::unique_ptr<InOrderRequestsProcessor> in_order_requests_processor_;
};

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_PROXY_SERVER_IFRT_BACKEND_H_
